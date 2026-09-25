"""
ROI Watcher UI

A small local web app that watches a folder for ROI images dropped into
it (e.g. crops produced by the annotation pipeline, or anything else),
runs OCR on each new image as it appears, and writes an annotated copy
-- the image with the recognized text printed on it -- into a separate
output folder. The browser page polls for new results and displays
them live, newest first, and lets a reviewer correct low-confidence
predictions inline -- corrections are persisted to review_queue.csv
next to the watched folders, mirroring the
ocr_dataset_builder/parseq_review_queue.csv schema used elsewhere in
this project.

Run from the project root:

    venv\\Scripts\\python.exe roi_ui\\app.py [--config FILE] [--input DIR] [--output DIR] [--port N]

Folders, port, polling and display settings come from the roi_ui section of
config.yaml (--config selects another file); --input/--output/--port
override it for one run.
"""

import argparse
import itertools
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont

from src.config import load_config
from src.ocr.factory import create_backend

# Loaded once at import; every tunable below comes from config.yaml.
# (--config FILE re-points these in main() before anything is used.)
cfg = load_config()
ui = cfg.roi_ui

REVIEW_COLUMNS = [
    "id",
    "filename",
    "prediction",
    "confidence",
    "needs_review",
    "corrected_text",
    "reviewed",
    "processed_at",
]

app = Flask(__name__)

state_lock = threading.Lock()
results = []            # newest first
review_csv_path = None  # set in main()
_id_counter = itertools.count(1)

status = {
    "input_dir": None,
    "output_dir": None,
    "review_csv": None,
    "backend": None,
    "backend_ready": False,
    "backend_error": None,
    "processed_count": 0,
}


def _load_font(size):
    for name in ui.font_candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TEXT = None
FONT_SMALL = None


def _apply_config(new_cfg):
    """Point the module at a (possibly different) config and reload fonts."""
    global cfg, ui, FONT_TEXT, FONT_SMALL

    cfg = new_cfg
    ui = cfg.roi_ui

    status["backend"] = cfg.ocr.backend

    FONT_TEXT = _load_font(ui.font_size_text)
    FONT_SMALL = _load_font(ui.font_size_small)


_apply_config(cfg)


def annotate_image(image_path, text, confidence, error=None):
    """Return a copy of the image with the OCR result printed below it."""

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    canvas_w = max(w, ui.min_canvas_width)
    canvas = Image.new(
        "RGB", (canvas_w, h + ui.caption_height), tuple(ui.canvas_bg)
    )
    canvas.paste(img, ((canvas_w - w) // 2, 0))

    draw = ImageDraw.Draw(canvas)

    if error:
        draw.text(
            (ui.text_x, h + ui.error_y), "OCR FAILED",
            fill=tuple(ui.error_color), font=FONT_TEXT,
        )
        draw.text(
            (ui.text_x, h + ui.error_detail_y),
            str(error)[:ui.error_detail_max_chars],
            fill=tuple(ui.error_detail_color), font=FONT_SMALL,
        )
        return canvas

    if confidence >= ui.confidence_high:
        conf_color = tuple(ui.color_high)
    elif confidence >= ui.confidence_mid:
        conf_color = tuple(ui.color_mid)
    else:
        conf_color = tuple(ui.color_low)

    label = text.strip() if text and text.strip() else "(no text detected)"
    draw.text(
        (ui.text_x, h + ui.label_y), label,
        fill=tuple(ui.text_color), font=FONT_TEXT,
    )
    draw.text(
        (ui.text_x, h + ui.confidence_y),
        f"confidence {confidence:.2f}",
        fill=conf_color,
        font=FONT_SMALL,
    )

    return canvas


# --------------------------------------------------------------------
# Review queue persistence (review_queue.csv)
# --------------------------------------------------------------------

def append_review_row(entry):
    row = pd.DataFrame([[entry[c] for c in REVIEW_COLUMNS]], columns=REVIEW_COLUMNS)
    write_header = not review_csv_path.exists()

    row.to_csv(review_csv_path, mode="a", header=write_header, index=False)


def update_review_row(entry_id, corrected_text):
    if not review_csv_path.exists():
        return

    df = pd.read_csv(review_csv_path, dtype={"corrected_text": str})
    df["corrected_text"] = df["corrected_text"].fillna("")
    mask = df["id"] == entry_id

    if mask.any():
        df.loc[mask, "corrected_text"] = corrected_text
        df.loc[mask, "reviewed"] = True
        df.to_csv(review_csv_path, index=False)


def watch_loop(input_dir: Path, output_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[roi-ui] Loading OCR backend '{cfg.ocr.backend}' ...")

    try:
        backend = create_backend(cfg.ocr.backend, cfg.ocr)
    except Exception as exc:
        with state_lock:
            status["backend_error"] = str(exc)
        print(f"[roi-ui] Failed to load OCR backend: {exc}")
        return

    with state_lock:
        status["backend_ready"] = True

    print(f"[roi-ui] Backend ready. Watching {input_dir} ...")

    image_exts = set(cfg.image_extensions)

    seen = {}       # filename -> (size, mtime) already processed
    pending = {}     # filename -> ((size, mtime), stable_polls_count)

    while True:
        try:
            candidates = [
                p for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_exts
            ]
        except FileNotFoundError:
            time.sleep(ui.poll_seconds)
            continue

        current_names = set()

        for path in candidates:
            name = path.name
            current_names.add(name)

            try:
                stat = path.stat()
            except OSError:
                continue

            signature = (stat.st_size, stat.st_mtime)

            if seen.get(name) == signature:
                continue

            prev_sig, streak = pending.get(name, (None, 0))

            if prev_sig == signature:
                streak += 1
            else:
                streak = 1

            pending[name] = (signature, streak)

            if streak >= ui.stable_polls:
                process_file(backend, path, output_dir)
                seen[name] = signature
                pending.pop(name, None)

        # drop bookkeeping for files that were removed from the input dir
        for stale in list(pending.keys()):
            if stale not in current_names:
                pending.pop(stale, None)

        for stale in list(seen.keys()):
            if stale not in current_names:
                seen.pop(stale, None)

        time.sleep(ui.poll_seconds)


def process_file(backend, path: Path, output_dir: Path):
    print(f"[roi-ui] New ROI: {path.name}")

    text, confidence, error = "", 0.0, None

    try:
        result = backend.read(path)
        text = result.get("text", "")
        confidence = float(result.get("confidence", 0.0))
    except Exception as exc:
        error = str(exc)
        print(f"[roi-ui] OCR failed on {path.name}: {exc}")

    try:
        annotated = annotate_image(path, text, confidence, error=error)
        out_name = path.stem + ".png"
        annotated.save(output_dir / out_name)
    except Exception as exc:
        print(f"[roi-ui] Failed to annotate/save {path.name}: {exc}")
        out_name = None

    if not error:
        print(f"[roi-ui]   -> '{text}' (confidence {confidence:.2f})")

    needs_review = (not error) and confidence < cfg.review.confidence_threshold

    entry = {
        "id": next(_id_counter),
        "filename": path.name,
        "output_filename": out_name,
        "prediction": text,
        "confidence": confidence,
        "error": error,
        "needs_review": needs_review,
        "corrected_text": "",
        "reviewed": False,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    with state_lock:
        results.insert(0, entry)
        del results[ui.max_results:]
        status["processed_count"] += 1

    append_review_row(entry)


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        ui_settings={
            "confidence_high": ui.confidence_high,
            "confidence_mid": ui.confidence_mid,
            "status_poll_ms": ui.status_poll_ms,
            "results_poll_ms": ui.results_poll_ms,
        },
    )


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(status))


@app.route("/api/results")
def api_results():
    with state_lock:
        return jsonify(list(results))


@app.route("/api/correct", methods=["POST"])
def api_correct():
    data = request.get_json(force=True) or {}

    try:
        entry_id = int(data.get("id"))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid id"}), 400

    corrected_text = str(data.get("corrected_text", ""))

    with state_lock:
        entry = next((r for r in results if r["id"] == entry_id), None)

        if entry is None:
            return jsonify({"error": "not found"}), 404

        entry["corrected_text"] = corrected_text
        entry["reviewed"] = True
        updated = dict(entry)

    update_review_row(entry_id, corrected_text)

    return jsonify(updated)


@app.route("/input/<path:filename>")
def serve_input(filename):
    return send_from_directory(status["input_dir"], filename)


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(status["output_dir"], filename)


def main():
    global review_csv_path

    parser = argparse.ArgumentParser(description="ROI watcher UI")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (default: $OCR_CONFIG, else <repo>/config.yaml)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Folder to watch for new ROI images (default: roi_ui.input_dir).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Folder to write annotated result images to (default: roi_ui.output_dir).",
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port (default: roi_ui.port)."
    )
    args = parser.parse_args()

    if args.config is not None:
        _apply_config(load_config(args.config))

    input_dir = Path(args.input).resolve() if args.input else ui.input_dir
    output_dir = Path(args.output).resolve() if args.output else ui.output_dir
    port = args.port if args.port is not None else ui.port

    review_csv_path = output_dir.parent / ui.review_csv_name

    status["input_dir"] = str(input_dir)
    status["output_dir"] = str(output_dir)
    status["review_csv"] = str(review_csv_path)

    watcher = threading.Thread(
        target=watch_loop,
        args=(input_dir, output_dir),
        daemon=True,
    )
    watcher.start()

    app.run(host=ui.host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
