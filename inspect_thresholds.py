"""
Threshold Inspector

Runs the same preprocessing and annotation-detection stages as main.py
on a PDF (or every PDF in a folder) and reports, per page, every detected
connected component alongside the pipeline threshold it was checked
against - so you can see exactly which marks get kept and which get
dropped, and why.

Reads every threshold live from the actual Preprocessor/AnnotationDetector
instances, so it always reflects whatever is currently configured in
src/preprocessor.py and src/annotation_detector.py.

Also renders a debug image per page into threshold_debug/<pdf_name>/
with every candidate annotation box drawn: green for accepted (would
trigger an OCR call), red for rejected, labeled with which threshold(s)
it failed.

Usage:
    python inspect_thresholds.py data/input_pdfs/Doc0680.pdf
    python inspect_thresholds.py data/input_pdfs/Doc0680.pdf --pages 4,5
    python inspect_thresholds.py data/input_pdfs/Doc0680.pdf --out report.csv
    python inspect_thresholds.py data/input_pdfs/Doc0680.pdf --no-images
    python inspect_thresholds.py data/input_pdfs
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.pdf_loader import PDFLoader
from src.deskewer import Deskewer
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector
from src.cropper import Cropper
from src.annotation_remover import AnnotationRemover
from src.preprocessor import Preprocessor
from src.annotation_detector import AnnotationDetector
from src.config import load_config


def raw_components(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    return num_labels, labels, stats


REASON_CODES = {
    "bbox_area>=min_confident_area": "AREA",
    "w>=min_confident_dim": "W",
    "h>=min_confident_dim": "H",
    "ink_pixels>=min_ink_pixels": "INK",
    "fill_ratio>=min_fill_ratio": "FILL",
}


def save_debug_image(gray, candidates, out_path, style):
    """
    candidates: list of {"bbox": (x1,y1,x2,y2), "accepted": bool, "reasons": [str]}
    style: the inspect_thresholds section of config.yaml
    """

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for item in candidates:
        x1, y1, x2, y2 = item["bbox"]

        if item["accepted"]:
            color = tuple(style.accepted_color)
            label = "OK"
        else:
            color = tuple(style.rejected_color)
            label = "/".join(item["reasons"])

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, style.box_thickness)

        cv2.putText(
            vis, label, (x1, max(style.label_min_y, y1 - style.label_offset)),
            cv2.FONT_HERSHEY_SIMPLEX, style.font_scale, color,
            style.font_thickness, cv2.LINE_AA
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def inspect_page(
    pdf_name,
    page_idx,
    page,
    deskewer,
    extractor,
    line_detector,
    cropper,
    annotation_remover,
    preprocessor,
    annotation_detector,
    cfg,
    debug_dir=None,
):
    rows = []

    corrected, _ = deskewer.process(page)
    vertical, _ = extractor.extract(corrected)
    _, separator = line_detector.detect(vertical)

    if separator is None:
        print(f"{pdf_name} page {page_idx + 1}: separator not found, skipping")
        return rows

    crop = cropper.crop(corrected, separator)

    if cfg.pipeline.use_annotation_removal:
        clean_crop, _ = annotation_remover.process_debug(crop)
    else:
        clean_crop = crop

    gray, clean_binary = preprocessor.process(clean_crop)

    # ------------------------------------------------------------
    # Stage 1: Preprocessor.min_area filter
    # ------------------------------------------------------------

    gray2 = (
        cv2.cvtColor(clean_crop, cv2.COLOR_BGR2GRAY)
        if len(clean_crop.shape) == 3
        else clean_crop.copy()
    )

    _, otsu_binary = cv2.threshold(
        gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    num_labels, labels, stats = raw_components(otsu_binary)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        rows.append({
            "pdf": pdf_name,
            "page": page_idx + 1,
            "stage": "preprocessor_min_area",
            "component_id": i,
            "x": x, "y": y, "w": w, "h": h,
            "measured_value": int(area),
            "threshold_name": "min_area",
            "threshold_value": preprocessor.min_area,
            "passed": bool(area >= preprocessor.min_area),
        })

    if not cfg.pipeline.use_annotation_detection:
        return rows

    # ------------------------------------------------------------
    # Stage 2: AnnotationDetector thresholds
    # ------------------------------------------------------------

    ad = annotation_detector

    binary_no_lines = ad.remove_lines(clean_binary)

    num_labels, labels, stats = raw_components(binary_no_lines)

    boxes = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        passed_w = w >= ad.min_width
        passed_h = h >= ad.min_height

        rows.append({
            "pdf": pdf_name,
            "page": page_idx + 1,
            "stage": "annotation_detector_min_dims",
            "component_id": i,
            "x": x, "y": y, "w": w, "h": h,
            "measured_value": f"w={w},h={h}",
            "threshold_name": "min_width & min_height",
            "threshold_value": f"{ad.min_width} & {ad.min_height}",
            "passed": bool(passed_w and passed_h),
        })

        if passed_w and passed_h:
            boxes.append((x, y, x + w, y + h))

    merged = ad._merge_boxes(boxes)
    merged = ad._refine_rois(merged, binary_no_lines)

    debug_candidates = []

    for x1, y1, x2, y2 in merged:
        w = x2 - x1
        h = y2 - y1
        bbox_area = w * h

        ink_pixels = cv2.countNonZero(binary_no_lines[y1:y2, x1:x2])
        fill_ratio = ink_pixels / float(bbox_area) if bbox_area else 0.0

        checks = [
            ("bbox_area>=min_confident_area", bbox_area, ad.min_confident_area),
            ("w>=min_confident_dim", w, ad.min_confident_dim),
            ("h>=min_confident_dim", h, ad.min_confident_dim),
            ("ink_pixels>=min_ink_pixels", ink_pixels, ad.min_ink_pixels),
            ("fill_ratio>=min_fill_ratio", round(fill_ratio, 3), ad.min_fill_ratio),
        ]

        component_id = f"{x1}_{y1}_{x2}_{y2}"
        accepted = True
        failed_reasons = []

        for name, measured, threshold in checks:
            passed = measured >= threshold
            accepted = accepted and passed

            if not passed:
                failed_reasons.append(REASON_CODES[name])

            rows.append({
                "pdf": pdf_name,
                "page": page_idx + 1,
                "stage": "annotation_detector_confidence",
                "component_id": component_id,
                "x": x1, "y": y1, "w": w, "h": h,
                "measured_value": measured,
                "threshold_name": name,
                "threshold_value": threshold,
                "passed": bool(passed),
            })

        rows.append({
            "pdf": pdf_name,
            "page": page_idx + 1,
            "stage": "annotation_detector_final",
            "component_id": component_id,
            "x": x1, "y": y1, "w": w, "h": h,
            "measured_value": "ACCEPT" if accepted else "REJECT",
            "threshold_name": "ALL",
            "threshold_value": "-",
            "passed": bool(accepted),
        })

        debug_candidates.append({
            "bbox": (x1, y1, x2, y2),
            "accepted": accepted,
            "reasons": failed_reasons,
        })

    if debug_dir is not None:
        save_debug_image(
            gray, debug_candidates,
            debug_dir / cfg.layout.page_image.format(page=page_idx + 1),
            cfg.inspect_thresholds,
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Inspect per-component thresholds for a PDF."
    )
    parser.add_argument(
        "pdf", help="Path to a PDF file, or a folder of PDFs (e.g. data/input_pdfs)"
    )
    parser.add_argument(
        "--pages",
        help="Comma-separated 1-based processed-page numbers to inspect "
             "(matches the page_NNN numbering in outputs/, default: all)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a config YAML (default: $OCR_CONFIG, else <repo>/config.yaml)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path (default: inspect_thresholds.report_csv)",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Base folder for accept/reject debug images; each PDF gets "
             "its own <debug-dir>/<pdf_name>/ subfolder "
             "(default: inspect_thresholds.debug_dir)",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Skip rendering debug images, only write the CSV",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    verbose = cfg.pipeline.verbose

    loader = PDFLoader(cfg.pdf)
    deskewer = Deskewer(cfg.deskew)
    extractor = VerticalExtractor(cfg.vertical, verbose=verbose)
    line_detector = LineDetector(cfg.separator, cfg.visualization, verbose=verbose)
    cropper = Cropper(cfg.crop)
    annotation_remover = AnnotationRemover(cfg.ink_removal)
    preprocessor = Preprocessor(cfg.preprocess)
    annotation_detector = AnnotationDetector(
        cfg.annotation_detector, cfg.visualization, verbose=verbose
    )

    out_csv = Path(args.out) if args.out else cfg.inspect_thresholds.report_csv

    pdf_arg = Path(args.pdf)

    if pdf_arg.is_dir():
        pdf_paths = sorted(pdf_arg.glob(cfg.paths.pdf_glob))
        if not pdf_paths:
            raise SystemExit(f"No PDFs found in {pdf_arg}")
    else:
        pdf_paths = [pdf_arg]

    wanted = None
    if args.pages:
        wanted = {int(p) for p in args.pages.split(",")}

    debug_base = None if args.no_images else (
        Path(args.debug_dir) if args.debug_dir
        else cfg.inspect_thresholds.debug_dir
    )

    all_rows = []

    for pdf_path in pdf_paths:

        pdf_name = pdf_path.stem

        print(f"\n=== {pdf_name} ===")

        pages = loader.load_pdf(str(pdf_path))

        debug_dir = debug_base / pdf_name if debug_base is not None else None

        for page_idx, page in enumerate(pages):

            if wanted is not None and (page_idx + 1) not in wanted:
                continue

            print(f"Inspecting {pdf_name} page {page_idx + 1}...")

            rows = inspect_page(
                pdf_name, page_idx, page, deskewer, extractor, line_detector,
                cropper, annotation_remover, preprocessor, annotation_detector,
                cfg, debug_dir=debug_dir,
            )

            all_rows.extend(rows)

        if debug_dir is not None:
            print(f"Debug images written to {debug_dir}")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    print(f"\nWrote {len(df)} threshold checks to {out_csv}")

    if len(df):
        fail_summary = (
            df[~df["passed"]]
            .groupby(["pdf", "stage", "threshold_name"])
            .size()
            .rename("fail_count")
        )

        print("\nFailure counts by pdf/stage/threshold:")
        print(fail_summary)


if __name__ == "__main__":
    main()
