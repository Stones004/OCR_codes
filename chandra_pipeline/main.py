"""
Chandra experiment pipeline (self-contained; delete this folder to remove it).

Same front half as the repo's main.py -- same PDFs, same shared config
(deskew, separator, crop) -- then, instead of ROI detection + PARSeq:

  1. the deskewed page with everything right of the separator blanked goes to
     Chandra (LM Studio): the question-number boxes are drawn and read;
  2. the page with the margin blanked goes to Chandra: the answer text;
  3. each answer is filed under the question whose row it sits in.

Run from anywhere:
    python chandra_pipeline/main.py [--config shared.yaml] [--chandra-config file.yaml]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# src.* lives in the repo root; this folder's own modules sit next to this file.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import cv2
import pandas as pd

from src.config import load_config
from src.pdf_loader import PDFLoader
from src.deskewer import Deskewer
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector
from src.cropper import Cropper

from chandra_config import load_chandra_config
from chandra_backend import ChandraBackend
from margin_qa_extractor import MarginQAExtractor

parser = argparse.ArgumentParser(description="Chandra margin question + answer extraction")
parser.add_argument("--config", default=None, help="Shared repo config (default: repo config.yaml)")
parser.add_argument("--chandra-config", default=None, help="This folder's config.yaml")
parser.add_argument("--pdf", default=None, help="Only process this PDF (file stem, e.g. Doc0729)")
args, _ = parser.parse_known_args()

cfg = load_config(args.config)
ccfg = load_chandra_config(args.chandra_config)

layout = cfg.layout
clayout = ccfg.layout
verbose = ccfg.run.verbose

INPUT_DIR = cfg.paths.input_dir
OUTPUT_DIR = ccfg.paths.output_dir

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

loader = PDFLoader(cfg.pdf)
deskewer = Deskewer(cfg.deskew)
extractor = VerticalExtractor(cfg.vertical, verbose=verbose)
line_detector = LineDetector(cfg.separator, cfg.visualization, verbose=verbose)
cropper = Cropper(cfg.crop)

chandra = ChandraBackend(ccfg.chandra)
margin_qa = MarginQAExtractor(ccfg.margin_qa)

summary = []

pdf_files = sorted(INPUT_DIR.glob(cfg.paths.pdf_glob))

if args.pdf:
    pdf_files = [p for p in pdf_files if p.stem == args.pdf]

print(f"\nFound {len(pdf_files)} PDF(s)\n")

for pdf_path in pdf_files:

    pdf_name = pdf_path.stem
    pdf_output = OUTPUT_DIR / pdf_name

    if ccfg.run.skip_existing_outputs and pdf_output.exists():
        print("=" * 70)
        print(f"Skipping (Already Processed - Folder Exists): {pdf_name}")
        print("=" * 70)
        continue

    print("=" * 70)
    print(f"Processing : {pdf_name}")
    print("=" * 70)

    deskew_dir = pdf_output / layout.deskewed_dir
    vertical_dir = pdf_output / layout.vertical_dir
    detected_dir = pdf_output / layout.detected_dir
    cropped_dir = pdf_output / layout.cropped_dir
    failed_dir = pdf_output / layout.failed_dir
    chandra_dir = pdf_output / clayout.chandra_dir

    for folder in [deskew_dir, vertical_dir, detected_dir, cropped_dir, failed_dir, chandra_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    pages = loader.load_pdf(str(pdf_path))

    results = []
    qa_rows = []

    for page_idx, page in enumerate(pages):

        page_no = page_idx + 1

        print(f"\nPage {page_no}")

        # ------------------------------------------
        # Same front half as main.py
        # ------------------------------------------

        corrected, angle = deskewer.process(page)
        cv2.imwrite(str(deskew_dir / layout.page_image.format(page=page_no)), corrected)

        vertical, debug = extractor.extract(corrected)
        cv2.imwrite(str(vertical_dir / layout.page_image.format(page=page_no)), vertical)

        detected, separator = line_detector.detect(vertical)
        cv2.imwrite(str(detected_dir / layout.page_image.format(page=page_no)), detected)

        if separator is None:

            print("Separator : FAILED")

            cv2.imwrite(str(failed_dir / layout.page_gray.format(page=page_no)), debug["gray"])
            cv2.imwrite(str(failed_dir / layout.page_binary.format(page=page_no)), debug["binary"])
            cv2.imwrite(str(failed_dir / layout.page_vertical.format(page=page_no)), vertical)

            results.append({
                "page": page_no,
                "status": "FAILED",
                "deskew_angle": angle,
                "separator_x": None,
                "width": None,
                "height": None,
            })

            continue

        crop = cropper.crop(corrected, separator)
        cv2.imwrite(str(cropped_dir / layout.page_image.format(page=page_no)), crop)

        # ------------------------------------------
        # Chandra: margin-only page -> boxed questions, answers-only page -> answers
        # ------------------------------------------

        # Two full-height, page-shaped images: Chandra's boxes were unreliable on the
        # 1:30 margin strip, and one whole-page call merges "a-1" into its answer line.
        # Blanking the other side keeps the page geometry, so boxes are in page pixels.
        split_x = crop.shape[1]

        margin_page = corrected.copy()
        margin_page[:, split_x:] = 255

        answers_page = corrected.copy()
        answers_page[:, :split_x] = 255

        margin_blocks = chandra.read(margin_page)
        margin_raw = chandra.last_raw
        margin_finish = (chandra.last_finish_reason, chandra.last_max_side)

        answer_blocks = chandra.read(answers_page)
        answer_raw = chandra.last_raw
        answer_finish = (chandra.last_finish_reason, chandra.last_max_side)

        for side, blocks, finish in (
            ("margin", margin_blocks, margin_finish),
            ("answers", answer_blocks, answer_finish),
        ):
            if not blocks:
                print(f"WARNING : Chandra returned no blocks for the {side} (finish_reason={finish[0]})")
            elif verbose:
                print(f"Chandra {side} : {len(blocks)} blocks (sent at {finish[1]} px)")

        if ccfg.margin_qa.save_raw_response:
            (chandra_dir / clayout.margin_raw.format(page=page_no)).write_text(margin_raw, encoding="utf-8")
            (chandra_dir / clayout.answers_raw.format(page=page_no)).write_text(answer_raw, encoding="utf-8")

        questions, answers = margin_qa.split(margin_blocks, answer_blocks)
        rows = margin_qa.pair(questions, answers)

        cv2.imwrite(
            str(chandra_dir / clayout.margin_image.format(page=page_no)),
            margin_qa.draw(corrected, questions),
        )

        page_rows = [
            {
                "page": page_no,
                "question": row["question"],
                "bbox": row["question_bbox"],
                "answer": row["answer"],
            }
            for row in rows
        ]

        pd.DataFrame(page_rows, columns=["page", "question", "bbox", "answer"]).to_csv(
            chandra_dir / clayout.qa_csv.format(page=page_no), index=False
        )

        qa_rows.extend(page_rows)

        print(f"Chandra : {len(questions)} margin question(s), {len(answers)} answer block(s)")
        print(f"Separator : {separator['x']}")

        results.append({
            "page": page_no,
            "status": "SUCCESS",
            "deskew_angle": angle,
            "separator_x": separator["x"],
            "width": separator["w"],
            "height": separator["h"],
        })

    pd.DataFrame(qa_rows, columns=["page", "question", "bbox", "answer"]).to_csv(
        chandra_dir / clayout.summary_csv, index=False
    )

    df = pd.DataFrame(results)
    df.to_csv(pdf_output / layout.report_csv, index=False)

    success = df[df["status"] == "SUCCESS"]

    summary.append({
        "pdf": pdf_name,
        "pages": len(df),
        "success": (df["status"] == "SUCCESS").sum(),
        "failed": (df["status"] == "FAILED").sum(),
        "mean_separator_x": success["separator_x"].mean() if len(success) else None,
        "std_separator_x": success["separator_x"].std() if len(success) else None,
    })

pd.DataFrame(summary).to_csv(OUTPUT_DIR / layout.summary_csv, index=False)

print("\n")
print("=" * 70)
print("Pipeline Finished")
print("=" * 70)
