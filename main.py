import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.pdf_loader import PDFLoader
from src.deskewer import Deskewer
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector
from src.cropper import Cropper
from src.annotation_detector import AnnotationDetector
from src.preprocessor import Preprocessor
from src.roi_text_extractor import ROITextExtractor
from src.config import load_config
from src.annotation_remover import AnnotationRemover
import numpy as np


"""
Annotation Extraction Pipeline

End-to-end pipeline for processing scanned PDFs. The pipeline loads
each document, deskews pages, detects the separator line, crops the
annotation margin, preprocesses the image, detects handwritten
annotations, performs OCR and exports the extracted results.

Every parameter comes from config.yaml (or the file given with --config /
the OCR_CONFIG environment variable); there are no tunables in this file.
"""

parser = argparse.ArgumentParser(description="Annotation extraction pipeline")
parser.add_argument(
    "--config",
    default=None,
    help="Path to a config YAML (default: $OCR_CONFIG, else <repo>/config.yaml)",
)
args, _ = parser.parse_known_args()

cfg = load_config(args.config)

pipeline_cfg = cfg.pipeline
layout = cfg.layout

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = cfg.paths.input_dir
OUTPUT_DIR = cfg.paths.output_dir

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Modules
# --------------------------------------------------

loader = PDFLoader(cfg.pdf)

deskewer = Deskewer(cfg.deskew)

extractor = VerticalExtractor(cfg.vertical, verbose=pipeline_cfg.verbose)

line_detector = LineDetector(
    cfg.separator,
    cfg.visualization,
    verbose=pipeline_cfg.verbose,
)

cropper = Cropper(cfg.crop)

preprocessor = Preprocessor(cfg.preprocess)

annotation_detector = AnnotationDetector(
    cfg.annotation_detector,
    cfg.visualization,
    verbose=pipeline_cfg.verbose,
)

annotation_remover = AnnotationRemover(cfg.ink_removal)

text_extractor = ROITextExtractor(cfg.ocr)

summary = []

pdf_files = sorted(INPUT_DIR.glob(cfg.paths.pdf_glob))

print(f"\nFound {len(pdf_files)} PDF(s)\n")

# ==================================================
# Process each PDF
# ==================================================

for pdf_path in pdf_files:

    pdf_name = pdf_path.stem
    pdf_output = OUTPUT_DIR / pdf_name

    # Skip processing if output directory for this PDF already exists
    if pipeline_cfg.skip_existing_outputs and pdf_output.exists():
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
    preprocessed_dir = pdf_output / layout.preprocessed_dir
    recognizer_dir = pdf_output / layout.recognizer_dir
    roi_dir = pdf_output / layout.roi_dir
    ocr_dir = pdf_output / layout.ocr_dir
    annotation_removed_dir = pdf_output / layout.annotation_removed_dir

    for folder in [
        deskew_dir,
        vertical_dir,
        detected_dir,
        cropped_dir,
        preprocessed_dir,
        recognizer_dir,
        roi_dir,
        ocr_dir,
        failed_dir,
        annotation_removed_dir
    ]:
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    pages = loader.load_pdf(str(pdf_path))

    results = []

    # ==============================================
    # Process Pages
    # ==============================================

    for page_idx, page in enumerate(pages):

        print(f"\nPage {page_idx + 1}")

        # ------------------------------------------
        # Deskew
        # ------------------------------------------

        corrected, angle = deskewer.process(page)

        cv2.imwrite(
            str(deskew_dir / layout.page_image.format(page=page_idx + 1)),
            corrected
        )

        # ------------------------------------------
        # Vertical Extraction
        # ------------------------------------------

        vertical, debug = extractor.extract(corrected)

        cv2.imwrite(
            str(vertical_dir / layout.page_image.format(page=page_idx + 1)),
            vertical
        )

        # ------------------------------------------
        # Line Detection
        # ------------------------------------------

        detected, separator = line_detector.detect(vertical)

        cv2.imwrite(
            str(detected_dir / layout.page_image.format(page=page_idx + 1)),
            detected
        )

        if separator is None:

            print("Separator : FAILED")

            cv2.imwrite(
                str(failed_dir / layout.page_gray.format(page=page_idx + 1)),
                debug["gray"]
            )

            cv2.imwrite(
                str(failed_dir / layout.page_binary.format(page=page_idx + 1)),
                debug["binary"]
            )

            cv2.imwrite(
                str(failed_dir / layout.page_vertical.format(page=page_idx + 1)),
                vertical
            )

            results.append({

                "page": page_idx + 1,
                "status": "FAILED",
                "deskew_angle": angle,
                "separator_x": None,
                "width": None,
                "height": None

            })

            continue

        # ------------------------------------------
        # Crop Margin
        # ------------------------------------------

        crop = cropper.crop(
            corrected,
            separator
        )

        cv2.imwrite(
            str(cropped_dir / layout.page_image.format(page=page_idx + 1)),
            crop
        )

        # ------------------------------------------
        # Annotation Removal
        # ------------------------------------------

        if pipeline_cfg.use_annotation_removal:

            clean_crop, mask = annotation_remover.process_debug(crop)

        else:

            clean_crop = crop

            mask = np.zeros(
                crop.shape[:2],
                dtype=np.uint8
            )

        cv2.imwrite(
            str(
                annotation_removed_dir /
                layout.page_image.format(page=page_idx + 1)
            ),
            clean_crop
        )

        cv2.imwrite(
            str(
                annotation_removed_dir /
                layout.page_mask.format(page=page_idx + 1)
            ),
            mask
        )

        # ------------------------------------------
        # Preprocessing
        # ------------------------------------------

        gray, binary = preprocessor.process(clean_crop)

        cv2.imwrite(
            str(preprocessed_dir / layout.page_gray.format(page=page_idx + 1)),
            gray
        )

        cv2.imwrite(
            str(preprocessed_dir / layout.page_binary.format(page=page_idx + 1)),
            binary
        )

        # ------------------------------------------
        # OCR
        # ------------------------------------------

        if pipeline_cfg.use_annotation_detection:

            print("Mode : ROI OCR")

            rois, debug = annotation_detector.detect(gray, binary)

            if pipeline_cfg.save_rois:

                for i, roi in enumerate(rois):

                    cv2.imwrite(

                        str(
                            roi_dir /
                            layout.roi_image.format(
                                page=page_idx + 1,
                                roi=i + 1
                            )
                        ),

                        roi["roi"]

                    )
            
            if pipeline_cfg.save_debug_images:

                cv2.imwrite(

                    str(
                        recognizer_dir /
                        layout.page_image.format(page=page_idx + 1)
                    ),

                    debug["detected"]

                )

            extracted_text = text_extractor.extract(
                gray,
                rois
            )

        else:

            print("Mode : Full Margin OCR")

            extracted_text = text_extractor.extract(
                gray
            )

        # ------------------------------------------
        # Save OCR Results
        # ------------------------------------------

        pd.DataFrame(extracted_text).to_csv(
            ocr_dir / layout.ocr_csv.format(page=page_idx + 1),
            index=False
        )

        print(f"Separator : {separator['x']}")
        results.append({

            "page": page_idx + 1,
            "status": "SUCCESS",
            "deskew_angle": angle,
            "separator_x": separator["x"],
            "width": separator["w"],
            "height": separator["h"]

        })

    # ==============================================
    # Save Report
    # ==============================================

    df = pd.DataFrame(results)

    df.to_csv(
        pdf_output / layout.report_csv,
        index=False
    )

    success = df[df["status"] == "SUCCESS"]

    summary.append({

        "pdf": pdf_name,

        "pages": len(df),

        "success":

            (df["status"] == "SUCCESS").sum(),

        "failed":

            (df["status"] == "FAILED").sum(),

        "mean_separator_x":

            success["separator_x"].mean()

            if len(success) else None,

        "std_separator_x":

            success["separator_x"].std()

            if len(success) else None

    })

# ==================================================
# Save Summary
# ==================================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT_DIR / layout.summary_csv,
    index=False
)

print("\n")
print("=" * 70)
print("Pipeline Finished")
print("=" * 70)