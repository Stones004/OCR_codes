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
from src.config import PipelineConfig
from src.annotation_remover import AnnotationRemover
import numpy as np


"""
Annotation Extraction Pipeline

End-to-end pipeline for processing scanned PDFs. The pipeline loads
each document, deskews pages, detects the separator line, crops the
annotation margin, preprocesses the image, detects handwritten
annotations, performs OCR and exports the extracted results.
"""

config = PipelineConfig()
# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = Path("data/input_pdfs")
#INPUT_DIR = Path(r"G:\ICT_Scripts")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Modules
# --------------------------------------------------

loader = PDFLoader(
    dpi=300,
    save_debug=False,
    # This session's page numbering (and all the page-specific bug
    # traces done against it) assumes every PDF page is loaded, so
    # keep the default skip_pages=2 (added for a different machine's
    # dataset) from silently shifting every output page_NNN label by 2.
    skip_pages=0
)

deskewer = Deskewer()

extractor = VerticalExtractor(
    threshold=180,
    kernel_height=None
)

line_detector = LineDetector()

cropper = Cropper(
    right_padding=5
)

preprocessor = Preprocessor(
)

annotation_detector = AnnotationDetector()

annotation_remover = AnnotationRemover()

#text_extractor = ROITextExtractor(
#    backend=config.OCR_BACKEND
#)

summary = []

pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDF(s)\n")

# ==================================================
# Process each PDF
# ==================================================

for pdf_path in pdf_files:

    pdf_name = pdf_path.stem
    pdf_output = OUTPUT_DIR / pdf_name

    # Skip processing if output directory for this PDF already exists
    if pdf_output.exists():
        print("=" * 70)
        print(f"Skipping (Already Processed - Folder Exists): {pdf_name}")
        print("=" * 70)
        continue

    print("=" * 70)
    print(f"Processing : {pdf_name}")
    print("=" * 70)

    deskew_dir = pdf_output / "deskewed"
    vertical_dir = pdf_output / "vertical"
    detected_dir = pdf_output / "detected"
    cropped_dir = pdf_output / "cropped"
    failed_dir = pdf_output / "failed"
    preprocessed_dir = pdf_output / "preprocessed"
    recognizer_dir = pdf_output / "recognizer"
    roi_dir = pdf_output / "roi"
    ocr_dir = pdf_output / "ocr"
    annotation_removed_dir = pdf_output / "annotation_removed"

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
            str(deskew_dir / f"page_{page_idx+1:03d}.png"),
            corrected
        )

        # ------------------------------------------
        # Vertical Extraction
        # ------------------------------------------

        vertical, debug = extractor.extract(corrected)

        cv2.imwrite(
            str(vertical_dir / f"page_{page_idx+1:03d}.png"),
            vertical
        )

        # ------------------------------------------
        # Line Detection
        # ------------------------------------------

        detected, separator = line_detector.detect(vertical)

        cv2.imwrite(
            str(detected_dir / f"page_{page_idx+1:03d}.png"),
            detected
        )

        if separator is None:

            print("Separator : FAILED")

            cv2.imwrite(
                str(failed_dir / f"page_{page_idx+1:03d}_gray.png"),
                debug["gray"]
            )

            cv2.imwrite(
                str(failed_dir / f"page_{page_idx+1:03d}_binary.png"),
                debug["binary"]
            )

            cv2.imwrite(
                str(failed_dir / f"page_{page_idx+1:03d}_vertical.png"),
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
            str(cropped_dir / f"page_{page_idx+1:03d}.png"),
            crop
        )

        # ------------------------------------------
        # Annotation Removal
        # ------------------------------------------

        if config.USE_ANNOTATION_REMOVAL:

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
                f"page_{page_idx+1:03d}.png"
            ),
            clean_crop
        )

        cv2.imwrite(
            str(
                annotation_removed_dir /
                f"page_{page_idx+1:03d}_mask.png"
            ),
            mask
        )

        # ------------------------------------------
        # Preprocessing
        # ------------------------------------------

        gray, binary = preprocessor.process(clean_crop)

        cv2.imwrite(
            str(preprocessed_dir / f"page_{page_idx + 1:03d}_gray.png"),
            gray
        )

        cv2.imwrite(
            str(preprocessed_dir / f"page_{page_idx + 1:03d}_binary.png"),
            binary
        )

        # ------------------------------------------
        # OCR
        # ------------------------------------------

        if config.USE_ANNOTATION_DETECTION:

            print("Mode : ROI OCR")

            rois, debug = annotation_detector.detect(gray, binary)

            if config.SAVE_ROIS:

                for i, roi in enumerate(rois):

                    cv2.imwrite(

                        str(
                            roi_dir /
                            f"page_{page_idx+1:03d}_roi_{i+1}.png"
                        ),

                        roi["roi"]

                    )
            
            if config.SAVE_DEBUG_IMAGES:

                cv2.imwrite(

                    str(
                        recognizer_dir /
                        f"page_{page_idx+1:03d}.png"
                    ),

                    debug["detected"]

                )

            #extracted_text = text_extractor.extract(
            #    gray,
            #    rois
            #)

        else:

            print("Mode : Full Margin OCR")

            #extracted_text = text_extractor.extract(
            #    gray
            #)

        # ------------------------------------------
        # Save OCR Results
        # ------------------------------------------

        

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
        pdf_output / "report.csv",
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
    OUTPUT_DIR / "summary.csv",
    index=False
)

print("\n")
print("=" * 70)
print("Pipeline Finished")
print("=" * 70)