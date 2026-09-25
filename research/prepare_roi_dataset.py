"""
prepare_roi_dataset.py

Creates a dataset of cropped left-margin ROIs from scanned PDFs.

Pipeline

PDF
 ↓
Render Page
 ↓
Deskew
 ↓
Preprocess
 ↓
Vertical Extraction
 ↓
Line Detection
 ↓
Crop ROI
 ↓
Save ROI

Output

data/
    roi_dataset/
        Paper1_page_001.png
        Paper1_page_002.png
        ...
"""

import argparse

import cv2
import fitz
import numpy as np
from tqdm import tqdm

from src.config import CropConfig, load_config
from src.deskewer import Deskewer
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector
from src.cropper import Cropper
from src.annotation_detector import AnnotationDetector
from src.pdf_loader import PDF_POINTS_PER_INCH

# ============================================================
# Configuration (config.yaml: research.prepare_roi_dataset + shared stages)
# ============================================================

cfg = None            # set in main()
settings = None       # cfg.research.prepare_roi_dataset

# Pipeline stages, built from config by build_components()
deskewer = None
vertical_extractor = None
line_detector = None
cropper = None
annotation_detector = None


# ============================================================
# PDF Renderer
# ============================================================

def pdf_to_images(pdf_path):

    document = fitz.open(pdf_path)

    zoom = cfg.pdf.dpi / PDF_POINTS_PER_INCH
    matrix = fitz.Matrix(zoom, zoom)

    for page_number in range(settings.skip_pages, len(document)):

        page = document.load_page(page_number)

        pix = page.get_pixmap(matrix=matrix)

        image = np.frombuffer(
            pix.samples,
            dtype=np.uint8
        ).reshape(
            pix.height,
            pix.width,
            pix.n
        )

        if pix.n == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR
            )
        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        yield page_number + 1, image

    document.close()


# ============================================================
# Process One Page
# ============================================================

def process_page(image):

    # --------------------------------------------------------
    # 1. Deskew
    # --------------------------------------------------------

    deskewed, angle = deskewer.process(image)

    # --------------------------------------------------------
    # 3. Vertical Line Extraction
    # --------------------------------------------------------

    vertical, _ = vertical_extractor.extract(deskewed)

    # --------------------------------------------------------
    # 4. Separator Detection
    # --------------------------------------------------------

    _, separator = line_detector.detect(vertical)

    if separator is None:
        return []

    # --------------------------------------------------------
    # 5. Crop ROI
    # --------------------------------------------------------

    strip = cropper.crop(deskewed, separator)

    if strip is None:
        return []

    gray = cv2.cvtColor(
        strip,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(

        gray,

        0,

        255,

        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

    )

    rois, _ = annotation_detector.detect(
        gray,
        binary
    )

    return rois


# ============================================================
# Process One PDF
# ============================================================

def process_pdf(pdf_path):

    pdf_name = pdf_path.stem

    print(f"\nProcessing {pdf_name}")

    saved = 0

    failed = 0

    for page_number, image in pdf_to_images(pdf_path):

        try:

            rois = process_page(image)

        except Exception as e:

            print(
                f"Page {page_number:03d} failed : {e}"
            )

            failed += 1
            continue

        if len(rois) == 0:

            print(
                f"Page {page_number:03d}: No annotations detected"
            )

            failed += 1
            continue

        for roi_number, roi in enumerate(rois, start=1):

            filename = settings.roi_filename.format(
                pdf=pdf_name,
                page=page_number,
                roi=roi_number
            )

            output_path = settings.output_dir / filename

            cv2.imwrite(

                str(output_path),

                roi["roi"]

            )

            saved += 1

    print(
        f"  Saved {saved} ROI(s)"
    )

    if failed:
        print(
            f"  Failed {failed} page(s)"
        )


# ============================================================
# Main
# ============================================================

def build_components():
    """Create the pipeline stages from config.yaml (module-level singletons)."""

    global deskewer, vertical_extractor, line_detector, cropper
    global annotation_detector

    verbose = cfg.pipeline.verbose

    deskewer = Deskewer(cfg.deskew)

    vertical_extractor = VerticalExtractor(cfg.vertical, verbose=verbose)

    line_detector = LineDetector(
        cfg.separator, cfg.visualization, verbose=verbose
    )

    # This script's crop padding is configured separately from main.py's
    # (research.prepare_roi_dataset.right_padding) to preserve its output.
    cropper = Cropper(CropConfig(right_padding=settings.right_padding))

    annotation_detector = AnnotationDetector(
        cfg.annotation_detector, cfg.visualization, verbose=verbose
    )


def main():

    global cfg, settings

    parser = argparse.ArgumentParser(description="Build the ROI dataset")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (default: $OCR_CONFIG, else <repo>/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = cfg.research.prepare_roi_dataset

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    build_components()

    pdfs = sorted(settings.input_dir.glob(cfg.paths.pdf_glob))

    if len(pdfs) == 0:

        print("No PDFs found.")

        return

    print(f"\nFound {len(pdfs)} PDF(s)\n")

    for pdf in tqdm(pdfs):

        process_pdf(pdf)

    print("\n===================================")
    print("ROI DATASET CREATION COMPLETE")
    print("===================================")
    print(f"Saved to : {settings.output_dir.resolve()}")


if __name__ == "__main__":

    main()
