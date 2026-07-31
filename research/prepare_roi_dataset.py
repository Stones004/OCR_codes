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

from pathlib import Path
import cv2
import fitz
import numpy as np
from tqdm import tqdm

from src.deskewer import Deskewer
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector
from src.cropper import Cropper
from src.annotation_detector import AnnotationDetector

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("data/input_pdfs")
OUTPUT_DIR = Path("data/roi_dataset")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300


# ============================================================
# PDF Renderer
# ============================================================

def pdf_to_images(pdf_path):

    document = fitz.open(pdf_path)

    zoom = DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_number in range(len(document)):

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
# Preprocessor
# ============================================================

class Preprocessor:

    def __init__(self, kernel_size=3):

        self.kernel_size = kernel_size

    def full_pipeline(self, gray):

        _, binary = cv2.threshold(

            gray,

            0,

            255,

            cv2.THRESH_BINARY + cv2.THRESH_OTSU

        )

        kernel = np.ones(

            (
                self.kernel_size,
                self.kernel_size
            ),

            np.uint8

        )

        return cv2.morphologyEx(

            binary,

            cv2.MORPH_CLOSE,

            kernel

        )

    def process(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        return self.full_pipeline(gray)

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

    print(type(rois))
    print(len(rois))

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

            filename = (

                f"{pdf_name}"
                f"_page_{page_number:03d}"
                f"_roi_{roi_number:03d}.png"

            )

            output_path = OUTPUT_DIR / filename

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

def main():

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))

    if len(pdfs) == 0:

        print("No PDFs found.")

        return

    print(f"\nFound {len(pdfs)} PDF(s)\n")

    for pdf in tqdm(pdfs):

        process_pdf(pdf)

    print("\n===================================")
    print("ROI DATASET CREATION COMPLETE")
    print("===================================")
    print(f"Saved to : {OUTPUT_DIR.resolve()}")





# ============================================================
# Pipeline Components
# ============================================================

deskewer = Deskewer()

preprocessor = Preprocessor()

vertical_extractor = VerticalExtractor()

line_detector = LineDetector()

cropper = Cropper()

annotation_detector = AnnotationDetector()
# ============================================================

if __name__ == "__main__":

    main()