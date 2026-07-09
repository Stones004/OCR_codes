from pathlib import Path

import cv2
import pandas as pd

from src.pdf_loader import PDFLoader
from src.image_analysis import ImageAnalyzer
from src.deskewer import Deskewer
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector
from src.cropper import Cropper
from src.recognizer import Recognizer
from src.preprocessor import Preprocessor


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = Path("data/input_pdfs")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Modules
# --------------------------------------------------

loader = PDFLoader(
    dpi=300,
    save_debug=False
)

analyzer = ImageAnalyzer()

deskewer = Deskewer()

extractor = VerticalExtractor(
    threshold=180,
    kernel_height=None
)

detector = LineDetector()

cropper = Cropper(
    right_padding=15
)

preprocessor = Preprocessor(
    scale=4
)

recognizer = Recognizer()

summary = []

pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDF(s)\n")

# ==================================================
# Process each PDF
# ==================================================

for pdf_path in pdf_files:

    pdf_name = pdf_path.stem

    print("=" * 70)
    print(f"Processing : {pdf_name}")
    print("=" * 70)

    pdf_output = OUTPUT_DIR / pdf_name

    deskew_dir = pdf_output / "deskewed"
    vertical_dir = pdf_output / "vertical"
    detected_dir = pdf_output / "detected"
    cropped_dir = pdf_output / "cropped"
    failed_dir = pdf_output / "failed"
    preprocessed_dir = pdf_output / "preprocessed"
    recognizer_dir = pdf_output / "recognizer"
    roi_dir = pdf_output / "roi"

    for folder in [
        deskew_dir,
        vertical_dir,
        detected_dir,
        cropped_dir,
        preprocessed_dir,
        recognizer_dir,
        roi_dir,
        failed_dir
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
        # Analysis (optional)
        # ------------------------------------------

        info = analyzer.analyze(page)

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

        detected, separator = detector.detect(vertical)

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
        # Preprocessing
        # ------------------------------------------

        processed = preprocessor.process(

            crop,

            method="morph_close"

        )

        cv2.imwrite(

            str(preprocessed_dir / f"page_{page_idx + 1:03d}.png"),

            processed

        )

        # ------------------------------------------
        # Recognition
        # ------------------------------------------

        roi, recog_debug = recognizer.detect(

            processed

        )

        cv2.imwrite(

            str(recognizer_dir / f"page_{page_idx + 1:03d}.png"),

            recog_debug["detected"]

        )

        if roi is not None:

            cv2.imwrite(

                str(roi_dir / f"page_{page_idx + 1:03d}.png"),

                roi

            )

        else:

            print("Recognizer : FAILED")

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