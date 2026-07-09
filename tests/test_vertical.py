'''from src.pdf_loader import PDFLoader
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector

import cv2
def show(name, image, max_width=900):

    display = image.copy()

    h, w = display.shape[:2]

    if w > max_width:

        scale = max_width / w

        display = cv2.resize(
            display,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    cv2.imshow(name, display)

loader = PDFLoader(
    dpi=300,
    save_debug=False
)

pages = loader.load_pdf(
    "data/input_pdfs/P5.pdf"
)

page = pages[2]          # Page 3

extractor = VerticalExtractor(
    threshold=180,
    kernel_height=120
)

vertical, debug = extractor.extract(page)

print("After Extractor")
print(vertical.shape)
print(vertical.dtype)

# Save everything

cv2.imwrite("gray.png", debug["gray"])
cv2.imwrite("binary.png", debug["binary"])
cv2.imwrite("vertical.png", vertical)

# Display

show("Gray", debug["gray"])
show("Binary", debug["binary"])
show("Vertical", vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()

detector = LineDetector()

detected, candidates = detector.detect(vertical)

cv2.imwrite("detected_lines.png", detected)

print("\nCandidates")

for c in candidates:
    print(c) '''


from pathlib import Path
import cv2

from src.pdf_loader import PDFLoader
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector
from src.deskewer import Deskewer

loader = PDFLoader(
    dpi=300,
    save_debug=False
)

pages = loader.load_pdf(
    "data/input_pdfs/P5.pdf"
)

extractor = VerticalExtractor(
    threshold=180,
    kernel_height=None      # Dynamic
)

deskewer = Deskewer()

detector = LineDetector()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

results = []

for page_idx, page in enumerate(pages):

    print("=" * 60)
    print(f"Processing Page {page_idx+1}")
    print("=" * 60)

    # --------------------------------------------------
    # Deskew
    # --------------------------------------------------

    corrected_page, angle = deskewer.process(page)

    import hashlib

    print(
        f"Page {page_idx + 1 if 'page_idx' in locals() else i + 1}"
    )

    print("Angle :", angle)

    print(
        "Checksum :",
        hashlib.md5(corrected_page.tobytes()).hexdigest()
    )

    print(f"Deskew Angle : {angle:.2f}°")

    # --------------------------------------------------
    # Vertical Extraction
    # --------------------------------------------------

    vertical, debug = extractor.extract(corrected_page)

    detected, separator = detector.detect(vertical)

    overlay = corrected_page.copy()

    if separator is not None:
        x = separator["x"]

        cv2.line(
            overlay,
            (x, 0),
            (x, overlay.shape[0]),
            (0, 0, 255),
            4
        )

    cv2.imwrite(
        str(output_dir / f"page_{page_idx + 1:03d}_overlay.png"),
        overlay
    )

    cv2.imwrite(
        str(output_dir / f"page_{page_idx+1:03d}.png"),
        detected
    )

    cv2.imwrite(
        str(output_dir / f"page_{page_idx + 1:03d}_deskewed.png"),
        corrected_page
    )

    if separator is None:

        print("No separator detected")

        results.append({

            "page": page_idx + 1,

            "status": "FAILED",

            "deskew_angle": angle,

            "separator_x": None
        })

        failed_dir = output_dir / "failed"
        failed_dir.mkdir(exist_ok=True)

        cv2.imwrite(
            str(failed_dir / f"page_{page_idx + 1:03d}_gray.png"),
            debug["gray"]
        )

        cv2.imwrite(
            str(failed_dir / f"page_{page_idx + 1:03d}_binary.png"),
            debug["binary"]
        )

        cv2.imwrite(
            str(failed_dir / f"page_{page_idx + 1:03d}_vertical.png"),
            vertical
        )

    else:

        print(separator)

        results.append({

            "page": page_idx + 1,

            "status": "SUCCESS",

            "deskew_angle": angle,

            "separator_x": separator["x"],

            "width": separator["w"],

            "height": separator["h"]
        })

print("\nFinished")


import pandas as pd

df = pd.DataFrame(results)

df.to_csv(
    output_dir / "separator_results.csv",
    index=False
)

print(df)

# ---------------------------------------
# Learn separator position
# ---------------------------------------

success = df[
    (df["status"] == "SUCCESS") &
    (df["separator_x"] > 100) &
    (df["separator_x"] < 500)
]

if len(success) > 0:

    mean_x = success["separator_x"].mean()
    std_x = success["separator_x"].std()

    print("\n" + "="*60)
    print("Separator Statistics")
    print("="*60)
    print(f"Mean X : {mean_x:.2f}")
    print(f"Std Dev: {std_x:.2f}")