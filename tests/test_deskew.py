from pathlib import Path
from src.pdf_loader import PDFLoader
from src.deskewer import Deskewer

import cv2

loader = PDFLoader(
    dpi=300,
    save_debug=False
)

pages = loader.load_pdf(
    "data/input_pdfs/P5.pdf"
)

deskewer = Deskewer()

output = Path("deskew_results")
output.mkdir(exist_ok=True)

angles = []

for i, page in enumerate(pages):

    corrected, angle = deskewer.process(page)

    import hashlib

    print(
        f"Page {i + 1 if 'page_idx' in locals() else i + 1}"
    )

    print("Angle :", angle)

    print(
        "Checksum :",
        hashlib.md5(corrected.tobytes()).hexdigest()
    )

    angles.append(angle)

    print(f"Page {i+1:02d} : {angle:.2f}°")

    cv2.imwrite(
        str(output / f"page_{i+1:03d}.png"),
        corrected
    )

print("\nAverage Angle :", sum(angles) / len(angles))