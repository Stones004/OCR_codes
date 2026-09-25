from pathlib import Path
from src.config import load_config
from src.pdf_loader import PDFLoader
from src.deskewer import Deskewer

import cv2

cfg = load_config()

loader = PDFLoader(cfg.pdf)

pages = loader.load_pdf(
    "data/input_pdfs/P5.pdf"
)

deskewer = Deskewer(cfg.deskew)

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