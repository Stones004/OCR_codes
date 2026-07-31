from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_DIR = Path(r"G:\OCR_codes\outputs\P4\cropped")
OUTPUT_DIR = Path(r"G:\OCR_codes\denoised")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Parameters
# --------------------------------------------------

MIN_AREA = 200

# --------------------------------------------------
# Process all PNGs
# --------------------------------------------------

png_files = sorted(INPUT_DIR.glob("*.png"))

print(f"Found {len(png_files)} image(s)\n")

for img_path in png_files:

    print(f"Processing: {img_path.name}")

    # -----------------------------
    # Load image
    # -----------------------------

    img = cv2.imread(
        str(img_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        print(f"Failed to read {img_path.name}")
        continue

    # -----------------------------
    # Threshold
    # -----------------------------

    _, binary = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # -----------------------------
    # Connected Components
    # -----------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    # -----------------------------
    # Remove small components
    # -----------------------------

    clean = np.zeros_like(binary)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_AREA:
            clean[labels == i] = 255

    # -----------------------------
    # Save
    # -----------------------------

    output_path = OUTPUT_DIR / img_path.name

    cv2.imwrite(
        str(output_path),
        255 - clean
    )

    print(f"  Components : {num_labels - 1}")

print("\nDone!")