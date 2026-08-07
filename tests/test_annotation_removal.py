import cv2
import numpy as np
from pathlib import Path

# ==========================================================
# INPUT / OUTPUT FOLDERS
# ==========================================================

INPUT_DIR = r"G:\OCR_codes\outputs\Doc0503\cropped"
OUTPUT_DIR = "annotation_removal"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ==========================================================
# HSV THRESHOLDS
# ==========================================================

# -------- RED --------
LOWER_RED1 = np.array([0, 60, 40])
UPPER_RED1 = np.array([10, 255, 255])

LOWER_RED2 = np.array([170, 60, 40])
UPPER_RED2 = np.array([179, 255, 255])

# -------- GREEN --------
LOWER_GREEN = np.array([35, 30, 30])
UPPER_GREEN = np.array([95, 255, 255])


# -------- RED --------
LOWER_RED1 = np.array([0, 20, 20])
UPPER_RED1 = np.array([15, 255, 255])

LOWER_RED2 = np.array([165, 20, 20])
UPPER_RED2 = np.array([179, 255, 255])

# -------- GREEN --------
LOWER_GREEN = np.array([30, 20, 20])
UPPER_GREEN = np.array([100,255,255])

# ==========================================================
# MORPHOLOGY
# ==========================================================

kernel = np.ones((3, 3), np.uint8)

# ==========================================================
# PROCESS
# ==========================================================

extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")

files = []

for ext in extensions:
    files.extend(Path(INPUT_DIR).glob(ext))

print(f"Found {len(files)} images")

for file in files:

    print(f"Processing {file.name}")

    img = cv2.imread(str(file))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---------------- RED ----------------

    red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)

    red_mask = cv2.bitwise_or(red1, red2)

    # ---------------- GREEN ----------------

    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # ---------------- FINAL MASK ----------------

    mask = cv2.bitwise_or(red_mask, green_mask)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # ======================================================
    # OPTION 1 : WHITE OUT
    # ======================================================

    cleaned = img.copy()
    cleaned[mask > 0] = (255, 255, 255)

    # ======================================================
    # OPTION 2 : INPAINT
    # Uncomment if preferred
    # ======================================================

    # cleaned = cv2.inpaint(
    #     img,
    #     mask,
    #     3,
    #     cv2.INPAINT_TELEA
    # )

    stem = file.stem

    cv2.imwrite(f"{OUTPUT_DIR}/{stem}_original.png", img)
    cv2.imwrite(f"{OUTPUT_DIR}/{stem}_redmask.png", red_mask)
    cv2.imwrite(f"{OUTPUT_DIR}/{stem}_greenmask.png", green_mask)
    cv2.imwrite(f"{OUTPUT_DIR}/{stem}_mask.png", mask)
    cv2.imwrite(f"{OUTPUT_DIR}/{stem}_cleaned.png", cleaned)

print("\nDone.")