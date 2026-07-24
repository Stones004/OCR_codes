import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# =====================================================
# Configuration
# =====================================================

IMAGE_PATH = r"D:\OCR_codes\outputs\P1\cropped\page_003.png"

WINDOW_HEIGHT = 96
STRIDE = 32

MIN_INK_PIXELS = 120

MIN_COMPONENT_AREA = 30

PADDING = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Load TrOCR
# =====================================================

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
).to(DEVICE)

# =====================================================
# Load Image
# =====================================================

img = cv2.imread(IMAGE_PATH)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

H, W = gray.shape

print(f"Image Size : {W} x {H}")

window_id = 0

# =====================================================
# Sliding Window
# =====================================================

for y in range(0, H, STRIDE):

    y2 = min(y + WINDOW_HEIGHT, H)

    window = gray[y:y2, :]

    if window.shape[0] < 40:
        continue

    # -----------------------------------------------
    # Threshold
    # -----------------------------------------------

    _, binary = cv2.threshold(
        window,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # -----------------------------------------------
    # Ink Density Filter
    # -----------------------------------------------

    ink_pixels = cv2.countNonZero(binary)

    if ink_pixels < MIN_INK_PIXELS:
        continue

    # -----------------------------------------------
    # Connected Components
    # -----------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    candidates = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_COMPONENT_AREA:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        yy = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        candidates.append((area, x, yy, w, h))

    if len(candidates) == 0:
        continue

    # -----------------------------------------------
    # Sort Largest First
    # -----------------------------------------------

    candidates.sort(reverse=True)

    # OCR top 3 blobs
    for blob_id, (_, x, yy, w, h) in enumerate(candidates[:3]):

        x1 = max(0, x - PADDING)
        y1 = max(0, yy - PADDING)

        x2 = min(W, x + w + PADDING)
        y2 = min(window.shape[0], yy + h + PADDING)

        roi = window[y1:y2, x1:x2]

        # Ignore tiny crops
        if roi.shape[0] < 16 or roi.shape[1] < 16:
            continue

        # Upscale
        roi = cv2.resize(
            roi,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        # RGB for TrOCR
        pil = Image.fromarray(roi).convert("RGB")

        pixel_values = processor(
            images=pil,
            return_tensors="pt"
        ).pixel_values.to(DEVICE)

        generated = model.generate(pixel_values)

        text = processor.batch_decode(
            generated,
            skip_special_tokens=True
        )[0].strip()

        print("-" * 70)
        print(f"Window : {window_id}")
        print(f"Global Y : {y}-{y2}")
        print(f"Blob : {blob_id}")
        print(f"ROI : {roi.shape}")
        print(f"Prediction : '{text}'")

        cv2.imwrite(
            f"window_{window_id:03d}_blob_{blob_id}.png",
            roi
        )

    window_id += 1