import cv2
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
IMAGE_PATH = r"D:\OCR_codes\outputs\P5 (1)\roi\page_008.png"         # Your image
Y_THRESHOLD = 20                   # Max Y difference to belong to same line
MIN_CONTOUR_AREA = 10              # Ignore tiny noise
PADDING = 10

# -----------------------------
# Load image
# ----------------------------- 
img = cv2.imread(IMAGE_PATH)
output = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Threshold
# -----------------------------
_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# -----------------------------
# Find contours
# -----------------------------
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------------
# Store contour information
# -----------------------------
boxes = []

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 40:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    # Remove tiny dots
    if w < 4 or h < 4:
        continue

    # Remove extremely thin vertical lines
    if h > 80 and w < 4:
        continue

    # Remove extremely thin horizontal lines
    if w > 80 and h < 4:
        continue

    # Remove contours touching image borders
    margin = 2

    if (
        x <= margin or
        y <= margin or
        x + w >= img.shape[1] - margin or
        y + h >= img.shape[0] - margin
    ):
        continue

    cy = y + h // 2



    x, y, w, h = cv2.boundingRect(cnt)

    cy = y + h // 2

    boxes.append({
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cy": cy
    })

# -----------------------------
# Sort by Y coordinate
# -----------------------------
boxes.sort(key=lambda b: b["cy"])

# -----------------------------
# Group boxes having similar Y
# -----------------------------
groups = []

for box in boxes:

    assigned = False

    for group in groups:

        if abs(box["cy"] - group["cy"]) < Y_THRESHOLD:

            group["boxes"].append(box)

            ys = [b["cy"] for b in group["boxes"]]
            group["cy"] = np.mean(ys)

            assigned = True
            break

    if not assigned:

        groups.append({
            "cy": box["cy"],
            "boxes": [box]
        })

# -----------------------------
# Merge boxes inside each group
# -----------------------------
for group in groups:

    xs = []
    ys = []
    xe = []
    ye = []

    for b in group["boxes"]:

        xs.append(b["x"])
        ys.append(b["y"])
        xe.append(b["x"] + b["w"])
        ye.append(b["y"] + b["h"])

    x1 = max(0, min(xs) - PADDING)
    y1 = max(0, min(ys) - PADDING)

    x2 = min(img.shape[1], max(xe) + PADDING)
    y2 = min(img.shape[0], max(ye) + PADDING)

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

# -----------------------------
# Save
# -----------------------------
cv2.imwrite("detected_lines.png", output)

print("Done.")