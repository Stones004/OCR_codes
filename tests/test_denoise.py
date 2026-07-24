import cv2
import numpy as np

# -----------------------------
# Load image
# -----------------------------
img = cv2.imread(r"D:\OCR_codes\outputs\P1\cropped\page_006.png", cv2.IMREAD_GRAYSCALE)

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
MIN_AREA = 200      # Increase to remove more blobs

clean = np.zeros_like(binary)

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area >= MIN_AREA:
        clean[labels == i] = 255

# -----------------------------
# Save
# -----------------------------
cv2.imwrite("clean.png", 255 - clean)

print(f"Original Components : {num_labels-1}")
print(f"Area Threshold      : {MIN_AREA}")