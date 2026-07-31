import base64
import json
import re
from pathlib import Path


"""
LM Studio OCR Test

Standalone script for evaluating vision-language OCR models served
through LM Studio. The script sends an image to the local model,
extracts text together with localization information, and visualizes
the returned bounding boxes and center coordinates."""


from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# ============================
# Configuration
# ============================
IMAGE_PATH = r"G:\OCR_codes\outputs\P5\cropped\page_008.png"
OUTPUT_PATH = r"G:\OCR_codes\page_008_annotated.png"

MODEL = "google/gemma-3-12b"
# MODEL = "qwen/qwen2.5-vl-7b"

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

# ============================
# Encode Image
# ============================
with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# ============================
# OCR Prompt
# ============================
prompt = """
You are a precise OCR engine.

Your task is to detect every visible text element and localize it accurately.

IMPORTANT

- Measure coordinates from the ORIGINAL image.
- Ignore any internal resizing.
- Coordinates must be normalized to the ORIGINAL image.
- Left edge = 0
- Right edge = 1
- Top edge = 0
- Bottom edge = 1

Every coordinate must correspond to the actual visible location of the text in the image.

Do not estimate.

If the exact position is uncertain, inspect the image again before returning coordinates.

For every text:

1. Find the tightest possible bounding rectangle.
2. Return the normalized bounding box.
3. Compute the center of that box.

Return ONLY valid JSON.

[
    {
        "text":"...",
        "bounding_box":{
            "left":0.000,
            "top":0.000,
            "right":0.000,
            "bottom":0.000
        },
        "center":{
            "x":0.000,
            "y":0.000
        }
    }
]
"""

# ============================
# Run OCR
# ============================
response = client.chat.completions.create(
    model=MODEL,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    },
                },
            ],
        }
    ],
)

content = response.choices[0].message.content

print("========== RAW RESPONSE ==========")
print(content)

# ============================
# Clean JSON
# ============================
content = re.sub(r"^```json\s*", "", content.strip())
content = re.sub(r"^```\s*", "", content)
content = re.sub(r"\s*```$", "", content)

try:
    results = json.loads(content)
except Exception as e:
    print("\nCould not parse JSON.")
    print(e)
    raise

# ============================
# Draw Results
# ============================
image = Image.open(IMAGE_PATH).convert("RGB")
draw = ImageDraw.Draw(image)

orig_w, orig_h = image.size

# -----------------------------
# Calibration (pixels)
# -----------------------------
X_OFFSET = 25      # positive -> move right
Y_OFFSET = 0       # positive -> move down

X_SCALE = 1.00     # stretch horizontally
Y_SCALE = 1.00     # stretch vertically

print(f"Original image : {orig_w} x {orig_h}")

# ----------------------------------------
# Change this to True to test swapped x/y
# ----------------------------------------
SWAP_XY = False

try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()


def convert_point(x, y):
    """
    Converts normalized (or pixel) coordinates into image coordinates
    and applies calibration.
    """

    # Normalized coordinates
    if 0 <= x <= 1 and 0 <= y <= 1:
        x *= orig_w
        y *= orig_h

    # Optional swap
    if SWAP_XY:
        x, y = y, x

    # Calibration
    x = x * X_SCALE + X_OFFSET
    y = y * Y_SCALE + Y_OFFSET

    return int(round(x)), int(round(y))


# ============================
# Draw OCR Results
# ============================

PADDING = 3

for item in results:

    text = item.get("text", "")

    # -------------------------
    # Bounding Box
    # -------------------------
    bbox = item.get("bounding_box")

    if bbox:

        x1, y1 = convert_point(
            bbox["left"],
            bbox["top"],
        )

        x2, y2 = convert_point(
            bbox["right"],
            bbox["bottom"],
        )

        # Slight padding
        x1 -= PADDING
        y1 -= PADDING
        x2 += PADDING
        y2 += PADDING

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=2,
        )

        draw.text(
            (x1, max(0, y1 - 18)),
            text,
            fill="blue",
            font=font,
        )

    # -------------------------
    # Center
    # -------------------------
    center = item.get("center")

    if center:

        cx, cy = convert_point(
            center["x"],
            center["y"],
        )

        draw.ellipse(
            (cx - 3, cy - 3, cx + 3, cy + 3),
            fill="green",
        )

# ============================
# Save
# ============================
Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

image.save(OUTPUT_PATH)

print("\nSaved to:")
print(OUTPUT_PATH)

image.show()