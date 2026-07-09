from pathlib import Path

from src.ocr import OCR

ocr = OCR(gpu=False)

image = Path(
    "outputs/P5/cropped/page_004.png"
)

result = ocr.read(image)

print("\n" + "=" * 60)
print("OCR RESULT")
print("=" * 60)

print("Text       :", result["text"])
print("Confidence :", result["confidence"])

print("\nDetected Words")

for word in result["words"]:

    print(word)