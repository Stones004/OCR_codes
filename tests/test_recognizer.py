from pathlib import Path
import cv2

from src.recognizer import Recognizer


recognizer = Recognizer()

image = cv2.imread(
    "outputs/P5/cropped/page_003.png"
)

components, debug = recognizer.detect(image)

print("=" * 60)
print("Detected Components")
print("=" * 60)

for c in components:

    print(c)

output = Path("outputs/recognizer")
output.mkdir(exist_ok=True)

cv2.imwrite(
    str(output / "gray.png"),
    debug["gray"]
)

cv2.imwrite(
    str(output / "binary.png"),
    debug["binary"]
)

cv2.imwrite(
    str(output / "detected.png"),
    debug["detected"]
)

print("\nFinished")