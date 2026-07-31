from pathlib import Path

import cv2

from src.ocr_preprocessor import OCRPreprocessor


IMAGE_PATH = Path("sample_crop.png")
OUTPUT_DIR = Path("research/results/images")


def main():

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(IMAGE_PATH)

    preprocessor = OCRPreprocessor()

    gray, outputs = preprocessor.benchmark(image)

    preprocessor.save_benchmark(
        outputs,
        OUTPUT_DIR
    )

    print("\nBenchmark complete!\n")

    print("Generated methods:")

    for method in outputs:
        print(f"  ✓ {method}")


if __name__ == "__main__":
    main()