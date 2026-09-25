import argparse

import cv2

from src.config import load_config
from src.ocr_preprocessor import OCRPreprocessor


def main():

    parser = argparse.ArgumentParser(description="Preprocessing benchmark")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (default: $OCR_CONFIG, else <repo>/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    image_path = cfg.research.benchmark.image_path
    output_dir = cfg.research.benchmark.output_dir

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    preprocessor = OCRPreprocessor(cfg.research.ocr_preprocessor)

    gray, outputs = preprocessor.benchmark(image)

    preprocessor.save_benchmark(
        outputs,
        output_dir
    )

    print("\nBenchmark complete!\n")

    print("Generated methods:")

    for method in outputs:
        print(f"  - {method}")


if __name__ == "__main__":
    main()
