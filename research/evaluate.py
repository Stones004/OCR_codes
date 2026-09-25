"""
Evaluate OCR preprocessing methods.

For every image in the dataset:

    Image
        ↓
    Every preprocessing method
        ↓
    OCR
        ↓
    Metrics
        ↓
    evaluation.csv
"""

import argparse
from pathlib import Path
import time

import cv2
import pandas as pd

from src.config import load_config
from src.ocr_preprocessor import OCRPreprocessor
from src.roi_text_extractor import ROITextExtractor


# ----------------------------------------------------------
# DATASET
# ----------------------------------------------------------

def load_dataset(folder, extensions):

    exts = set(extensions)

    return sorted(

        [

            f

            for f in Path(folder).iterdir()

            if f.suffix.lower() in exts

        ]

    )


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Evaluate preprocessing methods")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (default: $OCR_CONFIG, else <repo>/config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = cfg.research.evaluate

    preprocessor = OCRPreprocessor(cfg.research.ocr_preprocessor)

    ocr = ROITextExtractor(
        cfg.ocr,
        backend=settings.backend,
        use_backend_preprocessing=settings.use_backend_preprocessing
    )

    results = []

    images = load_dataset(settings.dataset_dir, cfg.image_extensions)

    print(f"\nFound {len(images)} image(s)\n")

    for image_path in images:

        print(f"Processing: {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:

            print("Could not load image.")

            continue

        for method in preprocessor.available_methods():

            # --------------------------------------
            # PREPROCESSING
            # --------------------------------------

            t0 = time.perf_counter()

            _, processed = preprocessor.process_method(

                image,

                method

            )

            save_dir = settings.preprocessed_dir / method
            save_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                str(save_dir / image_path.name),
                processed
            )

            preprocess_ms = (

                time.perf_counter()

                - t0

            ) * 1000

            # --------------------------------------
            # OCR
            # --------------------------------------

            t0 = time.perf_counter()

            predictions = ocr.extract(processed)

            ocr_ms = (

                time.perf_counter()

                - t0

            ) * 1000

            # --------------------------------------
            # TEXT
            # --------------------------------------

            text = "\n".join(

                block["text"]

                for block in predictions

            )

            # --------------------------------------
            # SAVE OCR TEXT
            # --------------------------------------

            ocr_dir = settings.ocr_dir / method
            ocr_dir.mkdir(parents=True, exist_ok=True)

            with open(
                ocr_dir / f"{image_path.stem}.txt",
                "w",
                encoding="utf-8"
            ) as f:
                f.write(text)

            # --------------------------------------
            # IMAGE METRICS
            # --------------------------------------

            black_pixels = (

                processed.size

                - cv2.countNonZero(processed)

            )

            density = black_pixels / processed.size

            # --------------------------------------
            # SAVE
            # --------------------------------------

            results.append({

                "image": image_path.name,

                "method": method,

                "preprocess_ms": round(preprocess_ms, 2),

                "ocr_ms": round(ocr_ms, 2),

                "total_ms": round(

                    preprocess_ms + ocr_ms,

                    2

                ),

                "blocks": len(predictions),

                "characters": len(text),

                "words": len(text.split()),

                "lines": len(text.splitlines()),

                "density": round(density, 4),

                "empty": len(text.strip()) == 0,

                "text": text

            })

    df = pd.DataFrame(results)

    settings.output_csv.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    df.to_csv(

        settings.output_csv,

        index=False

    )

    print("\nEvaluation Complete\n")

    print(df.head())

    print(f"\nSaved to:\n{settings.output_csv}")


if __name__ == "__main__":

    main()