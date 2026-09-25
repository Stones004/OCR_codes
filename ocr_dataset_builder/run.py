import argparse
import os


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (default: $OCR_CONFIG, else <repo>/config.yaml)",
    )

    parser.add_argument(
        "--backend",
        choices=["mistral", "parseq"],
        default=None,
        help="OCR backend to use (default: dataset_builder.ocr_backend in config.yaml)",
    )

    parser.add_argument(
        "--review",
        choices=["all", "low_confidence"],
        default=None,
        help=(
            "PARSeq only. 'all' flags every prediction for review -- use "
            "this when building a new training batch, since nothing is "
            "trusted until a human confirms it. 'low_confidence' (default) "
            "only flags predictions below review.confidence_threshold -- "
            "use this for production inference."
        ),
    )

    args = parser.parse_args()

    # The config is loaded when src.config is first imported, so the
    # chosen file has to be published through OCR_CONFIG before that.
    if args.config:
        os.environ["OCR_CONFIG"] = args.config

    from src.dataset_builder import AnnotationPipeline

    pipeline = AnnotationPipeline(backend=args.backend, review_mode=args.review)

    pipeline.run()


if __name__ == "__main__":
    main()
