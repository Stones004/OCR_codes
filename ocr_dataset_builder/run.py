import argparse

from src.dataset_builder import AnnotationPipeline


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=["mistral", "parseq"],
        default=None,
        help="OCR backend to use (default: OCR_BACKEND in src/config.py)",
    )

    parser.add_argument(
        "--review",
        choices=["all", "low_confidence"],
        default=None,
        help=(
            "PARSeq only. 'all' flags every prediction for review -- use "
            "this when building a new training batch, since nothing is "
            "trusted until a human confirms it. 'low_confidence' (default) "
            "only flags predictions below PARSEQ_CONFIDENCE_THRESHOLD -- "
            "use this for production inference."
        ),
    )

    args = parser.parse_args()

    pipeline = AnnotationPipeline(backend=args.backend, review_mode=args.review)

    pipeline.run()


if __name__ == "__main__":
    main()