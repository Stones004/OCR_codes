import time

from tqdm import tqdm

from .file_manager import FileManager
from .csv_manager import CSVManager
from .config import OCR_BACKEND
from .config import PARSEQ_REVIEW_MODE
from .config import PARSEQ_CONFIDENCE_THRESHOLD


class AnnotationPipeline:

    def __init__(self, backend=None, review_mode=None):

        backend = backend or OCR_BACKEND
        self.backend = backend

        if backend == "parseq":
            from .parseq_client import ParseqOCR
            from .review_manager import ReviewQueueManager
            self.client = ParseqOCR()
            self.review = ReviewQueueManager()
            self.review_mode = review_mode or PARSEQ_REVIEW_MODE
        elif backend == "mistral":
            from .mistral_client import MistralOCR
            self.client = MistralOCR()
            self.review = None
            self.review_mode = None
        else:
            raise ValueError(
                f"Unknown OCR backend: {backend!r} (expected 'mistral' or 'parseq')"
            )

        print(f"OCR backend: {backend}")

        if self.review_mode:
            print(f"Review mode : {self.review_mode}")

        self.files = FileManager()
        self.csv = CSVManager() if backend == "mistral" else None

    def run(self):

        batch = self.files.get_next_batch()

        if len(batch) == 0:
            print("\nNo images left to process.")
            return

        print("=" * 60)
        print(f"Found {len(batch)} images to annotate.")
        print("=" * 60)

        success = 0
        failed = 0
        flagged = 0

        start_time = time.time()

        try:

            for image in tqdm(batch):

                try:

                    text, confidence = self.client.ocr_image(str(image))

                    if self.backend == "parseq":

                        if self.review_mode == "all":
                            needs_review = True
                        else:
                            needs_review = confidence < PARSEQ_CONFIDENCE_THRESHOLD

                        if needs_review:
                            flagged += 1

                        self.review.append(
                            image.name,
                            text,
                            confidence,
                            needs_review,
                        )

                    else:

                        self.csv.append(
                            image.name,
                            text
                        )

                    self.files.create_preview(
                        image,
                        text
                    )

                    self.files.move_completed(image)

                    success += 1

                except Exception as e:

                    failed += 1

                    print(f"\nFailed : {image.name}")
                    print(e)

            elapsed = time.time() - start_time

            print("\n")
            print("=" * 60)
            print("Annotation Complete")
            print("=" * 60)
            print(f"Successful : {success}")
            print(f"Failed     : {failed}")

            if self.backend == "parseq":
                print(f"Flagged for review : {flagged}/{success}")

            print(f"Time Taken : {elapsed:.2f} sec")
            print(
                f"Remaining  : {self.files.remaining_images()}"
            )
            print("=" * 60)

        finally:

            if self.csv:
                self.csv.close()

            if self.review:
                self.review.close()

            self.client.close()