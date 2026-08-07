import time

from tqdm import tqdm

from .mistral_client import MistralOCR
from .file_manager import FileManager
from .csv_manager import CSVManager


class AnnotationPipeline:

    def __init__(self):

        self.client = MistralOCR()
        self.files = FileManager()
        self.csv = CSVManager()

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

        start_time = time.time()

        try:

            for image in tqdm(batch):

                try:

                    text = self.client.ocr_image(str(image))

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
            print(f"Time Taken : {elapsed:.2f} sec")
            print(
                f"Remaining  : {self.files.remaining_images()}"
            )
            print("=" * 60)

        finally:

            self.csv.close()