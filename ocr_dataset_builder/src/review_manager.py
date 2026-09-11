import csv

from .config import REVIEW_CSV


class ReviewQueueManager:
    """
    Staging area for PARSeq predictions awaiting human review, kept
    separate from annotations.csv so nothing untrusted mixes into the
    gold-label training set until a human confirms/corrects it.
    """

    def __init__(self):

        self.csv_file = None
        self.writer = None

        self._initialize()

    def _initialize(self):

        csv_exists = REVIEW_CSV.exists()

        self.csv_file = open(
            REVIEW_CSV,
            mode="a",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.csv_file)

        if not csv_exists:

            self.writer.writerow([
                "filename",
                "prediction",
                "confidence",
                "needs_review",
                "corrected_text",
            ])

            self.csv_file.flush()

    def append(self, filename, prediction, confidence, needs_review):

        self.writer.writerow([
            filename,
            prediction,
            confidence,
            needs_review,
            "",
        ])

        # Immediately save to disk
        self.csv_file.flush()

    def close(self):

        if self.csv_file:
            self.csv_file.close()
