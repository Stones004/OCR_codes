import csv

from .config import CSV_FILE


class CSVManager:

    def __init__(self):

        self.csv_file = None
        self.writer = None

        self._initialize()

    def _initialize(self):

        csv_exists = CSV_FILE.exists()

        self.csv_file = open(
            CSV_FILE,
            mode="a",
            newline="",
            encoding="utf-8"
        )

        self.writer = csv.writer(self.csv_file)

        if not csv_exists:

            self.writer.writerow([
                "filename",
                "text"
            ])

            self.csv_file.flush()

    def append(self, filename, text):

        self.writer.writerow([
            filename,
            text
        ])

        # Immediately save to disk
        self.csv_file.flush()

    def close(self):

        if self.csv_file:
            self.csv_file.close()