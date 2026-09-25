import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path

from src.config import PdfConfig


"""
PDF Loader

Loads scanned PDF documents using PyMuPDF and converts every page into
an OpenCV image for further processing. Optionally saves the rendered
pages for debugging purposes.
"""

# PDF user space is defined as 72 points per inch (a unit, not a tunable).
PDF_POINTS_PER_INCH = 72


class PDFLoader:
    """
    Loads scanned PDFs and converts each page into an OpenCV image.
    """

    def __init__(self, config: PdfConfig):
        self.dpi = config.dpi
        self.save_debug = config.save_debug
        self.output_dir = config.debug_dir
        self.skip_pages = config.skip_pages
        self.debug_page_image = config.debug_page_image

        if self.save_debug:
            if self.output_dir is None:
                raise ValueError("pdf.save_debug is true but pdf.debug_dir is null")

            self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_pdf(self, pdf_path: str):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} not found.")

        document = fitz.open(pdf_path)

        pages = []

        zoom = self.dpi / PDF_POINTS_PER_INCH
        matrix = fitz.Matrix(zoom, zoom)

        print("=" * 60)
        print("Loading PDF")
        print("=" * 60)
        print(f"File : {pdf_path.name}")
        print(f"Pages: {len(document)}")
        print(f"DPI  : {self.dpi}")
        print("=" * 60)

        for page_number in range(self.skip_pages, len(document)):

            page = document.load_page(page_number)

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image = np.frombuffer(
                pix.samples,
                dtype=np.uint8
            ).reshape(
                pix.height,
                pix.width,
                pix.n
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            pages.append(image)

            if self.save_debug:

                filename = (
                    self.output_dir /
                    self.debug_page_image.format(page=page_number + 1)
                )

                cv2.imwrite(
                    str(filename),
                    image
                )

        document.close()

        print(f"Successfully loaded {len(pages)} pages.")

        return pages
