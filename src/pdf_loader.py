from pathlib import Path

import fitz  # PyMuPDF
import cv2
import numpy as np


"""
PDF Loader

Loads scanned PDF documents using PyMuPDF and converts every page into
an OpenCV image for further processing. Optionally saves the rendered
pages for debugging purposes.
"""


class PDFLoader:
    """
    Loads scanned PDFs and converts each page into an OpenCV image.
    """

    def __init__(
        self,
        dpi: int = 300,
        save_debug: bool = False,
        output_dir: str | None = None,
        skip_pages: int = 2,
    ):
        self.dpi = dpi
        self.save_debug = save_debug
        self.output_dir = Path(output_dir) if output_dir else None
        self.skip_pages = skip_pages

        if self.save_debug and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_pdf(self, pdf_path: str):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} not found.")

        document = fitz.open(pdf_path)

        pages = []

        zoom = self.dpi / 72
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
                    f"page_{page_number+1:03d}.png"
                )

                cv2.imwrite(
                    str(filename),
                    image
                )

        document.close()

        print(f"Successfully loaded {len(pages)} pages.")

        return pages