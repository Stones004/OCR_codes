
from main import folder
from pathlib import Path

from src.pdf_loader import PDFLoader
from src.deskewer import Deskewer
from src.vertical_extractor import VerticalExtractor
from src.line_detector import LineDetector
from src.cropper import Cropper
from src.page_preprocessor import PagePreprocessor
from src.ocr_preprocessor import OCRPreprocessor
from src.annotation_detector import AnnotationDetector
from src.roi_text_extractor import ROITextExtractor
import cv2
import pandas as pd
from src.annotation_remover import AnnotationRemover
import numpy as np

class OCRPipeline:

    def __init__(self, config):

        self.config = config

        # -----------------------------
        # Paths
        # -----------------------------

        self.input_dir = Path("data/input_pdfs")
        self.output_dir = Path("outputs")

        self.output_dir.mkdir(exist_ok=True)

        # -----------------------------
        # Pipeline Modules
        # -----------------------------

        self.loader = PDFLoader(
            dpi=300,
            save_debug=False
        )

        self.deskewer = Deskewer()

        self.extractor = VerticalExtractor(
            threshold=180,
            kernel_height=None
        )

        self.line_detector = LineDetector()

        self.cropper = Cropper(
            right_padding=15
        )

        self.annotation_remover = AnnotationRemover()

        self.page_preprocessor = PagePreprocessor()

        self.ocr_preprocessor = OCRPreprocessor(
            method="connected_components"
        )

        self.annotation_detector = AnnotationDetector()

        self.text_extractor = ROITextExtractor(
            backend=config.OCR_BACKEND
        )

        # -----------------------------
        # Pipeline State
        # -----------------------------

        self.summary = []
    
    def save_debug_image(self, folder, filename, image):

        if not self.config.SAVE_DEBUG_IMAGES:
            return

        cv2.imwrite(
            str(folder / filename),
            image
        )

    def run(self):

        pdf_files = sorted(
            self.input_dir.glob("*.pdf")
        )

        print(
            f"\nFound {len(pdf_files)} PDF(s)\n"
        )

        for pdf_path in pdf_files:

            self.process_pdf(pdf_path)

        self.save_summary()



    def process_pdf(self, pdf_path):

        pdf_name = pdf_path.stem

        print("=" * 70)
        print(f"Processing : {pdf_name}")
        print("=" * 70)

        # --------------------------------------------------
        # Output Directories
        # --------------------------------------------------

        pdf_output = self.output_dir / pdf_name

        folders = {

            "deskew": pdf_output / "deskewed",

            "vertical": pdf_output / "vertical",

            "detected": pdf_output / "detected",

            "cropped": pdf_output / "cropped",

            "annotation_removed": pdf_output / "annotation_removed",

            "preprocessed": pdf_output / "preprocessed",

            "recognizer": pdf_output / "recognizer",

            "roi": pdf_output / "roi",

            "ocr": pdf_output / "ocr",

            "failed": pdf_output / "failed"

        }

        for folder in folders.values():

            folder.mkdir(

                parents=True,

                exist_ok=True

            )

        # --------------------------------------------------
        # Load PDF
        # --------------------------------------------------

        pages = self.loader.load_pdf(
            str(pdf_path)
        )

        results = []

        # --------------------------------------------------
        # Process Pages
        # --------------------------------------------------

        for page_idx, page in enumerate(pages):

            print(f"\nPage {page_idx + 1}")

            page_result = self.process_page(
                page=page,
                page_idx=page_idx,
                folders=folders
            )

            results.append(page_result)

        # --------------------------------------------------
        # Export Report
        # --------------------------------------------------

        self.save_report(

            pdf_name,

            pdf_output,

            results

        )
    
    def process_page(
        self,
        page,
        page_idx,
        folders
    ):
        print(
            f"Processing Page {page_idx + 1}"
        )
        page = self.page_preprocessor.process(page)
        
        corrected, angle = self.deskewer.process(page)

        cv2.imwrite(
            str(folders["deskew"] / f"page_{page_idx+1:03d}.png"),
            corrected
        )

        vertical, debug = self.extractor.extract(corrected)

        cv2.imwrite(
            str(folders["vertical"] / f"page_{page_idx+1:03d}.png"),
            vertical
        )

        detected, separator = self.line_detector.detect(vertical)

        cv2.imwrite(
            str(folders["detected"] / f"page_{page_idx+1:03d}.png"),
            detected
        )

        if separator is None:

            print("Separator : FAILED")

            cv2.imwrite(
                str(folders["failed"] / f"page_{page_idx+1:03d}_gray.png"),
                debug["gray"]
            )

            cv2.imwrite(
                str(folders["failed"] / f"page_{page_idx+1:03d}_binary.png"),
                debug["binary"]
            )

            cv2.imwrite(
                str(folders["failed"] / f"page_{page_idx+1:03d}_vertical.png"),
                vertical
            )

            return {
                "page": page_idx + 1,
                "status": "FAILED",
                "deskew_angle": angle,
                "separator_x": None,
                "width": None,
                "height": None,
            }

        
        crop = self.cropper.crop(
            corrected,
            separator
        )

        cv2.imwrite(
            str(folders["cropped"] / f"page_{page_idx+1:03d}.png"),
            crop
        )

        # -----------------------------------------
        # Remove evaluator annotations
        # -----------------------------------------

        if self.config.USE_ANNOTATION_REMOVAL:

            clean_crop, mask = self.annotation_remover.process_debug(crop)

        else:

            clean_crop = crop

            mask = np.zeros(
                crop.shape[:2],
                dtype=np.uint8
            )

        # Save debug images

        cv2.imwrite(
            str(
                folders["annotation_removed"] /
                f"page_{page_idx+1:03d}.png"
            ),
            clean_crop
        )

        cv2.imwrite(
            str(
                folders["annotation_removed"] /
                f"page_{page_idx+1:03d}_mask.png"
            ),
            mask
        )

        # Continue OCR pipeline

        gray, binary = self.ocr_preprocessor.process(clean_crop)

        cv2.imwrite(
            str(folders["preprocessed"] / f"page_{page_idx+1:03d}_gray.png"),
            gray
        )

        cv2.imwrite(
            str(folders["preprocessed"] / f"page_{page_idx+1:03d}_binary.png"),
            binary
        )

        # ------------------------------------------
        # OCR
        # ------------------------------------------

        if self.config.USE_ANNOTATION_DETECTION:

            print("Mode : ROI OCR")

            rois, debug = self.annotation_detector.detect(gray, binary)

            if self.config.SAVE_ROIS:

                for i, roi in enumerate(rois):

                    cv2.imwrite(

                        str(
                            folders['roi'] /
                            f"page_{page_idx+1:03d}_roi_{i+1}.png"
                        ),

                        roi["roi"]

                    )
            
            if self.config.SAVE_DEBUG_IMAGES:

                cv2.imwrite(

                    str(
                        folders['recognizer'] /
                        f"page_{page_idx+1:03d}.png"
                    ),

                    debug["detected"]

                )

            extracted_text = self.text_extractor.extract(
                gray,
                rois
            )

        else:

            print("Mode : Full Margin OCR")

            extracted_text = self.text_extractor.extract(
                gray
            )
        
        if len(extracted_text) > 0:

            ocr_df = pd.DataFrame(extracted_text)

            ocr_df.to_csv(

                folders['ocr'] /
                f"page_{page_idx+1:03d}.csv",

                index=False

            )

            print("\nOCR Results")

            for item in extracted_text:

                print(

                    f"Block {item['block_id']}: "
                    f"{item['text']}"

                )

        else:

            print("No text detected.")

        print(f"Separator : {separator['x']}")
        return {
            "page": page_idx + 1,
            "status": "SUCCESS",
            "deskew_angle": angle,
            "separator_x": separator["x"],
            "width": separator["w"],
            "height": separator["h"],
        }

        
            
    def save_report(

        self,
        pdf_name,
        pdf_output,
        results
    ):
        print(
            f"Saving report for {pdf_name}"
        )


        # ==============================================
        # Save Report
        # ==============================================

        df = pd.DataFrame(results)

        df.to_csv(
            pdf_output / "report.csv",
            index=False
        )

        success = df[df["status"] == "SUCCESS"]

        self.summary.append({

            "pdf": pdf_name,

            "pages": len(df),

            "success":
                (df["status"] == "SUCCESS").sum(),

            "failed":
                (df["status"] == "FAILED").sum(),

            "mean_separator_x":
                success["separator_x"].mean()
                if len(success) else None,

            "std_separator_x":
                success["separator_x"].std()
                if len(success) else None

        })
    
    def save_summary(self):

        summary_df = pd.DataFrame(self.summary)

        summary_df.to_csv(
            self.output_dir / "summary.csv",
            index=False
        )

        print("\n")
        print("=" * 70)
        print("Pipeline Finished")
        print("=" * 70)