import cv2
from src.preprocessor import Preprocessor
from src.ocr import OCR


class ROITextExtractor:

    def __init__(self):
        self.preprocessor = Preprocessor(scale=4)
        self.ocr = OCR()
    def extract(self, rois):

        results = []

        # Read annotations from top to bottom
        rois = sorted(
            rois,
            key=lambda roi: roi["bbox"][1]
        )

        for idx, roi in enumerate(rois):

            x1, y1, x2, y2 = roi["bbox"]

            crop = roi["roi"]

            # Preprocess ROI
            processed = self.preprocessor.process(
                crop,
                method="morph_close"
            )

            # OCR
            prediction = self.ocr.read(processed)

            text = prediction["text"].strip()
            confidence = prediction["confidence"]

            results.append({
                "block_id": idx + 1,
                "bbox": (x1, y1, x2, y2),
                "text": text,
                "confidence": round(confidence, 3)
            })

            print(
                f"Block {idx+1}: '{text}' "
                f"(Conf: {confidence:.2f})"
            )

        return results