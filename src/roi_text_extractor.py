"""
ROI Text Extractor

Provides a unified interface for OCR extraction.

If ROIs are supplied, OCR is performed on each ROI.
Otherwise, OCR is performed on the entire image.

The output is normalized into a common format so that the
rest of the pipeline remains independent of the OCR backend.
"""

from src.config import OcrConfig
from src.ocr import create_backend


class ROITextExtractor:

    def __init__(
        self,
        config: OcrConfig,
        backend=None,
        use_backend_preprocessing=None,
    ):
        """
        config : the `ocr` section of config.yaml.
        backend, use_backend_preprocessing : optional per-call overrides
            of config.backend / config.use_backend_preprocessing
            (used by research/evaluate.py).
        """

        backend = backend or config.backend

        self.ocr = create_backend(backend, config)

        if use_backend_preprocessing is None:
            use_backend_preprocessing = config.use_backend_preprocessing

        self.use_backend_preprocessing = use_backend_preprocessing

    # --------------------------------------------------

    def extract(self, image, rois=None):

        predictions = []

        # --------------------------------------------------
        # Whole-image OCR
        # --------------------------------------------------

        if rois is None:

            result = self.ocr.read(
                image,
                preprocess=self.use_backend_preprocessing
            )

            if isinstance(result, list):
                predictions.extend(result)
            else:
                predictions.append(result)

        # --------------------------------------------------
        # ROI OCR
        # --------------------------------------------------

        else:

            for roi in rois:

                result = self.ocr.read(roi["roi"])

                if isinstance(result, list):
                    predictions.extend(result)
                else:
                    predictions.append(result)

        # --------------------------------------------------
        # Normalize Output
        # --------------------------------------------------

        results = []

        for idx, item in enumerate(predictions):

            results.append({

                "block_id": idx + 1,
                "text": item.get("text", "")

            })

        return results
