from pathlib import Path

import cv2
import easyocr
import numpy as np

from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel

# pyrefly: ignore [missing-import]
import torch
import cv2
from PIL import Image
import numpy as np
from .base import OCRBackend


"""
OCR Engine

Recognizes handwritten text from annotation ROIs using Microsoft's
TrOCR handwriting recognition model and returns the predicted text
along with confidence information.
"""

class TrOCRBackend(OCRBackend):

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        ).to(self.device)

        self.model.eval()

    # -------------------------------------------------

    @torch.no_grad()
    def read(self, image):

        if isinstance(image, str):

            image = cv2.imread(image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )
        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        pil = Image.fromarray(image)

        pixel_values = self.processor(
            images=pil,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        generated_ids = self.model.generate(
            pixel_values,
            max_new_tokens=64
        )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return {
            "text": text.strip(),
            "confidence": 1.0,
            "words": []
        }