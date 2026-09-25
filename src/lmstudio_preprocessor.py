"""
LM Studio Preprocessor

Applies light image enhancement before sending the cropped annotation
strip to the vision-language OCR model.
"""

import cv2
import numpy as np

from src.config import LMStudioPreprocessConfig


class LMStudioPreprocessor:

    def __init__(self, config: LMStudioPreprocessConfig):

        self.denoise_h = config.denoise_h
        self.template_window = config.template_window
        self.search_window = config.search_window

        # Sharpening kernel
        self.kernel = np.array(config.sharpen_kernel, dtype=np.float32)

    # --------------------------------------------------

    def process(self, image):

        # Already grayscale
        if len(image.shape) == 2:
            gray = image.copy()

        # Colour image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Remove scanner noise while preserving handwriting
        denoised = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=self.denoise_h,
            templateWindowSize=self.template_window,
            searchWindowSize=self.search_window
        )

        # Sharpen handwriting
        sharpened = cv2.filter2D(
            denoised,
            -1,
            self.kernel
        )

        return sharpened
