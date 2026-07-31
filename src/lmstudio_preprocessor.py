"""
LM Studio Preprocessor

Applies light image enhancement before sending the cropped annotation
strip to the vision-language OCR model.
"""

import cv2
import numpy as np


class LMStudioPreprocessor:

    def __init__(self):

        # Sharpening kernel
        self.kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

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
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # Sharpen handwriting
        sharpened = cv2.filter2D(
            denoised,
            -1,
            self.kernel
        )

        return sharpened