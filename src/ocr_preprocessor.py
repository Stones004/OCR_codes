"""
OCR Preprocessor

Prepares cropped annotation margins for OCR.

Every preprocessing algorithm should be implemented as an
independent method so that they can be benchmarked easily.
"""

import cv2
import numpy as np
from pathlib import Path

from src.config import OcrPreprocessorConfig

class OCRPreprocessor:

    def __init__(self, config: OcrPreprocessorConfig):

        self.method = config.method.lower()

        self.scale = config.scale

        # Connected Components
        self.min_area = config.min_area

        # Morphology
        self.kernel_size = config.kernel_size
        self.iterations = config.iterations

        # Non-local-means denoising strength (otsu method)
        self.denoise_h = config.denoise_h

        self.methods = {
            "connected_components": self.connected_components,
            "otsu": self.otsu,
            "morph_close": self.morph_close,
            "morph_dilate": self.morph_dilate,
            "bilateral_otsu": self.bilateral_otsu,
            "full_pipeline": self.full_pipeline,
        }

        print("\nRegistered preprocessing methods:")
        for name, fn in self.methods.items():
            print(f"{name:25} -> {type(fn)}")


    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def to_gray(self, image):

        if len(image.shape) == 3:

            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        return image.copy()


    def upscale(self, gray):

        return cv2.resize(
            gray,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC

        )
    
    def connected_components(self, gray):

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8

        )

        clean = np.zeros_like(binary)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                clean[labels == i] = 255

        return clean

    def otsu(self, gray):
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=self.denoise_h
        )

        _, binary = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary
    

    def morph_close(self, gray):
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

        kernel = np.ones(
            (
                self.kernel_size,
                self.kernel_size
            ),
            np.uint8
        )
        return cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )
    
    def morph_dilate(self, gray):

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones(
            (
                self.kernel_size,
                self.kernel_size
            ),
            np.uint8
        )

        return cv2.dilate(
            binary,
            kernel,
            iterations=self.iterations
        )


    def bilateral_otsu(self, gray):

        _, binary = cv2.threshold(

            gray,

            0,

            255,

            cv2.THRESH_BINARY + cv2.THRESH_OTSU

        )

        return binary
    

    def full_pipeline(self, gray):

        _, binary = cv2.threshold(

            gray,

            0,

            255,

            cv2.THRESH_BINARY + cv2.THRESH_OTSU

        )

        kernel = np.ones(
            (
                self.kernel_size,
                self.kernel_size
            ),
            np.uint8
        )

        return cv2.morphologyEx(

            binary,

            cv2.MORPH_CLOSE,

            kernel

        )

    # --------------------------------------------------
    # Main Interface
    # --------------------------------------------------

    def process(self, image):

        gray = self.to_gray(image)

        gray = self.upscale(gray)

        if self.method not in self.methods:

            raise ValueError(

                f"Unknown preprocessing method: {self.method}"

            )

        processed = self.methods[self.method](gray)

        return gray, processed
    


    def available_methods(self):
        """Return all registered preprocessing methods."""
        return list(self.methods.keys())
    
    def process_method(self, image, method):
        """
        Process an image using a specific preprocessing method.
        """

        if method not in self.methods:
            raise ValueError(f"Unknown preprocessing method: {method}")

        gray = self.to_gray(image)
        gray = self.upscale(gray)

        processed = self.methods[method](gray.copy())

        return gray, processed
    

    def benchmark(self, image):
        """
        Run every preprocessing method on the same image.
        """

        gray = self.to_gray(image)
        gray = self.upscale(gray)

        outputs = {}

        for method in self.available_methods():

            outputs[method] = self.methods[method](gray.copy())

        return gray, outputs
    

    def save_benchmark(self, outputs, output_dir):
        """
        Save all benchmark images.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, image in outputs.items():

            cv2.imwrite(
                str(output_dir / f"{name}.png"),
                image
            )