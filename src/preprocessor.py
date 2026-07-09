import cv2
import numpy as np


class Preprocessor:

    def __init__(self, scale=4):

        self.scale = scale

    # --------------------------------------------------

    def upscale(self, img):

        return cv2.resize(
            img,
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_CUBIC
        )

    # --------------------------------------------------

    def otsu(self, img):

        blur = cv2.GaussianBlur(
            img,
            (3,3),
            0
        )

        _, binary = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return binary

    # --------------------------------------------------

    def morph_close(self, img):

        binary = self.otsu(img)

        kernel = np.ones((2,2), np.uint8)

        return cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

    # --------------------------------------------------

    def morph_dilate(self, img):

        binary = self.otsu(img)

        kernel = np.ones((2,2), np.uint8)

        return cv2.dilate(
            binary,
            kernel,
            iterations=1
        )

    # --------------------------------------------------

    def process(
        self,
        image,
        method="original"
    ):

        if len(image.shape) == 3:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = image.copy()

        gray = self.upscale(gray)

        method = method.lower()

        if method == "original":
            return gray

        elif method == "otsu":
            return self.otsu(gray)

        elif method == "morph_close":
            return self.morph_close(gray)

        elif method == "morph_dilate":
            return self.morph_dilate(gray)

        else:

            raise ValueError(
                f"Unknown preprocessing method: {method}"
            )