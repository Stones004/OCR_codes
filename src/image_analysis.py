import cv2
import numpy as np


class ImageAnalyzer:

    @staticmethod
    def analyze(image):

        height, width = image.shape[:2]

        channels = 1 if len(image.shape) == 2 else image.shape[2]

        gray = image

        if channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mean = float(np.mean(gray))
        std = float(np.std(gray))

        # Variance of Laplacian
        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        return {

            "width": width,

            "height": height,

            "channels": channels,

            "mean_brightness": round(mean,2),

            "contrast": round(std,2),

            "blur_score": round(blur_score,2)
        }