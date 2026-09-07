import cv2
import numpy as np


"""
Deskewer

Automatically estimates the skew angle of scanned pages using
projection profile analysis and rotates the image to improve
alignment for downstream processing.
"""

class Deskewer:

    def __init__(
        self,
        angle_range=(-5.0, 5.0),
        coarse_step=0.5,
        fine_step=0.05
    ):

        self.angle_range = angle_range
        self.coarse_step = coarse_step
        self.fine_step = fine_step

    # --------------------------------------------------
    # Horizontal Projection Score
    # --------------------------------------------------

    @staticmethod
    def projection_score(binary):

        projection = np.sum(binary == 255, axis=1)

        return np.var(projection)

    # --------------------------------------------------
    # Rotate image
    # --------------------------------------------------

    @staticmethod
    def rotate(image, angle):

        h, w = image.shape[:2]

        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        return rotated

    # --------------------------------------------------
    # Find Best Angle
    # --------------------------------------------------

    def find_best_angle(self, binary):

        best_angle = 0
        best_score = -1

        # ------------------------
        # Coarse Search
        # ------------------------

        for angle in np.arange(
            self.angle_range[0],
            self.angle_range[1] + self.coarse_step,
            self.coarse_step
        ):

            rotated = self.rotate(binary, angle)

            score = self.projection_score(rotated)

            if score > best_score:

                best_score = score
                best_angle = angle

        # If the coarse search's best angle sits at the very edge of the
        # search range, the projection-variance metric is diverging
        # toward the boundary rather than converging on a real skew (seen
        # on near-blank pages and pages with a dominant watermark
        # competing with the true ruled lines). Trust zero skew instead
        # of chasing that runaway result with a fine search.
        if best_angle in (self.angle_range[0], self.angle_range[1]):
            return 0

        # ------------------------
        # Fine Search
        # ------------------------

        fine_start = best_angle - 0.5
        fine_end = best_angle + 0.5

        best_score = -1

        for angle in np.arange(
            fine_start,
            fine_end + self.fine_step,
            self.fine_step
        ):

            rotated = self.rotate(binary, angle)

            score = self.projection_score(rotated)

            if score > best_score:

                best_score = score
                best_angle = angle

        return best_angle

    # --------------------------------------------------
    # Deskew
    # --------------------------------------------------

    def process(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        angle = self.find_best_angle(binary)

        corrected = self.rotate(image, angle)

        return corrected, angle