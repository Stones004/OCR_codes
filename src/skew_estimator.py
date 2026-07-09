import cv2
import numpy as np


class SkewEstimator:

    def estimate(self, points):

        if len(points) < 50:
            return 0.0

        points = points.astype(np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        angle = np.degrees(
            np.arctan2(vx, vy)
        )

        return float(angle)

    def rotate(self, image, angle):

        h, w = image.shape[:2]

        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        return cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )