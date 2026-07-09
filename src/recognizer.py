import cv2
import numpy as np


class Recognizer:

    def __init__(
        self,
        block_size=31,
        c=15,
        min_area=25,
        min_height=10,
        min_width=3
    ):

        self.block_size = block_size
        self.c = c

        self.min_area = min_area
        self.min_height = min_height
        self.min_width = min_width

    # ----------------------------------------------------

    def detect(self, image):

        # ------------------------------------------
        # Convert to grayscale
        # ------------------------------------------

        if len(image.shape) == 3:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = image.copy()

        # ------------------------------------------
        # Adaptive Threshold
        # ------------------------------------------

        binary = cv2.adaptiveThreshold(

            gray,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY_INV,

            self.block_size,

            self.c
        )

        # ------------------------------------------
        # Connected Components
        # ------------------------------------------

        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )

        output = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        components = []

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.min_area:
                continue

            if h < self.min_height:
                continue

            if w < self.min_width:
                continue

            components.append({

                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area

            })

            cv2.rectangle(

                output,

                (x, y),

                (x + w, y + h),

                (0, 255, 0),

                2
            )

        # ----------------------------------------------------
        # No components
        # ----------------------------------------------------

        if len(components) == 0:
            debug = {

                "gray": gray,

                "binary": binary,

                "detected": output

            }

            return None, debug

        # ----------------------------------------------------
        # Merge all detected components
        # ----------------------------------------------------

        x1 = min(c["x"] for c in components)
        y1 = min(c["y"] for c in components)

        x2 = max(c["x"] + c["w"] for c in components)
        y2 = max(c["y"] + c["h"] for c in components)

        padding = 15

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)

        x2 = min(gray.shape[1], x2 + padding)
        y2 = min(gray.shape[0], y2 + padding)

        roi = gray[
            y1:y2,
            x1:x2
        ]

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            (0, 0, 255),

            3

        )

        debug = {

            "gray": gray,

            "binary": binary,

            "detected": output,

            "roi": roi

        }

        return roi, debug