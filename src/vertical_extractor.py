import cv2
import numpy as np

class VerticalExtractor:

    def __init__(self,
                 threshold=180,
                 kernel_height=120):
        self.threshold = threshold
        self.kernel_height = kernel_height

    def extract(self, image):

        debug = {}

        # --------------------------------------------------
        # 1. Convert to grayscale
        # --------------------------------------------------

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        debug["gray"] = gray

        # --------------------------------------------------
        # 2. Binary Inverse
        #
        # Background -> Black
        # Ink/Lines -> White
        # --------------------------------------------------

        _, binary = cv2.threshold(
            gray,
            self.threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        debug["binary"] = binary

        # --------------------------------------------------
        # 3. Extract Vertical Structures
        # --------------------------------------------------

        height, width = image.shape[:2]

        kernel_height = max(80, height // 25)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, kernel_height)
        )

        # Repair broken lines
        repair_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 15)
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            repair_kernel
        )

        # Now isolate vertical objects
        '''vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 120)
        )'''

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, kernel_height)
        )

        vertical = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vertical_kernel
        )

        debug["vertical"] = vertical

        print("Inside VerticalExtractor")
        print(vertical.shape)
        print(vertical.dtype)
        print(np.unique(vertical))

        return vertical, debug