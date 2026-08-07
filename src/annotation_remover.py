import cv2
import numpy as np

class AnnotationRemover:

    def __init__(
        self,

        # -------------------------
        # Red HSV
        # -------------------------
        #red_lower1=(0, 20, 20),
        #red_lower1 = (0, 80, 40),
        red_lower1 = (0, 100, 60),
        red_upper1=(15, 255, 255),

        #red_lower2=(165, 20, 20),
        #red_lower2 = (165, 80, 40),
        red_lower2 = (168, 100, 60),
        red_upper2=(179, 255, 255),

        # -------------------------
        # Green HSV
        # -------------------------
        #green_lower=(30, 20, 20),
        #green_lower = (35, 80, 40),
        green_lower = (40, 100, 60),
        green_upper=(100, 255, 255),

        kernel_size=5,
        close_iterations=2,
        dilate_iterations=2,

        use_inpaint=True,
        inpaint_radius=5,
    ):

        self.red_lower1 = np.array(red_lower1, dtype=np.uint8)
        self.red_upper1 = np.array(red_upper1, dtype=np.uint8)

        self.red_lower2 = np.array(red_lower2, dtype=np.uint8)
        self.red_upper2 = np.array(red_upper2, dtype=np.uint8)

        self.green_lower = np.array(green_lower, dtype=np.uint8)
        self.green_upper = np.array(green_upper, dtype=np.uint8)

        self.kernel = np.ones(
            (kernel_size, kernel_size),
            np.uint8
        )

        self.close_iterations = close_iterations
        self.dilate_iterations = dilate_iterations

        self.use_inpaint = use_inpaint
        self.inpaint_radius = inpaint_radius

    def create_mask(self, image):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        red1 = cv2.inRange(
            hsv,
            self.red_lower1,
            self.red_upper1
        )

        red2 = cv2.inRange(
            hsv,
            self.red_lower2,
            self.red_upper2
        )

        red = cv2.bitwise_or(
            red1,
            red2
        )

        green = cv2.inRange(
            hsv,
            self.green_lower,
            self.green_upper
        )

        mask = cv2.bitwise_or(
            red,
            green
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=self.close_iterations
        )

        mask = cv2.dilate(
            mask,
            self.kernel,
            iterations=self.dilate_iterations
        )

        return mask

    def process(self, image):

        mask = self.create_mask(image)

        cleaned = image.copy()

        cleaned[mask > 0] = (255, 255, 255)

        return cleaned

    def process_debug(self, image):

        mask = self.create_mask(image)

        cleaned = image.copy()
        cleaned[mask > 0] = (255, 255, 255)

        return cleaned, mask