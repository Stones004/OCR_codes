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

        # Anti-aliased/chromatic-fringe edge pixels around the colored
        # ink are too faint (low saturation) to match the strict HSV
        # range directly, but loosening that range globally risks
        # matching real black/gray ink too, since near-gray pixels have
        # unstable, noisy hue at low saturation. Instead, only trust a
        # loose "any non-white pixel" pass within a generous dilation of
        # the strict mask -- real content elsewhere on the page is
        # spatially far from any confirmed red/green ink and is never
        # touched by it.
        remove_fringe=True,
        fringe_zone_kernel_size=11,
        fringe_zone_iterations=4,
        fringe_lightness_max=250,

        # A tiny strict-mask hit (a handful of pixels) is more likely a
        # false trigger -- e.g. a JPEG-compression edge artifact next to
        # dark ink of an unrelated color -- than genuine red/green ink.
        # Expanding the fringe zone around one of those would erase real
        # nearby content instead of cleaning up a real halo, so only
        # components at least this large earn the wider fringe search.
        fringe_min_component_area=500,
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

        self.remove_fringe = remove_fringe
        self.fringe_zone_kernel = np.ones(
            (fringe_zone_kernel_size, fringe_zone_kernel_size),
            np.uint8
        )
        self.fringe_zone_iterations = fringe_zone_iterations
        self.fringe_lightness_max = fringe_lightness_max
        self.fringe_min_component_area = fringe_min_component_area

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

        if self.remove_fringe:

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )

            confident_mask = np.zeros_like(mask)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.fringe_min_component_area:
                    confident_mask[labels == i] = 255

            fringe_zone = cv2.dilate(
                confident_mask,
                self.fringe_zone_kernel,
                iterations=self.fringe_zone_iterations
            )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            _, non_white = cv2.threshold(
                gray,
                self.fringe_lightness_max,
                255,
                cv2.THRESH_BINARY_INV
            )

            fringe = cv2.bitwise_and(non_white, fringe_zone)

            mask = cv2.bitwise_or(mask, fringe)

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