import cv2


class Cropper:

    def __init__(self, right_padding=15):
        """
        right_padding :
            Number of pixels to include AFTER the separator.
        """

        self.right_padding = right_padding

    def crop(self, image, separator):

        if separator is None:
            return None

        x = separator["x"] + self.right_padding

        # Prevent overflow
        x = min(x, image.shape[1])

        cropped = image[:, :x]

        return cropped