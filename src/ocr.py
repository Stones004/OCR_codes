from pathlib import Path

import cv2
import easyocr
import numpy as np


class OCR:

    def __init__(
        self,
        languages=None,
        gpu=False
    ):
        """
        OCR Wrapper

        Parameters
        ----------
        languages : list[str]
            OCR languages.

        gpu : bool
            Enable GPU if available.
        """

        if languages is None:
            languages = ["en"]

        print("=" * 60)
        print("Initializing EasyOCR")
        print("=" * 60)

        self.reader = easyocr.Reader(
            languages,
            gpu=gpu
        )

        print("OCR Ready\n")

    # -------------------------------------------------

    def read(self, image):

        """
        Parameters
        ----------
        image :
            Either

            • image path

            or

            • OpenCV image (numpy array)

        Returns
        -------
        dict
        """

        # ---------------------------------------------
        # Read image if a path is provided
        # ---------------------------------------------

        if isinstance(image, (str, Path)):

            image = cv2.imread(str(image))

            if image is None:

                raise FileNotFoundError(
                    f"Could not load image : {image}"
                )

        # ---------------------------------------------
        # Convert BGR → RGB
        # ---------------------------------------------

        if len(image.shape) == 3:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        # ---------------------------------------------
        # OCR
        # ---------------------------------------------

        result = self.reader.readtext(
            image,
            detail=1,
            paragraph=False
        )

        if len(result) == 0:

            return {

                "text": "",

                "confidence": 0.0,

                "words": []
            }

        words = []

        confidences = []

        texts = []

        for item in result:

            bbox = item[0]

            text = item[1]

            confidence = float(item[2])

            words.append({

                "bbox": bbox,

                "text": text,

                "confidence": confidence
            })

            texts.append(text)

            confidences.append(confidence)

        return {

            "text": " ".join(texts),

            "confidence": float(np.mean(confidences)),

            "words": words
        }