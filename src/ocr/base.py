"""
Base OCR Interface
"""

from abc import ABC, abstractmethod


class OCRBackend(ABC):

    @abstractmethod
    def read(self, image, preprocess=True):
        """
        Perform OCR on an image.

        Parameters
        ----------
        image : ndarray | str | Path
        preprocess : bool
            Whether the backend should apply its own preprocessing
            (backends without any simply ignore it).

        Returns
        -------
        list | dict
        """
        pass
