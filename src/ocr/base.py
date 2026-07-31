"""
Base OCR Interface
"""

from abc import ABC, abstractmethod


class OCRBackend(ABC):

    @abstractmethod
    def read(self, image):
        """
        Perform OCR on an image.

        Parameters
        ----------
        image : ndarray

        Returns
        -------
        list | dict
        """
        pass