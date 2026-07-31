"""
Page Preprocessor

Prepares the full scanned page before separator detection.

Currently acts as a pass-through.
Future versions may include:
    - illumination correction
    - scanner artifact removal
    - page denoising
"""

class PagePreprocessor:

    def process(self, image):

        return image