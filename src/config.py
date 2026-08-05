"""
Pipeline Configuration

Central configuration for the OCR pipeline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    # ------------------------------------------
    # OCR
    # ------------------------------------------

    OCR_BACKEND: str = "lmstudio"

    # ------------------------------------------
    # Pipeline Mode
    # ------------------------------------------

    USE_ANNOTATION_DETECTION: bool = True

    # ------------------------------------------
    # Debug
    # ------------------------------------------

    SAVE_DEBUG_IMAGES: bool = True

    SAVE_ROIS: bool = True

    # ------------------------------------------
    # OCR
    # ------------------------------------------

    OCR_MAX_TOKENS: int = 64

    USE_OCR_PREPROCESSOR: bool = True