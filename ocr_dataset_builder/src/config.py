"""
Dataset builder configuration.

Everything comes from the shared config.yaml at the repo root (the
`dataset_builder` section, plus `review`, `image_extensions` and the
`ocr.parseq` worker settings). This module only loads that file and
re-exports the values under the names the rest of the builder uses.

Two per-run environment overrides are honoured, as before:
    OCR_BACKEND         -> dataset_builder.ocr_backend
    PARSEQ_REVIEW_MODE  -> dataset_builder.review_mode
and OCR_CONFIG selects an alternative YAML file.

The root src/config.py is loaded by file path because this package is
also called `src`, so a normal import would resolve to the wrong one.
"""

import importlib.util
import os
import sys
from pathlib import Path

OCR_ROOT = Path(__file__).resolve().parents[2]


def _load_root_config_module():

    module_name = "ocr_root_config"

    spec = importlib.util.spec_from_file_location(
        module_name,
        OCR_ROOT / "src" / "config.py",
    )

    module = importlib.util.module_from_spec(spec)

    # dataclass type-hint resolution looks the module up by name
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


_root_config = _load_root_config_module()

CFG = _root_config.load_config()

_B = CFG.dataset_builder

# ==========================================================
# Paths
# ==========================================================

ENV_FILE = _B.env_file

DATASET_DIR = _B.dataset_dir

PREVIEW_FOLDER = _B.preview_dir

IMAGE_FOLDER = _B.image_dir

COMPLETED_FOLDER = _B.completed_dir

CSV_FILE = _B.annotations_csv

# PARSeq predictions land here for review, NOT in annotations.csv --
# they aren't trusted labels until a human confirms/corrects them.
REVIEW_CSV = _B.review_csv

# ==========================================================
# OCR Settings
# ==========================================================

MODEL = _B.mistral.model

BATCH_SIZE = _B.batch_size

# OCR backend: "mistral" (hosted API) or "parseq" (local fine-tuned model).
# Override per-run with `python run.py --backend parseq`.
OCR_BACKEND = os.getenv("OCR_BACKEND", _B.ocr_backend)

# ==========================================================
# Local PARSeq Settings
# ==========================================================

PARSEQ_VENV_PYTHON = _B.parseq_venv_python

PARSEQ_CHECKPOINT = _B.parseq_checkpoint

# Predictions below this confidence are flagged for human review.
PARSEQ_CONFIDENCE_THRESHOLD = CFG.review.confidence_threshold

# "low_confidence": only flag predictions below PARSEQ_CONFIDENCE_THRESHOLD
#   (production inference -- most predictions are trusted as-is).
# "all": flag every prediction for review
#   (building a new training batch -- nothing is trusted until a human
#   confirms it, since these will be fed back into fine-tuning).
# Override per-run with `python run.py --review all`.
PARSEQ_REVIEW_MODE = os.getenv("PARSEQ_REVIEW_MODE", _B.review_mode)

# ==========================================================
# Image Types
# ==========================================================

IMAGE_EXTENSIONS = set(CFG.image_extensions)
