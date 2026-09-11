import os
from pathlib import Path


from pathlib import Path

OCR_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = OCR_ROOT / ".env"

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

PREVIEW_FOLDER = DATASET_DIR / "preview"

IMAGE_FOLDER = Path(r"G:\OCR_codes\outputs_17-08\all_roi_images")

COMPLETED_FOLDER = DATASET_DIR / "completed"

CSV_FILE = DATASET_DIR / "annotations.csv"

# ==========================================================
# OCR Settings
# ==========================================================

MODEL = "mistral-ocr-latest"

BATCH_SIZE = 500

# OCR backend: "mistral" (hosted API) or "parseq" (local fine-tuned model).
# Override per-run with `python run.py --backend parseq`.
OCR_BACKEND = os.getenv("OCR_BACKEND", "mistral")

# ==========================================================
# Local PARSeq Settings
# ==========================================================

PARSEQ_VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"

PARSEQ_CHECKPOINT = (
    PROJECT_ROOT / "models" / "parseq_finetuned_epoch21_acc93.57_ned94.93.ckpt"
)

# Predictions below this confidence are flagged for human review.
PARSEQ_CONFIDENCE_THRESHOLD = 0.90

# "low_confidence": only flag predictions below PARSEQ_CONFIDENCE_THRESHOLD
#   (production inference -- most predictions are trusted as-is).
# "all": flag every prediction for review
#   (building a new training batch -- nothing is trusted until a human
#   confirms it, since these will be fed back into fine-tuning).
# Override per-run with `python run.py --review all`.
PARSEQ_REVIEW_MODE = os.getenv("PARSEQ_REVIEW_MODE", "low_confidence")

# PARSeq predictions land here for review, NOT in annotations.csv --
# they aren't trusted labels until a human confirms/corrects them.
REVIEW_CSV = DATASET_DIR / "parseq_review_queue.csv"

# ==========================================================
# Image Types
# ==========================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}