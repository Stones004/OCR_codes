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