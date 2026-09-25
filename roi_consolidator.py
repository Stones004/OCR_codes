import argparse
import shutil

from src.config import load_config

parser = argparse.ArgumentParser(
    description="Collect every <pdf>/roi image into one folder"
)
parser.add_argument(
    "--config",
    default=None,
    help="Path to a config YAML (default: $OCR_CONFIG, else <repo>/config.yaml)",
)
args = parser.parse_args()

cfg = load_config(args.config)
settings = cfg.roi_consolidator

# Root directory containing the per-PDF output folders
ROOT_DIR = settings.root_dir

# Destination folder
DEST_DIR = settings.dest_dir
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Supported image extensions
IMAGE_EXTS = set(cfg.image_extensions)

count = 0

# Iterate through every per-PDF folder
for doc_folder in sorted(ROOT_DIR.glob(settings.doc_glob)):

    if not doc_folder.is_dir():
        continue

    roi_folder = doc_folder / cfg.layout.roi_dir

    if not roi_folder.exists():
        continue

    # Copy every image in roi
    for img_path in roi_folder.iterdir():

        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # Prefix filename with document name to avoid collisions
        new_name = f"{doc_folder.name}_{img_path.name}"

        shutil.copy2(
            img_path,
            DEST_DIR / new_name
        )

        count += 1

print("\nCopied " + str(count) + " ROI images to:")
print(DEST_DIR)
