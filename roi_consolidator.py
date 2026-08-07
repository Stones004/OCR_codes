from pathlib import Path
import shutil

# Root directory containing Doc**** folders
ROOT_DIR = Path("outputs")

# Destination folder
DEST_DIR = ROOT_DIR / "all_roi_images"
DEST_DIR.mkdir(exist_ok=True)

# Supported image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

count = 0

# Iterate through every Doc**** folder
for doc_folder in sorted(ROOT_DIR.glob("Doc*")):

    if not doc_folder.is_dir():
        continue

    roi_folder = doc_folder / "roi"

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

print(f"\nCopied {count} ROI images to:")
print(DEST_DIR)