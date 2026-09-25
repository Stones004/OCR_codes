import shutil
from pathlib import Path


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import textwrap


from .config import (
    IMAGE_FOLDER,
    COMPLETED_FOLDER,
    PREVIEW_FOLDER,
    IMAGE_EXTENSIONS,
    BATCH_SIZE,
    CFG,
)


class FileManager:

    def __init__(self):
        self.preview = CFG.dataset_builder.preview

        IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
        COMPLETED_FOLDER.mkdir(parents=True, exist_ok=True)
        PREVIEW_FOLDER.mkdir(parents=True, exist_ok=True)

    def get_next_batch(self):
        """
        Returns the next batch of images to process.
        """

        images = sorted([
            image
            for image in IMAGE_FOLDER.iterdir()
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        ])

        return images[:BATCH_SIZE]

    def move_completed(self, image_path):
        """
        Move a processed image to the completed folder.
        """

        destination = COMPLETED_FOLDER / image_path.name

        shutil.move(str(image_path), str(destination))

    def remaining_images(self):
        """
        Number of images left.
        """

        return len([
            image
            for image in IMAGE_FOLDER.iterdir()
            if image.is_file()
            and image.suffix.lower() in IMAGE_EXTENSIONS
        ])

    def completed_images(self):
        """
        Number of processed images.
        """

        return len([
            image
            for image in COMPLETED_FOLDER.iterdir()
            if image.is_file()
        ])


    def create_preview(self, image_path, text):

        img = Image.open(image_path).convert("RGB")

        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(
                self.preview.font_name, self.preview.font_size
            )
        except OSError:
            font = ImageFont.load_default()

        # Clean OCR text
        text = text.replace("\n", " ")
        text = " ".join(text.split())

        # Limit text length
        if len(text) > self.preview.max_chars:
            text = text[:self.preview.max_chars] + "..."

        # Position at bottom-left
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = bbox[3] - bbox[1]

        x = self.preview.margin
        y = img.height - text_height - self.preview.margin

        draw.text(
            (x, y),
            text,
            fill=tuple(self.preview.text_color),
            font=font,
            stroke_width=self.preview.stroke_width,
            stroke_fill=self.preview.stroke_fill,
        )

        preview_path = PREVIEW_FOLDER / image_path.name

        img.save(preview_path)