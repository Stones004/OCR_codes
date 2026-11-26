"""
OCR Annotator

Renders an OCR result as an image: the crop the model read, with the
predicted text printed below it. Replaces the per-page OCR CSV.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import OcrOutputConfig


def _load_font(candidates, size):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class OcrAnnotator:

    def __init__(self, config: OcrOutputConfig):

        self.caption_height = config.caption_height
        self.min_canvas_width = config.min_canvas_width
        self.text_x = config.text_x
        self.text_y = config.text_y
        self.bg = tuple(config.bg)
        self.text_color = tuple(config.text_color)
        self.empty_label = config.empty_label
        self.font = _load_font(config.font_candidates, config.font_size)

    # --------------------------------------------------

    def annotate(self, image: np.ndarray, text: str) -> Image.Image:
        """image: the ndarray that was fed to OCR. Returns it with `text` printed below.

        Crops are often narrower than their own prediction, so the canvas width is
        measured from the rendered text; a crop already wider keeps its own width,
        a narrower one is centred in the extra space."""

        img = Image.fromarray(image).convert("RGB")
        w, h = img.size

        label = text.strip() if text and text.strip() else self.empty_label

        text_left, _, text_right, _ = ImageDraw.Draw(img).textbbox((0, 0), label, font=self.font)
        text_w = text_right - text_left

        canvas_w = max(w, self.min_canvas_width, text_w + 2 * self.text_x)
        canvas = Image.new("RGB", (canvas_w, h + self.caption_height), self.bg)
        canvas.paste(img, ((canvas_w - w) // 2, 0))

        ImageDraw.Draw(canvas).text(
            (self.text_x, h + self.text_y), label,
            fill=self.text_color, font=self.font,
        )

        return canvas
