"""
Noise Classifier

Wraps the ResNet-18 noise/clean crop classifier trained in the
trocr-handwriting-v2 project (src/noise_classifier/train.py there) so this
pipeline can tag every detected ROI as "noise" (blank, illegible, or never
worth transcribing) or "clean" (real handwriting worth sending to OCR),
before the OCR backend ever sees it.

Runs in-process -- torch/torchvision already live in this project's root
venv, unlike PARSeq, which needs a second venv and a subprocess worker.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision import models, transforms

from src.config import NoiseClassifierConfig

CLASSES = ["clean", "noise"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def _load_font(candidates, size):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class NoiseClassifier:

    def __init__(self, config: NoiseClassifierConfig):

        if not config.checkpoint.exists():
            raise FileNotFoundError(
                f"Noise classifier checkpoint not found: {config.checkpoint}\n"
                "Copy it from the trocr-handwriting-v2 project's "
                "models/noise_classifier/best.pt, or retrain it there."
            )

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        checkpoint = torch.load(config.checkpoint, map_location=self.device)

        self.model = _build_model()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

        img_size = checkpoint.get("img_size", 128)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.confidence_threshold = config.confidence_threshold
        self.save_review_crops = config.save_review_crops

        self.caption_height = config.caption_height
        self.caption_bg = tuple(config.caption_bg)
        self.bucket_colors = {
            "clean": tuple(config.color_clean),
            "noise": tuple(config.color_noise),
            "uncertain": tuple(config.color_uncertain),
        }
        self.font = _load_font(config.font_candidates, config.caption_font_size)

    # --------------------------------------------------

    def classify(self, roi_image: np.ndarray) -> float:
        """roi_image: a grayscale ndarray, as produced by AnnotationDetector.
        Returns noise_score = P(noise) in [0, 1] -- 0 means confidently clean, 1 means
        confidently noise, 0.5 means the model has no real signal either way. Unlike a
        per-class "confidence" (which can never fall below 0.5 in a 2-class model), this
        single score uses its whole range, so nearer 0 or nearer 1 both mean "trust this"
        and nearer 0.5 means "don't" -- see bucket()."""

        image = Image.fromarray(roi_image).convert("L")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        return float(probs[CLASSES.index("noise")])

    def bucket(self, noise_score: float) -> str:
        """noise_score -> "noise" (>= confidence_threshold), "clean" (<= 1 - confidence_threshold),
        or "uncertain" (the symmetric band around 0.5 in between)."""

        if noise_score >= self.confidence_threshold:
            return "noise"
        if noise_score <= 1 - self.confidence_threshold:
            return "clean"
        return "uncertain"

    def classify_rois(self, rois):
        """Adds "noise_score" and "noise_bucket" to each ROI dict in place. noise_bucket is the
        one to act on; noise_score is the raw number it was derived from. Returns rois."""

        for roi in rois:
            noise_score = self.classify(roi["roi"])
            roi["noise_score"] = noise_score
            roi["noise_bucket"] = self.bucket(noise_score)

        return rois

    # --------------------------------------------------

    def annotate_review_crop(self, roi_image: np.ndarray, bucket: str, noise_score: float) -> Image.Image:
        """Returns roi_image with the noise_score burned in below it as a caption, coloured by
        bucket -- same idea as roi_ui/app.py's annotate_image() for OCR results. For save_review_crops.

        Many ROIs (a single narrow stroke, a tall thin sliver) are narrower than the caption text
        itself, so the canvas width is measured from the actual rendered text, not assumed from the
        crop -- a crop already wider than its caption keeps its own width; a narrower one is padded
        out to fit, with the crop centred in the extra space."""

        img = Image.fromarray(roi_image).convert("RGB")
        w, h = img.size

        caption = f"{noise_score:.2f}"
        text_left, _, text_right, _ = ImageDraw.Draw(img).textbbox((0, 0), caption, font=self.font)
        text_w = text_right - text_left

        canvas_w = max(w, text_w + 4)
        canvas = Image.new("RGB", (canvas_w, h + self.caption_height), self.caption_bg)
        canvas.paste(img, ((canvas_w - w) // 2, 0))

        draw = ImageDraw.Draw(canvas)
        draw.text((2, h + 2), caption, fill=self.bucket_colors[bucket], font=self.font)

        return canvas
