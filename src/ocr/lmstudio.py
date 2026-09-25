"""
LM Studio OCR

Performs OCR on a complete cropped annotation strip using a vision-language
model served locally through LM Studio. Model, host, prompt and sampling
settings come from config.yaml (ocr.lmstudio).
"""

import base64
import json

import cv2
from openai import OpenAI

from src.config import LMStudioConfig
from src.lmstudio_preprocessor import LMStudioPreprocessor

from .base import OCRBackend


class LMStudioBackend(OCRBackend):

    def __init__(self, config: LMStudioConfig):

        self.model = config.model
        self.temperature = config.temperature
        self.prompt = config.prompt
        self.debug_image_path = config.debug_image_path

        self.client = OpenAI(
            base_url=config.host,
            api_key=config.api_key,
        )
        self.preprocessor = LMStudioPreprocessor(config.preprocess)

    # --------------------------------------------------

    def read(self, image, preprocess=True):

        # Production uses this.
        # Research benchmarking disables it.
        if preprocess:
            image = self.preprocessor.process(image)

        if self.debug_image_path is not None:
            self.debug_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.debug_image_path), image)

        success, buffer = cv2.imencode(".png", image)

        if not success:
            raise RuntimeError("Failed to encode image.")

        image_b64 = base64.b64encode(buffer).decode()

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
        )

        content = response.choices[0].message.content

        # Find where the first JSON array starts
        start = content.find("[")

        if start == -1:
            raise RuntimeError(
                f"No JSON array found.\n\nResponse:\n{content}"
            )

        decoder = json.JSONDecoder()

        try:
            predictions, end = decoder.raw_decode(content[start:])
            return predictions

        except json.JSONDecodeError as e:
            print("\n========== INVALID JSON ==========\n")
            print(content)
            print("\n==================================\n")
            raise
