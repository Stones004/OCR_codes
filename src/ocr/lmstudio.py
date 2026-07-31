"""
LM Studio OCR

Performs OCR on a complete cropped annotation strip using a vision-language
model served locally through LM Studio.
"""

from email import contentmanager
import base64
import json
import re

import cv2
from openai import OpenAI
from .base import OCRBackend
from src.lmstudio_preprocessor import LMStudioPreprocessor

class LMStudioBackend(OCRBackend):

    def __init__(
        self,
        #model="google/gemma-3-12b",
        model="mistralai/ministral-3-3b",
        host="http://127.0.0.1:1234/v1",
    ):

        self.model = model

        self.client = OpenAI(
            base_url=host,
            api_key="lm-studio",
        )
        self.preprocessor = LMStudioPreprocessor()

        self.prompt = """
Extract all visible text from the image as individual items (words, numbers, or short labels).

Output Requirements:

Return ONLY a raw JSON array (no markdown fences, no extra text).

Order items from top-to-bottom, then left-to-right.

Use normalized coordinates between 0.0 and 1.0 (top-left is 0.0, 0.0, bottom-right is 1.0, 1.0), rounded to 3 decimal places.

[
  {
    "text": "detected text",
    "bounding_box": {
      "left": 0.000,
      "top": 0.000,
      "right": 0.000,
      "bottom": 0.000
    },
    "center": {
      "x": 0.000,
      "y": 0.000
    }
  }
] Return exactly ONE JSON array
"""

    # --------------------------------------------------

    def read(self, image, preprocess=True):

        # Production uses this.
        # Research benchmarking disables it.
        if preprocess:
            image = self.preprocessor.process(image)

        cv2.imwrite(
            "lmstudio_processed/lmstudio_input.png",
            image
        )

        success, buffer = cv2.imencode(".png", image)

        if not success:
            raise RuntimeError("Failed to encode image.")

        image_b64 = base64.b64encode(buffer).decode()

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
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