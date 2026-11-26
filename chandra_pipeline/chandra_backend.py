"""
Chandra OCR (via LM Studio)

Sends a whole page to Chandra served by LM Studio's OpenAI-compatible
endpoint and parses its layout reply into text blocks with pixel bounding
boxes. Chandra answers in HTML, one <div data-bbox="x0 y0 x1 y1"
data-label="..."> per block, with coordinates on a 0..bbox_scale grid.
Everything tunable comes from chandra_pipeline/config.yaml.
"""

import base64
from html.parser import HTMLParser

import cv2
from openai import OpenAI

from chandra_config import ChandraConfig


# Tags that end a line of text inside a block (HTML protocol, not a tunable).
_LINE_BREAK_TAGS = {"br", "p", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"}


class _BlockParser(HTMLParser):
    """Collects every top-level <div data-bbox=...> as {bbox, label, text}."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self._depth = 0
        self._current = None

    def handle_starttag(self, tag, attrs):

        if tag == "div":
            self._depth += 1
            attrs = dict(attrs)
            if self._depth == 1 and attrs.get("data-bbox"):
                self._current = {
                    "bbox": attrs["data-bbox"],
                    "label": attrs.get("data-label", ""),
                    "parts": [],
                }
            return

        if self._current is not None and tag in _LINE_BREAK_TAGS:
            self._current["parts"].append("\n")

    def handle_endtag(self, tag):

        if tag == "div":
            if self._depth == 1 and self._current is not None:
                self.blocks.append(self._current)
                self._current = None
            self._depth = max(0, self._depth - 1)
            return

        if self._current is not None and tag in _LINE_BREAK_TAGS:
            self._current["parts"].append("\n")

    def handle_data(self, data):
        if self._current is not None:
            self._current["parts"].append(data)

    def close(self):
        super().close()
        # An unterminated reply (token limit hit) still yields its last block.
        if self._current is not None:
            self.blocks.append(self._current)
            self._current = None


def _clean_text(parts):
    lines = "".join(parts).split("\n")
    lines = [" ".join(line.split()) for line in lines]
    return "\n".join(line for line in lines if line)


class ChandraBackend:

    def __init__(self, config: ChandraConfig):

        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.max_side_attempts = config.max_side_attempts
        self.bbox_scale = config.bbox_scale
        self.prompt = config.prompt

        self.client = OpenAI(
            base_url=config.host,
            api_key=config.api_key,
            timeout=config.timeout,
        )

        # Raw model reply of the most recent read(), for debugging.
        self.last_raw = ""
        self.last_finish_reason = None
        self.last_max_side = None

    # --------------------------------------------------

    def read(self, image):
        """Returns [{"text", "label", "bbox": (x0, y0, x1, y1)}] with bbox in
        pixels of `image`. An empty reply is retried at the next size in
        max_side_attempts; if every size is empty the result is []."""

        height, width = image.shape[:2]

        for max_side in self.max_side_attempts:

            scale = max_side / max(height, width)
            sent = cv2.resize(
                image,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

            success, buffer = cv2.imencode(".png", sent)

            if not success:
                raise RuntimeError("Failed to encode image.")

            image_b64 = base64.b64encode(buffer).decode()

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": self.prompt,
                            },
                        ],
                    }
                ],
            )

            choice = response.choices[0]
            self.last_raw = choice.message.content or ""
            self.last_finish_reason = choice.finish_reason
            self.last_max_side = max_side

            if self.last_raw.strip():
                break

        return self._parse(self.last_raw, width, height)

    # --------------------------------------------------

    def _parse(self, content, width, height):

        parser = _BlockParser()
        parser.feed(content)
        parser.close()

        blocks = []

        for raw in parser.blocks:

            try:
                x0, y0, x1, y1 = (float(v) for v in raw["bbox"].split())
            except ValueError:
                continue

            blocks.append({
                "text": _clean_text(raw["parts"]),
                "label": raw["label"],
                "bbox": (
                    round(x0 / self.bbox_scale * width),
                    round(y0 / self.bbox_scale * height),
                    round(x1 / self.bbox_scale * width),
                    round(y1 / self.bbox_scale * height),
                ),
            })

        return blocks
