import os
import mimetypes
import requests

from dotenv import load_dotenv

from .config import MODEL
from .config import ENV_FILE

load_dotenv(ENV_FILE)

class MistralOCR:

    BASE_URL = "https://api.mistral.ai"

    def __init__(self):

        self.api_key = os.getenv("MISTRAL_API_KEY")

        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

    def upload_file(self, image_path):

        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

        with open(image_path, "rb") as f:

            files = {
                "file": (
                    os.path.basename(image_path),
                    f,
                    mime_type,
                )
            }

            data = {
                "purpose": "ocr"
            }

            response = requests.post(
                f"{self.BASE_URL}/v1/files",
                headers=self.headers,
                files=files,
                data=data,
            )

        response.raise_for_status()

        return response.json()["id"]

    def get_signed_url(self, file_id):

        response = requests.get(
            f"{self.BASE_URL}/v1/files/{file_id}/url",
            headers=self.headers,
        )

        response.raise_for_status()

        return response.json()["url"]

    def run_ocr(self, signed_url):

        headers = self.headers.copy()

        headers["Content-Type"] = "application/json"

        payload = {
            "model": MODEL,
            "document": {
                "type": "document_url",
                "document_url": signed_url,
            },
        }

        response = requests.post(
            f"{self.BASE_URL}/v1/ocr",
            headers=headers,
            json=payload,
        )

        response.raise_for_status()

        return response.json()

    def extract_text(self, result):

        pages = []

        for page in result["pages"]:
            pages.append(page["markdown"])

        return "\n".join(pages).strip()

    def ocr_image(self, image_path):

        file_id = self.upload_file(image_path)

        signed_url = self.get_signed_url(file_id)

        result = self.run_ocr(signed_url)

        # No per-character confidence available from the hosted API --
        # Mistral is treated as an already-trusted teacher, not something
        # that needs a confidence-gated review pass.
        return self.extract_text(result), None

    def close(self):
        pass