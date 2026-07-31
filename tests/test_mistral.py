import os
import sys
import mimetypes
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

if API_KEY is None:
    raise ValueError("MISTRAL_API_KEY not found in .env")

BASE_URL = "https://api.mistral.ai"


def upload_file(image_path):
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    with open(image_path, "rb") as f:
        files = {
            "file": (os.path.basename(image_path), f, mime_type)
        }

        data = {
            "purpose": "ocr"
        }

        response = requests.post(
            f"{BASE_URL}/v1/files",
            headers=headers,
            files=files,
            data=data,
        )

    response.raise_for_status()
    return response.json()


def get_signed_url(file_id):
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.get(
        f"{BASE_URL}/v1/files/{file_id}/url",
        headers=headers,
    )

    response.raise_for_status()
    return response.json()["url"]


def run_ocr(document_url):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "document_url",
            "document_url": document_url,
        },
    }

    response = requests.post(
        f"{BASE_URL}/v1/ocr",
        headers=headers,
        json=payload,
    )

    response.raise_for_status()
    return response.json()


def main(image_path):
    print("Uploading image...")
    uploaded = upload_file(image_path)

    file_id = uploaded["id"]
    print("Uploaded:", file_id)

    print("Generating signed URL...")
    signed_url = get_signed_url(file_id)

    print("Running OCR...")
    result = run_ocr(signed_url)

    print("\n========== OCR OUTPUT ==========\n")

    for i, page in enumerate(result["pages"], start=1):
        print(f"\n----- PAGE {i} -----\n")
        print(page["markdown"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("python test_mistral.py image.png")
        sys.exit(1)

    main(sys.argv[1])