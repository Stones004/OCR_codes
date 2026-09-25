"""
Local PARSeq OCR

Runs the fine-tuned PARSeq handwriting model fully offline. The model
and its `strhub` dependency live in the dedicated PARSeq venv
(configured as ocr.parseq.venv_python, see its README), so it's spawned as a
subprocess and kept alive for the life of the backend -- the checkpoint
loads once and is reused across every ROI instead of per-call.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import cv2

from src.config import ParseqConfig

from .base import OCRBackend

WORKER_SCRIPT = Path(__file__).resolve().parent / "parseq_worker.py"


class ParseqBackend(OCRBackend):

    def __init__(self, config: ParseqConfig):

        self.shutdown_timeout = config.shutdown_timeout

        if not config.venv_python.exists():
            raise FileNotFoundError(
                f"PARSeq venv python not found: {config.venv_python}\n"
                "Set it up per ocr_dataset_builder/README.md and point "
                "ocr.parseq.venv_python in config.yaml at it."
            )

        if not config.checkpoint.exists():
            raise FileNotFoundError(
                f"PARSeq checkpoint not found: {config.checkpoint}"
            )

        # Everything the worker needs from config.yaml travels as one JSON
        # argument, since the worker runs in a different venv.
        worker_settings = json.dumps({
            "device": config.device,
            "norm_mean": config.norm_mean,
            "norm_std": config.norm_std,
        })

        self.process = subprocess.Popen(
            [
                str(config.venv_python),
                str(WORKER_SCRIPT),
                str(config.checkpoint),
                worker_settings,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ready = self.process.stdout.readline()

        if ready.strip() != "READY":
            raise RuntimeError(
                f"PARSeq worker failed to start:\n{self.process.stderr.read()}"
            )

    # --------------------------------------------------

    def read(self, image, preprocess=True):

        if isinstance(image, (str, Path)):
            image_path = str(image)
            temp_path = None

        else:
            success, buffer = cv2.imencode(".png", image)

            if not success:
                raise RuntimeError("Failed to encode image.")

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(buffer.tobytes())
            tmp.close()

            image_path = tmp.name
            temp_path = tmp.name

        try:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"PARSeq worker exited unexpectedly:\n{self.process.stderr.read()}"
                )

            self.process.stdin.write(f"{image_path}\n")
            self.process.stdin.flush()

            line = self.process.stdout.readline()

            if not line:
                raise RuntimeError(
                    f"PARSeq worker closed unexpectedly:\n{self.process.stderr.read()}"
                )

            result = json.loads(line)

            if "error" in result:
                raise RuntimeError(result["error"])

            return {
                "text": result["prediction"],
                "confidence": result["confidence"],
                "words": [],
            }

        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    # --------------------------------------------------

    def close(self):

        process = getattr(self, "process", None)

        if process and process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=self.shutdown_timeout)

    def __del__(self):
        self.close()
