import json
import subprocess
from pathlib import Path

from .config import PARSEQ_VENV_PYTHON
from .config import PARSEQ_CHECKPOINT

WORKER_SCRIPT = Path(__file__).resolve().parent / "parseq_worker.py"


class ParseqOCR:

    def __init__(self):

        if not PARSEQ_VENV_PYTHON.exists():
            raise FileNotFoundError(
                f"PARSeq venv python not found: {PARSEQ_VENV_PYTHON}"
            )

        if not PARSEQ_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"PARSeq checkpoint not found: {PARSEQ_CHECKPOINT}"
            )

        self.process = subprocess.Popen(
            [str(PARSEQ_VENV_PYTHON), str(WORKER_SCRIPT), str(PARSEQ_CHECKPOINT)],
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

    def ocr_image(self, image_path):

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

        return result["prediction"], result["confidence"]

    def close(self):

        if self.process and self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=10)
