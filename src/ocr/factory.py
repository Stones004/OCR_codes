"""
OCR Backend Factory
"""

from .lmstudio import LMStudioBackend
from .trocr import TrOCRBackend


def create_backend(name: str):

    name = name.lower()

    backends = {

        "lmstudio": LMStudioBackend,

        "trocr": TrOCRBackend,
    }

    try:
        return backends[name]()

    except KeyError:

        raise ValueError(

            f"Unknown OCR backend '{name}'. "

            f"Available: {list(backends.keys())}"

        )