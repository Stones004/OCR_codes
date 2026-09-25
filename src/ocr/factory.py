"""
OCR Backend Factory

Backends are imported only when selected, so a backend's dependencies
(e.g. `openai` for LM Studio) are needed only if config.yaml chooses it.
"""

from src.config import OcrConfig


def _parseq(config: OcrConfig):
    from .parseq import ParseqBackend
    return ParseqBackend(config.parseq)


def _lmstudio(config: OcrConfig):
    from .lmstudio import LMStudioBackend
    return LMStudioBackend(config.lmstudio)


BACKENDS = {

    "lmstudio": _lmstudio,

    "parseq": _parseq,
}


def create_backend(name: str, config: OcrConfig):

    name = name.lower()

    try:
        builder = BACKENDS[name]

    except KeyError:

        raise ValueError(

            f"Unknown OCR backend '{name}'. "

            f"Available: {list(BACKENDS.keys())}"

        )

    return builder(config)
