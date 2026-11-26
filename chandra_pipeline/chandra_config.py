"""
Config for the Chandra experiment.

The shared pipeline settings (deskew, separator, crop, pdf, input folder, ...)
come from the repo's config.yaml via src.config.load_config. Only what is
specific to this folder lives in chandra_pipeline/config.yaml, validated with
the same loader machinery (no defaults: a missing key is an error).
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import ConfigError, _build

CONFIG_NAME = "config.yaml"


@dataclass(frozen=True)
class RunConfig:
    skip_existing_outputs: bool
    verbose: bool


@dataclass(frozen=True)
class PathsConfig:
    output_dir: Path                         # relative to this folder


@dataclass(frozen=True)
class LayoutConfig:
    chandra_dir: str                         # <pdf>/<chandra_dir>/
    margin_image: str                        # page with the left-margin question boxes drawn; {page}
    margin_raw: str                          # raw reply for the margin-only page; {page}
    answers_raw: str                         # raw reply for the answers-only page; {page}
    qa_csv: str                              # per-page question/answer rows; {page}
    summary_csv: str                         # all pages of one PDF


@dataclass(frozen=True)
class ChandraConfig:
    model: str
    host: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout: float                           # seconds per request
    max_side_attempts: tuple[int, ...]       # longest side (px) to send, tried in order until the reply has blocks
    bbox_scale: int                          # model bbox grid: coordinates run 0..bbox_scale
    prompt: str


@dataclass(frozen=True)
class MarginQaConfig:
    question_pattern: str | None             # regex a margin block must match; null = keep all
    answer_y_tolerance: int                  # px an answer may sit above its question's top
    answer_separator: str
    save_raw_response: bool
    box_color: tuple[int, ...]               # BGR
    box_thickness: int
    label_font_scale: float
    label_thickness: int
    label_offset: int


@dataclass(frozen=True)
class ChandraRunConfig:
    root: Path
    run: RunConfig
    paths: PathsConfig
    layout: LayoutConfig
    chandra: ChandraConfig
    margin_qa: MarginQaConfig


def load_chandra_config(path=None) -> ChandraRunConfig:

    config_path = (
        Path(path).resolve() if path
        else Path(__file__).resolve().parent / CONFIG_NAME
    )

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = config_path.parent

    return _build(
        ChandraRunConfig, {**raw, "root": str(root)}, config_path.name, root
    )
