"""
Configuration

Every tunable in the project lives in config.yaml (repo root). This module
only defines the *shape* of that file -- as typed, frozen dataclasses -- and
loads/validates it. There are deliberately NO default values here: a key
missing from config.yaml is an error, so the YAML is the single source of
truth and no number can silently come from code.

Usage:

    from src.config import load_config
    cfg = load_config()                 # OCR_CONFIG env var, else <repo>/config.yaml
    cfg = load_config("my_run.yaml")    # explicit file

Relative paths inside the YAML resolve against the folder that contains the
YAML file (the repo root by default), not against the current directory.
"""

from __future__ import annotations

import os
import types
import typing
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import yaml

CONFIG_ENV_VAR = "OCR_CONFIG"
DEFAULT_CONFIG_NAME = "config.yaml"


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed or incomplete."""


# ======================================================================
# Pipeline behaviour
# ======================================================================

@dataclass(frozen=True)
class PipelineConfig:
    use_annotation_detection: bool
    use_annotation_removal: bool
    use_noise_classifier: bool
    save_rois: bool
    save_debug_images: bool
    skip_existing_outputs: bool
    verbose: bool


@dataclass(frozen=True)
class ReviewConfig:
    # Predictions below this confidence are flagged for human review
    # (shared by roi_ui and ocr_dataset_builder).
    confidence_threshold: float


@dataclass(frozen=True)
class PathsConfig:
    input_dir: Path
    output_dir: Path
    pdf_glob: str


@dataclass(frozen=True)
class LayoutConfig:
    """Folder names and file-name patterns inside outputs/<pdf>/."""

    deskewed_dir: str
    vertical_dir: str
    detected_dir: str
    cropped_dir: str
    failed_dir: str
    annotation_removed_dir: str
    preprocessed_dir: str
    recognizer_dir: str
    roi_dir: str
    ocr_dir: str
    noise_dir: str                          # <pdf>/noise/{clean,noise}/ review crops + per-page CSVs

    # str.format patterns; {page} is the 1-based page number, {roi} the ROI number
    page_image: str
    page_mask: str
    page_gray: str
    page_binary: str
    page_vertical: str
    roi_image: str
    ocr_image: str                          # one image per ROI (crop + predicted text)
    ocr_page_image: str                     # whole-margin mode: one image per page
    noise_csv: str
    report_csv: str
    summary_csv: str


# ======================================================================
# Image-processing stages
# ======================================================================

@dataclass(frozen=True)
class PdfConfig:
    dpi: int
    skip_pages: int
    save_debug: bool
    debug_dir: Path | None
    debug_page_image: str                   # pattern; {page} = 1-based PDF page


@dataclass(frozen=True)
class DeskewConfig:
    angle_min: float
    angle_max: float
    coarse_step: float
    fine_step: float
    fine_window: float
    border_value: int
    edge_fallback_angle: float


@dataclass(frozen=True)
class VerticalConfig:
    threshold: int
    min_kernel_height: int
    kernel_height_divisor: int
    repair_kernel: tuple[int, ...]          # (width, height)


@dataclass(frozen=True)
class SeparatorConfig:
    left_ratio: float
    right_ratio: float
    min_height_ratio: float
    min_aspect_ratio: float
    expected_x: int | None
    tolerance: int | None


@dataclass(frozen=True)
class CropConfig:
    right_padding: int


@dataclass(frozen=True)
class InkRemovalConfig:
    red_lower1: tuple[int, ...]
    red_upper1: tuple[int, ...]
    red_lower2: tuple[int, ...]
    red_upper2: tuple[int, ...]
    green_lower: tuple[int, ...]
    green_upper: tuple[int, ...]
    kernel_size: int
    close_iterations: int
    dilate_iterations: int
    remove_fringe: bool
    fringe_zone_kernel_size: int
    fringe_zone_iterations: int
    fringe_lightness_max: int
    fringe_min_component_area: int
    fill_color: tuple[int, ...]


@dataclass(frozen=True)
class PreprocessConfig:
    min_area: int
    min_contrast_gap: float


@dataclass(frozen=True)
class AnnotationDetectorConfig:
    min_width: int
    min_height: int
    padding: int
    merge_dist: int
    min_confident_area: int
    min_confident_dim: int
    min_ink_pixels: int
    min_fill_ratio: float
    row_y_tolerance: float

    reject_above_top_line: bool
    top_line_search_frac: float
    top_line_min_len_ratio: float
    top_line_margin: int

    exclude_sparse_strokes: bool
    sparse_stroke_density_max: float
    sparse_stroke_min_dim: int

    exclude_long_vertical_strips: bool
    long_strip_max_width: int
    long_strip_min_height: int

    refine_height: int
    valley_threshold: float
    min_valley_width: int
    min_segment_height: int
    refine_blur_ksize: int

    line_kernel_min: int
    line_kernel_divisor: int
    line_dilate_kernel: int


@dataclass(frozen=True)
class VisualizationConfig:
    """Colours (BGR) and thicknesses of the debug overlays."""

    separator_candidate_color: tuple[int, ...]
    separator_candidate_thickness: int
    separator_selected_color: tuple[int, ...]
    separator_selected_thickness: int
    roi_color: tuple[int, ...]
    roi_thickness: int


# ======================================================================
# Noise classifier (pre-OCR triage)
# ======================================================================

@dataclass(frozen=True)
class NoiseClassifierConfig:
    # Trained in the trocr-handwriting-v2 project (src/noise_classifier/train.py) on
    # annotated ROI crops; a crop is "noise" if it was never given a usable
    # transcription (blank/NaN Normalize, or never annotated at all). Tags
    # every ROI for review -- does not itself change what gets OCR'd.
    checkpoint: Path
    device: str                              # "auto" | "cpu" | "cuda"
    # The model outputs a single noise_score in [0, 1] = P(noise) (see
    # NoiseClassifier.classify). A ROI is "noise" if noise_score >= this, "clean" if
    # noise_score <= 1 - this, else "uncertain" (the symmetric band around 0.5 in
    # between). On the held-out test split, 0.7 keeps 88.6% of ROIs out of that band
    # at 96.2% accuracy on those kept (vs 92.9% accuracy with the band empty at 0.5).
    confidence_threshold: float
    # Copy every classified ROI into <pdf>/noise/{clean,noise,uncertain}/
    # (named like the roi/ crop) so predictions can be reviewed by eye, captioned
    # with the noise_score the same way roi_ui/app.py captions OCR results.
    save_review_crops: bool
    caption_height: int                      # px reserved below the crop for the caption
    caption_font_size: int
    caption_bg: tuple[int, ...]              # RGB
    color_clean: tuple[int, ...]             # RGB caption text when bucket == clean
    color_noise: tuple[int, ...]             # RGB caption text when bucket == noise
    color_uncertain: tuple[int, ...]         # RGB caption text when bucket == uncertain
    font_candidates: tuple[str, ...]


# ======================================================================
# ROI enhancement (applied to each ROI crop before it is saved / OCR'd)
# ======================================================================

@dataclass(frozen=True)
class RoiEnhanceConfig:
    enabled: bool                            # false: ROIs stay plain grayscale crops
    blur_ksize: int                          # odd Gaussian kernel before Otsu; 0 = no blur
    # Per-ROI Otsu can split a near-uniform crop (blank paper) at a meaningless level and turn
    # paper noise into ink. If the threshold is not at least this far below the crop's mean
    # gray level, the crop is left as it was.
    min_contrast_gap: float


# ======================================================================
# OCR output images (crop with the predicted text printed below it)
# ======================================================================

@dataclass(frozen=True)
class OcrOutputConfig:
    caption_height: int                      # px reserved below the crop for the text
    min_canvas_width: int
    text_x: int
    text_y: int                              # offset of the text below the crop
    font_size: int
    bg: tuple[int, ...]                      # RGB
    text_color: tuple[int, ...]              # RGB
    empty_label: str                         # shown when the model returns nothing
    font_candidates: tuple[str, ...]


# ======================================================================
# OCR backends
# ======================================================================

@dataclass(frozen=True)
class ParseqConfig:
    venv_python: Path
    checkpoint: Path
    device: str                              # "auto" | "cpu" | "cuda"
    norm_mean: float
    norm_std: float
    shutdown_timeout: float


@dataclass(frozen=True)
class LMStudioPreprocessConfig:
    denoise_h: int
    template_window: int
    search_window: int
    sharpen_kernel: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class LMStudioConfig:
    model: str
    host: str
    api_key: str
    temperature: float
    prompt: str
    debug_image_path: Path | None
    preprocess: LMStudioPreprocessConfig


@dataclass(frozen=True)
class OcrConfig:
    backend: str                             # "parseq" | "lmstudio"
    use_backend_preprocessing: bool
    parseq: ParseqConfig
    lmstudio: LMStudioConfig


# ======================================================================
# Apps and tools
# ======================================================================

@dataclass(frozen=True)
class RoiUiConfig:
    input_dir: Path
    output_dir: Path
    review_csv_name: str                     # written next to output_dir
    host: str
    port: int
    poll_seconds: float
    stable_polls: int
    max_results: int

    # Pre-processing applied to every dropped image before OCR: the same
    # Preprocessor main.py uses (no annotation removal, no ROI detection).
    preprocess_enabled: bool
    alpha_background: tuple[int, ...]        # BGR fill for transparent pixels

    caption_height: int
    min_canvas_width: int
    canvas_bg: tuple[int, ...]
    text_color: tuple[int, ...]
    error_color: tuple[int, ...]
    error_detail_color: tuple[int, ...]
    error_detail_max_chars: int
    font_candidates: tuple[str, ...]
    font_size_text: int
    font_size_small: int
    text_x: int
    label_y: int
    confidence_y: int
    error_y: int
    error_detail_y: int

    confidence_high: float
    confidence_mid: float
    color_high: tuple[int, ...]
    color_mid: tuple[int, ...]
    color_low: tuple[int, ...]

    status_poll_ms: int
    results_poll_ms: int


@dataclass(frozen=True)
class MistralConfig:
    base_url: str
    model: str
    fallback_mime_type: str


@dataclass(frozen=True)
class PreviewConfig:
    font_name: str
    font_size: int
    max_chars: int
    text_color: tuple[int, ...]
    stroke_width: int
    stroke_fill: str
    margin: int


@dataclass(frozen=True)
class DatasetBuilderConfig:
    env_file: Path
    dataset_dir: Path
    image_dir: Path
    completed_dir: Path
    preview_dir: Path
    annotations_csv: Path
    review_csv: Path
    batch_size: int
    ocr_backend: str                         # "mistral" | "parseq"
    review_mode: str                         # "low_confidence" | "all"
    parseq_venv_python: Path
    parseq_checkpoint: Path
    mistral: MistralConfig
    preview: PreviewConfig


@dataclass(frozen=True)
class RoiConsolidatorConfig:
    root_dir: Path
    dest_dir: Path
    doc_glob: str


@dataclass(frozen=True)
class InspectThresholdsConfig:
    report_csv: Path
    debug_dir: Path
    accepted_color: tuple[int, ...]
    rejected_color: tuple[int, ...]
    box_thickness: int
    font_scale: float
    font_thickness: int
    label_offset: int
    label_min_y: int


@dataclass(frozen=True)
class PrepareRoiDatasetConfig:
    # research/prepare_roi_dataset.py re-implements the front half of
    # main.py; these two keys make its differences from production explicit.
    input_dir: Path
    output_dir: Path
    right_padding: int
    skip_pages: int
    roi_filename: str                       # pattern; {pdf}, {page}, {roi}


@dataclass(frozen=True)
class OcrPreprocessorConfig:
    method: str
    scale: int
    min_area: int
    kernel_size: int
    iterations: int
    denoise_h: int


@dataclass(frozen=True)
class EvaluateConfig:
    dataset_dir: Path
    output_csv: Path
    preprocessed_dir: Path
    ocr_dir: Path
    backend: str
    use_backend_preprocessing: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    image_path: Path
    output_dir: Path


@dataclass(frozen=True)
class ResearchConfig:
    prepare_roi_dataset: PrepareRoiDatasetConfig
    ocr_preprocessor: OcrPreprocessorConfig
    evaluate: EvaluateConfig
    benchmark: BenchmarkConfig


# ======================================================================
# Root
# ======================================================================

@dataclass(frozen=True)
class Config:
    root: Path                               # folder containing the YAML (injected)
    image_extensions: tuple[str, ...]
    review: ReviewConfig
    pipeline: PipelineConfig
    paths: PathsConfig
    layout: LayoutConfig
    pdf: PdfConfig
    deskew: DeskewConfig
    vertical: VerticalConfig
    separator: SeparatorConfig
    crop: CropConfig
    ink_removal: InkRemovalConfig
    preprocess: PreprocessConfig
    annotation_detector: AnnotationDetectorConfig
    visualization: VisualizationConfig
    noise_classifier: NoiseClassifierConfig
    roi_enhance: RoiEnhanceConfig
    ocr_output: OcrOutputConfig
    ocr: OcrConfig
    roi_ui: RoiUiConfig
    dataset_builder: DatasetBuilderConfig
    roi_consolidator: RoiConsolidatorConfig
    inspect_thresholds: InspectThresholdsConfig
    research: ResearchConfig


# ======================================================================
# Loader / validator
# ======================================================================

def _coerce(value, tp, where, root):

    origin = typing.get_origin(tp)

    # Optional[X] / X | None
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{where}: must not be null")
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], where, root)

    if is_dataclass(tp):
        return _build(tp, value, where, root)

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value

    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)

    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected text, got {value!r}")
        return value

    if tp is Path:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a path string, got {value!r}")
        path = Path(value)
        return path if path.is_absolute() else (root / path)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        item_type = typing.get_args(tp)[0]
        return tuple(
            _coerce(v, item_type, f"{where}[{i}]", root)
            for i, v in enumerate(value)
        )

    raise ConfigError(f"{where}: unsupported config type {tp!r}")


def _build(cls, data, where, root):

    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a section (mapping), got {data!r}")

    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]

    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown} (valid: {names})"
        )

    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")

    return cls(**{
        n: _coerce(data[n], hints[n], f"{where}.{n}", root)
        for n in names
    })


def resolve_config_path(path=None) -> Path:

    if path is not None:
        return Path(path).resolve()

    env_path = os.environ.get(CONFIG_ENV_VAR)

    if env_path:
        return Path(env_path).resolve()

    return Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_NAME


def load_config(path=None) -> Config:

    config_path = resolve_config_path(path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")

    if "root" in raw:
        raise ConfigError(f"{config_path}: 'root' is reserved, remove it")

    root = config_path.parent

    return _build(Config, {**raw, "root": str(root)}, config_path.name, root)
