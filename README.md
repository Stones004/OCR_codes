# OCR & Annotation Extraction Pipeline

An end-to-end Python pipeline designed for processing scanned PDF documents. The system performs page rendering, image deskewing, vertical separator line detection, margin/ROI cropping, preprocessing, annotation detection/removal, and OCR text extraction using configurable backends (e.g., LM Studio vision models, TrOCR). It also includes an AI-assisted OCR dataset builder suite and research benchmark tools.

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Code Files Reference](#code-files-reference)
  - [Root Scripts](#root-scripts)
  - [Core Pipeline Modules (`src/`)](#core-pipeline-modules-src)
  - [OCR Backends (`src/ocr/`)](#ocr-backends-srcocr)
  - [OCR Dataset Builder (`ocr_dataset_builder/`)](#ocr-dataset-builder-ocr_dataset_builder)
  - [Research & Benchmarking (`research/`)](#research--benchmarking-research)
  - [Unit Tests (`tests/`)](#unit-tests-tests)
- [System Setup Guide](#system-setup-guide)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Clone & Navigate](#2-clone--navigate)
  - [3. Set Up Virtual Environment](#3-set-up-virtual-environment)
  - [4. Install Dependencies](#4-install-dependencies)
  - [5. Environment Configuration](#5-environment-configuration)
  - [6. OCR Backend Prerequisites (LM Studio / TrOCR)](#6-ocr-backend-prerequisites-lm-studio--trocr)
- [Commands for Different Tasks](#commands-for-different-tasks)
  - [Task 1: Run End-to-End OCR Pipeline](#task-1-run-end-to-end-ocr-pipeline)
  - [Task 2: Consolidate Extracted ROI Images](#task-2-consolidate-extracted-roi-images)
  - [Task 3: Build ROI Dataset from Input PDFs](#task-3-build-roi-dataset-from-input-pdfs)
  - [Task 4: Run AI-Assisted Dataset Builder](#task-4-run-ai-assisted-dataset-builder)
  - [Task 5: Run Preprocessing Benchmark](#task-5-run-preprocessing-benchmark)
  - [Task 6: Run OCR Preprocessing Evaluation](#task-6-run-ocr-preprocessing-evaluation)
  - [Task 7: Execute Unit Tests](#task-7-execute-unit-tests)
- [Pipeline Configuration](#pipeline-configuration)

---

## Features

- **PDF Ingestion & Rendering**: Renders PDF pages to high-resolution images via PyMuPDF (`fitz`).
- **Deskewing**: Automatically estimates and corrects page rotation and skew angles.
- **Vertical Line Detection**: Extracts margin vertical separation lines to isolate annotation zones.
- **ROI & Margin Cropping**: Crops target text margins and extracts region-of-interest (ROI) annotation blocks.
- **Image Preprocessing**: Enhances contrast (CLAHE, morphology, binarization, denoising) for high-accuracy OCR.
- **Annotation Detection & Masking**: Detects handwritten/typed margin annotations and offers masking capabilities.
- **Modular OCR Engines**: Pluggable backend support including **LM Studio** local Vision LLMs and HuggingFace **TrOCR**.
- **Dataset Builder Suite**: Automated AI-assisted dataset labeling pipeline powered by Mistral AI API integration.
- **Research Tools**: Utilities for dataset preparation, benchmarking preprocessing techniques, and evaluating OCR performance metrics.

---

## Folder Structure

```text
OCR_codes/
│
├── data/                            # Input PDFs and generated ROI datasets
│   ├── input_pdfs/                  # Source scanned PDF files to process
│   └── roi_dataset/                 # Extracted Regions of Interest (ROIs) for research/training
│
├── ocr_dataset_builder/             # Automated AI dataset labeling framework
│   ├── dataset/                     # Built dataset storage (annotations CSV, completed/preview images)
│   │   ├── annotations.csv          # Generated ground-truth OCR text annotations
│   │   ├── completed/               # Processed dataset images
│   │   └── preview/                 # Output preview visualizations
│   ├── src/                         # Core dataset builder modules
│   │   ├── __init__.py              # Package initialization
│   │   ├── config.py                # Dataset builder configurations and directory paths
│   │   ├── csv_manager.py           # Annotation CSV read/write management
│   │   ├── dataset_builder.py       # Main annotation pipeline orchestrator
│   │   ├── file_manager.py          # Image caching and file organization operations
│   │   └── mistral_client.py        # Mistral AI Vision API client for automated annotation
│   ├── README.md                    # Dataset builder user guide
│   └── run.py                       # Entry point to execute dataset builder pipeline
│
├── outputs/                         # Per-PDF pipeline execution outputs
│   └── <pdf_name>/
│       ├── cropped/                 # Cropped page margin images
│       ├── deskewed/                # Deskewed page images
│       ├── detected/                # Visualized detected vertical separator lines
│       ├── failed/                  # Pages where line detection/processing failed
│       ├── ocr/                     # CSV files containing extracted OCR text per page
│       ├── preprocessed/            # Binary and grayscale preprocessed page images
│       ├── recognizer/              # Annotation detection visual debug outputs
│       ├── roi/                     # Saved individual ROI snippets
│       └── report.csv               # Page-by-page execution report
│   ├── all_roi_images/              # Consolidated ROI image collection from all PDFs
│   └── summary.csv                  # Aggregate execution summary across all processed PDFs
│
├── research/                        # Preprocessing benchmark and evaluation tools
│   ├── results/                     # Benchmark results (metrics CSVs and visual outputs)
│   ├── benchmark.py                 # Evaluates image preprocessing filters on sample crops
│   ├── dataset.py                   # PyTorch dataset loader for ROI images and labels
│   ├── evaluate.py                  # Evaluates OCR preprocessors across ROI dataset
│   └── prepare_roi_dataset.py       # Extracts ROI crops from input PDFs for research
│
├── src/                             # Core pipeline library code
│   ├── ocr/                         # OCR engine implementations
│   │   ├── __init__.py              # Package initialization
│   │   ├── base.py                  # Abstract base class interface for OCR backends
│   │   ├── factory.py               # Factory for instantiating configured OCR backend
│   │   ├── lmstudio.py              # LM Studio local Vision LLM integration
│   │   └── trocr.py                 # HuggingFace TrOCR vision-transformer integration
│   ├── annotation_detector.py       # Detects bounding boxes for handwritten/typed annotations
│   ├── annotation_remover.py        # Removes or masks detected annotations from page images
│   ├── config.py                    # Global pipeline configuration dataclass (PipelineConfig)
│   ├── cropper.py                   # Crops pages using detected vertical line coordinates
│   ├── deskewer.py                  # Estimates page skew angle and rotates images
│   ├── line_detector.py             # Locates vertical separator lines on document pages
│   ├── lmstudio_preprocessor.py     # Image preprocessing tailor-made for LM Studio Vision models
│   ├── ocr_preprocessor.py          # Advanced image enhancement (CLAHE, Otsu, morph operations)
│   ├── page_preprocessor.py         # Page-level preprocessor handler
│   ├── pdf_loader.py                # Renders PDF pages to NumPy OpenCV images
│   ├── pipeline.py                  # High-level pipeline controller orchestrating OCR workflow
│   ├── preprocessor.py              # Primary binarization and grayscale preprocessing operations
│   ├── roi_text_extractor.py        # Passes ROI regions to active OCR backend for text extraction
│   └── vertical_extractor.py        # Isolates vertical lines using morphological filtering
│
├── tests/                           # Unit test suite
│   ├── __init__.py              # Package initialization
│   ├── test_annotation_removal.py   # Tests for annotation detection and masking
│   ├── test_denoise.py              # Tests for noise reduction algorithms
│   ├── test_deskew.py               # Tests for page deskew angle estimation and rotation
│   ├── test_detect_text.py          # Tests for text region bounding box detection
│   ├── test_mistral.py              # Tests for Mistral API integration
│   ├── test_mistral_local.py        # Tests for local Mistral server interaction
│   ├── test_ocr.py                  # Tests for OCR base class and factory instantiation
│   ├── test_overlap.py              # Tests for bounding box overlap and merging algorithms
│   ├── test_preprocessor.py         # Tests for image binarization and preprocessing routines
│   ├── test_recognizer.py           # Tests for text recognition wrapper components
│   └── test_vertical.py             # Tests for vertical separator line extraction
│
├── .env                             # Environment variables configuration (API keys, endpoints)
├── .gitignore                       # Git exclusion specification
├── main.py                          # Primary execution script for end-to-end OCR processing
├── README.md                        # Master project documentation
├── requirements.txt                 # Python dependencies manifest
└── roi_consolidator.py              # Aggregates ROI images across all processed document output folders
```

---

## Code Files Reference

Below is a one-line description for every code file in the repository:

### Root Scripts
- [main.py](file:///g:/OCR_codes/main.py): Primary execution script that runs the end-to-end OCR and annotation extraction pipeline on input PDFs.
- [roi_consolidator.py](file:///g:/OCR_codes/roi_consolidator.py): Collects and consolidates extracted ROI images from all document output folders into a single output directory.

### Core Pipeline Modules (`src/`)
- [src/annotation_detector.py](file:///g:/OCR_codes/src/annotation_detector.py): Detects bounding boxes for handwritten or typed text annotations within document page regions.
- [src/annotation_remover.py](file:///g:/OCR_codes/src/annotation_remover.py): Removes or masks detected annotations from page images using morphological mask operations.
- [src/config.py](file:///g:/OCR_codes/src/config.py): Defines global pipeline settings and execution parameters via the `PipelineConfig` dataclass.
- [src/cropper.py](file:///g:/OCR_codes/src/cropper.py): Crops specific document page regions based on detected vertical separator line coordinates.
- [src/deskewer.py](file:///g:/OCR_codes/src/deskewer.py): Estimates document page skew angle and rotates images to straighten text lines.
- [src/line_detector.py](file:///g:/OCR_codes/src/line_detector.py): Locates vertical separator lines on scanned pages using line detection algorithms.
- [src/lmstudio_preprocessor.py](file:///g:/OCR_codes/src/lmstudio_preprocessor.py): Applies image preprocessing filters tailored specifically for LM Studio Vision LLM inputs.
- [src/ocr_preprocessor.py](file:///g:/OCR_codes/src/ocr_preprocessor.py): Implements multimodal image contrast enhancement, CLAHE, binarization, and noise filtering.
- [src/page_preprocessor.py](file:///g:/OCR_codes/src/page_preprocessor.py): Manages page-level preprocessor workflows prior to region cropping and OCR.
- [src/pdf_loader.py](file:///g:/OCR_codes/src/pdf_loader.py): Converts input PDF document pages into high-resolution NumPy OpenCV image matrices.
- [src/pipeline.py](file:///g:/OCR_codes/src/pipeline.py): Orchestrates the complete PDF rendering, deskewing, line detection, cropping, preprocessing, and OCR workflow.
- [src/preprocessor.py](file:///g:/OCR_codes/src/preprocessor.py): Performs primary image binarization and grayscale preprocessing routines.
- [src/roi_text_extractor.py](file:///g:/OCR_codes/src/roi_text_extractor.py): Extracts Region-of-Interest (ROI) text crops and passes them to the configured OCR engine backend.
- [src/vertical_extractor.py](file:///g:/OCR_codes/src/vertical_extractor.py): Isolates vertical line structures from page images using specialized morphological kernel operations.

### OCR Backends (`src/ocr/`)
- [src/ocr/__init__.py](file:///g:/OCR_codes/src/ocr/__init__.py): Package initialization file exposing OCR backends and factory modules.
- [src/ocr/base.py](file:///g:/OCR_codes/src/ocr/base.py): Defines the abstract base class and standard interface (`BaseOCR`) for all OCR backends.
- [src/ocr/factory.py](file:///g:/OCR_codes/src/ocr/factory.py): Implements factory pattern (`OCRFactory`) to instantiate the selected OCR engine backend.
- [src/ocr/lmstudio.py](file:///g:/OCR_codes/src/ocr/lmstudio.py): OCR backend implementation integrating with local LM Studio Vision LLM HTTP endpoints.
- [src/ocr/trocr.py](file:///g:/OCR_codes/src/ocr/trocr.py): OCR backend implementation integrating with HuggingFace TrOCR vision-transformer models.

### OCR Dataset Builder (`ocr_dataset_builder/`)
- [ocr_dataset_builder/run.py](file:///g:/OCR_codes/ocr_dataset_builder/run.py): Entry point script to launch the AI-assisted dataset builder pipeline.
- [ocr_dataset_builder/src/__init__.py](file:///g:/OCR_codes/ocr_dataset_builder/src/__init__.py): Package initialization file for dataset builder source components.
- [ocr_dataset_builder/src/config.py](file:///g:/OCR_codes/ocr_dataset_builder/src/config.py): Defines path configurations, API settings, and parameters for dataset building.
- [ocr_dataset_builder/src/csv_manager.py](file:///g:/OCR_codes/ocr_dataset_builder/src/csv_manager.py): Handles loading, updating, and saving dataset ground-truth annotations to CSV.
- [ocr_dataset_builder/src/dataset_builder.py](file:///g:/OCR_codes/ocr_dataset_builder/src/dataset_builder.py): Core pipeline controller that orchestrates image processing and automated AI text labeling.
- [ocr_dataset_builder/src/file_manager.py](file:///g:/OCR_codes/ocr_dataset_builder/src/file_manager.py): Manages image file copying, caching, preview generation, and directory organization for dataset assets.
- [ocr_dataset_builder/src/mistral_client.py](file:///g:/OCR_codes/ocr_dataset_builder/src/mistral_client.py): API client for communicating with Mistral AI Vision models to automatically generate ground-truth annotations.

### Research & Benchmarking (`research/`)
- [research/benchmark.py](file:///g:/OCR_codes/research/benchmark.py): Runs benchmarking comparisons across various image preprocessing filters on sample crops.
- [research/dataset.py](file:///g:/OCR_codes/research/dataset.py): PyTorch `Dataset` class implementation for loading ROI image patches and ground-truth text targets.
- [research/evaluate.py](file:///g:/OCR_codes/research/evaluate.py): Evaluates OCR quality, processing speed, and token density across different preprocessing techniques.
- [research/prepare_roi_dataset.py](file:///g:/OCR_codes/research/prepare_roi_dataset.py): Extracts and formats annotated ROI crops from source PDFs to generate research datasets.

### Unit Tests (`tests/`)
- [tests/__init__.py](file:///g:/OCR_codes/tests/__init__.py): Package initialization file for the unit test package.
- [tests/test_annotation_removal.py](file:///g:/OCR_codes/tests/test_annotation_removal.py): Unit tests verifying annotation detection and mask removal routines.
- [tests/test_denoise.py](file:///g:/OCR_codes/tests/test_denoise.py): Unit tests verifying image noise reduction and filtering algorithms.
- [tests/test_deskew.py](file:///g:/OCR_codes/tests/test_deskew.py): Unit tests verifying page skew angle estimation and image rotation.
- [tests/test_detect_text.py](file:///g:/OCR_codes/tests/test_detect_text.py): Unit tests verifying text region bounding box detection.
- [tests/test_mistral.py](file:///g:/OCR_codes/tests/test_mistral.py): Integration tests verifying Mistral AI API request/response handling.
- [tests/test_mistral_local.py](file:///g:/OCR_codes/tests/test_mistral_local.py): Integration tests verifying local API server communications.
- [tests/test_ocr.py](file:///g:/OCR_codes/tests/test_ocr.py): Unit tests verifying base OCR engine interfaces and factory instantiation.
- [tests/test_overlap.py](file:///g:/OCR_codes/tests/test_overlap.py): Unit tests verifying bounding box overlap logic and box merging operations.
- [tests/test_preprocessor.py](file:///g:/OCR_codes/tests/test_preprocessor.py): Unit tests verifying image binarization and preprocessing pipelines.
- [tests/test_recognizer.py](file:///g:/OCR_codes/tests/test_recognizer.py): Unit tests verifying text recognizer wrapper components.
- [tests/test_vertical.py](file:///g:/OCR_codes/tests/test_vertical.py): Unit tests verifying vertical separator line detection algorithms.

---

## System Setup Guide

Follow these steps to set up and run the codebase on **Windows**, **macOS**, or **Linux**.

### 1. Prerequisites
- **Python 3.8+** (Python 3.10+ recommended)
- **Git**
- *(Optional)* **CUDA-capable GPU** with drivers installed if running local neural models (TrOCR/PyTorch) with hardware acceleration.

Verify Python version:
```bash
python --version
# or
python3 --version
```

### 2. Clone & Navigate
Clone the repository (or extract the source code) and navigate into the root directory:
```bash
git clone <repository_url>
cd OCR_codes
```

### 3. Set Up Virtual Environment

Create and activate an isolated Python virtual environment:

- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

- **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```

- **macOS / Linux (Bash or Zsh):**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies

Upgrade `pip` and install all required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch (GPU Acceleration):**  
> `requirements.txt` installs standard CPU PyTorch packages by default. To enable GPU acceleration with CUDA on Windows/Linux, install CUDA PyTorch binaries from the [official PyTorch website](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 5. Environment Configuration

Create or update the `.env` file in the root project directory with your credentials:

```env
MISTRAL_API_KEY=your_api_key_here
```

### 6. OCR Backend Prerequisites (LM Studio / TrOCR)

- **LM Studio (Default Backend):**
  1. Download and open [LM Studio](https://lmstudio.ai/).
  2. Load a Vision LLM model (e.g., Qwen2-VL, Llama-3-Vision).
  3. Start the **Local Inference Server** inside LM Studio (default URL: `http://localhost:1234`).
- **TrOCR Backend:**
  1. Standard PyTorch environment; required models will automatically download from HuggingFace on first run.

---

## Commands for Different Tasks

### Task 1: Run End-to-End OCR Pipeline
Processes input PDF files placed in `data/input_pdfs/` through deskewing, line detection, cropping, preprocessing, and OCR text extraction.

1. **Place input PDFs:**
   ```bash
   mkdir -p data/input_pdfs
   # Copy target .pdf files into data/input_pdfs/
   ```

2. **Execute pipeline:**
   ```bash
   python main.py
   ```

3. **View outputs:**
   - Detailed results per document are written to `outputs/<pdf_name>/` (CSV reports, crop images, deskewed images, extracted OCR text per page).
   - An overall summary is saved to `outputs/summary.csv`.

---

### Task 2: Consolidate Extracted ROI Images
Aggregates all extracted ROI image snippets from individual document subfolders under `outputs/` into a single consolidated folder (`outputs/all_roi_images/`).

```bash
python roi_consolidator.py
```

---

### Task 3: Build ROI Dataset from Input PDFs
Extracts region-of-interest (ROI) annotation crops from PDFs in `data/input_pdfs/` and organizes them into `data/roi_dataset/`.

```bash
python research/prepare_roi_dataset.py
```

---

### Task 4: Run AI-Assisted Dataset Builder
Runs the automated dataset annotation pipeline using Mistral Vision API to generate ground-truth text annotations for ROI images in `ocr_dataset_builder/dataset/`.

```bash
python ocr_dataset_builder/run.py
```

Outputs generated annotations to `ocr_dataset_builder/dataset/annotations.csv`.

---

### Task 5: Run Preprocessing Benchmark
Applies various image preprocessing filters (CLAHE, Otsu binarization, morphological transformations, sharpening) to sample images and writes benchmark visualizations to `research/results/`.

```bash
python research/benchmark.py
```

---

### Task 6: Run OCR Preprocessing Evaluation
Evaluates available preprocessing techniques across the dataset stored in `data/roi_dataset/`, measuring processing speeds (ms), token densities, block counts, and text output quality.

```bash
python research/evaluate.py
```

Outputs evaluation metrics to `research/results/evaluation.csv`.

---

### Task 7: Execute Unit Tests
Runs the automated test suite to verify line detection, deskewing, OCR backends, preprocessors, annotation removal, and overlap algorithms.

- **Run all unit tests:**
  ```bash
  python -m unittest discover -s tests
  ```

- **Run a specific test module (e.g., deskewer tests):**
  ```bash
  python -m unittest tests/test_deskew.py
  ```

- **Run vertical line extraction tests:**
  ```bash
  python -m unittest tests/test_vertical.py
  ```

---

## Pipeline Configuration

Customize execution parameters by modifying [src/config.py](file:///g:/OCR_codes/src/config.py):

| Setting | Default Value | Description |
| :--- | :--- | :--- |
| `OCR_BACKEND` | `"lmstudio"` | Active OCR backend (`"lmstudio"` or `"trocr"`) |
| `USE_ANNOTATION_DETECTION` | `False` | `True` enables ROI-based region extraction before OCR; `False` runs full margin OCR |
| `SAVE_DEBUG_IMAGES` | `True` | Saves intermediate debug visualization images in `outputs/` |
| `SAVE_ROIS` | `True` | Saves cropped ROI snippets to `outputs/<pdf>/roi/` |
| `OCR_MAX_TOKENS` | `64` | Maximum token length for vision model generations |
| `USE_OCR_PREPROCESSOR` | `True` | Applies image contrast/denoising preprocessor prior to OCR |
