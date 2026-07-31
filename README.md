# OCR & Annotation Extraction Pipeline

An end-to-end Python pipeline designed for processing scanned PDF documents. The system performs page rendering, image deskewing, vertical separator line detection, margin/ROI cropping, preprocessing, annotation detection, and OCR text extraction using configurable backends (e.g., LM Studio vision models, TrOCR).

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [System Setup Guide](#system-setup-guide)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Clone & Navigate](#2-clone--navigate)
  - [3. Set Up Virtual Environment](#3-set-up-virtual-environment)
  - [4. Install Dependencies](#4-install-dependencies)
  - [5. Environment Configuration](#5-environment-configuration)
  - [6. OCR Backend Prerequisites (LM Studio / TrOCR)](#6-ocr-backend-prerequisites-lm-studio--trocr)
- [Commands for Different Tasks](#commands-for-different-tasks)
  - [Task 1: Run End-to-End OCR Pipeline](#task-1-run-end-to-end-ocr-pipeline)
  - [Task 2: Build ROI Dataset from Input PDFs](#task-2-build-roi-dataset-from-input-pdfs)
  - [Task 3: Run Preprocessing Benchmark](#task-3-run-preprocessing-benchmark)
  - [Task 4: Run OCR Preprocessing Evaluation](#task-4-run-ocr-preprocessing-evaluation)
  - [Task 5: Execute Unit Tests](#task-5-execute-unit-tests)
- [Pipeline Configuration](#pipeline-configuration)

---

## Features

- **PDF Ingestion & Rendering**: Renders PDF pages to high-resolution images via PyMuPDF (`fitz`).
- **Deskewing**: Automatically estimates and corrects page rotation and skew angles.
- **Vertical Line Detection**: Extracts margin vertical separation lines to isolate annotation zones.
- **ROI & Margin Cropping**: Crops target text margins and extracts region-of-interest (ROI) annotation blocks.
- **Image Preprocessing**: Enhances contrast (CLAHE, morphology, binarization, denoising) for high-accuracy OCR.
- **Modular OCR Engines**: Pluggable backend support including **LM Studio** local Vision LLMs and HuggingFace **TrOCR**.
- **Research Tools**: Utilities for dataset preparation, benchmarking preprocessing techniques, and evaluating OCR performance metrics.

---

## Folder Structure

```text
OCR_codes/
│
├── data/                            # Input and processed dataset directory
│   ├── input_pdfs/                  # Source scanned PDF files to process
│   └── roi_dataset/                 # Extracted Regions of Interest (ROIs) for research/training
│
├── denoised/                        # Denoised intermediate output images
├── results/                         # Sample image processing pipeline test results
│
├── outputs/                         # Main output directory generated per PDF
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
│   └── summary.csv                  # Aggregate execution summary across all processed PDFs
│
├── research/                        # Research, evaluation, and benchmark scripts
│   ├── benchmark.py                 # Evaluates preprocessing methods on sample images
│   ├── dataset.py                   # PyTorch dataset utility for fine-tuning / evaluation
│   ├── evaluate.py                  # Runs preprocessor evaluation suite across ROI dataset
│   ├── prepare_roi_dataset.py       # Extracts annotated ROIs from PDFs to build dataset
│   └── results/                     # Evaluation results (CSV metrics and preprocessed images)
│
├── src/                             # Source code modules
│   ├── ocr/                         # OCR engine implementations
│   │   ├── base.py                  # Base class for OCR backends
│   │   ├── factory.py               # Factory pattern to instantiate configured OCR backends
│   │   ├── lmstudio.py              # LM Studio local Vision LLM backend integration
│   │   └── trocr.py                 # HuggingFace TrOCR vision-transformer backend integration
│   ├── annotation_detector.py       # Detects bounding boxes for handwritten or typed annotations
│   ├── config.py                    # Global configuration dataclass (PipelineConfig)
│   ├── cropper.py                   # Crops pages based on detected separator coordinates
│   ├── deskewer.py                  # Page skew angle estimation and image rotation
│   ├── line_detector.py             # Separator line locator
│   ├── lmstudio_preprocessor.py     # Image preprocessing tailor-made for LM Studio input
│   ├── ocr_preprocessor.py          # Multimodal image preprocessing library (CLAHE, Otsu, morph)
│   ├── page_preprocessor.py         # Page-level preprocessor handler
│   ├── pdf_loader.py                # Converts PDF pages to NumPy OpenCV images
│   ├── pipeline.py                  # High-level pipeline controller
│   ├── preprocessor.py              # Primary binarization and grayscale preprocessing pipeline
│   ├── roi_text_extractor.py        # Connects ROIs to the active OCR backend
│   └── vertical_extractor.py        # Isolates vertical lines via morphological operations
│
├── tests/                           # Unit test suite
│   ├── test_denoise.py              # Tests for noise reduction algorithms
│   ├── test_deskew.py               # Tests for deskewing module
│   ├── test_detect_text.py          # Tests for text detection logic
│   ├── test_mistral.py              # API client integration tests
│   ├── test_mistral_local.py        # Local API backend integration tests
│   ├── test_ocr.py                  # General OCR engine tests
│   ├── test_overlap.py              # Tests for bounding box overlap and merging logic
│   ├── test_preprocessor.py         # Tests for image preprocessing pipelines
│   ├── test_recognizer.py           # Tests for text recognizer wrapper
│   └── test_vertical.py             # Tests for vertical line extraction
│
├── .env                             # Environment variables (API keys, endpoint URLs)
├── .gitignore                       # Git exclusion file
├── main.py                          # Primary execution script for end-to-end OCR pipeline
└── requirements.txt                 # Python dependencies manifest
```

---

## System Setup Guide

Follow these steps to set up and run the codebase on **Windows**, **macOS**, or **Linux**.

### 1. Prerequisites
- **Python 3.8+** (Python 3.10+ recommended)
- **Git**
- *(Optional)* **CUDA-capable GPU** with drivers installed if running local neural models (TrOCR/PyTorch) with hardware acceleration.

Verify your Python installation:
```bash
python --version
# or
python3 --version
```

### 2. Clone & Navigate
Clone the repository (or extract the source code) and change into the project root directory:
```bash
git clone <repository_url>
cd OCR_codes
```

### 3. Set Up Virtual Environment

It is recommended to use an isolated Python virtual environment:

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

Upgrade `pip` and install all required libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch (GPU acceleration):**  
> `requirements.txt` installs default PyTorch packages. If you want PyTorch with CUDA support on Windows/Linux, install CUDA-enabled PyTorch build from the [official PyTorch guide](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 5. Environment Configuration

Create or update the `.env` file in the root directory if using external APIs or custom server endpoints:

```env
MISTRAL_API_KEY=your_api_key_here
```

### 6. OCR Backend Prerequisites (LM Studio / TrOCR)

- **LM Studio (Default Backend):**
  - Download and start [LM Studio](https://lmstudio.ai/).
  - Load a Vision LLM model (e.g., Qwen2-VL, Llama-3-Vision, or similar).
  - Start the **Local Inference Server** inside LM Studio (default address: `http://localhost:1234`).
- **TrOCR Backend:**
  - Standard PyTorch environment; required models will automatically download from HuggingFace on first run.

---

## Commands for Different Tasks

### Task 1: Run End-to-End OCR Pipeline
Processes all input PDF files placed in `data/input_pdfs/` through deskewing, line detection, cropping, preprocessing, and OCR text extraction.

1. **Place your PDF files:**
   ```bash
   mkdir -p data/input_pdfs
   # Copy input .pdf files into data/input_pdfs/
   ```

2. **Execute the pipeline:**
   ```bash
   python main.py
   ```

3. **Check outputs:**
   - Detailed results are saved in `outputs/<pdf_name>/` (CSV reports, crop images, deskewed images, extracted OCR text per page).
   - An overall summary is saved to `outputs/summary.csv`.

---

### Task 2: Build ROI Dataset from Input PDFs
Extracts region-of-interest (ROI) annotation crops from PDFs in `data/input_pdfs/` and saves them to `data/roi_dataset/`.

```bash
python research/prepare_roi_dataset.py
```

---

### Task 3: Run Preprocessing Benchmark
Applies various image preprocessing filters (CLAHE, Otsu binarization, morphological transformations, sharpening) to an input sample image (`sample_crop.png`) and outputs benchmarking visualizations to `research/results/images/`.

```bash
python research/benchmark.py
```

---

### Task 4: Run OCR Preprocessing Evaluation
Evaluates all available preprocessing techniques across the dataset stored in `data/roi_dataset/`, measuring processing speeds (ms), token densities, block counts, and text output quality.

```bash
python research/evaluate.py
```

Outputs metrics to `research/results/csv/evaluation.csv` and intermediate outputs to `research/results/preprocessed/`.

---

### Task 5: Execute Unit Tests
Run the automated unittest suite to verify system logic across line detection, deskewing, OCR engines, preprocessors, and overlap algorithms.

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

---
