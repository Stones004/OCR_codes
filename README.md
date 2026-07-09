# OCR Pipeline

This repository contains a Python-based Optical Character Recognition (OCR) pipeline. It is designed to take input PDF files, process them (loading, deskewing, line detection, cropping, preprocessing), and finally recognize text from the processed images.

## Directory Structure

```text
OCR_codes/
│
├── data/                  # Directory for input data. 
│   └── input_pdfs/        # Place your input PDF files here before running the pipeline.
│
├── outputs/               # Directory where the pipeline saves processed outputs.
│   ├── cropped/           # Intermediate cropped images.
│   ├── deskewed/          # Intermediate deskewed images.
│   ├── detected/          # Intermediate line-detected images.
│   ├── failed/            # Files that failed processing.
│   ├── preprocessed/      # Images after preprocessing steps.
│   ├── recognizer/        # Outputs from the OCR recognizer.
│   ├── roi/               # Regions of interest.
│   └── vertical/          # Vertical line extraction outputs.
│
├── src/                   # Source code modules for the pipeline.
│   ├── cropper.py         # Module to crop regions of interest.
│   ├── deskewer.py        # Module for image deskewing (straightening).
│   ├── line_detector.py   # Module to detect lines of text.
│   ├── ocr.py             # General OCR utilities.
│   ├── pdf_loader.py      # Module to load and parse PDFs.
│   ├── preprocessor.py    # Image preprocessing before OCR.
│   ├── recognizer.py      # Module to recognize text (EasyOCR/PyTorch based).
│   ├── skew_estimator.py  # Utility to estimate image skew angle.
│   └── vertical_extractor.py # Module to extract vertical lines/separators.
│
├── tests/                 # Unit tests for the pipeline modules.
│   ├── test_deskew.py
│   ├── test_ocr.py
│   ├── test_preprocessor.py
│   ├── test_recognizer.py
│   └── test_vertical.py
│
├── .gitignore             # Git ignore file.
├── main.py                # The main execution script. Run this to start the pipeline.
└── requirements.txt       # Python package dependencies.
```

## Setup Instructions

Follow these steps to set up and run the codebase on any system:

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system. You can verify your Python version by running:
```bash
python --version
```

### 2. Clone/Navigate to the Repository
Open your terminal or command prompt and navigate to the project directory:
```bash
cd path/to/OCR_codes
```

### 3. Create a Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On macOS and Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all the required Python libraries (like `easyocr`, `opencv-python`, `pandas`, `torch`, etc.) using `pip`:
```bash
pip install -r requirements.txt
```

*(Note: Depending on your system and hardware, installing PyTorch/EasyOCR may require specific installation commands to enable GPU acceleration. By default, `requirements.txt` will install the standard packages).*

### 5. Prepare Input Data
Create the input directory (if it does not exist) and place your PDF files there.
```bash
mkdir -p data/input_pdfs
```
Copy any PDF files you want to process into `data/input_pdfs/`.

### 6. Run the Pipeline
Execute the main script to start processing the PDFs:
```bash
python main.py
```
The script will locate all PDFs in `data/input_pdfs/`, run the full OCR pipeline on them, and output intermediate and final results into the `outputs/` folder.

## Running Tests
To ensure everything is working correctly, you can run the test suite:
```bash
python -m unittest discover -s tests
```
