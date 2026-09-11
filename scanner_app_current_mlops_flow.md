# Scanner App — Current ML / Data Flow

## 1. Overview

The Scanner App currently consists of **two independent repositories/codebases**.

The repositories are intentionally separate and communicate primarily through **shared data/artifacts**:

- **Repo 1 — Data & Dataset Pipeline**
  - Data preprocessing
  - Data normalization
  - Human intervention/review
  - Dataset creation
  - Training-data preparation

- **Repo 2 — Scanner / Inference Pipeline**
  - Raw PDF ingestion
  - PDF preprocessing
  - ROI/question detection
  - Bounding-box generation
  - OCR inference
  - Result generation
  - JSON output

At the current stage, the two repositories should **remain separate**. The future MLOps architecture should connect them through versioned datasets, trained models, model metadata, and controlled deployment rather than merging the codebases.

---

# 2. Current High-Level Flow

```text
                         ┌─────────────────────────────┐
                         │          REPO 1             │
                         │  Data / Dataset Pipeline    │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                              Raw / Source Data
                                        │
                                        ▼
                              Data Preprocessing
                                        │
                                        ▼
                               Normalization
                                        │
                                        ▼
                              Human Intervention
                              / Review / Correction
                                        │
                                        ▼
                               Dataset Creation
                                        │
                                        │
                                  Shared Data
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │          REPO 2             │
                         │   Scanner / Inference       │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                              Raw PDF Ingestion
                                        │
                                        ▼
                              PDF Preprocessing
                                        │
                                        ▼
                           ROI / Question Detection
                                        │
                                        ▼
                              Bounding Boxes
                                        │
                                        ▼
                              OCR Inference Layer
                              (TrOCR / PARSeq / ...)
                                        │
                                        ▼
                              Post-processing
                                        │
                                        ▼
                                 JSON Results
```

---

# 3. Repo 1 — Data & Dataset Pipeline

Repo 1 acts as the **data factory** for the OCR models.

Its primary responsibility is to transform raw or collected samples into datasets suitable for model training and evaluation.

```text
Raw Data
   │
   ▼
Preprocessing
   │
   ├── Cleaning
   ├── Image preparation
   └── Other transformations
   │
   ▼
Normalization
   │
   ├── Standardize image representation
   ├── Normalize dimensions / format
   └── Prepare samples consistently
   │
   ▼
Human Intervention
   │
   ├── Review samples
   ├── Correct labels
   ├── Remove invalid samples
   └── Handle difficult cases
   │
   ▼
Dataset Creation
   │
   ├── Train
   ├── Validation
   └── Test
   │
   ▼
Shared Dataset / Training Artifact
```

## Key characteristic

Human intervention is an explicit part of the pipeline.

This means the dataset is not simply generated automatically. Human corrections and decisions contribute directly to the quality of the eventual OCR model.

---

# 4. Repo 2 — Scanner / Inference Pipeline

Repo 2 represents the **runtime application pipeline**.

Its responsibility is to take a raw PDF and transform it into structured OCR results.

```text
Raw PDF
   │
   ▼
PDF Ingestion
   │
   ▼
PDF Preprocessing
   │
   ├── Page processing
   ├── Image preparation
   └── Other PDF-specific transformations
   │
   ▼
ROI / Question Detection
   │
   ▼
Bounding Boxes
   │
   ├── Question number regions
   ├── Relevant answer regions
   └── Other detected regions
   │
   ▼
ROI Extraction
   │
   ▼
OCR Inference
   │
   ├── TrOCR
   ├── PARSeq
   └── Future OCR models
   │
   ▼
Post-processing
   │
   ▼
Structured JSON
```

---

# 5. Relationship Between the Two Repositories

Currently, the relationship can be represented as:

```text
┌──────────────────────────────┐
│           REPO 1             │
│                              │
│ Data → Processing → Dataset  │
└──────────────┬───────────────┘
               │
               │ Shared data / artifacts
               ▼
┌──────────────────────────────┐
│           REPO 2             │
│                              │
│ PDF → ROI → OCR → JSON       │
└──────────────────────────────┘
```

The repositories do **not** need to become a single repository.

Instead, the future goal is to make the interface between them explicit and reproducible.

---

# 6. Future MLOps-Oriented Flow

The eventual architecture should introduce versioning and traceability between data, experiments, models, and production.

```text
                    ┌─────────────────────────┐
                    │         REPO 1          │
                    │ Data / Dataset Pipeline │
                    └────────────┬────────────┘
                                 │
                                 ▼
                         Versioned Dataset
                                 │
                                 ▼
                        Training / Evaluation
                                 │
                                 ▼
                         Experiment Tracking
                                 │
                                 ▼
                          Model Validation
                                 │
                                 ▼
                         Model Registry
                                 │
                                 ▼
                       Approved Model Version
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │         REPO 2          │
                    │ Scanner / Inference     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                            Raw PDF
                                 │
                                 ▼
                       PDF Preprocessing
                                 │
                                 ▼
                         ROI Detection
                                 │
                                 ▼
                         OCR Inference
                                 │
                                 ▼
                     Prediction + Confidence
                                 │
                                 ▼
                           JSON Result
```

---

# 7. Future Closed-Loop Learning Flow

A major future improvement is to allow production results to feed difficult examples back into the data pipeline.

```text
                     PRODUCTION
                         │
                         ▼
                    OCR Prediction
                         │
                         ▼
                Confidence / Quality
                         │
              ┌──────────┴──────────┐
              │                     │
          High confidence       Low confidence
              │                     │
              ▼                     ▼
            Accept              Human Review
                                    │
                                    ▼
                              Correction
                                    │
                                    ▼
                           Hard Example Queue
                                    │
                                    ▼
                                  REPO 1
                                    │
                                    ▼
                           Dataset Version N+1
                                    │
                                    ▼
                              Retraining
                                    │
                                    ▼
                               Evaluation
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                      Better                Worse
                         │                     │
                         ▼                     ▼
                    New Model              Reject
                         │
                         ▼
                 Model Registry
                         │
                         ▼
                    Production
```

This creates the eventual:

> **Production → Feedback → Dataset → Training → Evaluation → Model → Production**

loop.

---

# 8. Model Version Traceability

The system should eventually be able to identify exactly which versions produced a result.

For example:

```json
{
  "pipeline_version": "2.4.0",

  "roi_detector": {
    "model": "question-detector",
    "version": "3.2"
  },

  "ocr": {
    "model": "parseq",
    "version": "2.7"
  },

  "preprocessing": {
    "version": "1.8"
  }
}
```

This allows a production result to be traced backwards:

```text
JSON Result
    │
    ▼
Pipeline Version
    │
    ├── ROI Detector Version
    │
    ├── OCR Model Version
    │
    └── Preprocessing Version
              │
              ▼
        Training Experiment
              │
              ▼
          Dataset Version
```

This is important for debugging and reproducibility.

---

# 9. Main Components of the Future System

| Component | Responsibility |
|---|---|
| Repo 1 | Data processing and dataset generation |
| Repo 2 | Production scanning and inference |
| Dataset Versioning | Track exactly which data was used |
| Experiment Tracking | Record training parameters and metrics |
| Model Registry | Store and identify approved model versions |
| Model Evaluation | Automatically compare candidate models |
| Human Review | Correct difficult/uncertain predictions |
| Feedback Dataset | Store production errors and corrections |
| Inference Metadata | Track model/pipeline versions |
| Monitoring | Observe production quality and performance |
| CI/CD | Automate testing and deployment |

---

# 10. Important Design Principle

The goal is **not**:

```text
Two repositories → One repository
```

The goal is:

```text
Two repositories
      │
      ▼
Well-defined interfaces
      │
      ▼
Versioned data
      │
      ▼
Reproducible experiments
      │
      ▼
Versioned models
      │
      ▼
Controlled deployment
      │
      ▼
Production feedback
      │
      ▼
Continuous improvement
```

The two repositories can therefore remain independently maintainable while participating in one overall ML lifecycle.

---

# 11. Current State vs Target State

## Current

```text
Repo 1
  ↓
Processed / shared data
  ↓
Repo 2
  ↓
Inference
  ↓
JSON
```

## Target

```text
Repo 1
  ↓
Versioned Dataset
  ↓
Training
  ↓
Experiment Tracking
  ↓
Evaluation
  ↓
Model Registry
  ↓
Approved Model
  ↓
Repo 2
  ↓
Inference
  ↓
Prediction + Confidence
  ↓
JSON
  ↓
Human Feedback
  ↓
Versioned Dataset
  ↓
Retraining
```

---

# 12. Recommended Evolution Path

The system should be upgraded incrementally.

### Stage 1 — Formalize the current architecture

Document:

- Inputs and outputs of Repo 1
- Inputs and outputs of Repo 2
- Shared artifacts
- Dataset formats
- Model formats
- Configuration
- Evaluation metrics

### Stage 2 — Version datasets and models

Introduce:

- Dataset versions
- Model versions
- Pipeline versions
- Reproducible configurations

### Stage 3 — Add experiment tracking

Track:

- Dataset version
- Model architecture
- Hyperparameters
- Preprocessing configuration
- Training metrics
- CER
- NED
- Accuracy
- Checkpoints

### Stage 4 — Introduce a model registry

Models move through:

```text
Candidate
   ↓
Validated
   ↓
Staging
   ↓
Production
   ↓
Archived
```

### Stage 5 — Add production feedback

Capture:

```text
Input
Prediction
Confidence
Model Version
Human Correction
```

and use these samples to improve future datasets.

### Stage 6 — Automate the lifecycle

Eventually:

```text
New Data
   ↓
Validation
   ↓
Dataset Version
   ↓
Training
   ↓
Evaluation
   ↓
Model Validation
   ↓
Registry
   ↓
Deployment
   ↓
Monitoring
```

---

# 13. Current Architectural Goal

The immediate objective is **not to implement every MLOps tool**.

The immediate objective is to make the existing system:

1. **Reproducible**
2. **Versioned**
3. **Traceable**
4. **Evaluatable**
5. **Deployable**
6. **Capable of receiving human feedback**

Once those foundations are in place, tools such as MLflow, DVC, Docker, CI/CD, monitoring, and workflow orchestration can be introduced where they actually solve a problem.

---

## Final Concept

The Scanner App should evolve from:

```text
DATA → OCR → JSON
```

into:

```text
DATA
  ↓
DATASET
  ↓
TRAIN
  ↓
EVALUATE
  ↓
REGISTER MODEL
  ↓
DEPLOY
  ↓
SCAN
  ↓
OCR
  ↓
CONFIDENCE
  ↓
HUMAN FEEDBACK
  ↓
NEW DATA
  ↓
RETRAIN
  ↓
BETTER MODEL
  ↓
DEPLOY
```

That is the foundation of the Scanner App's future MLOps architecture.
