# OCR Dataset Builder

## Setup

1. Add your Mistral API key to `.env`

2. Copy all ROI images into

```
dataset/all_roi_images
```

3. Run

```
python run.py
```

The pipeline will

- Process the first 1000 images
- Send them to Mistral OCR
- Save OCR output to `annotations.csv`
- Move processed images to `completed/`

Run it again to process the next batch.

## OCR backend

Two backends are available and can be toggled without touching code:

- `mistral` (default) — sends each image to the hosted Mistral OCR API.
  Treated as an already-trusted teacher; output goes straight into
  `annotations.csv`.
- `parseq` — runs the locally fine-tuned PARSeq model fully offline,
  via the `venv` in this repo (see Setup below) spawned as a subprocess
  so the checkpoint loads once and is reused across the whole batch.

Pick one per run with `--backend`:

```
python run.py --backend parseq
python run.py --backend mistral
```

Or change the default in `src/config.py` (`OCR_BACKEND`), or set the
`OCR_BACKEND` environment variable. The checkpoint used for the `parseq`
backend is configured via `PARSEQ_CHECKPOINT` in `src/config.py`.

### Setting up the `parseq` backend's venv

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

This installs both backends' dependencies (CUDA PyTorch + PARSeq's
`strhub`, plus `requests`/`python-dotenv` for Mistral) into one venv, so
either `--backend` works from it.

## PARSeq predictions and human review

PARSeq (~93.5% val accuracy) isn't trusted the way Mistral is, so its
predictions never go straight into `annotations.csv`. Instead they're
written to `dataset/parseq_review_queue.csv` (filename, prediction,
confidence, `needs_review`, blank `corrected_text`) for a human to check —
the preview images in `dataset/preview/` (prediction burned into the
image) are meant to make that review fast.

Which predictions get flagged depends on `--review`:

- `--review low_confidence` (default) — only predictions below
  `PARSEQ_CONFIDENCE_THRESHOLD` (`src/config.py`) are flagged. Use this
  for production inference, where most predictions are trusted as-is and
  only the uncertain ones need a second look.
- `--review all` — every prediction is flagged. Use this when building a
  new training batch: nothing is trusted until a human confirms or
  corrects it, since these rows are meant to be fed back into the next
  PARSeq fine-tuning round (training itself happens in the separate
  `trocr-handwriting-v2` project, not here).

```
python run.py --backend parseq --review all             # building a training batch
python run.py --backend parseq --review low_confidence   # production inference
```

Merging reviewed/corrected rows from `parseq_review_queue.csv` back into
a training set is a manual step for now — there's no automated
promotion/retraining trigger yet.