# Codebase Guide: Data Flow and Configuration

A map of how data moves through this repo, and where every tunable lives.
Every setting is now in one file, `config.yaml`; you should not need to edit
Python to change behaviour.

---

## 0. Short answers

**Is everything controlled by config? Yes, with a short list of stated exceptions.**

- Every tunable (thresholds, kernel sizes, HSV ranges, DPI, page skipping, padding,
  folder and file names, OCR backend/model/host/prompt, UI polling and colours,
  dataset-builder paths and batch size, research-script paths) is in
  [config.yaml](config.yaml).
- The code has **no default values**. `src/config.py` only defines the shape of the
  file (typed dataclasses) and validates it. A missing, misspelled or wrongly typed
  key stops the program at startup with a message that names the key.
- Every entry point accepts `--config other.yaml` (or set the `OCR_CONFIG`
  environment variable), so you can keep one file per experiment.
- Relative paths in the YAML resolve against the YAML file's folder, not the
  current directory.
- What is deliberately **not** configurable: the choice of algorithm (Otsu, connected
  components, 8-connectivity), fixed formats and schemas (CSV column names,
  `SUCCESS`/`FAILED`, PNG as the output format, the PDF unit of 72 points per inch),
  third-party API protocol fields, console banner widths, and HTTP status codes.

- A fresh clone still needs the git-ignored files (PARSeq venv, checkpoints, PDFs); see
  section 7 and README "Setting up on a fresh clone". No code edits are needed.

Section 5 explains how to change things. The rest of this file explains the code.

---

## 1. What this project does

Scanned exam-style PDFs have a printed table with a **left margin** where a
reviewer wrote small handwritten marks (mostly numbers and roman numerals). The
project:

1. Renders each PDF page to an image.
2. Finds the vertical printed line that separates the margin from the body.
3. Crops the margin, removes red/green pen ink, and finds each handwritten mark.
4. Runs OCR on each mark and writes the text to CSV.
5. Separately, supports **building a labelled dataset** from the cropped marks
   (Mistral API or a local PARSeq model, with human review) so the local model can be
   fine-tuned. Fine-tuning happens in a different repo (`trocr-handwriting-v2`).

---

## 2. Repo map (live vs dead)

| Path | Role | Status |
|---|---|---|
| `main.py` | **Primary entry point.** PDF -> per-page CSVs. Runs on import (no `__main__` guard). | live |
| `config.yaml` | **All settings** (single source of truth) | live |
| `src/config.py` | Schema (typed dataclasses) + loader/validator for `config.yaml`. No values. | live |
| `src/pdf_loader.py` | PDF -> list of BGR images | live |
| `src/deskewer.py` | Skew estimation + rotation | live |
| `src/vertical_extractor.py` | Isolate long vertical strokes | live |
| `src/line_detector.py` | Pick the separator line | live |
| `src/cropper.py` | Crop margin at the separator | live |
| `src/annotation_remover.py` | Erase red/green ink via HSV mask | live |
| `src/preprocessor.py` | gray + cleaned binary | live |
| `src/annotation_detector.py` | Find handwritten-mark ROIs | live |
| `src/roi_text_extractor.py` | ROIs -> backend -> `[{block_id, text}]` | live |
| `src/ocr/` | Backends: `parseq` (default), `lmstudio` + factory | live |
| `src/ocr/parseq_worker.py` | Subprocess that hosts the PARSeq model | live (run by another interpreter) |
| `src/lmstudio_preprocessor.py` | Denoise + sharpen before LM Studio | live (lmstudio only) |
| `roi_consolidator.py` | Copy all `outputs/Doc*/roi/*.png` into one folder | live, manual step |
| `roi_ui/` | Flask app: watch a folder, OCR new images, review UI | live, standalone |
| `ocr_dataset_builder/` | Label ROI images (Mistral / PARSeq) + review queue | live, standalone |
| `inspect_thresholds.py` | Diagnostic: shows which components pass/fail each threshold | live, tool |
| `research/prepare_roi_dataset.py` | Re-implementation of the front half of `main.py` that saves ROIs to `data/roi_dataset/` | live, diverges from `main.py` |
| `research/evaluate.py`, `research/benchmark.py` | Compare preprocessing methods through an OCR backend | experimental |
| `src/ocr_preprocessor.py` | Six binarization variants for benchmarking | only used by `research/` |
| `src/page_preprocessor.py` | Pass-through stub | unused |
| `tests/` | Ad-hoc scripts, not a test suite | see section 6 |
| `scanner_app_current_mlops_flow.md` | Future MLOps design notes (two-repo flow) | prose only |

---

## 3. Inputs and entry points

| Entry point | How it is run | Input | Output |
|---|---|---|---|
| `main.py` | `python main.py [--config FILE]` | `data/input_pdfs/*.pdf` | `outputs/<pdf>/...`, `outputs/summary.csv` |
| `roi_consolidator.py` | `python roi_consolidator.py [--config FILE]` | `outputs/Doc*/roi/*.png` | `outputs/all_roi_images/` |
| `ocr_dataset_builder/run.py` | `python run.py [--config FILE] [--backend mistral\|parseq] [--review all\|low_confidence]` | `ocr_dataset_builder/dataset/all_roi_images/` | `annotations.csv` or `parseq_review_queue.csv`, `preview/`, `completed/` |
| `roi_ui/app.py` | `python roi_ui/app.py [--config FILE] [--input] [--output] [--port]` | images dropped into `data/roi_watch/input/` | annotated PNGs in `.../output/`, `data/roi_watch/review_queue.csv` |
| `inspect_thresholds.py` | `python inspect_thresholds.py <pdf-or-dir> [--config FILE] [--pages] [--out] [--no-images] [--debug-dir]` | PDFs | CSV report + `threshold_debug/` images |
| `research/prepare_roi_dataset.py` | `python research/prepare_roi_dataset.py [--config FILE]` | `data/input_pdfs/*.pdf` | `data/roi_dataset/*.png` |
| `research/evaluate.py` | `python research/evaluate.py [--config FILE]` | `data/roi_dataset/` | `research/results/...` |
| `research/benchmark.py` | `python research/benchmark.py [--config FILE]` | `research.benchmark.image_path` | `research/results/images/` |

Secrets and environment: `MISTRAL_API_KEY` comes from the repo-root `.env` (read by
`ocr_dataset_builder/src/mistral_client.py`). Env vars `OCR_BACKEND` and
`PARSEQ_REVIEW_MODE` are honoured **only** by the dataset builder.

Data-egress note: the `mistral` backend uploads every ROI image to `api.mistral.ai`.
Everything else runs locally (LM Studio on `127.0.0.1:1234`, PARSeq in a local
subprocess).

---

## 4. Data flow

### 4.1 Main pipeline (`main.py`)

Numbers in the diagrams below are the values currently in `config.yaml`; the key that
controls each one is named in the text.

```
data/input_pdfs/*.pdf            (paths.input_dir)
   |  sorted glob (paths.pdf_glob). A PDF is SKIPPED if outputs/<stem>/ already exists
   |  and pipeline.skip_existing_outputs is true
   v
PDFLoader.load_pdf            render at pdf.dpi (300), drop first pdf.skip_pages (2) pages
   |                          -> list of BGR ndarrays
   |  per page (page_001 == 3rd PDF page)
   v
Deskewer.process              Otsu-binarize -> projection-variance search over
   |                          +-5 deg (coarse 0.5, fine 0.05) -> rotate
   |                          -> corrected (BGR), angle                 -> deskewed/
   v
VerticalExtractor.extract     gray -> threshold 180 -> close (3x15) -> open with a
   |                          (1 x max(80, H/25)) kernel -> vertical (binary)  -> vertical/
   v
LineDetector.detect           connected components; keep tall (>=50% page height),
   |                          thin (h/w >= 40), x in [3%, 30%] of width;
   |                          pick left-most  -> overlay + separator dict  -> detected/
   |
   +-- separator is None --> failed/ (gray, binary, vertical); report.csv row FAILED;
   |                          no OCR for this page; continue with next page
   v
Cropper.crop                  keep columns [0 : separator.x + crop.right_padding (5)]
   |                          -> crop (BGR)                               -> cropped/
   v
AnnotationRemover             (if pipeline.use_annotation_removal) HSV mask of red + green ink,
   |                          close + dilate, "fringe" cleanup near large components,
   |                          paint masked pixels white
   |                          -> clean_crop (BGR) + mask        -> annotation_removed/
   v
Preprocessor.process          gray; Otsu-inverse binary (forced blank if the threshold
   |                          is within 30 levels of the mean); drop components < 20 px
   |                          -> gray, binary                        -> preprocessed/
   v
   +-- pipeline.use_annotation_detection = true  (default)
   |     AnnotationDetector.detect(gray, binary)     -> rois = [{bbox, roi}]
   |         roi/ (pipeline.save_rois), recognizer/ (pipeline.save_debug_images)
   |     ROITextExtractor.extract(gray, rois)   one backend.read() per ROI
   |
   +-- pipeline.use_annotation_detection = false
         ROITextExtractor.extract(gray)         whole-margin OCR (one call)
   v
ocr/page_NNN.csv              columns: block_id, text
report.csv                    per page: status, deskew_angle, separator x/w/h
outputs/summary.csv           per PDF: pages, success, failed, mean/std separator_x
```

Things worth knowing about this flow:

- **Only `text` survives to disk.** `ROITextExtractor.extract`
  (`src/roi_text_extractor.py`) throws away `confidence` and `bbox`. The
  per-page CSV cannot be traced back to a location on the page, or to a confidence,
  except by ROI order: `block_id = i` corresponds to `roi/page_NNN_roi_{i}.png` (for
  PARSeq, which returns one item per ROI; LM Studio can return several per ROI and
  breaks that alignment).
- **`summary.csv` is overwritten every run** and only lists PDFs processed in that
  run. Skipped PDFs vanish from it.
- **The OCR backend is created at import time** (`main.py`, right after the stages are
  built). For `parseq`, that spawns the PARSeq subprocess even if there are no PDFs.
- **Page numbering** starts after the skipped cover pages, so `page_001` is PDF
  page 3.

### 4.2 Inside `AnnotationDetector.detect` (the most tunable stage)

```
binary (from Preprocessor)
  1. _find_top_line_y     find the printed header rule in the top 20% (long horizontal
  |                       run >= 60% of width). Must run BEFORE line removal.
  2. remove_lines         open with horizontal (w/4) and vertical (h/4) kernels,
  |                       dilate 3x3, subtract -> printed table lines gone
  3. connected components; drop a component if any of:
  |     w < min_width | h < min_height
  |     entirely above the header rule (+ margin)
  |     large but sparse (max(w,h) > 80 and ink density < 0.12)  -> arrows/connectors
  |     narrow and tall (w < 30 and h > 100)                      -> edge shadows
  4. _merge_boxes         iteratively merge boxes within merge_dist (16 px)
  5. _refine_rois         split ROIs taller than 140 px at horizontal-projection valleys
  6. confidence filter    reject by area, width/height, ink pixels, fill ratio
  7. _group_boxes_by_y    boxes whose vertical centres are within 18 px become ONE row box
  8. pad by 10 px, crop from the GRAY image
  -> rois: [{"bbox": (x1,y1,x2,y2), "roi": gray_crop}]
```

**One ROI is one horizontal writing row**, not one digit. Detection runs on
`binary`, but the ROI pixels are cut from `gray`.

### 4.3 The OCR backend layer

```
ROITextExtractor  --create_backend(ocr.backend, cfg.ocr)-->  OCRBackend.read(image) -> {"text","confidence","words"}
                       (src/ocr/factory.py)
   "parseq"   ParseqBackend    subprocess -> ocr_dataset_builder/venv python
   "lmstudio" LMStudioBackend  OpenAI client -> http://127.0.0.1:1234/v1
```

PARSeq runs out of process because `strhub` is installed only in
`ocr_dataset_builder/venv`, not in the root `venv`. Protocol (`src/ocr/parseq.py`,
`src/ocr/parseq_worker.py`):

```
parent: cv2.imencode(.png) -> temp file -> write "<path>\n" to worker stdin
worker: PIL open -> RGB -> resize to model.hparams.img_size -> normalize(norm_mean, norm_std)
        -> model -> softmax -> tokenizer.decode
        -> stdout: {"prediction": "...", "confidence": mean(token probs)}
parent: parse JSON, delete temp file
worker prints "READY" once when the checkpoint is loaded
```

The factory passes each backend its own section of `config.yaml`
(`ocr.parseq`, `ocr.lmstudio`): model, host, prompt, checkpoint, venv path,
device, token limit. The PARSeq worker runs in another venv, so its settings (device,
normalisation) are handed over as one JSON argument on its command line.

### 4.4 Hand-off from the pipeline to the other tools (all manual)

```
main.py --> outputs/<Doc>/roi/page_NNN_roi_K.png
                |
                +--> roi_consolidator.py --> outputs/all_roi_images/<Doc>_page_NNN_roi_K.png
                |         |
                |         +-- (you copy) --> ocr_dataset_builder/dataset/all_roi_images/  --> 4.5
                |
                +-- (you copy) --> data/roi_watch/input/  --> 4.6
```

Nothing connects these stages automatically. Each hop is a copy you do by hand.

### 4.5 Dataset builder (`ocr_dataset_builder/run.py`)

```
dataset/all_roi_images/  (first dataset_builder.batch_size images, sorted by name)
   |
   v  per image: client.ocr_image(path) -> (text, confidence)
   |
   +-- backend "mistral"  upload file -> signed URL -> POST /v1/ocr -> markdown text
   |                      confidence = None; rows go to dataset/annotations.csv
   |
   +-- backend "parseq"   local worker (same protocol as 4.3, checkpoint parseq_final.ckpt)
                          needs_review = (review_mode == "all") or (confidence < review.confidence_threshold)
                          rows go to dataset/parseq_review_queue.csv
                          (filename, prediction, confidence, needs_review, corrected_text="")
   |
   v  create_preview(): image with the prediction drawn on it -> dataset/preview/
   v  move image -> dataset/completed/
```

A human then edits `corrected_text` in the review queue and (manually) uses the rows
to fine-tune PARSeq in the other repo. The new checkpoint is dropped into
`ocr_dataset_builder/models/` and referenced by editing the checkpoint path in
`config.yaml`. There is
no automated promotion step.

### 4.6 ROI watcher UI (`roi_ui/app.py`)

```
data/roi_watch/input/*.png  <-- polled every roi_ui.poll_seconds; a file is read once its
   |                            (size, mtime) is unchanged for roi_ui.stable_polls polls
   v
create_backend(cfg.ocr.backend, cfg.ocr).read(path) -> text, confidence
   v
annotate_image() -> data/roi_watch/output/<stem>.png   (text + confidence drawn under image)
append_review_row() -> data/roi_watch/review_queue.csv (id, filename, prediction,
   |                    confidence, needs_review = confidence < review.confidence_threshold, corrected_text, ...)
   v
Flask JSON API (/api/status, /api/results, /api/correct) <-- browser polls every 1.5-2 s
   correction in the browser -> POST /api/correct -> rewrites the row in review_queue.csv
```

This uses the same `ocr.backend` as `main.py`, so switching backend in
`config.yaml` changes both.

### 4.7 Research scripts

- `prepare_roi_dataset.py` **re-implements** the front half of `main.py`: no annotation
  removal, no `Preprocessor` (raw Otsu binary), and its own `right_padding` (15) and
  `skip_pages` (0) from `research.prepare_roi_dataset` instead of `crop.right_padding` (5)
  and `pdf.skip_pages` (2). ROIs produced here are therefore **not identical** to the
  ones `main.py` produces, unless you set those two keys equal.
- `evaluate.py` runs every `OCRPreprocessor` method over `data/roi_dataset/` through
  the backend named in `research.evaluate.backend` (`lmstudio`) and writes timing/size metrics to `research/results/`.

---

## 5. Configuration

### 5.1 How it works

```
config.yaml  --load_config()-->  Config (frozen, typed dataclasses)  -->  each stage gets its own section
   ^                                     |
   |  --config FILE / OCR_CONFIG         +-- unknown / missing / wrongly-typed key => startup error
```

Each stage takes its section in the constructor and copies the values it needs, e.g.
`Deskewer(cfg.deskew)`, `AnnotationDetector(cfg.annotation_detector, cfg.visualization, verbose=...)`,
`create_backend(name, cfg.ocr)`. There are no default arguments anywhere, so there is
exactly one place to set any number.

### 5.2 Where to change what

| I want to change... | Section in `config.yaml` |
|---|---|
| Which PDFs are read / where output goes / folder and file names | `paths`, `layout` |
| Which features run (ROI detection, ink removal, saving ROIs/debug images, skip finished PDFs, console logging) | `pipeline` |
| Render DPI, cover pages skipped | `pdf` |
| Deskew search range and steps | `deskew` |
| How the separator line is found and chosen | `vertical`, `separator` |
| How much of the separator is kept when cropping | `crop` |
| Red/green ink colours and clean-up | `ink_removal` |
| Noise filtering before detection | `preprocess` |
| What counts as a handwritten mark, how marks are merged into rows | `annotation_detector` |
| Colours/thickness of debug overlays | `visualization` |
| OCR backend and its model, host, prompt, checkpoint, device | `ocr` (`parseq`, `lmstudio`) |
| Review threshold (shared by UI and dataset builder) | `review` |
| Accepted image extensions | `image_extensions` |
| Review UI folders, port, polling, colours, fonts | `roi_ui` |
| Dataset builder paths, batch size, backend, review mode, Mistral model, preview text | `dataset_builder` |
| ROI consolidation folders | `roi_consolidator` |
| Threshold inspector output and drawing style | `inspect_thresholds` |
| Research scripts (ROI dataset, preprocessing methods, evaluation, benchmark) | `research` |

Per-run overrides that remain (they only choose *which* config value to use):

| Entry point | Options |
|---|---|
| all scripts | `--config FILE` (or `OCR_CONFIG`) |
| `ocr_dataset_builder/run.py` | `--backend`, `--review`; env `OCR_BACKEND`, `PARSEQ_REVIEW_MODE` |
| `roi_ui/app.py` | `--input`, `--output`, `--port` |
| `inspect_thresholds.py` | `--pages`, `--out`, `--debug-dir`, `--no-images` |

`MISTRAL_API_KEY` stays in `.env` (a secret, not a setting).

### 5.3 Things worth knowing about the values

- **Values were carried over unchanged** from the previous code, so a default run behaves
  as before (verified, see 5.5).
- **`research.prepare_roi_dataset`** keeps its own `right_padding: 15` and
  `skip_pages: 0`, which differ from `crop.right_padding: 5` and `pdf.skip_pages: 2`
  used by `main.py`. That is how the script always behaved. Set them equal to `main.py`
  if you want the research ROIs to match production ROIs exactly.
- **PARSeq checkpoints differ on purpose (for now):** `ocr.parseq.checkpoint` (scanner)
  is `parseq_base_mix_v2.ckpt`; `dataset_builder.parseq_checkpoint` is `parseq_final.ckpt`.
  They are now visible side by side in one file, so pointing both at the same model is a
  one-line change.
- **`ocr.parseq.venv_python`** uses the Windows layout (`venv/Scripts/python.exe`).
  On Linux/macOS change it to `venv/bin/python`.
- **`ocr.lmstudio.debug_image_path`** is `null` (nothing saved). Before, the code tried to
  write to a folder that did not exist and silently saved nothing. Set a path to enable it.
- **Absolute vs relative:** because relative paths resolve against the YAML's folder,
  running from a different working directory no longer changes where files go.

### 5.4 Adding a new setting

1. Add the field to the right dataclass in `src/config.py` (no default).
2. Add the key and value to `config.yaml`.
3. Read it in the module from the section object it already receives.

Forgetting step 2 fails at startup with `missing key(s) [...]`, which is the intent.

### 5.5 How this was verified

- **Stage outputs identical to the old code.** The old (git `HEAD`) stages and the new
  config-driven stages were run on 6 pages of `Doc0624.pdf`; pixel hashes of every
  intermediate image (deskew, vertical, separator overlay, crop, ink-removed, gray,
  binary, ROI crops, detection overlay), the separator positions, and all 24 ROI boxes
  matched exactly.
- **Full `main.py` run** (PARSeq backend, 18 pages) into a scratch folder: all 219 images
  and `report.csv` are byte-identical to the earlier output in `outputs/Doc0624`. 4 of 18
  OCR CSVs differ, because that earlier output predates the current checkpoint
  (`parseq_base_mix_v2.ckpt` is newer than those CSVs). The original PARSeq worker and
  the new one return identical predictions and confidences on all 39 ROIs.
- **Validation:** missing, unknown, wrongly-typed and null-where-not-allowed keys are rejected
  with the key path in the message.
- **Behaviour follows config:** changing the red HSV range in config changes the ink
  removal output with no code edit.
- **Every other entry point ran** against a sandbox config: `roi_consolidator`,
  `inspect_thresholds`, `research/benchmark`, `research/prepare_roi_dataset`, the dataset
  builder (PARSeq, `--review all`), and the `roi_ui` app (rendering and the injected
  page settings).
- **Fresh environment:** a brand-new venv built only from `requirements.txt`, with
  `openai` uninstalled, ran `main.py` end to end; all 220 output files were byte-identical
  to the previous run.
- **Not run:** `research/evaluate.py` (needs a running LM Studio server); it was only
  syntax/import checked.

### 5.6 Audit result (mechanical check of the code as it is now)

A one-off AST scan (not a script kept in the repo) of every live Python file (`src/`, `main.py`, `roi_ui/`, `ocr_dataset_builder/`,
`research/`, the tools):

| Check | Result |
|---|---|
| Function parameters with a literal default value | **0** |
| Config fields defined in the schema | 204, **all** read by code |
| Numeric literals left in code, other than 0, 1, 2, -1, 255 | only the exempt kinds below |

The literals that remain, and why they are fine:

| Literal | Where | Why it is not a tunable |
|---|---|---|
| `8` | `connectivity=8` in connected-component calls | algorithm identity |
| `3` | tuple/array indexes and `len(shape) == 3` (channel check) | structural |
| `70`, `60` | `"=" * 70` console banners | print decoration |
| `72` | `PDF_POINTS_PER_INCH` | PDF unit definition |
| `1000` | seconds to milliseconds in `research/evaluate.py` | unit conversion |
| `4` | `round(density, 4)`, `pix.n == 4` (RGBA) | output precision / data format |
| `400`, `404` | HTTP status codes in `roi_ui/app.py` | protocol |

Not covered by the scan: the `tests/` scratch scripts (see section 6), and the CSS in
`roi_ui/templates/index.html` (styling, not behaviour). The JavaScript tunables (poll
intervals, confidence bands) are injected from config.

---

## 6. Changes made while moving to config, and what is still rough

Changes beyond "move value to YAML":

1. **Deleted `src/pipeline.py`.** It could not run (it imported `main`, which executes the
   whole pipeline, plus classes that do not exist) and nothing used it.
2. **Removed dead parameters** that looked tunable but did nothing:
   `AnnotationRemover.use_inpaint` / `inpaint_radius`, `VerticalExtractor.kernel_height`,
   `PipelineConfig.OCR_MAX_TOKENS` / `USE_OCR_PREPROCESSOR`, and the unused
   bilateral/gamma/rolling-ball options in `OCRPreprocessor`.
3. **Removed dead code:** three unused helper methods and a commented-out block in
   `annotation_detector.py`, the commented-out legacy class in `preprocessor.py`, and an
   unused local `Preprocessor` class in `research/prepare_roi_dataset.py`.
4. **Fixed a duplicate append** in `AnnotationDetector.detect` (each accepted box was added
   twice). Output on the test pages is identical, but in principle it could shift the row
   grouping on unusual pages.
5. **The TrOCR backend was removed** (`src/ocr/trocr.py`, its config section, and
   `tests/test_overlap.py`, which was a TrOCR-only experiment). Unused imports were also
   removed from `lmstudio.py`.
6. **`ParseqBackend.close()`** no longer raises `AttributeError` when the constructor failed
   (for example a wrong `venv_python`).
7. **Debug prints** in the detectors are now controlled by `pipeline.verbose`.
   **Backends load lazily:** `src/ocr/factory.py` imports a backend only when it is
   selected, so `openai` is needed only for `ocr.backend: lmstudio`.
8. **`tests/test_deskew.py`, `test_vertical.py`, `test_preprocessor.py`** now build their
   stages from config.
9. **`requirements.txt` rewritten** as direct dependencies only. It was missing `openai`
   (which made a fresh install fail even with the default backend) and `tqdm`, and it listed
   `easyocr`, `torch` and `torchvision`, which root code no longer uses.
10. **README** now has a "Setting up on a fresh clone" section, an updated configuration
    section, and no TrOCR references; `ocr_dataset_builder/README.md` points at `config.yaml`.

Still rough (not touched):

- **Only `text` survives to the OCR CSV.** Confidence and bounding box are dropped in
  `ROITextExtractor.extract`.
- **`summary.csv` is overwritten each run** and lists only the PDFs processed in that run.
- **`tests/` are scratch scripts**, not a test suite: they name specific PDFs and absolute
  `G:\` / `D:\` paths, and `test_ocr.py` / `test_recognizer.py` import modules that do not
  exist. They are exempt from the "no hardcoded values" rule for that reason.
- **PARSeq worker and client are duplicated** in `src/ocr/` and `ocr_dataset_builder/src/`.
  Both were updated the same way, but any future fix has to be made twice.

---

## 7. Fresh clone: what git does not contain

`config.yaml` and all code are in git. These are not (git-ignored), so a new user must
provide them; nothing in the Python needs editing.

| Item | Needed for | Where |
|---|---|---|
| Root venv from `requirements.txt` | everything | `venv/` |
| PARSeq venv (`torch`, `strhub`) from `ocr_dataset_builder/requirements.txt` | `ocr.backend: parseq` (default), dataset builder | `ocr_dataset_builder/venv/` |
| PARSeq checkpoints | `parseq` backend | `ocr_dataset_builder/models/` (`parseq_base_mix_v2.ckpt`, `parseq_final.ckpt`) |
| Input PDFs | `main.py` | `data/input_pdfs/` |
| `MISTRAL_API_KEY` | dataset builder `mistral` backend only | `.env` |
| LM Studio server | `ocr.backend: lmstudio` only | running locally |

If the layout differs from the defaults (for example Linux, where the venv python is
`venv/bin/python`), change the matching path in `config.yaml`
(`ocr.parseq.venv_python`, `dataset_builder.parseq_venv_python`).
