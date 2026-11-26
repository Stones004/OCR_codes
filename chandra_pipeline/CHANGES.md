# chandra_pipeline -- Chandra experiment (2026-10-09)

Self-contained. **Delete this folder to remove everything.** The rest of the repo has no
Chandra code (it was moved out of `src/`, `main.py`, `config.yaml`, `src/config.py`).

Run: `venv\Scripts\python.exe chandra_pipeline\main.py`
(needs LM Studio server up with `chandra-ocr-2` loaded)

## What it does
Same front half as the repo's `main.py`, using the repo's `config.yaml` and `data/input_pdfs`:
PDF -> deskew -> vertical extraction -> separator -> margin crop. Then:
1. Cropped margin -> Chandra -> question-number blocks, boxed on the crop.
2. Right half (`corrected[:, crop_width:]`) -> Chandra -> answer blocks.
3. Each answer is filed under the question whose row it sits in (`answer_y_tolerance`).

Outputs go to `chandra_pipeline/outputs/<pdf>/` (deskewed/, vertical/, detected/, cropped/,
failed/, `chandra/` with `page_NNN_margin_boxes.png`, `page_NNN_qa.csv`, `questions.csv`,
`page_NNN_{margin,answers}_raw.txt`), plus `report.csv` / `summary.csv`.

## Files
- `main.py` -- the pipeline (imports `src.*` from the repo, nothing in the repo imports this).
- `config.yaml` -- Chandra-specific settings (model, host, prompt, retries, box style, ...).
- `chandra_config.py` -- typed loader for that file (reuses `src.config._build`).
- `chandra_backend.py` -- LM Studio call + HTML `data-bbox` parser.
- `margin_qa_extractor.py` -- split / pair / draw.

## Findings from the first real run (Doc0729 page 1)
- Right half: works (10 blocks with sensible boxes, ~23 s).
- Margin strip (125x3858): Chandra reads the text ("a. 1.", "2", ...) but the boxes it gives
  do not line up with the handwriting; padding the strip to a normal aspect made replies empty
  or flaky.
- Full page and some strip calls come back **empty** (`finish_reason=stop`, no text),
  intermittently. `chandra.empty_retries` re-asks; check LM Studio's context length / GPU
  memory settings if it persists.
- `min_aspect` (pad the strip) is implemented but off (0): unproven.
- `bbox_scale: 1024` is an assumption; margin y-coordinates did not match 1024 either, so the
  margin boxes are the open problem. Idea: tile the strip into near-square chunks.
