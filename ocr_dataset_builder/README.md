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