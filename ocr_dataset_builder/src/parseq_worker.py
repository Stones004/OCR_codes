"""
Persistent PARSeq OCR worker.

Must be run with the venv_parseq interpreter (needs strhub importable).
Not meant to be run directly -- spawned by parseq_client.ParseqOCR as a
subprocess so the checkpoint is loaded once and reused across images.

Protocol: one image path per line on stdin, one JSON object per line on
stdout -- {"prediction": ..., "confidence": ...} or {"error": ...}.
Prints "READY" once the model has finished loading.
"""
import json
import sys

import torch
from PIL import Image
import torchvision.transforms as T

from strhub.models.utils import load_from_checkpoint

checkpoint_path = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_from_checkpoint(checkpoint_path).eval().to(device)

img_transform = T.Compose([
    T.Resize(model.hparams.img_size, T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])

print("READY", flush=True)

for line in sys.stdin:

    image_path = line.strip()

    if not image_path:
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        x = img_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            p = model(x).softmax(-1)

        pred_list, prob_list = model.tokenizer.decode(p)

        prediction = pred_list[0].strip()
        probs = prob_list[0]
        confidence = probs.mean().item() if len(probs) else 0.0

        print(json.dumps({"prediction": prediction, "confidence": confidence}), flush=True)

    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
