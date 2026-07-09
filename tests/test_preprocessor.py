from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from src.preprocessor import Preprocessor


processor = Preprocessor(scale=2)

image = cv2.imread(
    "outputs/P5/cropped/page_003.png"
)

results = processor.process(image)

output = Path("outputs/preprocessing")

output.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------
# Save every preprocessing method
# ------------------------------------------

for name, img in results.items():

    filename = (
        name.lower()
        .replace(" + ", "_plus_")
        .replace(" ", "_")
    )

    cv2.imwrite(
        str(output / f"{filename}.png"),
        img
    )

# ------------------------------------------
# Build comparison figure
# ------------------------------------------

cols = 2
rows = 2

fig, axes = plt.subplots(

    rows,

    cols,

    figsize=(15, 5 * rows)

)

axes = axes.flatten()

for ax, (title, img) in zip(
    axes,
    results.items()
):

    ax.imshow(
        img,
        cmap="gray"
    )

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold"
    )

    ax.axis("off")

for i in range(len(results), len(axes)):
    axes[i].axis("off")

plt.tight_layout()

comparison = output / "comparison.png"

plt.savefig(
    comparison,
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("Saved comparison to")
print(comparison)
print("=" * 60)