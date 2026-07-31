from pathlib import Path


def load_dataset(folder):

    folder = Path(folder)

    extensions = {".png", ".jpg", ".jpeg", ".tif"}

    return sorted(

        [

            file

            for file in folder.iterdir()

            if file.suffix.lower() in extensions

        ]

    )