from pathlib import Path


def load_dataset(folder, extensions):
    """
    Sorted image files in `folder` whose suffix is in `extensions`
    (pass cfg.image_extensions).
    """

    folder = Path(folder)

    extensions = set(extensions)

    return sorted(

        [

            file

            for file in folder.iterdir()

            if file.suffix.lower() in extensions

        ]

    )
