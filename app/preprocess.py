import io

import numpy as np
from PIL import Image

# EMNIST characters fill ~24px of 28 (2px border); MNIST digits fit a 20px box.
SIZE = {"emnist": 24, "mnist": 20}


def preprocess(png, dataset):
    img = Image.open(io.BytesIO(png)).convert("L")
    box = img.getbbox()
    if box is None:
        return None
    img = img.crop(box)
    scale = SIZE[dataset] / max(img.size)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    x = np.asarray(canvas, dtype=np.float32) / 255
    return ((x - 0.1307) / 0.3081)[None, None]


def features(png, x):
    img = Image.open(io.BytesIO(png)).convert("L")
    left, top, right, bottom = img.getbbox()
    return {
        "ink": float(((x * 0.3081 + 0.1307) > 0.5).mean()),
        "size": (right - left) * (bottom - top) / (img.width * img.height),
    }
