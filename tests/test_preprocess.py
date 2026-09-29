import io

import numpy as np
from PIL import Image, ImageDraw

from app.preprocess import features, preprocess


def png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def corner_box():
    img = Image.new("L", (280, 280))
    ImageDraw.Draw(img).rectangle([10, 10, 60, 110], fill=255)
    return png(img)


def test_drawing_in_a_corner_is_centered_and_scaled():
    x = preprocess(corner_box(), "emnist")
    assert x.shape == (1, 1, 28, 28)
    rows, cols = np.nonzero((x[0, 0] * 0.3081 + 0.1307) > 0.5)
    assert 23 <= rows.max() - rows.min() + 1 <= 25
    assert abs((rows.min() + rows.max()) / 2 - 13.5) <= 1
    assert abs((cols.min() + cols.max()) / 2 - 13.5) <= 1


def test_blank_drawing_gives_none():
    assert preprocess(png(Image.new("L", (280, 280))), "emnist") is None


def test_features():
    data = corner_box()
    f = features(data, preprocess(data, "emnist"))
    assert f["size"] == 51 * 101 / (280 * 280)
    assert 0 < f["ink"] < 1
