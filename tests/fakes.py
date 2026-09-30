"""Tiny stand-ins for trained models and drawings, so tests need no PyTorch or GPU."""
import csv
import io
import json
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image, ImageDraw, ImageFilter

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abdefghnqrt")


def model(path, favorite):
    """Linear model whose bias makes it always predict `favorite`."""
    rng = np.random.default_rng(0)
    w = (rng.normal(size=(784, len(CLASSES))) * 0.001).astype(np.float32)
    b = np.zeros(len(CLASSES), dtype=np.float32)
    b[CLASSES.index(favorite)] = 10
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["image"], ["f"]), helper.make_node("MatMul", ["f", "w"], ["m"]),
         helper.make_node("Add", ["m", "b"], ["logits"])],
        "fake", [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 1, 28, 28])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(CLASSES)])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), path)
    return w, b


def run_dir(path, test_accuracy, favorite="A", broken=False):
    """A collected training run as Colab would return it."""
    path.mkdir(parents=True, exist_ok=True)
    w, b = model(path / "model.onnx", favorite)
    x = np.random.default_rng(1).normal(size=(100, 1, 28, 28)).astype(np.float32)
    logits = x.reshape(100, -1) @ w + b + (1.0 if broken else 0.0)
    np.savez(path / "sample.npz", x=x, y=np.zeros(100), logits=logits)
    (path / "classes.json").write_text(json.dumps({"dataset": "emnist", "model": "resnet", "classes": CLASSES}))
    (path / "run.json").write_text(json.dumps({
        "run_uid": uuid.uuid4().hex,
        "params": {"dataset": "emnist", "model": "resnet", "epochs": 3, "lr": 0.001, "batch": 128, "own": 1},
        "env": {"gpu": "fake"},
        "epochs": [{"loss": 0.3, "val_accuracy": test_accuracy, "time": 1.0}],
        "metrics": {"test_accuracy": test_accuracy},
    }))


def bundle(path, favorite="A"):
    """An app/model folder like `make bundle` creates."""
    path.mkdir(parents=True, exist_ok=True)
    model(path / "model.onnx", favorite)
    (path / "info.json").write_text(json.dumps(
        {"dataset": "emnist", "model": "resnet", "classes": CLASSES, "version": 1, "run": "fake", "test_accuracy": 0.9}))


def blank():
    buf = io.BytesIO()
    Image.new("L", (280, 280)).save(buf, "PNG")
    return buf.getvalue()


def drawing(seed, thick=False):
    """A random stroke drawing on a black 280x280 canvas, as PNG bytes."""
    r = random.Random(seed)
    img = Image.new("L", (280, 280))
    points = [(r.randrange(60, 220), r.randrange(60, 220)) for _ in range(4)]
    ImageDraw.Draw(img).line(points, fill=255, width=20, joint="curve")
    if thick:
        img = img.filter(ImageFilter.MaxFilter(21))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def collect(root, n, label="A", thick=False, start=0):
    """Save n drawings plus labels.csv, like the app's /feedback does."""
    root.mkdir(parents=True, exist_ok=True)
    labels = root / "labels.csv"
    new = not labels.exists()
    with open(labels, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "label", "predicted", "model_version", "created", "mode"])
        for i in range(start, start + n):
            name = f"{uuid.UUID(int=i).hex}.png"
            (root / name).write_bytes(drawing(i, thick))
            w.writerow([name, label, label, 1, "2026-01-01T00:00:00+00:00", "collect"])


if __name__ == "__main__":
    bundle(Path(sys.argv[1]))
