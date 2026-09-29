import base64
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

HERE = Path(__file__).parent
INFO = json.loads((HERE / "model/info.json").read_text())
SESSION = ort.InferenceSession(str(HERE / "model/model.onnx"))
# EMNIST characters fill ~24px of 28 (2px border); MNIST digits fit a 20px box.
SIZE = 24 if INFO["dataset"] == "emnist" else 20

app = FastAPI()


class Drawing(BaseModel):
    image: str


def preprocess(png):
    img = Image.open(io.BytesIO(png)).convert("L")
    box = img.getbbox()
    if box is None:
        return None
    img = img.crop(box)
    scale = SIZE / max(img.size)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    x = np.asarray(canvas, dtype=np.float32) / 255
    return ((x - 0.1307) / 0.3081)[None, None]


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/health")
def health():
    return {k: INFO[k] for k in ("version", "run", "dataset", "model", "test_accuracy")}


@app.post("/predict")
def predict(d: Drawing):
    x = preprocess(base64.b64decode(d.image.split(",", 1)[-1]))
    if x is None:
        return {"version": INFO["version"], "top": []}
    logits = SESSION.run(None, {"image": x})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = p.argsort()[::-1][:3]
    return {"version": INFO["version"], "top": [{"label": INFO["classes"][i], "prob": float(p[i])} for i in top]}
