import base64
import csv
import json
import os
import random
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.preprocess import features, preprocess

HERE = Path(__file__).parent
INFO = json.loads((HERE / "model/info.json").read_text())
SESSION = ort.InferenceSession(str(HERE / "model/model.onnx"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "collected"))
LABELS = DATA_DIR / "labels.csv"
LOG = Path(os.environ.get("LOG_DIR", "logs")) / "predictions.jsonl"
FIELDS = ["file", "label", "predicted", "model_version", "created", "mode"]

app = FastAPI()


class Drawing(BaseModel):
    image: str


class Feedback(Drawing):
    label: str
    predicted: str
    mode: Literal["free", "collect"]


def decode(d):
    return base64.b64decode(d.image.split(",", 1)[-1])


def counts():
    if not LABELS.exists():
        return Counter()
    with open(LABELS) as f:
        return Counter(row["label"] for row in csv.DictReader(f))


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/health")
def health():
    return {k: INFO[k] for k in ("version", "run", "dataset", "model", "test_accuracy", "classes")}


@app.post("/predict")
def predict(d: Drawing):
    png = decode(d)
    x = preprocess(png, INFO["dataset"])
    if x is None:
        return {"version": INFO["version"], "top": []}
    logits = SESSION.run(None, {"image": x})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    top = p.argsort()[::-1][:3]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "version": INFO["version"],
                            "label": INFO["classes"][top[0]], "confidence": float(p[top[0]]), **features(png, x)}) + "\n")
    return {"version": INFO["version"], "top": [{"label": INFO["classes"][i], "prob": float(p[i])} for i in top]}


@app.post("/feedback")
def feedback(d: Feedback):
    if d.label not in INFO["classes"]:
        raise HTTPException(400, f"unknown label: {d.label}")
    if not d.predicted:
        raise HTTPException(400, "predict before saving")
    png = decode(d)
    if preprocess(png, INFO["dataset"]) is None:
        raise HTTPException(400, "blank drawing")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.png"
    (DATA_DIR / name).write_bytes(png)
    new = not LABELS.exists()
    with open(LABELS, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({"file": name, "label": d.label, "predicted": d.predicted,
                    "model_version": INFO["version"], "created": datetime.now(timezone.utc).isoformat(),
                    "mode": d.mode})
    return stats()


@app.get("/stats")
def stats():
    c = counts()
    fewest = min(c[k] for k in INFO["classes"])
    return {"total": sum(c.values()), "next": random.choice([k for k in INFO["classes"] if c[k] == fewest])}
