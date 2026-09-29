import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.preprocess import features, preprocess
from own import OWN_DIR, rows

SIGNALS = ["confidence", "ink", "size"]
MIN_WINDOW = 20

window, log = int(sys.argv[1]), Path(sys.argv[2])
info = json.loads(Path("app/model/info.json").read_text())
session = ort.InferenceSession("app/model/model.onnx")


def describe(png):
    x = preprocess(png, info["dataset"])
    logits = session.run(None, {"image": x})[0][0]
    p = np.exp(logits - logits.max())
    return {"confidence": float(p.max() / p.sum()), **features(png, x)}


def psi(ref, cur, bins=5):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1]
    r = np.bincount(np.searchsorted(edges, ref), minlength=bins) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur), minlength=bins) / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


def status(score):
    return "OK" if score < 0.1 else "WARN" if score < 0.25 else "DRIFT"


labeled = defaultdict(lambda: [0, 0])
for r in rows():
    labeled[r["model_version"]][0] += r["label"] == r["predicted"]
    labeled[r["model_version"]][1] += 1
print("live accuracy: " + " · ".join(f"v{v} {ok / n:.0%} ({n} labeled)" for v, (ok, n) in sorted(labeled.items())))

entries = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
current = [e for e in entries if e["version"] == info["version"]][-window:]
if len(current) < MIN_WINDOW:
    print(f"not enough predictions for v{info['version']} yet ({len(current)}/{MIN_WINDOW})")
    sys.exit(0)

reference = [describe((OWN_DIR / r["file"]).read_bytes()) for r in rows()]
print(f"\nwindow: last {len(current)} predictions (v{info['version']}) vs {len(reference)} reference drawings")
print(f"{'signal':<12} {'reference':>9} {'current':>9} {'PSI':>6}  status")
drifted = []
for s in SIGNALS:
    ref, cur = np.array([d[s] for d in reference]), np.array([e[s] for e in current])
    score = psi(ref, cur)
    print(f"{s:<12} {ref.mean():>9.2f} {cur.mean():>9.2f} {score:>6.2f}  {status(score)}")
    if status(score) == "DRIFT":
        drifted.append(s)

if drifted:
    print(f"\nDRIFT detected: {', '.join(drifted)}")
    sys.exit(1)
print("\nno drift")
