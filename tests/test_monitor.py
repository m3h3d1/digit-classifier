import json

import fakes
import numpy as np
import onnxruntime as ort
from conftest import run

from app.preprocess import features, preprocess


def traffic(project, thick):
    session = ort.InferenceSession(str(project / "app/model/model.onnx"))
    lines = []
    for i in range(30):
        png = fakes.drawing(i, thick)
        x = preprocess(png, "emnist")
        logits = session.run(None, {"image": x})[0][0]
        p = np.exp(logits - logits.max())
        lines.append(json.dumps({"version": 1, "confidence": float(p.max() / p.sum()), **features(png, x)}))
    (project / "logs").mkdir()
    (project / "logs/predictions.jsonl").write_text("\n".join(lines) + "\n")


def monitor(project):
    return run(project, "ops.monitor", "50", "logs/predictions.jsonl")


def test_not_enough_predictions_is_not_an_error(project):
    fakes.collect(project / "collected", 30)
    r = monitor(project)
    assert r.returncode == 0 and "not enough predictions" in r.stdout


def test_live_accuracy_is_split_by_mode(project):
    root = project / "collected"
    fakes.collect(root, 4)
    with open(root / "labels.csv", "a") as f:
        f.write(f"{(root / 'x.png').name},B,A,1,2026-01-01T00:00:00+00:00,free\n")
    (root / "x.png").write_bytes(fakes.drawing(99))
    r = monitor(project)
    assert "v1 collect 100% (4)" in r.stdout and "v1 free 0% (1)" in r.stdout


def test_normal_traffic_has_no_drift(project):
    fakes.collect(project / "collected", 30)
    traffic(project, thick=False)
    r = monitor(project)
    assert r.returncode == 0 and "no drift" in r.stdout


def test_thick_strokes_are_detected(project):
    fakes.collect(project / "collected", 30)
    traffic(project, thick=True)
    r = monitor(project)
    assert r.returncode == 1 and "DRIFT detected" in r.stdout and "ink" in r.stdout.splitlines()[-1]
