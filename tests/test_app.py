import base64
import csv
import importlib
import sys

import fakes
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(project, monkeypatch):
    monkeypatch.chdir(project)
    monkeypatch.syspath_prepend(str(project))
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name)
    return TestClient(importlib.import_module("app.main").app)


def url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


def test_health(client):
    h = client.get("/health").json()
    assert h["version"] == 1 and len(h["classes"]) == 47


def test_predict_returns_top3_and_logs(client, project):
    r = client.post("/predict", json={"image": url(fakes.drawing(1))}).json()
    assert r["top"][0]["label"] == "A" and len(r["top"]) == 3
    assert len((project / "logs/predictions.jsonl").read_text().splitlines()) == 1


def test_predict_blank_returns_nothing(client):
    assert client.post("/predict", json={"image": url(fakes.blank())}).json()["top"] == []


def test_feedback_saves_drawing_and_row(client, project):
    r = client.post("/feedback", json={"image": url(fakes.drawing(2)), "label": "B", "predicted": "A"})
    assert r.status_code == 200 and r.json()["total"] == 1
    rows = list(csv.DictReader((project / "collected/labels.csv").read_text().splitlines()))
    assert rows[0]["label"] == "B" and (project / "collected" / rows[0]["file"]).exists()


@pytest.mark.parametrize("image, label, predicted, error", [
    (fakes.blank(), "A", "A", "blank drawing"),
    (fakes.drawing(3), "?", "A", "unknown label"),
    (fakes.drawing(3), "A", "", "predict before saving"),
])
def test_feedback_rejects_bad_input(client, image, label, predicted, error):
    r = client.post("/feedback", json={"image": url(image), "label": label, "predicted": predicted})
    assert r.status_code == 400 and error in r.json()["detail"]
