import json
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

NAME = "digit-classifier"
cmd, store, path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
mlflow.set_tracking_uri(f"sqlite:///{store}/mlflow.db")
client = MlflowClient()


def production():
    try:
        return client.get_model_version_by_alias(NAME, "production")
    except MlflowException:
        return None


def fetch(run_id, name, dest):
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=name, dst_path=str(dest)))


def promote(src):
    run_uid = json.loads((src / "run.json").read_text())["run_uid"]
    found = mlflow.search_runs(experiment_names=[NAME], filter_string=f"tags.run_uid = '{run_uid}'", output_format="list")
    if not found:
        sys.exit("not tracked yet: run `make track` first")
    run = found[0]
    name, acc = run.info.run_name, run.data.metrics.get("test_accuracy")
    if acc is None:
        sys.exit(f"blocked: {name} has no test accuracy")

    prod = production()
    if prod and prod.run_id == run.info.run_id:
        sys.exit(f"{name} is already production (v{prod.version})")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            onnx, sample = fetch(run.info.run_id, "model.onnx", tmp), np.load(fetch(run.info.run_id, "sample.npz", tmp))
        except MlflowException:
            sys.exit(f"blocked: {name} has no ONNX export")
        out = ort.InferenceSession(str(onnx)).run(None, {"image": sample["x"]})[0]
    diff = float(np.abs(out - sample["logits"]).max())
    if diff > 1e-3:
        sys.exit(f"blocked: ONNX differs from PyTorch (max diff {diff:.1e})")
    print(f"check 1 ok: ONNX matches PyTorch (max diff {diff:.1e})")

    if prod:
        prun = client.get_run(prod.run_id)
        pacc, pds = prun.data.metrics["test_accuracy"], prun.data.params["dataset"]
        if pds != run.data.params["dataset"]:
            sys.exit(f"blocked: dataset {run.data.params['dataset']} differs from production ({pds})")
        if acc < pacc:
            sys.exit(f"blocked: {acc:.2%} < production v{prod.version} {pacc:.2%}")
        print(f"check 2 ok: {acc:.2%} >= production v{prod.version} {pacc:.2%}")
    else:
        print("check 2 ok: no production model yet")

    try:
        client.get_registered_model(NAME)
    except MlflowException:
        client.create_registered_model(NAME)
    mv = client.create_model_version(NAME, source=run.info.artifact_uri, run_id=run.info.run_id)
    client.set_registered_model_alias(NAME, "production", mv.version)
    print(f"promoted {name} ({acc:.2%}) -> v{mv.version} @production")


def bundle(dest):
    prod = production()
    if not prod:
        sys.exit("no production model yet: run `make promote` first")
    run = client.get_run(prod.run_id)
    dest.mkdir(parents=True, exist_ok=True)
    fetch(run.info.run_id, "model.onnx", dest)
    with tempfile.TemporaryDirectory() as tmp:
        info = json.loads(fetch(run.info.run_id, "classes.json", tmp).read_text())
    info |= {"version": int(prod.version), "run": run.info.run_name, "test_accuracy": run.data.metrics["test_accuracy"]}
    (dest / "info.json").write_text(json.dumps(info, indent=2))
    print(f"bundled v{prod.version} ({run.info.run_name}) into {dest}")


{"promote": promote, "bundle": bundle}[cmd](path)
