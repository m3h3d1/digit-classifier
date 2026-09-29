import json
import sys
from pathlib import Path

import mlflow

src, store = Path(sys.argv[1]), Path(sys.argv[2])
run = json.loads((src / "run.json").read_text())

store.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{store}/mlflow.db")
if mlflow.get_experiment_by_name("digit-classifier") is None:
    mlflow.create_experiment("digit-classifier", artifact_location=(store / "artifacts").as_uri())
mlflow.set_experiment("digit-classifier")

if mlflow.search_runs(filter_string=f"tags.run_uid = '{run['run_uid']}'", output_format="list"):
    print(f"run {run['run_uid']} already tracked")
    sys.exit(0)

p = run["params"]
with mlflow.start_run(run_name=f"{p['dataset']}-{p['model']}") as r:
    mlflow.log_params(p)
    mlflow.set_tags({"run_uid": run["run_uid"], **{k: str(v) for k, v in run["env"].items()}})
    for step, e in enumerate(run["epochs"], 1):
        mlflow.log_metrics({"loss": e["loss"], "epoch_time": e["time"]}, step=step)
    mlflow.log_metrics(run["metrics"])
    mlflow.log_artifacts(str(src))
print(f"tracked {r.info.run_name}: accuracy {run['metrics']['test_accuracy']:.2%}")
