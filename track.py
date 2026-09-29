import json
import sys
from pathlib import Path

import mlflow

src, store = Path(sys.argv[1]), Path(sys.argv[2]).resolve()

store.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{store}/mlflow.db")
if mlflow.get_experiment_by_name("digit-classifier") is None:
    mlflow.create_experiment("digit-classifier", artifact_location=(store / "artifacts").as_uri())
mlflow.set_experiment("digit-classifier")


def run_name(p):
    name = f"{p['dataset']}-{p['model']}-lr{p['lr']}-b{p['batch']}"
    if p.get("aug"):
        name += "-aug"
    if p.get("sched", "none") != "none":
        name += f"-{p['sched']}"
    return name


for path in sorted(src.rglob("run.json")):
    run = json.loads(path.read_text())
    if mlflow.search_runs(filter_string=f"tags.run_uid = '{run['run_uid']}'", output_format="list"):
        print(f"skip {path.parent.name}: already tracked")
        continue

    p = run["params"]
    with mlflow.start_run(run_name=run_name(p)) as r:
        mlflow.log_params(p)
        mlflow.set_tags({"run_uid": run["run_uid"], **{k: str(v) for k, v in run["env"].items()}})
        for step, e in enumerate(run["epochs"], 1):
            mlflow.log_metrics({k: v for k, v in e.items() if k != "time"} | {"epoch_time": e["time"]}, step=step)
        mlflow.log_metrics(run.get("metrics", {}))
        for f in path.parent.iterdir():
            if f.is_file():
                mlflow.log_artifact(str(f))
    print(f"tracked {r.info.run_name}")
