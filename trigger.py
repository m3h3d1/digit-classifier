import subprocess
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from own import data_version

NAME = "digit-classifier"
NEW_MIN = 20

mlflow.set_tracking_uri(f"sqlite:///{sys.argv[1]}/mlflow.db")
client = MlflowClient()
reasons = []

try:
    prod = client.get_model_version_by_alias(NAME, "production")
    trained_on = int(client.get_run(prod.run_id).data.params.get("own_files", 0))
except MlflowException:
    reasons.append("no production model")
    trained_on = 0

new = data_version()["own_files"] - trained_on
if new >= NEW_MIN:
    reasons.append(f"{new} new drawings")

monitor = subprocess.run(
    [sys.executable, "monitor.py", "50", "logs/predictions.jsonl"], capture_output=True, text=True, check=False
)
if monitor.returncode == 1:
    reasons.append(monitor.stdout.strip().splitlines()[-1])

print("; ".join(reasons))
