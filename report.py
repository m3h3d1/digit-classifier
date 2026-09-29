import sys

import mlflow

mlflow.set_tracking_uri(f"sqlite:///{sys.argv[1]}/mlflow.db")
query = f"params.dataset = '{sys.argv[2]}'" if len(sys.argv) > 2 else ""
runs = mlflow.search_runs(experiment_names=["digit-classifier"], filter_string=query, output_format="list")
runs.sort(key=lambda r: r.data.metrics.get("val_accuracy", -1), reverse=True)


def pct(m, key):
    return f"{m[key]:.2%}" if key in m else "-"


print(f"{'run':<40} {'epochs':>6} {'val_acc':>8} {'test_acc':>8} {'own_test':>8} {'epoch_s':>7}")
for r in runs:
    m = r.data.metrics
    print(
        f"{r.info.run_name:<40} {r.data.params.get('epochs', '-'):>6} {pct(m, 'val_accuracy'):>8} "
        f"{pct(m, 'test_accuracy'):>8} {pct(m, 'own_test_accuracy'):>8} {m.get('epoch_time', 0):>7.1f}"
    )
