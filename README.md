# digit-classifier

Draw a digit or letter in the browser; a model predicts it. A small end-to-end MLOps project:
GPU training, experiment tracking, a gated model registry, serving, own-data collection, drift monitoring and a pipeline.

- **Model:** ResNet-18 or a small CNN on EMNIST balanced (47 classes)
- **Training:** on a GPU (Google Colab); **serving:** FastAPI + ONNX Runtime in Docker

## Layout

| Folder | Runs on | Purpose |
|---|---|---|
| `training/` | GPU | train, evaluate, export to ONNX |
| `ops/` | local | tracking, promotion gate, data checks, monitoring, pipeline, deploy |
| `app/` | local / Docker | drawing page and prediction API |
| `tests/` | local / CI | tests with fake models |

## Usage

```sh
make results DATASET=emnist MODEL=resnet SCHED=cosine OWN=1   # on the GPU
make track && make report DATASET=emnist                      # log and compare runs (MLflow)
make promote && make bundle                                   # gate, register, package the production model
make serve                                                    # http://127.0.0.1:8000
make data                                                     # validate and version saved drawings (DVC)
make monitor                                                  # live accuracy and drift
make pipeline                                                 # retrain → gate → build → deploy with rollback
make lint test                                                # checks, also run by CI
```

A model is promoted only if its ONNX output matches PyTorch, its EMNIST accuracy drops at most 0.5%,
and it is at least as accurate on your own held-out drawings.
