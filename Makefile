DATASET ?= mnist
MODEL ?= small
EPOCHS ?= 3
LR ?= 0.001
BATCH ?= 128
AUG ?= 0
SCHED ?= none
SEED ?= 42
OWN ?= 0
COMMIT ?= none
PY ?= python3
TRAIN = $(PY) -m training.train --dataset $(DATASET) --model $(MODEL) --epochs $(EPOCHS) --seed $(SEED) --aug $(AUG) --sched $(SCHED) --own $(OWN) --commit $(COMMIT)

LRS ?= 0.003 0.001 0.0003
BATCHES ?= 64 256
SWEEP = $(foreach l,$(LRS),$(foreach b,$(BATCHES),out/sweep/$(DATASET)-$(MODEL)-lr$(l)-b$(b)/run.json))

STORE ?= $(HOME)/.local/share/digit-classifier
MLFLOW ?= mlflow==3.16.1

# One job at a time: runs share one GPU and one dataset download.
.NOTPARALLEL:
.PHONY: info train eval results sweep clean track report ui promote bundle serve docker data monitor simulate lint test pipeline

info:
	@nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU"
	@$(PY) -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA:', torch.cuda.is_available())"

train:
	$(TRAIN) --lr $(LR) --batch $(BATCH)

out/model.pt:
	$(TRAIN) --lr $(LR) --batch $(BATCH)

eval: out/model.pt
	$(PY) -m training.eval

results: train
	$(PY) -m training.eval
	$(PY) -m training.export

# Finished runs are skipped when the sweep is rerun.
sweep: $(SWEEP)

out/sweep/%/run.json:
	$(TRAIN) --lr $(patsubst lr%,%,$(filter lr%,$(subst -, ,$*))) --batch $(patsubst b%,%,$(filter b%,$(subst -, ,$*))) --out out/sweep/$*

clean:
	rm -rf out data

# Local only (run with make, not cloudmake)
track:
	uv run --no-project --with $(MLFLOW) python -m ops.track .cloudmake/artifacts $(STORE)

# Filters only when DATASET is given on the command line.
report:
	uv run --no-project --with $(MLFLOW) python -m ops.report $(STORE) $(if $(filter command line,$(origin DATASET)),$(DATASET))

ui:
	uvx --from $(MLFLOW) mlflow ui --backend-store-uri sqlite:///$(STORE)/mlflow.db

REGISTRY = uv run --no-project --with $(MLFLOW) --with-requirements app/requirements.txt python -m ops.registry

promote:
	$(REGISTRY) promote $(STORE) .cloudmake/artifacts

bundle:
	$(REGISTRY) bundle $(STORE) app/model

serve:
	uv run --no-project --with-requirements app/requirements.txt uvicorn app.main:app --port 8000

data:
	uv run --no-project --with-requirements app/requirements.txt python -m ops.validate app/model/info.json
	dvc add collected
	dvc push

WINDOW ?= 50
STYLE ?= normal
N ?= 50

monitor:
	uv run --no-project --with-requirements app/requirements.txt python -m ops.monitor $(WINDOW) logs/predictions.jsonl

# Sends changed copies of collected/ drawings to the running app (make serve).
simulate:
	uv run --no-project --with-requirements app/requirements.txt python -m ops.simulate $(STYLE) $(N) http://127.0.0.1:8000

lint:
	uvx ruff@0.16.9 check .

test:
	uv run --no-project --with-requirements app/requirements.txt --with-requirements tests/requirements.txt python -m pytest -q tests

# Retrain, gate and deploy when needed. FORCE=1 skips the retrain check.
pipeline:
	ops/pipeline.sh

docker:
	docker build -t digit-app:v$$(python3 -c "import json; print(json.load(open('app/model/info.json'))['version'])") .
