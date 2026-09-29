DATASET ?= mnist
MODEL ?= small
EPOCHS ?= 3
LR ?= 0.001
BATCH ?= 128
AUG ?= 0
SCHED ?= none
SEED ?= 42
OWN ?= 0
PY ?= python3
TRAIN = $(PY) train.py --dataset $(DATASET) --model $(MODEL) --epochs $(EPOCHS) --seed $(SEED) --aug $(AUG) --sched $(SCHED) --own $(OWN)

LRS ?= 0.003 0.001 0.0003
BATCHES ?= 64 256
SWEEP = $(foreach l,$(LRS),$(foreach b,$(BATCHES),out/sweep/$(DATASET)-$(MODEL)-lr$(l)-b$(b)/run.json))

STORE ?= $(HOME)/.local/share/digit-classifier
MLFLOW ?= mlflow==3.16.1

# One job at a time: runs share one GPU and one dataset download.
.NOTPARALLEL:
.PHONY: info train eval results sweep clean track report ui promote bundle serve docker data

info:
	@nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU"
	@$(PY) -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA:', torch.cuda.is_available())"

train:
	$(TRAIN) --lr $(LR) --batch $(BATCH)

out/model.pt:
	$(TRAIN) --lr $(LR) --batch $(BATCH)

eval: out/model.pt
	$(PY) eval.py

results: train
	$(PY) eval.py
	$(PY) export.py

# Finished runs are skipped when the sweep is rerun.
sweep: $(SWEEP)

out/sweep/%/run.json:
	$(TRAIN) --lr $(patsubst lr%,%,$(filter lr%,$(subst -, ,$*))) --batch $(patsubst b%,%,$(filter b%,$(subst -, ,$*))) --out out/sweep/$*

clean:
	rm -rf out data

# Local only (run with make, not cloudmake)
track:
	uv run --no-project --with $(MLFLOW) python track.py .cloudmake/artifacts $(STORE)

# Filters only when DATASET is given on the command line.
report:
	uv run --no-project --with $(MLFLOW) python report.py $(STORE) $(if $(filter command line,$(origin DATASET)),$(DATASET))

ui:
	uvx --from $(MLFLOW) mlflow ui --backend-store-uri sqlite:///$(STORE)/mlflow.db

REGISTRY = uv run --no-project --with $(MLFLOW) --with-requirements app/requirements.txt python registry.py

promote:
	$(REGISTRY) promote $(STORE) .cloudmake/artifacts

bundle:
	$(REGISTRY) bundle $(STORE) app/model

serve:
	uv run --no-project --with-requirements app/requirements.txt uvicorn app.main:app --port 8000

data:
	uv run --no-project --with-requirements app/requirements.txt python validate.py app/model/info.json
	dvc add collected
	dvc push

docker:
	docker build -t digit-app:v$$(python3 -c "import json; print(json.load(open('app/model/info.json'))['version'])") .
