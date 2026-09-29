DATASET ?= mnist
MODEL ?= small
EPOCHS ?= 3
LR ?= 0.001
SEED ?= 42
PY ?= python3
TRAIN = $(PY) train.py --dataset $(DATASET) --model $(MODEL) --epochs $(EPOCHS) --lr $(LR) --seed $(SEED)

STORE ?= $(HOME)/.local/share/digit-classifier
MLFLOW ?= mlflow==3.16.1

.PHONY: info train eval results clean track ui

info:
	@nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU"
	@$(PY) -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA:', torch.cuda.is_available())"

train:
	$(TRAIN)

out/model.pt:
	$(TRAIN)

eval: out/model.pt
	$(PY) eval.py

results: train
	$(PY) eval.py

clean:
	rm -rf out data

# Local only (run with make, not cloudmake)
track:
	uv run --no-project --with $(MLFLOW) python track.py .cloudmake/artifacts $(STORE)

ui:
	uvx --from $(MLFLOW) mlflow ui --backend-store-uri sqlite:///$(STORE)/mlflow.db
