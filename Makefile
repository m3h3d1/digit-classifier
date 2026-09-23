EPOCHS ?= 3
LR ?= 0.001
PY ?= python3

.PHONY: info train eval results clean

info:
	@nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU"
	@$(PY) -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA:', torch.cuda.is_available())"

train:
	$(PY) train.py --epochs $(EPOCHS) --lr $(LR)

out/model.pt:
	$(PY) train.py --epochs $(EPOCHS) --lr $(LR)

eval: out/model.pt
	$(PY) eval.py

results: train
	$(PY) eval.py

clean:
	rm -rf out data
