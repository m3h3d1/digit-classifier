import argparse
import json
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.data import get_dataset
from training.model import build

try:
    import onnxscript  # noqa: F401  (needed by torch.onnx.export)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxscript"])

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
args = p.parse_args()

ckpt = torch.load(f"{args.out}/model.pt", map_location="cpu")
model = build(ckpt["model"], len(ckpt["classes"]))
model.load_state_dict(ckpt["state"])
model.eval()

data, _ = get_dataset(ckpt["dataset"], train=False)
x, y = next(iter(DataLoader(data, batch_size=100, shuffle=True, generator=torch.Generator().manual_seed(0))))
with torch.no_grad():
    logits = model(x)

torch.onnx.export(
    model, (x[:1],), f"{args.out}/model.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    external_data=False,
)
np.savez(f"{args.out}/sample.npz", x=x.numpy(), y=y.numpy(), logits=logits.numpy())
with open(f"{args.out}/classes.json", "w") as f:
    json.dump({"dataset": ckpt["dataset"], "model": ckpt["model"], "classes": ckpt["classes"]}, f)
print(f"saved {args.out}/model.onnx, sample.npz and classes.json")
