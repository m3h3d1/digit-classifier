import argparse
import json
import os
import platform
import time
import uuid

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from data import get_dataset
from model import build, device

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="mnist", choices=["mnist", "emnist"])
p.add_argument("--model", default="small", choices=["small", "resnet"])
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=0.001)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="out")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

dev = device()
print(f"device: {dev}  dataset: {args.dataset}  model: {args.model}")
data, classes = get_dataset(args.dataset, train=True)
loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=2)

model = build(args.model, len(classes)).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

epochs = []
for epoch in range(1, args.epochs + 1):
    model.train()
    start, total = time.time(), 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
    epochs.append({"loss": total / len(data), "time": time.time() - start})
    print(f"epoch {epoch}/{args.epochs}  loss {epochs[-1]['loss']:.4f}  time {epochs[-1]['time']:.1f}s")

os.makedirs(args.out, exist_ok=True)
torch.save(
    {"state": model.state_dict(), "dataset": args.dataset, "model": args.model, "classes": classes},
    f"{args.out}/model.pt",
)
run = {
    "run_uid": uuid.uuid4().hex,
    "params": {k: v for k, v in vars(args).items() if k != "out"},
    "env": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
    },
    "epochs": epochs,
}
with open(f"{args.out}/run.json", "w") as f:
    json.dump(run, f, indent=2)
print(f"saved {args.out}/model.pt and {args.out}/run.json")
