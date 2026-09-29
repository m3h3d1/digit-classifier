import argparse
import json
import os
import platform
import time
import uuid

import torch
import torchvision
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data import OwnDataset, get_dataset, split
from model import accuracy, build, device
from own import data_version

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="mnist", choices=["mnist", "emnist"])
p.add_argument("--model", default="small", choices=["small", "resnet"])
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=0.001)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--aug", type=int, default=0, choices=[0, 1])
p.add_argument("--sched", default="none", choices=["none", "cosine"])
p.add_argument("--val", type=float, default=0.1)
p.add_argument("--own", type=int, default=0, choices=[0, 1])
p.add_argument("--own_repeat", type=int, default=20)
p.add_argument("--commit", default="none")
p.add_argument("--out", default="out")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

dev = device()
print(f"device: {dev}  dataset: {args.dataset}  model: {args.model}")
train_full, classes = get_dataset(args.dataset, train=True, aug=args.aug)
plain, _ = get_dataset(args.dataset, train=True)
train_idx, val_idx = split(len(plain), args.val, args.seed)
data = Subset(train_full, train_idx)
version = data_version()
if args.own:
    own = OwnDataset(args.dataset, classes, test=False)
    data = ConcatDataset([data] + [own] * args.own_repeat)
    print(f"own drawings: {len(own)} for training (x{args.own_repeat}), data {version['own_md5'][:8]}")
    if version["own_stale"]:
        print("warning: collected/ changed since the last `make data`; the recorded data version is out of date")
loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=2)
val_loader = DataLoader(Subset(plain, val_idx), batch_size=1000, num_workers=2)

model = build(args.model, len(classes)).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs) if args.sched == "cosine" else None
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
    elapsed = time.time() - start
    if sched:
        sched.step()
    epochs.append({"loss": total / len(data), "val_accuracy": accuracy(model, val_loader, dev), "time": elapsed})
    e = epochs[-1]
    print(f"epoch {epoch}/{args.epochs}  loss {e['loss']:.4f}  val {e['val_accuracy']:.2%}  time {e['time']:.1f}s")

os.makedirs(args.out, exist_ok=True)
torch.save(
    {"state": model.state_dict(), "dataset": args.dataset, "model": args.model, "classes": classes},
    f"{args.out}/model.pt",
)
run = {
    "run_uid": uuid.uuid4().hex,
    "params": {k: v for k, v in vars(args).items() if k != "out"} | (version if args.own else {}),
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
