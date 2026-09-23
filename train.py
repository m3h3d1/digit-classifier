import argparse
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from model import TRANSFORM, Net, device

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=0.001)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--out", default="out")
args = p.parse_args()

dev = device()
print(f"device: {dev}")
data = datasets.MNIST("data", train=True, download=True, transform=TRANSFORM)
loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=2)

model = Net().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

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
    print(f"epoch {epoch}/{args.epochs}  loss {total / len(data):.4f}  time {time.time() - start:.1f}s")

os.makedirs(args.out, exist_ok=True)
torch.save(model.state_dict(), f"{args.out}/model.pt")
print(f"saved {args.out}/model.pt")
