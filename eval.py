import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model import TRANSFORM, Net, device

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
args = p.parse_args()

dev = device()
data = datasets.MNIST("data", train=False, download=True, transform=TRANSFORM)
model = Net().to(dev)
model.load_state_dict(torch.load(f"{args.out}/model.pt", map_location=dev))
model.eval()

correct = 0
with torch.no_grad():
    for x, y in DataLoader(data, batch_size=1000):
        correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
acc = correct / len(data)
print(f"accuracy {acc:.2%} ({correct}/{len(data)})")
with open(f"{args.out}/metrics.txt", "w") as f:
    f.write(f"device: {dev}\naccuracy: {acc:.4f}\ncorrect: {correct}/{len(data)}\n")

x, y = next(iter(DataLoader(data, batch_size=16, shuffle=True)))
with torch.no_grad():
    pred = model(x.to(dev)).argmax(1).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, img, t, pr in zip(axes.flat, x, y.tolist(), pred.tolist()):
    ax.imshow(img[0], cmap="gray")
    ax.set_title(f"{t} / {pr}", color="green" if t == pr else "red")
    ax.axis("off")
fig.suptitle("true / predicted")
fig.tight_layout()
fig.savefig(f"{args.out}/samples.png")
print(f"saved {args.out}/metrics.txt and {args.out}/samples.png")
