import argparse
import json

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import OwnDataset, get_dataset
from model import accuracy, build, device

p = argparse.ArgumentParser()
p.add_argument("--out", default="out")
args = p.parse_args()

dev = device()
ckpt = torch.load(f"{args.out}/model.pt", map_location=dev)
classes = ckpt["classes"]
data, _ = get_dataset(ckpt["dataset"], train=False)
model = build(ckpt["model"], len(classes)).to(dev)
model.load_state_dict(ckpt["state"])

acc = accuracy(model, DataLoader(data, batch_size=1000), dev)
print(f"test accuracy {acc:.2%}")

with open(f"{args.out}/run.json") as f:
    run = json.load(f)
run["metrics"] = {"test_accuracy": acc}
own_test = OwnDataset(ckpt["dataset"], classes, test=True)
if len(own_test):
    run["metrics"]["own_test_accuracy"] = accuracy(model, DataLoader(own_test, batch_size=1000), dev)
    print(f"own test accuracy {run['metrics']['own_test_accuracy']:.2%} ({len(own_test)} drawings)")
with open(f"{args.out}/run.json", "w") as f:
    json.dump(run, f, indent=2)

x, y = next(iter(DataLoader(data, batch_size=16, shuffle=True)))
with torch.no_grad():
    pred = model(x.to(dev)).argmax(1).cpu()

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, img, t, pr in zip(axes.flat, x, y.tolist(), pred.tolist()):
    ax.imshow(img[0], cmap="gray")
    ax.set_title(f"{classes[t]} / {classes[pr]}", color="green" if t == pr else "red")
    ax.axis("off")
fig.suptitle(f"{ckpt['dataset']} · {ckpt['model']} — true / predicted")
fig.tight_layout()
fig.savefig(f"{args.out}/samples.png")
print(f"saved {args.out}/run.json and {args.out}/samples.png")
