import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from training.own import load_own

NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))


def upright(x):
    return x.transpose(1, 2)


def add_noise(x):
    return x + 0.1 * torch.randn_like(x)


def get_dataset(name, train, aug=False):
    steps = [transforms.RandomAffine(10, translate=(0.1, 0.1))] if aug else []
    steps.append(transforms.ToTensor())
    if name == "emnist":
        steps.append(upright)
    if aug:
        steps.append(add_noise)
    steps.append(NORMALIZE)
    tf = transforms.Compose(steps)

    if name == "mnist":
        data = datasets.MNIST("data", train=train, download=True, transform=tf)
    elif name == "emnist":
        data = datasets.EMNIST("data", split="balanced", train=train, download=True, transform=tf)
    else:
        raise ValueError(f"unknown dataset: {name}")
    return data, [c.split(" ")[0] for c in data.classes]


class OwnDataset(Dataset):
    def __init__(self, dataset, classes, test):
        x, y = load_own(dataset, classes, test)
        self.x, self.y = torch.from_numpy(x), y.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def split(n, val, seed):
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    cut = int(n * (1 - val))
    return idx[:cut], idx[cut:]
