import torch
from torch import nn
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def build(name, num_classes):
    if name == "small":
        return Net(num_classes)
    if name == "resnet":
        m = resnet18(num_classes=num_classes)
        m.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return m
    raise ValueError(f"unknown model: {name}")


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
