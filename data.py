from torchvision import datasets, transforms

NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))


def upright(x):
    return x.transpose(1, 2)


def get_dataset(name, train):
    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(), NORMALIZE])
        data = datasets.MNIST("data", train=train, download=True, transform=tf)
    elif name == "emnist":
        tf = transforms.Compose([transforms.ToTensor(), upright, NORMALIZE])
        data = datasets.EMNIST("data", split="balanced", train=train, download=True, transform=tf)
    else:
        raise ValueError(f"unknown dataset: {name}")
    return data, [c.split(" ")[0] for c in data.classes]
