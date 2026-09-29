import csv
import hashlib
import re
from pathlib import Path

import numpy as np

from app.preprocess import preprocess

OWN_DIR = Path("collected")


def rows():
    labels = OWN_DIR / "labels.csv"
    if not labels.exists():
        return []
    with open(labels) as f:
        return list(csv.DictReader(f))


def is_test(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 5 == 0


def load_own(dataset, classes, test):
    xs, ys = [], []
    for r in rows():
        if is_test(r["file"]) == test and r["label"] in classes:
            xs.append(preprocess((OWN_DIR / r["file"]).read_bytes(), dataset)[0])
            ys.append(classes.index(r["label"]))
    return np.array(xs, dtype=np.float32).reshape(-1, 1, 28, 28), np.array(ys, dtype=np.int64)


def data_version():
    dvc = Path(f"{OWN_DIR}.dvc")
    files = sum(1 for f in OWN_DIR.iterdir() if f.is_file()) if OWN_DIR.exists() else 0
    if not dvc.exists():
        return {"own_md5": "none", "own_files": files, "own_stale": files > 0}
    text = dvc.read_text()
    md5 = re.search(r"md5: (\S+)", text).group(1)
    nfiles = int(re.search(r"nfiles: (\d+)", text).group(1))
    return {"own_md5": md5.removesuffix(".dir"), "own_files": files, "own_stale": files != nfiles}
