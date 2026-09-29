import json

import fakes
from conftest import run


def promote(project, store, name, **run_kw):
    art = project / name
    fakes.run_dir(art, **run_kw)
    assert run(project, "track.py", str(art), str(store)).returncode == 0
    r = run(project, "registry.py", "promote", str(store), str(art))
    return r.returncode, r.stdout + r.stderr


def test_gate(project, tmp_path):
    store = tmp_path / "store"
    fakes.collect(project / "collected", 30, label="A")

    code, out = promote(project, store, "first", test_accuracy=0.894)
    assert code == 0 and "-> v1 @production" in out

    code, out = promote(project, store, "worse", test_accuracy=0.87)
    assert code == 1 and "more than 0.5% below" in out

    code, out = promote(project, store, "broken", test_accuracy=0.95, broken=True)
    assert code == 1 and "ONNX differs from PyTorch" in out

    code, out = promote(project, store, "wrong-on-own", test_accuracy=0.95, favorite="B")
    assert code == 1 and "own-test 0.00% < production v1 100.00%" in out

    code, out = promote(project, store, "better", test_accuracy=0.90)
    assert code == 0 and "-> v2 @production" in out

    assert run(project, "registry.py", "bundle", str(store), "bundled").returncode == 0
    assert json.loads((project / "bundled/info.json").read_text())["version"] == 2
