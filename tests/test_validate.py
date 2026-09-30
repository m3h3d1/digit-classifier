import fakes
from conftest import run


def validate(project):
    return run(project, "ops.validate", "app/model/info.json")


def test_clean_data_passes(project):
    fakes.collect(project / "collected", 10)
    r = validate(project)
    assert r.returncode == 0 and "data ok" in r.stdout


def test_problems_are_caught(project):
    root = project / "collected"
    fakes.collect(root, 5)
    first = (root / "labels.csv").read_text().splitlines()[1].split(",")[0]
    (root / "blank.png").write_bytes(fakes.blank())
    (root / "copy.png").write_bytes((root / first).read_bytes())
    (root / "odd.png").write_bytes(fakes.drawing(50))
    (root / "stray.png").write_bytes(fakes.drawing(51))
    (root / "moody.png").write_bytes(fakes.drawing(52))
    extra = [("blank.png", "A", "collect"), ("copy.png", "A", "collect"), ("odd.png", "?", "collect"),
             (first, "A", "collect"), ("gone.png", "A", "collect"), ("moody.png", "A", "guess")]
    with open(root / "labels.csv", "a") as f:
        f.writelines(f"{name},{label},{label},1,2026-01-01T00:00:00+00:00,{mode}\n" for name, label, mode in extra)

    r = validate(project)
    assert r.returncode == 1
    for problem in ["blank.png: blank drawing", "copy.png: duplicate of", "odd.png: unknown label '?'",
                    f"{first}: listed twice", "gone.png: file missing", "stray.png: not in labels.csv",
                    "moody.png: unknown mode 'guess'"]:
        assert problem in r.stdout
