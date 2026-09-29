import shutil
import subprocess
import sys
from pathlib import Path

import fakes
import pytest

ROOT = Path(__file__).parent.parent


@pytest.fixture
def project(tmp_path):
    """A copy of the project code with a fake production bundle."""
    for part in ("app", "training", "ops"):
        shutil.copytree(ROOT / part, tmp_path / part, ignore=shutil.ignore_patterns("model", "__pycache__"))
    fakes.bundle(tmp_path / "app/model")
    return tmp_path


def run(project, module, *args):
    return subprocess.run([sys.executable, "-m", module, *args], cwd=project, capture_output=True, text=True, check=False)
