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
    for f in ROOT.glob("*.py"):
        shutil.copy(f, tmp_path)
    shutil.copytree(ROOT / "app", tmp_path / "app", ignore=shutil.ignore_patterns("model", "__pycache__"))
    fakes.bundle(tmp_path / "app/model")
    return tmp_path


def run(project, *args):
    return subprocess.run([sys.executable, *args], cwd=project, capture_output=True, text=True, check=False)
