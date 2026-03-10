from __future__ import annotations

import tomllib
from pathlib import Path


def test_project_metadata_keeps_numpy_as_runtime_dependency():
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text())

    dependencies = data["project"]["dependencies"]

    assert "numpy>=1.24" in dependencies


def test_project_metadata_exposes_konash_console_script():
    pyproject = Path("pyproject.toml")
    data = tomllib.loads(pyproject.read_text())

    scripts = data["project"]["scripts"]

    assert scripts["konash"] == "konash.cli:main"
