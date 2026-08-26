from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# Repository layout
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Make the two implementation scripts importable for tests that genuinely need
# direct access, while keeping subprocess-based behavioural tests independent
# of Python packaging.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def src_dir() -> Path:
    return SRC_DIR


@pytest.fixture(scope="session")
def designer_script(repo_root: Path) -> Path:
    path = repo_root / "bout_grid_designer.py"
    assert path.is_file(), f"Missing Grid Suite designer: {path}"
    return path


@pytest.fixture(scope="session")
def generator_script(src_dir: Path) -> Path:
    path = src_dir / "bout_tokamak_grid_generator.py"
    assert path.is_file(), f"Missing grid generator: {path}"
    return path


@pytest.fixture(scope="session")
def diagnostics_script(src_dir: Path) -> Path:
    path = src_dir / "bout_tokamak_grid_diagnostics.py"
    assert path.is_file(), f"Missing grid diagnostics: {path}"
    return path


@pytest.fixture(scope="session")
def grids_dir() -> Path:
    path = Path(__file__).resolve().parent / "data" / "grids"
    assert path.is_dir(), f"Missing test grid directory: {path}"
    return path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def read_report_json():
    return read_json
