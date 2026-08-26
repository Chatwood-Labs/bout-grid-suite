from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_designer(designer_script: Path):
    spec = importlib.util.spec_from_file_location(
        "bout_grid_designer_smoke",
        designer_script,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_designer_imports_and_resolves_bundled_components(
    designer_script: Path,
    generator_script: Path,
    diagnostics_script: Path,
) -> None:
    """
    Import the GUI module without constructing a Tk window and verify that the
    default bundled component paths match the Grid Suite repository layout.
    """
    designer = _load_designer(designer_script)

    assert isinstance(designer.__version__, str)
    assert designer.__version__

    assert designer._DEFAULT_GENERATOR.resolve() == generator_script.resolve()
    assert designer._DEFAULT_DIAGNOSTICS.resolve() == diagnostics_script.resolve()

    assert designer._DEFAULT_GENERATOR.is_file()
    assert designer._DEFAULT_DIAGNOSTICS.is_file()


def test_designer_help_is_available_without_starting_gui(
    designer_script: Path,
) -> None:
    """
    --help must exit successfully before a Tk window is constructed.

    This gives users and CI a safe command-line smoke test for the desktop
    entry point without requiring an active graphical display.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(designer_script),
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, (
        f"\nreturn code: {proc.returncode}"
        f"\nstdout:\n{proc.stdout}"
        f"\nstderr:\n{proc.stderr}"
    )

    help_text = proc.stdout

    assert "BOUT++ Grid Designer GUI" in help_text
    assert "--generator" in help_text
    assert "--diagnostics" in help_text
