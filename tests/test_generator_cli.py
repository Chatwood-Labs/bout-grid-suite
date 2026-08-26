from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


def _failure_context(proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"\nreturn code: {proc.returncode}"
        f"\nstdout:\n{proc.stdout}"
        f"\nstderr:\n{proc.stderr}"
    )


def test_generator_creates_valid_small_grid(
    tmp_path: Path,
    generator_script: Path,
) -> None:
    """Smoke-test the generator through its public command-line interface."""
    grid = tmp_path / "generated_grid.nc"

    # nx=12 is intentional:
    # with MXG=2, the 8-point interior is already a power of two, so the
    # generator's PCR compatibility rounding should leave nx unchanged.
    cmd = [
        sys.executable,
        str(generator_script),
        "--outfile",
        str(grid),
        "--nx",
        "12",
        "--ny",
        "16",
        "--nz",
        "8",
        "--R0",
        "6.2",
        "--a",
        "2.0",
        "--kappa",
        "1.7",
        "--delta",
        "0.33",
        "--q0",
        "1.05",
        "--qa",
        "3.5",
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, _failure_context(proc)
    assert grid.is_file(), f"Generator did not create expected grid: {grid}"
    assert grid.stat().st_size > 0, "Generated NetCDF file is empty."

    with Dataset(grid, "r") as nc:
        # Stable dimensional contract for this deliberately small fixture.
        assert len(nc.dimensions["x"]) == 12
        assert len(nc.dimensions["y"]) == 16
        assert len(nc.dimensions["z"]) == 8

        # Analytic input parameters are preserved as NetCDF provenance metadata.
        assert float(nc.getncattr("R0")) == 6.2
        assert float(nc.getncattr("a")) == 2.0
        assert float(nc.getncattr("kappa")) == 1.7
        assert float(nc.getncattr("delta")) == 0.33
        assert float(nc.getncattr("B0")) == 5.3
        assert float(nc.getncattr("q0")) == 1.05
        assert float(nc.getncattr("qa")) == 3.5
        assert nc.getncattr("qform") == "quadratic"
        assert int(nc.getncattr("bout5_canonical")) == 1
        assert nc.getncattr("grid_contract") == "BOUT5_FIELD_ALIGNED"
        assert "g11..g23=contravariant" in nc.getncattr("metric_naming")

        # Core fields required by the Grid Suite/BOUT++ workflow.
        required_variables = {
            "x",
            "y",
            "z",
            "xcoord",
            "ycoord",
            "zcoord",
            "dx",
            "dy",
            "dz",
            "R",
            "Z",
            "g11",
            "g22",
            "g33",
            "g12",
            "g13",
            "g23",
            "g_11",
            "g_22",
            "g_33",
            "g_12",
            "g_13",
            "g_23",
            "J",
            "zShift",
            "ShiftAngle",
            "ShiftTorsion",
            "IntShiftTorsion",
            "Bxy",
            "Bpxy",
            "ixseps1",
            "ixseps2",
        }

        missing = sorted(required_variables.difference(nc.variables))
        assert not missing, f"Generated grid is missing required variables: {missing}"

        # Coordinate maps are always full 3D arrays.
        assert nc.variables["xcoord"].shape == (12, 16, 8)
        assert nc.variables["ycoord"].shape == (12, 16, 8)
        assert nc.variables["zcoord"].shape == (12, 16, 8)

        # Default axisymmetric mode writes metric/geometry fields in 2D.
        assert nc.variables["R"].shape == (12, 16)
        assert nc.variables["Z"].shape == (12, 16)
        assert nc.variables["J"].shape == (12, 16)

        # The baseline grid must contain finite, physically usable geometry.
        for name in (
            "R",
            "Z",
            "g11",
            "g22",
            "g33",
            "g_11",
            "g_22",
            "g_33",
            "J",
            "Bxy",
            "Bpxy",
            "zShift",
            "ShiftAngle",
            "ShiftTorsion",
            "IntShiftTorsion",
        ):
            values = np.asanyarray(nc.variables[name][:])
            assert np.all(np.isfinite(values)), (
                f"Generated variable {name!r} contains non-finite values."
            )

        jacobian = np.asanyarray(nc.variables["J"][:])
        assert np.all(jacobian > 0.0), (
            "Generated baseline grid contains a non-positive Jacobian."
        )

        # Emulate the canonical BOUT++ 5.x loader checks. BOUT++ treats
        # un-underscored metric names as contravariant and underscored names
        # as covariant, then derives J and Bxy from the covariant tensor.
        g11c = np.asanyarray(nc.variables["g_11"][:])
        g22c = np.asanyarray(nc.variables["g_22"][:])
        g33c = np.asanyarray(nc.variables["g_33"][:])
        g12c = np.asanyarray(nc.variables["g_12"][:])
        g13c = np.asanyarray(nc.variables["g_13"][:])
        g23c = np.asanyarray(nc.variables["g_23"][:])
        det_cov = (
            g11c * (g22c * g33c - g23c**2)
            - g12c * (g12c * g33c - g13c * g23c)
            + g13c * (g12c * g23c - g13c * g22c)
        )
        Jcalc = np.sqrt(np.abs(det_cov))
        Bcalc = np.sqrt(np.abs(g22c)) / jacobian
        Bxy = np.asanyarray(nc.variables["Bxy"][:])
        assert np.allclose(Jcalc, jacobian, rtol=1e-8, atol=1e-10)
        assert np.allclose(Bcalc, Bxy, rtol=1e-8, atol=1e-10)

        # In BOUT++ field-aligned coordinates the covariant toroidal metric
        # component is g_33 = R^2.  The contravariant g33 is not expected to
        # equal R^2.
        R = np.asanyarray(nc.variables["R"][:])
        assert np.allclose(g33c, R**2, rtol=1e-10, atol=1e-10)

        assert nc.variables["zShift"].shape == (12, 16)
        assert nc.variables["ShiftAngle"].shape == (12,)
        assert nc.variables["ShiftTorsion"].shape == (12, 16)
        assert nc.variables["IntShiftTorsion"].shape == (12, 16)
        assert int(np.asanyarray(nc.variables["ixseps1"][:]).item()) == 12
        assert int(np.asanyarray(nc.variables["ixseps2"][:]).item()) == 12
        assert nc.getncattr("topology_model") == "closed_core_no_xpoint"

        assert float(nc.getncattr("toroidal_period_rad")) == pytest.approx(2 * np.pi)
        assert int(nc.getncattr("zperiod")) == 1

def test_generator_console_survives_legacy_windows_encoding(
    tmp_path: Path,
    generator_script: Path,
) -> None:
    """
    Regression guard for Windows/redirected consoles using cp1252.

    Generator status output must not crash merely because a terminal or pipe
    cannot encode scientific Unicode symbols.
    """
    grid = tmp_path / "cp1252_grid.nc"

    cmd = [
        sys.executable,
        str(generator_script),
        "--outfile",
        str(grid),
        "--nx",
        "12",
        "--ny",
        "16",
        "--nz",
        "8",
        "--R0",
        "6.2",
        "--a",
        "2.0",
        "--kappa",
        "1.7",
        "--delta",
        "0.33",
        "--q0",
        "1.05",
        "--qa",
        "3.5",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252:strict"

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=False,
        check=False,
        env=env,
    )

    stdout = proc.stdout.decode("cp1252", errors="replace")
    stderr = proc.stderr.decode("cp1252", errors="replace")

    assert proc.returncode == 0, (
        f"\nreturn code: {proc.returncode}"
        f"\nstdout:\n{stdout}"
        f"\nstderr:\n{stderr}"
    )
    assert grid.is_file(), f"Generator did not create expected grid: {grid}"
    assert "BOUT++ 5.x canonical metric contract" in stdout
    assert "max rel dJ=" in stdout
    assert "max rel dBxy=" in stdout
    assert "UnicodeEncodeError" not in stderr

