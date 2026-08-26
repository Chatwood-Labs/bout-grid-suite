from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset


def _load_designer(designer_script: Path):
    spec = importlib.util.spec_from_file_location(
        "bout_grid_designer_custom_boundary_test",
        designer_script,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_boundary_validation_accepts_concave_star_shape_and_rejects_invalid(
    designer_script: Path,
) -> None:
    designer = _load_designer(designer_script)

    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)

    # Deliberately concave but still star-shaped about the magnetic axis.
    radius = 2.0 * (1.0 + 0.22 * np.cos(3.0 * theta))
    R = 6.2 + radius * np.cos(theta)
    Z = radius * np.sin(theta)

    is_convex, _ = designer.check_convexity(R, Z)
    assert not is_convex

    result = designer.validate_custom_boundary(R, Z)
    assert result["valid"] is True
    assert result["axis_R"] == pytest.approx(6.2, abs=1e-10)
    assert result["axis_Z"] == pytest.approx(0.0, abs=1e-10)

    # A U-like boundary excludes its inferred centroid / fails radial uniqueness.
    bad_R = np.array([4.0, 8.0, 8.0, 7.0, 7.0, 5.0, 5.0, 4.0])
    bad_Z = np.array([-2.0, -2.0, 2.0, 2.0, -1.0, -1.0, 2.0, 2.0])
    bad_Rs, bad_Zs = designer.smooth_contour(bad_R, bad_Z, n=160)
    bad = designer.validate_custom_boundary(bad_Rs, bad_Zs)

    assert bad["valid"] is False
    assert bad["reason"]


def test_custom_boundary_is_preserved_through_generator_and_diagnostics(
    tmp_path: Path,
    generator_script: Path,
    diagnostics_script: Path,
    read_report_json,
) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 320, endpoint=False)

    # Non-Miller star-shaped contour: enough higher-order structure that a
    # four-parameter Miller approximation cannot reproduce it.
    radius = 2.0 * (
        1.0
        + 0.15 * np.cos(3.0 * theta)
        + 0.08 * np.sin(2.0 * theta)
    )
    R = 6.2 + radius * np.cos(theta)
    Z = radius * np.sin(theta)

    boundary = tmp_path / "boundary.json"
    boundary.write_text(
        json.dumps(
            {
                "format": "chatwood-grid-boundary-v1",
                "mode": "custom",
                "R": R.tolist(),
                "Z": Z.tolist(),
                "axis_R": 6.2,
                "axis_Z": 0.0,
            }
        ),
        encoding="utf-8",
    )

    grid = tmp_path / "custom_grid.nc"

    generate = subprocess.run(
        [
            sys.executable,
            str(generator_script),
            "--boundary-file",
            str(boundary),
            "--outfile",
            str(grid),
            "--R0",
            "6.2",
            "--a",
            "2.0",
            "--kappa",
            "1.0",
            "--delta",
            "0.0",
            "--nx",
            "68",
            "--ny",
            "64",
            "--nz",
            "8",
            "--q0",
            "1.05",
            "--qa",
            "3.5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert generate.returncode == 0, (
        f"\nstdout:\n{generate.stdout}\nstderr:\n{generate.stderr}"
    )
    assert grid.is_file()

    with Dataset(grid, "r") as nc:
        assert nc.getncattr("geometry_model") == "custom_radial_boundary"

        outer_R = np.asarray(nc.variables["R"][-1, :], dtype=float)
        outer_Z = np.asarray(nc.variables["Z"][-1, :], dtype=float)
        requested_R = np.asarray(nc.variables["requested_boundary_R"][:], dtype=float)
        requested_Z = np.asarray(nc.variables["requested_boundary_Z"][:], dtype=float)

        fidelity = np.hypot(outer_R - requested_R, outer_Z - requested_Z)
        assert float(np.max(fidelity)) < 1e-10

        # Prove the custom geometry did not silently collapse back to the
        # near-circular Miller parameters supplied above.
        y = np.asarray(nc.variables["y"][:], dtype=float)
        miller_R = 6.2 + 2.0 * np.cos(y)
        miller_Z = 2.0 * np.sin(y)
        deviation_from_miller = np.hypot(
            outer_R - miller_R,
            outer_Z - miller_Z,
        )
        assert float(np.max(deviation_from_miller)) > 0.20

        # The full designer contour remains in requested_boundary.json; the
        # NetCDF keeps only the y-sampled requested boundary to avoid introducing
        # non-BOUT++ dimensions solely for provenance.

    reports = tmp_path / "reports"
    diagnosed = subprocess.run(
        [
            sys.executable,
            str(diagnostics_script),
            str(grid),
            "--outdir",
            str(reports),
            "--json-only",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert diagnosed.returncode == 0, (
        f"\nstdout:\n{diagnosed.stdout}\nstderr:\n{diagnosed.stderr}"
    )

    report = read_report_json(reports / "custom_grid_report.json")
    assert report["geometry_model"] == "custom_radial_boundary"
    assert report["boundary_fidelity_max_m"] < 1e-10
    assert report["critical_failures"] == []

    fidelity_checks = [
        check
        for check in report["checks"]
        if check.get("name") == "geometry:requested_boundary_fidelity"
    ]
    assert len(fidelity_checks) == 1
    assert fidelity_checks[0]["status"] == "PASS"


def test_periodic_plot_helper_closes_poloidal_seam() -> None:
    import bout_tokamak_grid_diagnostics as diagnostics

    R = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    Z = -R
    field = R * 10.0

    Rc, Zc, Fc = diagnostics._close_periodic_y(R, Z, field)

    assert Rc.shape == (2, 4)
    assert Zc.shape == (2, 4)
    assert Fc.shape == (2, 4)
    np.testing.assert_allclose(Rc[:, -1], R[:, 0])
    np.testing.assert_allclose(Zc[:, -1], Z[:, 0])
    np.testing.assert_allclose(Fc[:, -1], field[:, 0])


def test_preset_buttons_restore_canonical_parameters(
    designer_script: Path,
) -> None:
    """
    Preset buttons must never inherit R0/a from a previous custom geometry.
    This is a headless regression for the state-sync bug found manually.
    """
    designer = _load_designer(designer_script)

    class FakeDesigner:
        def __init__(self):
            self.calls = []

        def _load_preset(self, R, Z, params):
            self.calls.append(
                (
                    np.asarray(R, dtype=float),
                    np.asarray(Z, dtype=float),
                    dict(params),
                )
            )

    cases = [
        ("_preset_circle", "circle"),
        ("_preset_ellipse", "ellipse"),
        ("_preset_d", "d_shape"),
        ("_preset_neg_d", "neg_d"),
    ]

    for method_name, preset_name in cases:
        fake = FakeDesigner()
        method = getattr(designer.BoutGridDesigner, method_name)
        method(fake)

        assert len(fake.calls) == 1
        R, Z, params = fake.calls[0]
        expected = designer.PRESET_DEFAULTS[preset_name]

        assert params == expected
        assert params["R0"] == pytest.approx(6.2)
        assert params["a"] == pytest.approx(2.0)
        assert len(R) >= 3
        assert len(Z) == len(R)

    # Explicitly lock the canonical D acceptance values.
    fake = FakeDesigner()
    designer.BoutGridDesigner._preset_d(fake)
    _, _, params = fake.calls[0]
    assert params == {
        "R0": 6.2,
        "a": 2.0,
        "kappa": 1.7,
        "delta": 0.33,
    }


def test_shifted_custom_boundary_does_not_require_global_z_symmetry(
    tmp_path: Path,
    generator_script: Path,
    diagnostics_script: Path,
    read_report_json,
) -> None:
    """
    A valid custom boundary may be vertically shifted or asymmetric. Diagnostics
    must validate the recorded magnetic axis instead of assuming mean(Z) == 0.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 320, endpoint=False)
    axis_R = 6.4
    axis_Z = 0.75

    radius = 1.8 * (
        1.0
        + 0.10 * np.cos(3.0 * theta)
        + 0.05 * np.sin(2.0 * theta)
    )
    R = axis_R + radius * np.cos(theta)
    Z = axis_Z + radius * np.sin(theta)

    boundary = tmp_path / "shifted_boundary.json"
    boundary.write_text(
        json.dumps(
            {
                "format": "chatwood-grid-boundary-v1",
                "mode": "custom",
                "R": R.tolist(),
                "Z": Z.tolist(),
                "axis_R": axis_R,
                "axis_Z": axis_Z,
            }
        ),
        encoding="utf-8",
    )

    grid = tmp_path / "shifted_custom_grid.nc"

    generate = subprocess.run(
        [
            sys.executable,
            str(generator_script),
            "--boundary-file",
            str(boundary),
            "--outfile",
            str(grid),
            "--R0",
            str(axis_R),
            "--a",
            "1.8",
            "--kappa",
            "1.0",
            "--delta",
            "0.0",
            "--nx",
            "68",
            "--ny",
            "64",
            "--nz",
            "8",
            "--q0",
            "1.05",
            "--qa",
            "3.5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert generate.returncode == 0, (
        f"\nstdout:\n{generate.stdout}\nstderr:\n{generate.stderr}"
    )

    reports = tmp_path / "reports"
    diagnosed = subprocess.run(
        [
            sys.executable,
            str(diagnostics_script),
            str(grid),
            "--outdir",
            str(reports),
            "--json-only",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert diagnosed.returncode == 0, (
        f"\nstdout:\n{diagnosed.stdout}\nstderr:\n{diagnosed.stderr}"
    )

    report = read_report_json(reports / "shifted_custom_grid_report.json")
    assert report["geometry_model"] == "custom_radial_boundary"
    assert report["critical_failures"] == []

    names = {check.get("name") for check in report["checks"]}
    assert "geometry:Z_mean_near_zero" not in names
    assert "geometry:custom_axis_within_vertical_extent" in names

    axis_checks = [
        check
        for check in report["checks"]
        if check.get("name") == "geometry:custom_axis_within_vertical_extent"
    ]
    assert len(axis_checks) == 1
    assert axis_checks[0]["status"] == "PASS"
    assert axis_checks[0]["severity"] == "CRITICAL"

    assert not any("Z mean" in warning for warning in report["warnings"])


def test_miller_generation_snapshot_matches_displayed_fields(
    designer_script: Path,
) -> None:
    """Generation must use the exact Miller values displayed at click time."""
    designer = _load_designer(designer_script)

    class FakeVar:
        def __init__(self, value):
            self.value = str(value)

        def get(self):
            return self.value

    class FakeDesigner:
        pass

    fake = FakeDesigner()
    fake._miller_vars = {
        "R0": FakeVar("6.2000"),
        "a": FakeVar("2.0000"),
        "kappa": FakeVar("1.7000"),
        "delta": FakeVar("0.3300"),
    }

    snapshot = designer.BoutGridDesigner._snapshot_miller_generation_state(fake)

    assert snapshot == {
        "R0": 6.2,
        "a": 2.0,
        "kappa": 1.7,
        "delta": 0.33,
    }
    assert np.min(fake.smooth_R) == pytest.approx(4.2, abs=1e-12)
    assert np.max(fake.smooth_R) == pytest.approx(8.2, abs=1e-12)
    assert np.min(fake.smooth_Z) == pytest.approx(-3.4, abs=1e-12)
    assert np.max(fake.smooth_Z) == pytest.approx(3.4, abs=1e-12)
