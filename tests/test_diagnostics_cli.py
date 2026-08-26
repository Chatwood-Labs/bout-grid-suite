from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import bout_tokamak_grid_diagnostics as diag


def _run_diagnostics(
    diagnostics_script: Path,
    grid: Path,
    outdir: Path,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run diagnostics exactly as a user would from the command line."""
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(diagnostics_script),
        str(grid),
        "--outdir",
        str(outdir),
        "--json-only",
        "--quiet",
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    report = outdir / f"{grid.stem}_report.json"
    return proc, report


def _assert_report_schema(payload: dict) -> None:
    """Check the stable minimum contract consumed by tests and automation."""
    assert isinstance(payload, dict)
    assert isinstance(payload.get("checks"), list)
    assert isinstance(payload.get("critical_failures"), list)
    assert isinstance(payload.get("warnings"), list)

    # A failed WARN check must be reflected in the top-level warnings list;
    # the report must not simultaneously say WARN/FAIL and "no warnings".
    failed_warn_checks = [
        check
        for check in payload["checks"]
        if check.get("severity") == "WARN" and check.get("status") == "FAIL"
    ]
    if failed_warn_checks:
        assert payload["warnings"], (
            "Report contains failed WARN checks but the top-level warnings list is empty: "
            f"{failed_warn_checks}"
        )


def _failure_context(proc: subprocess.CompletedProcess[str]) -> str:
    return (
        f"\nreturn code: {proc.returncode}"
        f"\nstdout:\n{proc.stdout}"
        f"\nstderr:\n{proc.stderr}"
    )


def test_safe_text_stream_handles_legacy_windows_encoding() -> None:
    raw = io.BytesIO()
    cp1252_stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    safe = diag.Logging._SafeTextStream(cp1252_stream)

    safe.write("kappa=κ delta=δ approx=≈ ok=✓\n")
    safe.flush()

    rendered = raw.getvalue().decode("cp1252")
    assert "kappa=" in rendered
    assert "\\u03ba" in rendered
    assert "\\u03b4" in rendered
    assert "\\u2248" in rendered
    assert "\\u2713" in rendered


def test_known_good_grid_passes(
    tmp_path: Path,
    grids_dir: Path,
    diagnostics_script: Path,
    read_report_json,
) -> None:
    grid = grids_dir / "grid_tiny_shaped.nc"
    proc, report = _run_diagnostics(
        diagnostics_script,
        grid,
        tmp_path / "good",
    )

    assert proc.returncode == 0, _failure_context(proc)
    assert report.is_file(), f"Expected JSON report was not created: {report}"

    payload = read_report_json(report)
    _assert_report_schema(payload)

    assert payload["critical_failures"] == [], (
        "Known-good regression fixture produced critical failures: "
        f"{payload['critical_failures']}"
    )

    # This fixture is generated using the current Grid Suite v2 canonical
    # BOUT++ 5.x field-aligned metric and topology contract.
    assert payload.get("bout5_compatible") is True
    assert payload.get("bout5_metric_naming") == "CANONICAL"
    assert payload.get("grid_contract") == "BOUT5_FIELD_ALIGNED"
    assert payload.get("config_type") == "CLOSED_CORE / NO X-POINT"


def test_jacobian_signflip_is_rejected(
    tmp_path: Path,
    grids_dir: Path,
    diagnostics_script: Path,
    read_report_json,
) -> None:
    grid = grids_dir / "grid_tiny_bad_J_signflip.nc"
    proc, report = _run_diagnostics(
        diagnostics_script,
        grid,
        tmp_path / "bad_jacobian",
    )

    assert proc.returncode == 1, _failure_context(proc)
    assert report.is_file(), f"Expected JSON report was not created: {report}"

    payload = read_report_json(report)
    _assert_report_schema(payload)

    assert payload["critical_failures"], (
        "Known Jacobian sign-flip fixture was not reported as critically invalid."
    )


def test_nan_metric_is_rejected(
    tmp_path: Path,
    grids_dir: Path,
    diagnostics_script: Path,
    read_report_json,
) -> None:
    grid = grids_dir / "grid_tiny_bad_g11_nan.nc"
    proc, report = _run_diagnostics(
        diagnostics_script,
        grid,
        tmp_path / "bad_metric",
    )

    assert proc.returncode == 1, _failure_context(proc)
    assert report.is_file(), f"Expected JSON report was not created: {report}"

    payload = read_report_json(report)
    _assert_report_schema(payload)

    assert payload["critical_failures"], (
        "Known NaN metric fixture was not reported as critically invalid."
    )
