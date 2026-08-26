# Chatwood Labs Grid Suite for BOUT++

**Version 2.0.0**

[![Grid Suite tests](https://github.com/Chatwood-Labs/bout-grid-suite/actions/workflows/tests.yml/badge.svg)](https://github.com/Chatwood-Labs/bout-grid-suite/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

Chatwood Labs Grid Suite is an integrated desktop and command-line workflow for designing, generating, inspecting and validating tokamak grids for **BOUT++**.

Grid Suite v2 builds on the original Chatwood Labs axisymmetric grid generator and grid diagnostics utility first published in the [`bout-tools`](https://github.com/Chatwood-Labs/bout-tools) repository, combining them with an interactive graphical designer and a unified generation and validation workflow. It supports both controlled Miller-style analytic geometry and validated user-drawn custom boundaries, then runs the resulting grid through automated geometry, metric, Jacobian and magnetic-field diagnostics.

The original [`bout-tools`](https://github.com/Chatwood-Labs/bout-tools) repository is retained as the historical v1 release and remains available for reference, but is no longer actively maintained. Current Grid Suite development continues in this repository.

The suite is intended for:

- controlled BOUT++ geometry studies
- solver and workflow development
- synthetic grid generation
- numerical experiments with shaping
- regression and CI fixtures
- rapid visual exploration of analytic and custom tokamak cross-sections

Grid Suite v2 is **not** a Grad-Shafranov equilibrium solver and does not reconstruct a physical plasma equilibrium from experimental data.

---

## Features

### Grid Studio

- interactive 2D poloidal cross-section designer
- circle, ellipse, positive-D and negative-D presets
- editable Miller parameters
- freehand custom-boundary drawing
- optional contour smoothing
- live non-convexity warnings
- hard rejection of custom boundaries that cannot produce a valid radial mapping
- 3D toroidal preview of the current visible contour
- save and reload design configurations
- integrated generation and diagnostics workflow
- embedded report, geometry and magnetic-field views

### Grid generation

- Miller-style analytic shaping
- validated custom poloidal boundaries
- configurable major/minor radius, elongation and triangularity
- configurable toroidal magnetic field
- quadratic, linear and cubic q profiles
- configurable radial, poloidal and toroidal resolution
- configurable curvature calculation
- 2D axisymmetric metrics by default
- optional full 3D metric output
- optional CLI-only toroidal/helical ripple for synthetic 3D geometry tests
- canonical BOUT++ 5.x field-aligned NetCDF output for axisymmetric Miller and Custom Boundary grids
- requested-boundary provenance for custom grids

### Diagnostics and validation

- required-variable checks
- finite and positive metric checks
- leading principal-minor metric check
- Jacobian sign and boundary-singularity checks
- `det(g)` versus `J²` convention/consistency checking
- optional `surfvol` checks
- magnetic-field checks
- shift-angle checks
- metric-conditioning checks
- requested-versus-generated boundary fidelity for custom grids
- HTML reports
- JSON reports
- geometry/curvature visualisation
- magnetic-field visualisation
- strict mode and explicit CLI exit codes
- optional dask chunking for large grids

---

# Installation

Clone the repository and install the runtime dependencies:

```bash
git clone https://github.com/Chatwood-Labs/bout-grid-suite.git
cd bout-grid-suite
python -m pip install -r requirements.txt
```

On Windows, `py` may be used instead of `python`:

```powershell
py -m pip install -r requirements.txt
```

## Tk/Tkinter requirement

The **Grid Studio graphical interface** uses `tkinter`, Python's standard interface to the Tcl/Tk desktop GUI toolkit.

`tkinter` is deliberately not listed in `requirements.txt`. The requirements file installs Python packages from PyPI, while Tk support depends on the Python build and operating system. The Python `tkinter` module also relies on the native Tcl/Tk libraries supplied with, or installed alongside, Python.

If you only use the command-line grid generator and diagnostics utility, Tk/Tkinter is not required.

You can check whether Tkinter is available by running:

```bash
python -m tkinter
```

or on Windows:

```powershell
py -m tkinter
```

A small Tk test window should open.

### Windows

The standard Python installer from [python.org](https://www.python.org/downloads/) normally includes Tcl/Tk and `tkinter` by default.

If `py -m tkinter` fails, rerun the Python installer, choose **Modify**, and ensure **Tcl/Tk and IDLE** is enabled.

### Ubuntu / Debian

Install the system Tk package if it is not already present:

```bash
sudo apt update
sudo apt install python3-tk
```

Then verify with:

```bash
python3 -m tkinter
```

### Fedora / RHEL-family distributions

Install the system Tkinter package:

```bash
sudo dnf install python3-tkinter
```

### macOS

Python installations from [python.org](https://www.python.org/downloads/) normally include working Tk support. Verify with:

```bash
python3 -m tkinter
```

Other Python distributions, including some package-manager installations, may require the matching Tcl/Tk or Python-Tk package for that Python version.



For development and regression testing:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The v2.0.0 regression suite contains 17 automated tests. GitHub Actions runs the suite on Windows and Ubuntu using Python 3.10 and Python 3.14.

---

# Quick Start: Grid Studio

From the repository root:

```bash
python bout_grid_designer.py
```

The application automatically locates the bundled generator and diagnostics scripts under `src/`.

The normal workflow is:

1. choose a preset or draw a custom contour;
2. optionally edit parameters or smooth the contour;
3. inspect the 2D shape and 3D toroidal preview;
4. select **Generate Grid + Report**;
5. inspect the diagnostics verdict, geometry and magnetic-field views;
6. open the full HTML report if required.

Generated files are separated by run ID:

```text
output/<run_id>/grid.nc
reports/<run_id>/
```

Custom-boundary runs also preserve the exact designer request:

```text
output/<run_id>/requested_boundary.json
```

Saved Grid Studio designs are stored under:

```text
saves/
```

---

# Geometry Modes

## Miller analytic geometry

The built-in presets use a Miller-style analytic surface:

```text
R = R0 + r cos(theta + asin(delta) sin(theta))
Z = kappa r sin(theta)
```

The canonical v2 presets are:

| Preset | R0 (m) | a (m) | κ | δ |
| --- | ---: | ---: | ---: | ---: |
| Circle | 6.2 | 2.0 | 1.0 | 0.0 |
| Ellipse | 6.2 | 2.0 | 1.7 | 0.0 |
| D-shape | 6.2 | 2.0 | 1.7 | 0.33 |
| Negative-D | 6.2 | 2.0 | 1.7 | -0.33 |

Clicking a preset restores its canonical values. Editing the Miller parameter fields then updates the analytic contour directly.

For Miller mode, the displayed `R0`, `a`, `κ` and `δ` are the authoritative generation parameters.

## Custom Boundary geometry

When a contour is drawn, or an existing contour is directly modified with the drawing tools, Grid Studio switches to **Custom Boundary** mode.

The visible contour becomes the requested outer boundary. It is **not** discarded and replaced with a four-parameter Miller approximation.

The custom-boundary workflow:

```text
visible contour
    ↓
geometric validation
    ↓
periodic angular parameterisation about the inferred magnetic axis
    ↓
nested surfaces generated by radial scaling
    ↓
metric / Jacobian / magnetic geometry calculation
    ↓
NetCDF
    ↓
requested-boundary fidelity validation
```

The Miller values shown for a custom contour are labelled **Equivalent Miller Fit** and are descriptive only.

### Custom-boundary validation

Grid Studio uses two levels of geometry handling.

**Advisory warning**

A non-convex contour may still be usable when it remains single-valued about the inferred magnetic axis. Grid Studio warns that strongly non-convex shaping may produce distorted cells and recommends using **Smooth**, but the user may continue.

**Hard generation block**

Generation is refused when the contour cannot form the radial nested-surface mapping used by v2. Examples include:

- self-intersecting contours
- inferred magnetic axis outside the boundary
- degenerate/negligible enclosed area
- boundaries reaching non-positive major radius
- multiple or missing boundary intersections along a radial ray
- a boundary approaching the magnetic axis too closely

### What custom mode proves and what it does not

For an accepted custom boundary, Grid Suite constructs and validates a numerically coherent coordinate mapping.

It does **not** prove that the shape corresponds to a physically achievable magnetohydrodynamic equilibrium.

Grid Suite v2 does not include GEQDSK import, X-point/divertor topology generation or a Grad-Shafranov equilibrium solver.

---

# Grid Studio command-line options

The GUI itself has only two command-line overrides:

```bash
python bout_grid_designer.py \
    --generator /path/to/bout_tokamak_grid_generator.py \
    --diagnostics /path/to/bout_tokamak_grid_diagnostics.py
```

| Option | Purpose |
| --- | --- |
| `--generator PATH` | Override the bundled grid-generator script |
| `--diagnostics PATH` | Override the bundled diagnostics script |

These are primarily useful for development, testing or comparing alternate component versions.

---

# Grid Generator CLI

The bundled generator remains directly usable without the GUI:

```bash
python src/bout_tokamak_grid_generator.py [options]
```

The command-line interface from the original [`bout-tools` axisymmetric grid generator](https://github.com/Chatwood-Labs/bout-tools/tree/main/bout_tokamak_grid) is retained, with additional v2 custom-boundary and synthetic 3D options.

## Basic Miller example

```bash
python src/bout_tokamak_grid_generator.py \
    --R0 6.2 \
    --a 2.0 \
    --B0 5.3 \
    --kappa 1.7 \
    --delta 0.33 \
    --q0 1.05 \
    --qa 3.5 \
    --qform quadratic \
    --nx 68 \
    --ny 64 \
    --nz 128 \
    --curvature exact \
    --precision f8 \
    --outfile grid.nc
```

## Generator arguments

| Option | Default | Description |
| --- | --- | --- |
| `--R0 FLOAT` | `6.2` | Major/reference radius in metres |
| `--a FLOAT` | `2.0` | Minor-radius/radial scale in metres |
| `--B0 FLOAT` | `5.3` | Toroidal magnetic field at the reference radius, tesla |
| `--xmin_frac FLOAT` | `0.1` | Inner radial fraction used to avoid the `r=0` coordinate singularity |
| `--kappa FLOAT` | `1.7` | Miller elongation |
| `--delta FLOAT` | `0.33` | Miller triangularity |
| `--boundary-file PATH` | none | Use a JSON custom boundary instead of Miller shaping |
| `--q0 FLOAT` | `1.05` | Central safety factor |
| `--qa FLOAT` | `3.5` | Edge safety factor at `r=a` |
| `--qform FORM` | `quadratic` | q-profile form: `quadratic`, `linear`, or `cubic` |
| `--nx INT` | `64` requested | Requested radial grid points; normalised to a PCR-compatible total, described below |
| `--ny INT` | `64` | Poloidal grid points |
| `--nz INT` | `128` | Toroidal grid points |
| `--precision {f4,f8}` | `f8` | NetCDF floating-point precision |
| `--outfile PATH` | `grid.nc` | Output NetCDF filename |
| `--curvature MODEL` | `exact` | `exact`, `simple`, or `none` |
| `--metrics-3d` | off | Write metric/geometry variables on `(x,y,z)` rather than the default `(x,y)` |
| `--toroidal-ripple` | off | Enable synthetic toroidal/helical ripple |
| `--ripple-eps FLOAT` | `0.0` | Fractional ripple amplitude |
| `--ripple-n INT` | `8` | Toroidal ripple mode number `N` |
| `--ripple-m INT` | `0` | Poloidal/helical ripple mode number `M` |

## Radial-point normalisation

The generator uses two radial guard cells on each side (`MXG=2`) and normalises the requested `nx` so that:

```text
nx - 2*MXG
```

is a power of two for PCR-solver compatibility.

For example, Grid Studio requests:

```text
nx = 68
```

which gives:

```text
68 - 4 = 64
```

interior radial points.

The generator CLI parser retains the historical requested default of `64`; because that value does not provide a power-of-two interior with four guard points, it is normalised and a warning is printed. For predictable v2 command-line use, `--nx 68` is the recommended baseline.

## q-profile forms

The generator supports:

- `quadratic`
- `linear`
- `cubic`

with `q0` defining the central value and `qa` the edge value.

The q profile is checked for monotonicity. A non-monotonic profile is allowed but produces a warning because it can introduce shear reversal or magnetic-well behaviour.

## Curvature modes

`--curvature exact`

Calculates the full geometry-derived curvature terms used by the generator. This is the default and most complete mode.

`--curvature simple`

Uses the simplified geometric approximation.

`--curvature none`

Disables curvature output calculation.

## 2D versus 3D metrics

For normal axisymmetric operation, metric and geometry quantities are written as 2D `(x,y)` arrays by default.

Use:

```bash
--metrics-3d
```

to write them as `(x,y,z)` arrays.

The coordinate maps `xcoord`, `ycoord` and `zcoord` are always written in 3D.

## Custom boundary CLI

The same custom-boundary engine used by Grid Studio can be called directly:

```bash
python src/bout_tokamak_grid_generator.py \
    --boundary-file requested_boundary.json \
    --a 2.0 \
    --B0 5.3 \
    --nx 68 \
    --ny 64 \
    --nz 128 \
    --q0 1.05 \
    --qa 3.5 \
    --outfile custom_grid.nc
```

A boundary file has the form:

```json
{
  "format": "chatwood-grid-boundary-v1",
  "mode": "custom",
  "R": [8.2, 8.19, 8.16],
  "Z": [0.0, 0.2, 0.4],
  "axis_R": 6.2,
  "axis_Z": 0.0
}
```

`R` and `Z` must contain the complete closed-boundary sampling, not merely the three illustrative points above.

When `--boundary-file` is supplied:

- the boundary is authoritative;
- `R0`, `kappa` and `delta` no longer define the generated outer contour;
- `a` remains the radial coordinate scale used by the nested-surface mapping;
- `B0`, q-profile and numerical options remain active;
- equivalent Miller values may still be retained as descriptive metadata.

Grid Studio writes the correct boundary JSON automatically and is the recommended way to create this file.

## Synthetic toroidal ripple

The generator includes an advanced CLI-only perturbation for deliberately introducing toroidal/helical variation:

```bash
python src/bout_tokamak_grid_generator.py \
    --R0 6.2 \
    --a 2.0 \
    --kappa 1.7 \
    --delta 0.33 \
    --nx 68 \
    --ny 64 \
    --nz 128 \
    --toroidal-ripple \
    --ripple-eps 0.005 \
    --ripple-n 8 \
    --ripple-m 0 \
    --metrics-3d \
    --outfile ripple_grid.nc
```

The perturbation is based on a controlled phase of the form:

```text
cos(N*phi + M*theta)
```

This feature is intended for synthetic 3D geometry and z-dependence testing, not equilibrium reconstruction.

For meaningful toroidally varying output, use `--toroidal-ripple` together with `--metrics-3d`. Without `--metrics-3d`, the file retains the normal 2D metric-output contract and therefore does not preserve the full toroidal metric variation.

Keep ripple amplitudes small; large perturbations can produce poor or invalid geometry.

---

# Generator NetCDF output

The generator writes the BOUT++ grid plus geometry and provenance metadata.

For normal axisymmetric Miller and Custom Boundary generation, v2 writes the explicit contract metadata:

```text
grid_contract = BOUT5_FIELD_ALIGNED
bout5_canonical = 1
metric_naming = BOUT++: g11..g23=contravariant; g_11..g_23=covariant
```

Before writing, the generator verifies that the covariant metric reconstructs the generated Jacobian and that `sqrt(g_22)/J` reconstructs `Bxy` to numerical precision.

The optional toroidal-ripple mode remains a deliberately synthetic 3D/forensic path. Because the current ripple model is not transformed through the axisymmetric Clebsch construction, it is explicitly labelled `NONCANONICAL_SYNTHETIC_3D` rather than being claimed as a canonical BOUT++ 5.x tokamak grid.

## Dimensions and coordinates

- `x`, `y`, `z`
- `xcoord`, `ycoord`, `zcoord`
- scalar/grid metadata including `nx`, `ny`, `nz`
- guard-cell metadata including `MXG`, `MYG`, `MZG`

## Grid spacing

- `dx`
- `dy`
- `dz`

## Topology metadata

Grid Suite v2 Miller and Custom Boundary modes generate nested closed-core flux surfaces without an X-point or an internal separatrix.

Canonical output therefore writes the BOUT++ scalar topology variables:

- `ixseps1 = nx`
- `ixseps2 = nx`

In BOUT++ this explicitly places the separatrix outside the represented radial domain, so all grid points belong to the closed-core region. Grid Suite diagnostics reports this configuration as **CLOSED_CORE / NO X-POINT**.

These values do not claim that a physical separatrix exists at the outer grid index. Diverted and X-point topology generation is outside the scope of v2.

## Geometry

- `R`
- `Z`

## Contravariant metric components (`g^{ij}`)

BOUT++ 5.x uses the un-underscored names for the contravariant tensor:

- `g11`
- `g22`
- `g33`
- `g12`
- `g13`
- `g23`

## Covariant metric components (`g_ij`)

BOUT++ 5.x uses the underscored names for the covariant tensor:

- `g_11`
- `g_22`
- `g_33`
- `g_12`
- `g_13`
- `g_23`

## Jacobian and volume

- `J`
- `surfvol`

## Field alignment and magnetic geometry

Canonical axisymmetric Grid Suite output includes:

- `zShift(x,y)` - accumulated local field-line pitch from the poloidal reference location
- `ShiftAngle(x)` - total toroidal shift after one complete poloidal circuit
- `ShiftTorsion(x,y)` - radial derivative of the local field-line pitch, used by BOUT++ vector differential operators
- `IntShiftTorsion(x,y)` - integrated shear, equivalent to `partial(zShift)/partial(x)`
- `Bxy` - magnetic-field magnitude consistent with the BOUT++ metric/Jacobian identity
- `Bpxy` - poloidal magnetic-field magnitude

For canonical output, Grid Suite constructs a field-aligned Clebsch coordinate system and verifies the same loader identities used by BOUT++ 5.x before writing the file.

## Using canonical grids with BOUT++ 5.x

Grid Suite writes the field-alignment quantities required by BOUT++, but the parallel-transform mode is selected by the consuming BOUT++ model rather than by the NetCDF grid file.

A configuration validated during the v2.0.0 release testing is:

```ini
twistshift = true

[mesh]
type = "bout"
file = "grid.nc"
symmetricGlobalX = false
symmetricGlobalY = false
calcParallelSlices_on_communicate = true

[mesh:paralleltransform]
type = "shiftedinterp"
```

`twistshift` is a root-level BOUT++ option. `shiftedinterp` uses the generated `zShift` field when constructing parallel slices and supports interpolation weights required by BOUT++ operators such as the PETSc 3D AMG Laplacian.

Other BOUT++ models may use a different parallel-transform configuration. In particular, `type = shifted` is supported by BOUT++ but does not provide the Y-approximation interpolation weights required by `petsc3damg`.

### BOUT++ runtime notes

Grid Suite does not currently write `d2x`, `d2y` or `d2z`. These are optional non-uniform differencing quantities. When they are absent, BOUT++ derives the required corrections from `dx`, `dy` and `dz`.

Some BOUT++ 5.2.x builds may also print legacy tokamak-topology wording such as `EQUILIBRIUM IS SINGLE NULL` while loading a closed-core grid. For Grid Suite v2 output, `ixseps1 = ixseps2 = nx` is the authoritative declaration that no separatrix or X-point lies inside the represented domain.

## Curvature

- `G1`
- `G2`

By default the spacing, physical geometry, metric, Jacobian, magnetic and curvature quantities are written on `(x,y)` for axisymmetric grids.

With `--metrics-3d`, these quantities are written on `(x,y,z)`.

## Custom-boundary provenance

Custom grids additionally contain:

- `requested_boundary_R(y)`
- `requested_boundary_Z(y)`
- `geometry_model = "custom_radial_boundary"`
- `axis_R`
- `axis_Z`
- equivalent Miller-fit metadata

The full original designer contour is also retained in the run's `requested_boundary.json`.

---

# Grid Diagnostics CLI

The bundled diagnostics utility remains directly usable:

```bash
python src/bout_tokamak_grid_diagnostics.py path/to/grid.nc
```

This workflow continues the original [`bout-tools` Grid Diagnostics Utility](https://github.com/Chatwood-Labs/bout-tools/tree/main/bout_tokamak_grid_diagnostics) and extends it as part of the integrated Grid Suite.

It can inspect Grid Suite output or other BOUT++ 4.x/5.x-style grid files, with the strongest assumptions and visualisations aimed at tokamak-style geometry.

If no grid path is supplied, the tool attempts to open:

```text
grid.nc
```

from the current directory.

## Output-control options

```bash
# Chosen output directory
python src/bout_tokamak_grid_diagnostics.py grid.nc --outdir diagnostics_out

# No PNG plots
python src/bout_tokamak_grid_diagnostics.py grid.nc --no-plots

# No HTML report; JSON is still written
python src/bout_tokamak_grid_diagnostics.py grid.nc --no-html

# JSON only
python src/bout_tokamak_grid_diagnostics.py grid.nc --json-only
```

| Option | Description |
| --- | --- |
| `grid` | Optional positional path; defaults to `grid.nc` |
| `--version` | Print diagnostics component version |
| `--outdir PATH` | Output directory; defaults to the grid file's directory |
| `--no-plots` | Disable PNG generation |
| `--no-html` | Disable HTML generation |
| `--json-only` | JSON only; implies `--no-html` and `--no-plots` |
| `--strict` | Treat warnings as a failing verdict/exit code |

## Diagnostic threshold controls

The diagnostic policy can be tuned from the CLI:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--det-relerr-mean-max FLOAT` | `1e-3` | Mean determinant-consistency threshold |
| `--det-relerr-max-max FLOAT` | `1e-2` | Maximum determinant-consistency threshold |
| `--metric-cond-warn FLOAT` | `1e6` | Warn when metric conditioning ratio reaches/exceeds this value |
| `--jacobian-singularity-min FLOAT` | `1e-10` | Critical threshold for minimum boundary `|J|` |
| `--aspect-ratio-warn-min FLOAT` | `2.0` | Warn below this inferred aspect ratio |
| `--ny-warn-min INT` | `32` | Warn below this poloidal resolution |

The determinant check becomes critical when both the configured mean and maximum conditions are exceeded.

## Large-grid / chunking controls

The diagnostics utility includes a memory heuristic and can use dask-backed chunking for large datasets.

```bash
# Force chunked loading
python src/bout_tokamak_grid_diagnostics.py grid.nc --force-chunk

# Supply explicit chunk sizes
python src/bout_tokamak_grid_diagnostics.py grid.nc \
    --force-chunk \
    --chunks "x=256,y=256,z=1"
```

| Option | Description |
| --- | --- |
| `--force-chunk` | Force dask chunked opening regardless of the internal memory heuristic |
| `--chunks SPEC` | Explicit comma-separated dimension chunk sizes |

Only dimensions present in the dataset are used.

## Logging controls

```bash
# Warnings/errors only
python src/bout_tokamak_grid_diagnostics.py grid.nc --quiet

# Debug output
python src/bout_tokamak_grid_diagnostics.py grid.nc --verbose

# Also write to a log file
python src/bout_tokamak_grid_diagnostics.py grid.nc --log-file diagnostics.log
```

| Option | Description |
| --- | --- |
| `--quiet` | Only show warnings/errors |
| `--verbose` | Enable DEBUG-level output |
| `--log-file PATH` | Write log output to a file |

## Strict mode

Warnings normally do not invalidate a grid.

For CI or conservative regression gates:

```bash
python src/bout_tokamak_grid_diagnostics.py grid.nc --strict
```

Strict mode returns a failure exit code when warnings exist.

## Exit codes

The diagnostics script uses an explicit exit-code contract:

| Code | Meaning |
| ---: | --- |
| `0` | No critical validation failures |
| `1` | Critical validation failure, invalid invocation/input, or warning under `--strict` |
| `2` | Partial/unsupported grid structure where the available report can still be written |

This makes the diagnostics utility suitable for batch processing and CI.

---

# What the diagnostics check

## Basic structure

- grid dimensions
- required variables
- optional variables
- 2D/3D dimensionality
- BOUT++-style format inference

## Geometry

- radial ordering
- outboard radial monotonicity
- geometry descriptors
- custom magnetic-axis consistency
- requested/generated boundary fidelity for custom grids

Custom grids are not required to be vertically symmetric around global `Z=0`.

## Metric tensor

- NaN/Inf checks
- positive diagonal metric components
- leading 2×2 principal minor
- metric conditioning

## Jacobian

- sign consistency
- boundary singularity threshold

## Metric determinant and BOUT++ contract

The diagnostics utility retains its forensic convention detection because historical and third-party grids sometimes use reversed metric names. It compares the un-underscored tensor against both `J²` and `1/J²` to identify the mathematical convention actually present.

That forensic interpretation is reported separately from BOUT++ compatibility. BOUT++ 5.x has fixed naming semantics:

- `g11..g23` are contravariant `g^{ij}`
- `g_11..g_23` are covariant `g_ij`

The diagnostics therefore also reconstructs the Jacobian from the covariant tensor, checks the BOUT++ loader relation `Bxy = sqrt(g_22) / J`, verifies that the convention-appropriate covariant toroidal component satisfies `g_zz = R^2` when the toroidal coordinate is in radians, and checks for finite `ShiftTorsion` in grids that claim the canonical Grid Suite BOUT++ 5.x contract.

A reversed-convention grid can still be mathematically self-consistent and useful for forensic analysis, but it is labelled **NON-CANONICAL for BOUT++ 5.x**. BOUT++ may open such a file and accept supplied `J` and `Bxy`, while still interpreting the metric variable names according to its canonical convention. Differential operators can therefore use the wrong geometry even though the run starts. Such a grid must not be treated as BOUT++ 5.x valid.

## `surfvol`

When available:

- sign consistency
- sign agreement with Jacobian
- finite `surfvol/J`

## Magnetic field

When available:

- positive `Bxy`
- min/max field statistics
- poloidal-field visualisation using `Bpxy`

## Field-line shift

For canonical BOUT++ 5.x grids the diagnostics recognises:

- `zShift(x,y)` as the accumulated local field-line pitch
- `ShiftAngle(x)` as the complete poloidal-circuit shift used by twist-shift matching
- `ShiftTorsion(x,y)` as `partial(nu)/partial(x)`, required by BOUT++ for vector differential operators
- `IntShiftTorsion(x,y)` as the integrated shear `partial(zShift)/partial(x)`

Historical files containing a 2D `shiftAngle`-style field are still readable for forensic purposes, but are not presented as the canonical BOUT++ 5.x field-alignment contract.

## Reports

Normal diagnostics output can include:

```text
<grid>_report.html
<grid>_report.json
<grid>_render.png
<grid>_bfield.png
```

The JSON report contains:

- dimensions
- geometry model
- inferred grid/configuration type
- `critical_failures`
- `warnings`
- structured check results with severity and details
- metric/Jacobian/magnetic min/max values
- determinant-consistency metadata
- custom requested-boundary fidelity values when available

---

# HPC / CI note: Matplotlib cache

The diagnostics utility uses Matplotlib's non-interactive backend for report images.

On HPC systems, CI runners or locked-down accounts where the default Matplotlib cache directory is not writable, set `MPLCONFIGDIR` to a writable location:

```bash
export MPLCONFIGDIR=/tmp/$USER-mpl
```

or:

```bash
export MPLCONFIGDIR=$PWD/.mpl-cache
```

---

# Repository layout

```text
bout-grid-suite/
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   ├── bout_tokamak_grid_generator.py
│   └── bout_tokamak_grid_diagnostics.py
├── tests/
│   ├── data/
│   │   └── grids/
│   ├── conftest.py
│   ├── test_bout5_contract.py
│   ├── test_custom_boundary.py
│   ├── test_designer_smoke.py
│   ├── test_diagnostics_cli.py
│   ├── test_generator_cli.py
│   └── test_pipeline.py
├── bout_grid_designer.py
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

Runtime-created directories are not tracked by Git:

```text
output/
reports/
saves/
```

---

# Validation status

Grid Suite v2.0.0 has automated regression coverage for:

- known-good diagnostic fixtures
- deliberately invalid Jacobian and metric fixtures
- generator CLI execution
- NetCDF dimensional/variable contract
- generator-to-diagnostics pipeline execution
- Grid Studio import and CLI smoke behaviour
- Windows Unicode/logging behaviour
- canonical preset state and generation values
- custom-boundary validation
- custom-boundary preservation
- requested/generated boundary fidelity
- custom magnetic-axis handling
- warning/report consistency

The v2.0.0 suite passes 17 automated tests in the release candidate used for this public version.

The project does not currently run a full BOUT++ solver execution as part of its GitHub Actions test matrix.

The v2.0.0 release candidate was additionally validated manually against BOUT++ 5.2.x. BOUT++ loaded the generated metric tensors, Jacobian, magnetic field, `zShift`, `ShiftAngle` and `ShiftTorsion`; its independent grid-loader checks reported a maximum absolute Jacobian difference of approximately `3.2e-13` and a `Bxy` difference of `0`. The grid was then used with twist-shift, the `shiftedinterp` parallel transform and a PETSc 3D AMG Laplacian, and the simulation successfully initialised and advanced through multiple timesteps.

---

# Known limitations

- Grid Suite v2 is an analytic/synthetic grid system, not an equilibrium reconstruction package.
- No GEQDSK import or export is included in v2.
- No Grad-Shafranov solve is performed.
- No automatic O-point/X-point equilibrium reconstruction is performed.
- Diverted single-null/double-null topology generation is not included.
- Custom boundaries must be suitable for a star-shaped radial mapping about the inferred magnetic axis.
- Very strong non-convexity can produce poor cell quality even when the boundary remains mathematically generatable.
- The optional toroidal-ripple mode is a controlled synthetic perturbation, not a physical equilibrium model.
- Some diagnostics use tokamak-centric geometry heuristics and should be treated qualitatively for strongly non-standard or genuinely 3D grids.

---

# Project history

The generator and diagnostics components originated as independent public tools in the Chatwood Labs [`bout-tools`](https://github.com/Chatwood-Labs/bout-tools) repository:

- [Axisymmetric Tokamak Grid Generator](https://github.com/Chatwood-Labs/bout-tools/tree/main/bout_tokamak_grid)
- [BOUT++ Tokamak Grid Diagnostics Utility](https://github.com/Chatwood-Labs/bout-tools/tree/main/bout_tokamak_grid_diagnostics)

Grid Suite v2.0.0 supersedes those standalone v1 tools as the actively developed integrated application:

```text
bout-tools
    ↓
Axisymmetric Grid Generator
Grid Diagnostics Utility
    ↓
Grid Suite v2
    ├── Grid Studio
    ├── Miller analytic generation
    ├── Custom-boundary generation
    └── Integrated diagnostics
```

The original [`bout-tools`](https://github.com/Chatwood-Labs/bout-tools) repository remains available as historical provenance and for users who need the standalone v1 releases, but it is no longer actively maintained. New development and fixes are made in Grid Suite.

---

# Version 2.0.0 highlights

- introduced the integrated Grid Studio desktop interface
- retained the generator and diagnostics direct command-line workflows
- added canonical Miller presets
- added direct custom-boundary generation
- added custom-geometry advisory and hard-block validation
- added requested-boundary provenance and fidelity checking
- added interactive 3D toroidal preview
- integrated generation, diagnostics and report viewing
- added save/load design workflow
- added cross-platform regression tests and GitHub Actions
- improved Windows Unicode/logging behaviour
- improved geometry and radial-ordering diagnostics
- added canonical BOUT++ 5.x covariant/contravariant metric output
- added field-alignment metadata including `zShift`, `ShiftAngle` and `ShiftTorsion`
- added explicit closed-core/no-X-point topology metadata
- validated canonical output in an actual BOUT++ 5.2.x solver run
- closed periodic seams in report visualisations

---

# Citation

Citation metadata is provided in:

```text
CITATION.cff
```

If Grid Suite is used in published work, please cite the software using that metadata.

---

# License

Released under the MIT License.

Copyright © 2025-2026 Chatwood Labs Ltd.

---

# Contributing

Issues and pull requests are welcome.

For substantial feature proposals or structural changes, please open an issue first so the design can be discussed before implementation.
