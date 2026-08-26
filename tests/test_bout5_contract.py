from __future__ import annotations

import numpy as np

import bout_tokamak_grid_generator as gen


def _build_contract(kappa: float, delta: float):
    coords = gen.generate_coordinates(68, 64, 8, 0.2, 2.0)
    geom = gen.compute_geometry(coords["x"], coords["y"], 6.2, kappa, delta)
    geom["geometry_model"] = "miller"
    q = gen.compute_q_profile(coords["x"], 2.0, 1.05, 3.5, "quadratic")
    basis = gen.compute_basis_vectors(
        geom,
        coords["z3"][:1, :1, :],
        theta_1d=coords["y"],
        ripple={"enabled": False, "eps": 0.0, "N": 8, "M": 0},
    )
    bout = gen.compute_bout_field_aligned_system(basis, geom, coords, q, 5.3, 6.2)
    return q, bout


def test_bout5_field_aligned_contract_across_miller_shapes() -> None:
    for kappa, delta in ((1.0, 0.0), (1.7, 0.0), (1.7, 0.33), (1.7, -0.33)):
        q, bout = _build_contract(kappa, delta)
        cov = bout["metric"]["g_cov"]
        contra = bout["metric"]["g_contra"]
        J = bout["J"]
        Bxy = bout["bfield"]["Bxy"]

        # Requested safety factor is the complete field-line pitch integral.
        assert np.allclose(bout["q_actual"], q, rtol=1e-11, atol=1e-12)
        assert np.allclose(bout["ShiftAngle"] / (2.0 * np.pi), q, rtol=1e-11, atol=1e-12)

        # Metric tensors are genuine inverses.
        ident = np.einsum("...ik,...kj->...ij", contra, cov)
        assert np.allclose(ident, np.eye(3), rtol=1e-9, atol=2e-10)

        # These reproduce the exact checks performed by the BOUT++ 5.x loader.
        Jcalc = np.sqrt(np.abs(np.linalg.det(cov)))
        Bcalc = np.sqrt(np.abs(cov[..., 1, 1])) / J
        assert np.allclose(Jcalc, J, rtol=1e-8, atol=1e-10)
        assert np.allclose(Bcalc, Bxy, rtol=1e-10, atol=1e-12)

        # The covariant toroidal basis vector is the geometric toroidal
        # direction, so g_zz is the squared major radius.
        ez_mag2 = (
            bout["basis"]["ephi_X"] ** 2
            + bout["basis"]["ephi_Y"] ** 2
            + bout["basis"]["ephi_Z"] ** 2
        )
        assert np.allclose(cov[..., 2, 2], ez_mag2, rtol=1e-11, atol=1e-12)

        assert np.all(J > 0.0)
        assert np.all(Bxy > 0.0)
        assert np.all(bout["bfield"]["Bpxy"] > 0.0)


def test_bout5_shift_contract_has_canonical_shapes() -> None:
    q, bout = _build_contract(1.7, 0.33)

    assert bout["zShift"].shape == (68, 64, 8)
    assert bout["ShiftAngle"].shape == (68,)
    assert bout["ShiftTorsion"].shape == (68, 64, 8)
    assert bout["IntShiftTorsion"].shape == (68, 64, 8)
    assert np.all(np.isfinite(bout["ShiftTorsion"]))

    # Integrated shear is the poloidal integral of the local torsion.
    integrated, _ = gen._periodic_cumulative_trapezoid(
        bout["ShiftTorsion"][:, :, 0],
        np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False),
    )
    assert np.allclose(
        integrated,
        bout["IntShiftTorsion"][:, :, 0],
        rtol=1e-8,
        atol=2e-10,
    )

    assert bout["psi"].shape == (68,)
    assert np.all(np.diff(bout["psi"]) > 0.0)
    assert np.all(bout["dx_1d"] > 0.0)
    assert np.max(np.abs(bout["q_actual"] - q)) < 1e-10
