#!/usr/bin/env python3
__version__ = "2.0.0"

import numpy as np
import argparse
import time
import warnings
import json
from pathlib import Path

from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from netCDF4 import Dataset

# netCDF4 1.7.4 triggers a NumPy 2.5 deprecation warning internally when
# assigning array-shaped data.  Suppress only that exact upstream warning;
# do not hide unrelated deprecations from Grid Suite itself.
warnings.filterwarnings(
    "ignore",
    message=r"Setting the shape on a NumPy array has been deprecated in NumPy 2\.5\.",
    category=DeprecationWarning,
)

# ======================================================================
#  Chatwood Labs - Axisymmetric Tokamak Grid Generator (v2.0.0, BOUT++ 5.x Compatible)
#
#  bout_tokamak_grid_generator.py
#
#  This script generates an axisymmetric tokamak geometry grid compatible
#  with BOUT++ 5.x. Supports Miller-style analytic shaping and validated
#  custom poloidal boundaries. Produces full
#  coordinate mapping, metric tensor, Jacobian, magnetic geometry, and
#  canonical BOUT++ zShift / ShiftAngle field-alignment metadata.
#
#  License & Usage:
#      - Released under the MIT License (see repository LICENSE file).
#      - Free to use, modify, and redistribute under MIT terms.
#      - Attribution appreciated but not required.
#
#  This software is provided "as is", without warranty of any kind,
#  express or implied, including but not limited to the warranties of
#  merchantability, fitness for a particular purpose and non-infringement.
#  In no event shall the authors or copyright holders be liable for any
#  claim, damages or other liability arising from the software or its use.
#
#  © 2025-2026 Chatwood Labs Ltd
# ======================================================================


def parse_arguments():
    """
    Parse command-line arguments for tokamak grid generation.
    
    Returns all geometric, magnetic, and numerical setup values.
    Each parameter has a safe default so the script can be run with
    no arguments and produce the same D-Shape geometry and original
    q-profile as before.
    
    Example usage:
      python3 bout_tokamak_grid_generator.py \
          --kappa 1.7 \
          --delta 0.33 \
          --qform cubic \
          --q0 1.05 --qa 4.0 \
          --nx 128 --ny 128
    """
    parser = argparse.ArgumentParser(
        description="Generate an axisymmetric BOUT++ tokamak grid from Miller parameters or a custom boundary"
    )

    parser.add_argument("--R0", type=float, default=6.2,
                        help="Major radius (m)")
    parser.add_argument("--a", type=float, default=2.0,
                        help="Minor radius (m)")
    parser.add_argument("--B0", type=float, default=5.3,
                        help="Toroidal field at R0 (T)")

    parser.add_argument("--xmin_frac", type=float, default=0.1,
                        help="Fraction of minor radius to avoid r=0 singularity (default: 0.1 = 10%% of a)")

    parser.add_argument("--kappa", type=float, default=1.7,
                        help="Elongation (default: 1.7, ITER-ish baseline)")
    parser.add_argument("--delta", type=float, default=0.33,
                        help="Triangularity (default: 0.33, ITER-ish baseline)")
    parser.add_argument("--boundary-file", type=str, default=None,
                        help="JSON custom poloidal boundary. When supplied, the actual boundary "
                             "is used to construct nested flux surfaces instead of Miller shaping.")

    parser.add_argument("--q0", type=float, default=1.05,
                        help="Central safety factor (default: 1.05)")
    parser.add_argument("--qa", type=float, default=3.5,
                        help="Edge safety factor at r=a (default: 3.5)")
    parser.add_argument("--qform", type=str, default="quadratic",
                        choices=["quadratic", "linear", "cubic"],
                        help="Functional form of q-profile: quadratic (default), linear, or cubic")

    parser.add_argument("--nx", type=int, default=64,
                        help="Radial grid points")
    parser.add_argument("--ny", type=int, default=64,
                        help="Poloidal grid points")
    parser.add_argument("--nz", type=int, default=128,
                        help="Toroidal grid points")

    parser.add_argument("--precision", choices=["f4", "f8"], default="f8",
                        help="Floating point precision (f4 or f8)")
    parser.add_argument("--outfile", type=str, default="grid.nc",
                        help="Output netCDF filename")

    parser.add_argument("--curvature", type=str, default="exact",
                        choices=["exact", "simple", "none"],
                        help="Curvature model: exact (full tensor), simple (geometric approx), or none")
    
    parser.add_argument("--metrics-3d", action="store_true",
                        help="Output 3D metric tensors (x,y,z). Default is 2D (x,y) for axisymmetric tokamak")

    # ------------------------------------------------------------------
    # Toroidal ripple (3D geometry) options
    #
    # Purpose:
    #   Axisymmetric grids often look 'too well behaved' for z-only tests
    #   because nothing depends on z/phi. Adding a controlled, small ripple
    #   makes R and/or Z depend on toroidal angle, creating z-dependent metrics
    #   (g33, g13, g23) that z-advection must actually deal with.
    #
    # Model:
    #   R_phys = R0(r,theta) * [1 + eps * cos(N*phi + M*theta)]
    #   Z_phys = Z0(r,theta) * [1 + eps * cos(N*phi + M*theta)]
    #
    # Notes:
    #   - eps should be small (1e-3 .. few e-2) to avoid invalid grids.
    #   - Use N ~ 6..16 for noticeable toroidal variation.
    #   - M adds a helical component (optional, default 0 gives pure toroidal ripple).
    # ------------------------------------------------------------------
    parser.add_argument("--toroidal-ripple", action="store_true",
                        help="Enable 3D toroidal ripple: R and Z become functions of phi (z). "
                             "This makes zblob much more sensitive to geometry.")

    parser.add_argument("--ripple-eps", type=float, default=0.0,
                        help="Ripple amplitude eps (fractional). Typical: 0.002 to 0.02. "
                             "Only used when --toroidal-ripple is enabled.")

    parser.add_argument("--ripple-n", type=int, default=8,
                        help="Toroidal ripple mode number N in cos(N*phi + M*theta). "
                             "Only used when --toroidal-ripple is enabled.")

    parser.add_argument("--ripple-m", type=int, default=0,
                        help="Poloidal/helical ripple mode number M in cos(N*phi + M*theta). "
                             "Set M!=0 for helical ripple. Only used when --toroidal-ripple is enabled.")

    return parser.parse_args()


def validate_shaping_parameters(kappa, delta):
    """
    Validate elongation and triangularity parameters.
    
    Early sanity checks on extreme shaping to prevent singular grids.
    Raises ValueError for unphysical configurations.
    """
    if kappa <= 0:
        raise ValueError("kappa must be > 0. Elongation cannot be zero or negative.")

    if kappa > 3.0:
        print("WARNING: kappa > 3.0 is extremely elongated and likely to produce a singular grid.")

    if abs(delta) > 0.6:
        print("WARNING: |delta| > 0.6 creates extreme triangularity and may cause Jacobian failure.")

    if kappa > 5.0 or abs(delta) > 0.9:
        raise ValueError("Unphysical shaping parameters: geometry will be singular. Reduce kappa/delta.")


def generate_coordinates(nx, ny, nz, xmin, a):
    """
    Generate coordinate arrays for tokamak grid.
    
    Parameters:
        nx, ny, nz: Grid dimensions
        xmin: Inner radial cutoff (to avoid r=0 singularity)
        a: Minor radius
    
    Returns:
        dict with keys: x, y, z, x3, y3, z3, dr, dtheta, dphi
        
    CRITICAL: y uses endpoint=False for periodic poloidal sampling
    """
    x = np.linspace(xmin, a, nx)
    y = np.linspace(0, 2*np.pi, ny, endpoint=False)  # Non-periodic coordinate for monotonic shift

    # Toroidal angle φ must be periodic WITHOUT duplicating endpoints.
    # endpoint=False avoids having both 0 and 2π in the mesh.
    z = np.linspace(0.0, 2.0*np.pi, nz, endpoint=False)

    # Fully broadcast coordinate arrays (no degenerate dimensions)
    x3 = np.broadcast_to(x[:, None, None], (nx, ny, nz))
    y3 = np.broadcast_to(y[None, :, None], (nx, ny, nz))
    z3 = np.broadcast_to(z[None, None, :], (nx, ny, nz))

    # Compute differentials (coordinate spacings)
    dr = x[1] - x[0]
    dtheta = y[1] - y[0]
    dphi = z[1] - z[0]


    return {
        'x': x, 'y': y, 'z': z,
        'x3': x3, 'y3': y3, 'z3': z3,
        'dr': dr, 'dtheta': dtheta, 'dphi': dphi
    }


def compute_q_profile(x, a, q0, qa, qform):
    """
    Compute safety factor q-profile on radial grid.
    
    Parameters:
        x: Radial coordinate array (nx, )
        a: Minor radius
        q0: Central safety factor
        qa: Edge safety factor
        qform: Profile type ("quadratic", "linear", "cubic")
    
    Returns:
        q_vals: Safety factor array (nx)
    
    The q-profile controls magnetic shear and is used throughout
    metric, B-field, and field-line pitch calculations.
    """
    s = x / a  #normalized radius

    if qform == "quadratic":
        #Original behaviour: q = q0 + (qa - q0) * s^2
        q_vals = q0 + (qa - q0) * s**2

    elif qform == "linear":
        #Linear shear: q = q0 + (qa - q0) * s
        q_vals = q0 + (qa - q0) * s

    elif qform == "cubic":
        #Smooth central behavior: q = q0 + (qa - q0) * s^3
        q_vals = q0 + (qa - q0) * s**3

    else:
        raise ValueError(f"Unknown q-profile type '{qform}'")

    #Safety checks
    if np.any(~np.isfinite(q_vals)):
        raise RuntimeError("Safety factor q-profile contains non-finite values.")

    if np.any(q_vals <= 0):
        raise RuntimeError("Safety factor q-profile is non-positive - invalid for tokamak configuration.")

    if not np.all(np.diff(q_vals) >= -1e-12):
        print("WARNING: q-profile is not monotonic. This is allowed but may cause shear inversion or magnetic wells.")

    return q_vals



def _cross2(ax, ay, bx, by):
    return ax*by - ay*bx


def _polygon_area_centroid(R, Z):
    R=np.asarray(R,float); Z=np.asarray(Z,float)
    Rc=np.append(R,R[0]); Zc=np.append(Z,Z[0])
    cross=Rc[:-1]*Zc[1:]-Rc[1:]*Zc[:-1]
    area=0.5*np.sum(cross)
    if abs(area)<1e-15:
        return float(area),float(np.mean(R)),float(np.mean(Z))
    cx=float(np.sum((Rc[:-1]+Rc[1:])*cross)/(6*area))
    cy=float(np.sum((Zc[:-1]+Zc[1:])*cross)/(6*area))
    return float(area),cx,cy


def _point_in_polygon(px, py, R, Z):
    inside=False; n=len(R); j=n-1
    for i in range(n):
        xi,yi=float(R[i]),float(Z[i]); xj,yj=float(R[j]),float(Z[j])
        if ((yi>py)!=(yj>py)):
            xhit=(xj-xi)*(py-yi)/(yj-yi)+xi
            if px<xhit: inside=not inside
        j=i
    return inside


def _segments_intersect(p1,p2,q1,q2,tol=1e-10):
    p1=np.asarray(p1,float); p2=np.asarray(p2,float)
    q1=np.asarray(q1,float); q2=np.asarray(q2,float)
    r=p2-p1; e=q2-q1; qp=q1-p1
    den=_cross2(r[0],r[1],e[0],e[1])
    if abs(den)<=tol:
        if abs(_cross2(qp[0],qp[1],r[0],r[1]))>tol:
            return False
        rr=float(np.dot(r,r))
        if rr<=tol: return False
        t0=float(np.dot(q1-p1,r)/rr); t1=float(np.dot(q2-p1,r)/rr)
        lo=max(min(t0,t1),0.0); hi=min(max(t0,t1),1.0)
        return hi-lo>tol
    t=_cross2(qp[0],qp[1],e[0],e[1])/den
    u=_cross2(qp[0],qp[1],r[0],r[1])/den
    return (-tol<=t<=1+tol) and (-tol<=u<=1+tol)


def _ray_intersections(axis_R,axis_Z,theta,R,Z,tol=1e-9):
    dx,dy=np.cos(theta),np.sin(theta)
    hits=[]; n=len(R)
    for i in range(n):
        x1,y1=float(R[i]),float(Z[i]); x2,y2=float(R[(i+1)%n]),float(Z[(i+1)%n])
        ex,ey=x2-x1,y2-y1
        den=_cross2(dx,dy,ex,ey)
        if abs(den)<=tol: continue
        qx,qy=x1-axis_R,y1-axis_Z
        t=_cross2(qx,qy,ex,ey)/den
        u=_cross2(qx,qy,dx,dy)/den
        if t>tol and -tol<=u<=1+tol:
            hits.append(float(t))
    if not hits: return []
    hits.sort(); unique=[hits[0]]
    merge_tol=max(tol,1e-7*max(1.0,hits[-1]))
    for value in hits[1:]:
        if abs(value-unique[-1])>merge_tol:
            unique.append(value)
    return unique


def load_custom_boundary(path):
    """Load and validate a custom poloidal boundary JSON file."""
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    R=np.asarray(payload.get("R",[]),float)
    Z=np.asarray(payload.get("Z",[]),float)
    if len(R)!=len(Z) or len(R)<8:
        raise ValueError("Custom boundary must contain at least 8 R/Z points.")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
        raise ValueError("Custom boundary contains NaN or infinite coordinates.")
    if np.any(R<=0.0):
        raise ValueError("Custom boundary reaches R <= 0 and cannot form a tokamak surface.")
    # Remove duplicated closing sample if a caller supplied one.
    if len(R)>8 and np.hypot(R[-1]-R[0],Z[-1]-Z[0])<1e-10:
        R=R[:-1]; Z=Z[:-1]
    area,cx,cy=_polygon_area_centroid(R,Z)
    scale=max(float(np.ptp(R)),float(np.ptp(Z)),1.0)
    if abs(area)<1e-8*scale*scale:
        raise ValueError("Custom boundary encloses negligible area.")
    axis_R=float(payload.get("axis_R",cx)); axis_Z=float(payload.get("axis_Z",cy))
    if not _point_in_polygon(axis_R,axis_Z,R,Z):
        raise ValueError("Custom-boundary magnetic axis lies outside the boundary.")
    # Simple polygon check.
    tol=1e-10*scale; n=len(R)
    for i in range(n):
        p1=(R[i],Z[i]); p2=(R[(i+1)%n],Z[(i+1)%n])
        for j in range(i+1,n):
            if j==i or j==(i+1)%n or (j+1)%n==i: continue
            if _segments_intersect(p1,p2,(R[j],Z[j]),(R[(j+1)%n],Z[(j+1)%n]),tol):
                raise ValueError("Custom boundary self-intersects; cannot construct nested surfaces.")
    return {
        "R":R,"Z":Z,"axis_R":axis_R,"axis_Z":axis_Z,
        "payload":payload,"scale":scale,
    }


def _build_radial_boundary(boundary, n_dense=1024):
    """Return a periodic spline s(theta) for a star-shaped custom boundary."""
    R=boundary["R"]; Z=boundary["Z"]
    axis_R=boundary["axis_R"]; axis_Z=boundary["axis_Z"]
    theta=np.linspace(0.0,2*np.pi,int(n_dense),endpoint=False)
    radii=np.empty_like(theta)
    tol=1e-10*boundary["scale"]
    for i,ang in enumerate(theta):
        hits=_ray_intersections(axis_R,axis_Z,float(ang),R,Z,tol=tol)
        if len(hits)!=1:
            raise ValueError(
                "Custom boundary is not single-valued about the magnetic axis; "
                f"ray theta={ang:.4f} rad has {len(hits)} intersections. Smooth or redraw it."
            )
        radii[i]=hits[0]
    if np.nanmin(radii)<1e-4*boundary["scale"]:
        raise ValueError("Custom boundary approaches the magnetic axis too closely.")
    theta_closed=np.append(theta,2*np.pi)
    radius_closed=np.append(radii,radii[0])
    return CubicSpline(theta_closed,radius_closed,bc_type="periodic")


def compute_custom_geometry(x,y,a,boundary):
    """Construct nested surfaces that preserve the user's requested boundary."""
    nx=len(x); ny=len(y)
    spline=_build_radial_boundary(boundary,n_dense=max(1024,ny*16))
    s=np.asarray(spline(y),float)
    ds=np.asarray(spline(y,1),float)
    c=np.cos(y); si=np.sin(y)
    axis_R=float(boundary["axis_R"]); axis_Z=float(boundary["axis_Z"])
    Rb=axis_R+s*c; Zb=axis_Z+s*si
    dRb=ds*c-s*si; dZb=ds*si+s*c

    r_mesh=x[:,None,None]
    theta_mesh=y[None,:,None]
    rho=(x/max(float(a),1e-12))[:,None,None]
    R_vals=axis_R+rho*(Rb[None,:,None]-axis_R)
    Z_vals=axis_Z+rho*(Zb[None,:,None]-axis_Z)
    dR_dr=np.broadcast_to(((Rb-axis_R)/a)[None,:,None],(nx,ny,1))
    dZ_dr=np.broadcast_to(((Zb-axis_Z)/a)[None,:,None],(nx,ny,1))
    dR_dtheta=rho*dRb[None,:,None]
    dZ_dtheta=rho*dZb[None,:,None]
    if np.any(~np.isfinite(R_vals)) or np.any(~np.isfinite(Z_vals)):
        raise RuntimeError("Non-finite values found in custom geometry coordinates.")
    return {
        "R_vals":R_vals,"Z_vals":Z_vals,
        "theta_mesh":theta_mesh,"theta_tilde":theta_mesh,
        "dR_dr":dR_dr,"dR_dtheta":dR_dtheta,
        "dZ_dr":dZ_dr,"dZ_dtheta":dZ_dtheta,
        "r_mesh":r_mesh,
        "geometry_model":"custom_radial_boundary",
        "axis_R":axis_R,"axis_Z":axis_Z,
        "boundary_R":Rb,"boundary_Z":Zb,
        "source_boundary_R":np.asarray(boundary["R"],float),
        "source_boundary_Z":np.asarray(boundary["Z"],float),
    }


def compute_geometry(x, y, R0, kappa, delta):
    """
    Compute tokamak geometry: R, Z, shaped poloidal angle, and derivatives.
    
    Parameters:
        x: Radial grid (nx)
        y: Poloidal grid (ny)
        R0: Major radius
        kappa: Elongation
        delta: Triangularity
    
    Returns:
        dict with keys: R_vals, Z_vals, theta_tilde, dR_dr, dR_dtheta, dZ_dr, dZ_dtheta
        All arrays shape (nx, ny, 1) for broadcasting
    
    NOTE ON GEOMETRY APPROXIMATION - R0 AS CONSTANT MAJOR RADIUS:
    
    This grid generator assumes a large-aspect-ratio tokamak where the
    major radius R0 is treated as a constant reference value. This is the
    standard approximation used in analytic geometry models and is fully
    consistent with the BOUT++ workflow.
    
    Why this is "non-full-Grad-Shafranov":
        In a true MHD equilibrium, R0, shape, and B-fields all emerge from
        solving the Grad-Shafranov equation. That produces:
            - poloidally varying R shifts
            - pressure-driven Shafranov shift
            - flux-surface-dependent metrics
    
    This code  instead uses:
            R = R0 + r*cos(theta_tilde)
    which is the canonical large-aspect-ratio approximation. It captures:
            - shaping (κ, δ)
            - correct poloidal variation of R
            - physically meaningful metrics and B-fields
    but does notmodel:
            - Shafranov shift
            - self-consistent equilibrium from pressure/current profiles
    
    Why this doesn't break physics:
        Because all metric elements, B-fields, Jacobian, and curvature
        are computed directly from the R,Z coordinates you define here.
        As long as the geometry is smooth and monotonic, BOUT++ doesn't
        care whether the surfaces came from a GS solver or an analytic form.
    
    Future upgrade note:
        A future version of this generator can include a "GS mode" where
        equilibrium R(r,θ) and Z(r,θ) are loaded from an EFIT or HELENA
        equilibrium file. The rest of the code already supports arbitrary
        coordinate maps,  you just need to replace the analytic R,Z here
        with data from a full Grad-Shafranov solution.
    
    Translation for the annoyed plasma theorist:
        This is a large-aspect-ratio analytic tokamak, not a full GS
        equilibrium. The physics is consistent; the approximations are
        intentional. If you want EFIT, bring your own damn equilibrium file.
    
    NOTE ON TRIANGULARITY MODEL (delta):

    Uses the standard EFIT / Miller parametrisation:
        R = R0 + r * cos(theta + arcsin(delta)*sin(theta))
        Z = kappa * r * sin(theta)

    The triangularity shift only appears in R; Z uses plain theta.
    This ensures max(Z) occurs at theta=pi/2, giving:
        R_top = R0 - r * delta   (correct Miller triangularity definition)

    Using theta_tilde in both R AND Z (the previous implementation) was incorrect:
    it caused max(Z) at theta_tilde=pi/2, forcing R_top=R0 regardless of delta,
    which produced effectively zero triangularity in all generated grids.
    """
    nx = len(x)
    ny = len(y)

    #Mesh coordinates for broadcasting
    r_mesh = x[:, None, None]      #(nx, 1, 1)
    theta_mesh = y[None, :, None]  #(1, ny, 1)

    # Standard Miller / EFIT parametrisation:
    #   R = R0 + r * cos(theta + arcsin(delta)*sin(theta))   <- theta_tilde in R
    #   Z = kappa * r * sin(theta)                            <- plain theta in Z
    #
    # This is the correct form. Using theta_tilde in BOTH R and Z (the previous
    # implementation) cancels the triangularity: max(Z) occurs at theta_tilde=pi/2
    # which forces R_top = R0 + r*cos(pi/2) = R0, giving zero effective delta.
    #
    # With the correct form, max(Z) occurs at theta=pi/2, so:
    #   R_top = R0 + r*cos(pi/2 + arcsin(delta)) = R0 - r*delta
    # which is the standard triangularity definition.
    delta_param = np.arcsin(np.clip(delta, -0.999, 0.999))
    theta_tilde = theta_mesh + delta_param * np.sin(theta_mesh)

    #Geometry
    R_vals = R0 + r_mesh * np.cos(theta_tilde)
    Z_vals = kappa * r_mesh * np.sin(theta_mesh)   # plain theta, not theta_tilde

    # Derivatives for metric tensor
    # dR/dr = cos(theta_tilde)                      (unchanged)
    # dR/dtheta = -r * sin(theta_tilde) * dtheta_tilde/dtheta
    # dZ/dr = kappa * sin(theta)                    (plain theta)
    # dZ/dtheta = kappa * r * cos(theta)            (plain theta, no tilde)
    dtheta_tilde_dtheta = 1 + delta_param * np.cos(theta_mesh)
    dR_dr = np.cos(theta_tilde)
    dR_dtheta = -r_mesh * np.sin(theta_tilde) * dtheta_tilde_dtheta

    dZ_dr = kappa * np.sin(theta_mesh)
    dZ_dtheta = kappa * r_mesh * np.cos(theta_mesh)

    #Geometry sanity checks
    if np.any(~np.isfinite(R_vals)) or np.any(~np.isfinite(Z_vals)):
        raise RuntimeError("Non-finite values found in geometry coordinates R or Z.")

    return {
        'R_vals': R_vals,
        'Z_vals': Z_vals,
        'theta_mesh': theta_mesh,
        'theta_tilde': theta_tilde,
        'dR_dr': dR_dr,
        'dR_dtheta': dR_dtheta,
        'dZ_dr': dZ_dr,
        'dZ_dtheta': dZ_dtheta,
        'r_mesh': r_mesh
    }


def compute_basis_vectors(geom, phi_mesh, theta_1d=None, ripple=None):
    """
    Compute coordinate basis vectors e_r, e_theta, e_phi in Cartesian space.

    This function supports:
      1) Original axisymmetric mapping (default):
           R = R(r,theta), Z = Z(r,theta)
           => dR/dphi = dZ/dphi = 0
      2) Optional 3D toroidal/helical ripple:
           R_phys = R0 * [1 + eps*cos(N*phi + M*theta)]
           Z_phys = Z0 * [1 + eps*cos(N*phi + M*theta)]
           => dR/dphi, dZ/dphi non-zero
           => z-dependent metrics (g33, g13, g23) and stronger geometric coupling

    Parameters:
        geom: dict from compute_geometry() containing R_vals, derivatives, etc.
        phi_mesh: Toroidal coordinate mesh (1, 1, nz)
        theta_1d: 1D theta array (ny,) needed for ripple phase if ripple enabled
        ripple: dict or None
            Expected keys:
              - enabled: bool
              - eps: float
              - N: int
              - M: int

    Returns:
        dict with basis vector components in Cartesian (X, Y, Z)
        arrays shape (nx, ny, nz)
    """
    R_vals = geom['R_vals']
    dR_dr = geom['dR_dr']
    dR_dtheta = geom['dR_dtheta']
    dZ_dr = geom['dZ_dr']
    dZ_dtheta = geom['dZ_dtheta']

    nx, ny, _ = R_vals.shape
    nz = phi_mesh.shape[2]

    # Broadcast 2D geometry to full 3D
    R0_3d = np.broadcast_to(R_vals, (nx, ny, nz))
    dR_dr_3d = np.broadcast_to(dR_dr, (nx, ny, nz))
    dR_dtheta_3d = np.broadcast_to(dR_dtheta, (nx, ny, nz))
    Z0_3d = np.broadcast_to(geom['Z_vals'], (nx, ny, nz))
    dZ_dr_3d = np.broadcast_to(dZ_dr, (nx, ny, nz))
    dZ_dtheta_3d = np.broadcast_to(dZ_dtheta, (nx, ny, nz))

    # Default: no ripple
    R_3d = R0_3d
    Z_3d = Z0_3d
    dR_dphi_3d = np.zeros((nx, ny, nz))
    dZ_dphi_3d = np.zeros((nx, ny, nz))

    # Optional: 3D toroidal/helical ripple
    if ripple is not None and ripple.get("enabled", False):
        if theta_1d is None:
            raise ValueError("compute_basis_vectors: theta_1d must be provided when ripple is enabled.")

        eps = float(ripple.get("eps", 0.0))
        N = int(ripple.get("N", 8))
        M = int(ripple.get("M", 0))

        # Construct phase = N*phi + M*theta
        theta_mesh = theta_1d[None, :, None]          # (1, ny, 1)
        theta_mesh = np.broadcast_to(theta_mesh, (nx, ny, nz))
        phase = N * phi_mesh + M * theta_mesh

        c = np.cos(phase)
        s = np.sin(phase)

        # Physical R,Z with ripple
        # NOTE: Multiplicative ripple keeps R positive if eps small.
        R_3d = R0_3d * (1.0 + eps * c)
        Z_3d = Z0_3d * (1.0 + eps * c)

        # Derivatives: chain rule
        # d/dphi cos(phase) = -sin(phase) * N
        dR_dphi_3d = R0_3d * (eps * (-s) * N)
        dZ_dphi_3d = Z0_3d * (eps * (-s) * N)

        # d/dtheta also changes due to ripple
        # d/dtheta cos(phase) = -sin(phase) * M
        dR_dtheta_3d = dR_dtheta_3d * (1.0 + eps * c) + R0_3d * (eps * (-s) * M)
        dZ_dtheta_3d = dZ_dtheta_3d * (1.0 + eps * c) + Z0_3d * (eps * (-s) * M)

        # d/dr also changes due to ripple scaling
        dR_dr_3d = dR_dr_3d * (1.0 + eps * c)
        dZ_dr_3d = dZ_dr_3d * (1.0 + eps * c)

    # Trig for physical position
    cosphi = np.cos(phi_mesh)
    sinphi = np.sin(phi_mesh)

    # ------------------------------------------------------------------
    # Position mapping:
    #   X = R(r,theta,phi) * cos(phi)
    #   Y = R(r,theta,phi) * sin(phi)
    #   Z = Z(r,theta,phi)
    #
    # With ripple, R and Z depend on phi, so e_phi includes dR/dphi, dZ/dphi.
    # ------------------------------------------------------------------

    # e_r = ∂/∂r
    er_X = dR_dr_3d * cosphi
    er_Y = dR_dr_3d * sinphi
    er_Z = dZ_dr_3d

    # e_theta = ∂/∂theta
    etheta_X = dR_dtheta_3d * cosphi
    etheta_Y = dR_dtheta_3d * sinphi
    etheta_Z = dZ_dtheta_3d

    # e_phi = ∂/∂phi
    # IMPORTANT: If R depends on phi, ∂X/∂phi includes both:
    #   -R*sin(phi) + (dR/dphi)*cos(phi)
    # similarly for Y. Z includes dZ/dphi.
    ephi_X = -R_3d * sinphi + dR_dphi_3d * cosphi
    ephi_Y =  R_3d * cosphi + dR_dphi_3d * sinphi
    ephi_Z = dZ_dphi_3d

    # Sanity check
    if np.any(np.isnan(er_X)) or np.any(np.isnan(etheta_X)) or np.any(np.isnan(ephi_X)):
        raise RuntimeError("Basis vector field contains NaN values - geometry definition failed.")

    return {
        'er_X': er_X, 'er_Y': er_Y, 'er_Z': er_Z,
        'etheta_X': etheta_X, 'etheta_Y': etheta_Y, 'etheta_Z': etheta_Z,
        'ephi_X': ephi_X, 'ephi_Y': ephi_Y, 'ephi_Z': ephi_Z
    }



def compute_metric_tensor(basis):
    """
    Compute covariant and contravariant metric tensor components.
    
    Parameters:
        basis: dict from compute_basis_vectors()
    
    Returns:
        dict with internal legacy keys:
            g11, g22, g33, g12, g13, g23 (covariant, internal only)
            g_11, g_22, g_33, g_12, g_13, g_23 (contravariant, internal only)
            g_cov (full covariant tensor, shape (nx,ny,nz,3,3))
            g_contra (full contravariant tensor, shape (nx,ny,nz,3,3))

    IMPORTANT: these dictionary key names are historical implementation details.
    write_netcdf_grid() maps them onto the canonical BOUT++ 5.x file contract,
    where g11..g23 are contravariant and g_11..g_23 are covariant.
    
    Metric tensor g_ij = e_i · e_j
    Contravariant components via full 3×3 matrix inversion
    Includes validation: g^ik g_kj ≈ δ^i_j
    """

    er_X = basis['er_X']
    er_Y = basis['er_Y']
    er_Z = basis['er_Z']
    etheta_X = basis['etheta_X']
    etheta_Y = basis['etheta_Y']
    etheta_Z = basis['etheta_Z']
    ephi_X = basis['ephi_X']
    ephi_Y = basis['ephi_Y']
    ephi_Z = basis['ephi_Z']

    #Covariant metric components
    g11 = er_X*er_X + er_Y*er_Y + er_Z*er_Z
    g22 = etheta_X*etheta_X + etheta_Y*etheta_Y + etheta_Z*etheta_Z
    g33 = ephi_X*ephi_X + ephi_Y*ephi_Y + ephi_Z*ephi_Z

    g12 = er_X*etheta_X + er_Y*etheta_Y + er_Z*etheta_Z
    g13 = er_X*ephi_X + er_Y*ephi_Y + er_Z*ephi_Z
    g23 = etheta_X*ephi_X + etheta_Y*ephi_Y + etheta_Z*ephi_Z

    #Metric sanity checks
    if np.any(~np.isfinite(g11)) or np.any(~np.isfinite(g22)) or np.any(~np.isfinite(g33)):
        raise RuntimeError("Metric tensor contains non-finite values - geometry or derivatives invalid.")

    if np.any(g11 <= 0) or np.any(g22 <= 0) or np.any(g33 <= 0):
        raise RuntimeError("Metric diagonal element is non-positive - indicates coordinate singularity.")

    #Stack into full covariant tensor
    g_cov = np.stack([
        np.stack([g11, g12, g13], axis=-1),
        np.stack([g12, g22, g23], axis=-1),
        np.stack([g13, g23, g33], axis=-1)
    ], axis=-2)   # shape: (nx, ny, nz, 3, 3)

    #Invert to get contravariant metric
    g_contra = np.linalg.inv(g_cov)

    #Extract contravariant components
    g_11 = g_contra[..., 0, 0]
    g_22 = g_contra[..., 1, 1]
    g_33 = g_contra[..., 2, 2]
    g_12 = g_contra[..., 0, 1]
    g_13 = g_contra[..., 0, 2]
    g_23 = g_contra[..., 1, 2]

    #Full sanity check: g^ik g_kj ≈ δ^i_j
    tol = 1e-10
    delta = np.eye(3)
    identity_test = np.einsum("...ik,...kj->...ij", g_contra, g_cov)

    if not np.allclose(identity_test, delta, atol=tol):
        raise RuntimeError("Metric inversion check failed: full g^ik g_kj != δ^i_j")

    return {
        'g11': g11, 'g22': g22, 'g33': g33,
        'g12': g12, 'g13': g13, 'g23': g23,
        'g_11': g_11, 'g_22': g_22, 'g_33': g_33,
        'g_12': g_12, 'g_13': g_13, 'g_23': g_23,
        'g_cov': g_cov,
        'g_contra': g_contra
    }



def _periodic_cumulative_trapezoid(values_2d, theta):
    """Periodic cumulative trapezoid in the poloidal direction.

    Returns values integrated from theta[0] to each sample and the full
    0..2*pi loop integral.  `theta` is expected to be uniformly sampled with
    endpoint=False, as used by Grid Suite.
    """
    values = np.asarray(values_2d, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if values.ndim != 2:
        raise ValueError("_periodic_cumulative_trapezoid expects a 2D (x,y) array")
    if theta.ndim != 1 or theta.size != values.shape[1]:
        raise ValueError("theta length must match the y dimension")
    if theta.size < 2:
        raise ValueError("At least two poloidal points are required")

    dtheta = float(theta[1] - theta[0])
    intervals = 0.5 * (values + np.roll(values, -1, axis=1)) * dtheta
    cumulative = np.zeros_like(values)
    if values.shape[1] > 1:
        cumulative[:, 1:] = np.cumsum(intervals[:, :-1], axis=1)
    total = np.sum(intervals, axis=1)
    return cumulative, total


def compute_bout_field_aligned_system(basis, geom, coords, q_vals, B0, reference_R0):
    """Construct a canonical BOUT++ 5.x field-aligned coordinate system.

    Grid Suite's geometry engine first constructs a physical axisymmetric
    mapping in (r, theta, phi).  BOUT++'s tokamak metric contract is instead a
    Clebsch/field-aligned system where:

        g11..g23   are contravariant g^{ij}
        g_11..g_23 are covariant g_ij
        B = (1/J) e_y
        Bxy = sqrt(g_22) / J

    This routine transforms the physical mapping into coordinates
    (x=psi, y=theta, z=field-line label), while preserving the requested
    radial q profile and toroidal-field magnitude B0 at reference_R0.

    The flux derivative dpsi/dr is chosen so that the loop-average field-line
    pitch equals the requested q(r).  This makes B0, q, J and the metric a
    single self-consistent coordinate system rather than independent inputs.
    """
    r = np.asarray(coords['x'], dtype=float)
    theta = np.asarray(coords['y'], dtype=float)
    nx, ny, nz = basis['er_X'].shape

    # Physical basis from the geometry engine.  The common R-Z convention used
    # by Grid Suite with increasing geometric theta is left-handed relative to
    # +phi.  Use zeta=-phi when necessary rather than flipping e_theta, so the
    # metric remains faithful to the actual R(theta), Z(theta) ordering.
    er = np.stack([basis['er_X'], basis['er_Y'], basis['er_Z']], axis=-1)
    et = np.stack([basis['etheta_X'], basis['etheta_Y'], basis['etheta_Z']], axis=-1)
    ep = np.stack([basis['ephi_X'], basis['ephi_Y'], basis['ephi_Z']], axis=-1)

    J_base = np.einsum('...i,...i', er, np.cross(et, ep))
    toroidal_sign = 1.0
    if float(np.nanmean(J_base)) < 0.0:
        ep = -ep
        J_base = -J_base
        toroidal_sign = -1.0
        print("[grid] Using zeta=-phi toroidal coordinate to maintain a right-handed BOUT++ basis.")

    if np.any(~np.isfinite(J_base)) or np.nanmin(J_base) <= 0.0:
        raise RuntimeError("Base geometry Jacobian is non-positive or non-finite")

    R = np.broadcast_to(np.asarray(geom['R_vals'], dtype=float), (nx, ny, nz))
    R2d = R[:, :, 0]
    J2d = J_base[:, :, 0]

    # F = R*B_tor is a flux function in the axisymmetric model.
    F = abs(float(B0) * float(reference_R0))
    if F <= 0.0:
        raise ValueError("B0 and reference_R0 must define a non-zero toroidal field")

    # For x=psi(r), J_xyz = J_rtheta_zeta / (dpsi/dr) and
    # nu = F * J_xyz / R^2.  Choose dpsi/dr so the loop average of nu is q(r).
    geom_integrand = J2d / np.maximum(R2d**2, 1e-30)
    _, geom_loop = _periodic_cumulative_trapezoid(geom_integrand, theta)
    q_vals = np.asarray(q_vals, dtype=float)
    if np.any(q_vals <= 0.0):
        raise ValueError("Canonical BOUT++ field-aligned output currently requires q(r) > 0")

    dpsi_dr = F * geom_loop / (2.0 * np.pi * q_vals)
    if np.any(~np.isfinite(dpsi_dr)) or np.any(dpsi_dr <= 0.0):
        raise RuntimeError("Computed dpsi/dr is non-positive or non-finite")

    # Integrate psi(r); only differences/spacings matter to BOUT++ operators.
    psi = cumulative_trapezoid(dpsi_dr, x=r, initial=0.0)
    dx_1d = np.gradient(psi, edge_order=2 if nx >= 3 else 1)
    if np.any(dx_1d <= 0.0):
        raise RuntimeError("Computed flux-coordinate spacing dx is non-positive")

    J = J_base / dpsi_dr[:, None, None]
    nu = F * J / np.maximum(R**2, 1e-30)

    # zShift is the accumulated local field-line pitch.  ShiftAngle is the
    # complete poloidal-circuit shift and is 1D in canonical BOUT++ grid files.
    zshift_2d, shift_angle = _periodic_cumulative_trapezoid(nu[:, :, 0], theta)
    zshift = np.broadcast_to(zshift_2d[:, :, None], (nx, ny, nz))

    # Local torsion used by BOUT++ vector differential operators is the
    # radial derivative of the local field-line pitch.  Integrated shear I is
    # the radial derivative of zShift, equivalently the poloidal integral of
    # ShiftTorsion.  Here x=psi, so convert r-derivatives with dpsi/dr.
    dnu_dr = np.gradient(nu[:, :, 0], r, axis=0, edge_order=2 if nx >= 3 else 1)
    shift_torsion_2d = dnu_dr / dpsi_dr[:, None]
    shift_torsion = np.broadcast_to(shift_torsion_2d[:, :, None], (nx, ny, nz))

    dzshift_dr = np.gradient(zshift_2d, r, axis=0, edge_order=2 if nx >= 3 else 1)
    I2d = dzshift_dr / dpsi_dr[:, None]
    I = np.broadcast_to(I2d[:, :, None], (nx, ny, nz))

    # Field-aligned basis: e_x=e_r/dpsi_dr + I e_zeta,
    # e_y=e_theta + nu e_zeta, e_z=e_zeta.
    ex = er / dpsi_dr[:, None, None, None] + I[..., None] * ep
    ey = et + nu[..., None] * ep
    ez = ep

    fa_basis = {
        'er_X': ex[..., 0], 'er_Y': ex[..., 1], 'er_Z': ex[..., 2],
        'etheta_X': ey[..., 0], 'etheta_Y': ey[..., 1], 'etheta_Z': ey[..., 2],
        'ephi_X': ez[..., 0], 'ephi_Y': ez[..., 1], 'ephi_Z': ez[..., 2],
    }
    metric = compute_metric_tensor(fa_basis)

    # The transformed triple product should match J from the coordinate
    # transformation; verify before writing anything.
    J_basis = np.einsum('...i,...i', ex, np.cross(ey, ez))
    if not np.allclose(J_basis, J, rtol=2e-10, atol=1e-12):
        raise RuntimeError("Field-aligned Jacobian transformation consistency check failed")

    # In Clebsch coordinates B=(1/J)e_y.  This is exactly the relation used by
    # BOUT++ to derive Bxy from g_22 and J.
    ey_mag = np.linalg.norm(ey, axis=-1)
    et_mag = np.linalg.norm(et, axis=-1)
    Bxy = ey_mag / J
    Bpxy = et_mag / J
    Btor = np.sqrt(np.maximum(Bxy**2 - Bpxy**2, 0.0))

    b_cart = ey / ey_mag[..., None]
    bfield = {
        'Bxy': Bxy,
        'Bpxy': Bpxy,
        'Bphi': Btor,
        'Btheta': Bpxy / np.maximum(et_mag, 1e-30),
        'bX': b_cart[..., 0], 'bY': b_cart[..., 1], 'bZ': b_cart[..., 2],
        'b_r': np.zeros_like(Bxy),
        'b_theta': 1.0 / np.maximum(J * Bxy, 1e-30),
        'b_phi': np.zeros_like(Bxy),
        'etheta_mag': ey_mag,
    }

    # BOUT++ loader identities.  These are intentionally identical to the
    # checks performed by Coordinates::readFromMesh().
    bout_J_from_metric = np.sqrt(np.maximum(np.linalg.det(metric['g_cov']), 0.0))
    bout_B_from_metric = np.sqrt(np.maximum(metric['g_cov'][..., 1, 1], 0.0)) / J
    j_rel = float(np.nanmax(np.abs(bout_J_from_metric - J) / np.maximum(np.abs(J), 1e-30)))
    b_rel = float(np.nanmax(np.abs(bout_B_from_metric - Bxy) / np.maximum(np.abs(Bxy), 1e-30)))
    if j_rel > 1e-9 or b_rel > 1e-9:
        raise RuntimeError(
            f"BOUT++ metric contract failed before write: J rel={j_rel:.3e}, Bxy rel={b_rel:.3e}"
        )

    return {
        'basis': fa_basis,
        'metric': metric,
        'J': J,
        'surfvol': J * dx_1d[:, None, None] * coords['dtheta'] * coords['dphi'],
        'bfield': bfield,
        'psi': psi,
        'dx_1d': dx_1d,
        'zShift': zshift,
        'ShiftAngle': shift_angle,
        'ShiftTorsion': shift_torsion,
        'IntShiftTorsion': I,
        'nu': nu,
        'q_actual': shift_angle / (2.0 * np.pi),
        'dpsi_dr': dpsi_dr,
        'toroidal_coordinate_sign': toroidal_sign,
        'bout_J_max_rel_error': j_rel,
        'bout_Bxy_max_rel_error': b_rel,
    }

def compute_jacobian(basis, dx_vals, dy_vals, dz_vals):
    """
    Compute Jacobian J and differential volume element surfvol.
    
    Parameters:
        basis: dict from compute_basis_vectors()
        dx_vals, dy_vals, dz_vals: 3D spacing arrays (nx, ny, nz)
    
    Returns:
        dict with:
            J: Jacobian determinant (nx, ny, nz)
            surfvol: Differential volume J * dx * dy * dz
    
    J = e_r · (e_theta × e_phi)
    
    Includes sign consistency check - fails if geometry is invalid.
    """
    er_X = basis['er_X']
    er_Y = basis['er_Y']
    er_Z = basis['er_Z']
    etheta_X = basis['etheta_X']
    etheta_Y = basis['etheta_Y']
    etheta_Z = basis['etheta_Z']
    ephi_X = basis['ephi_X']
    ephi_Y = basis['ephi_Y']
    ephi_Z = basis['ephi_Z']

    #Cross product e_theta × e_phi
    cross_e_theta_phi_X = etheta_Y * ephi_Z - etheta_Z * ephi_Y
    cross_e_theta_phi_Y = etheta_Z * ephi_X - etheta_X * ephi_Z
    cross_e_theta_phi_Z = etheta_X * ephi_Y - etheta_Y * ephi_X

    #Triple scalar product
    J = (
        er_X * cross_e_theta_phi_X +
        er_Y * cross_e_theta_phi_Y +
        er_Z * cross_e_theta_phi_Z
    )

    # ------------------------------------------------------------------
    # Jacobian sign / validity checks
    #
    # Note: Mathematical coordinate systems can be left-handed (J<0),
    # but BOUT++ 3D metrics require a right-handed mapping with J>0
    # everywhere because J is used as a positive volume element.
    #
    # We therefore:
    #   1) Enforce global sign consistency (no local flips/folds)
    #   2) Enforce positivity (J>0) for BOUT++ compatibility
    #   3) Catch NaN/Inf and near-zero singularities
    # ------------------------------------------------------------------

    # Use a stable reference sign (avoid corner cases where J might be tiny)
    ref_val = J[0, 0, 0]
    ref_sign = np.sign(ref_val)

    if ref_sign == 0:
        raise RuntimeError(
            "Jacobian sign at reference point is zero - degenerate basis detected at (0,0,0). "
            f"J[0,0,0]={ref_val}"
        )

    # 1) No sign flips anywhere (local folding / inverted cells)
    if np.any(np.sign(J) != ref_sign):
        # This is a real geometry failure: local orientation flips
        raise RuntimeError(
            "Jacobian sign inconsistency detected - basis orientation flips across the grid. "
            "This indicates local cell folding / invalid geometry (non-physical mapping)."
        )

    # 2) Enforce right-handed orientation for BOUT++ (J must be positive)
    if ref_sign < 0:
        J_min = float(np.nanmin(J))
        J_max = float(np.nanmax(J))
        raise RuntimeError(
            "Grid is globally left-handed: Jacobian is negative everywhere. "
            "BOUT++ 3D metrics requires J>0. "
            f"Observed range: J_min={J_min}, J_max={J_max}. "
            "Fix: flip one basis vector globally (e.g. e_theta -> -e_theta) "
            "in the grid generator BEFORE computing metric/J."
        )

    # 3) Additional sanity checks
    if np.any(~np.isfinite(J)):
        raise RuntimeError("Jacobian contains NaN or Inf values - invalid metric or basis vectors.")

    J_abs_min = float(np.nanmin(np.abs(J)))
    if J_abs_min < 1e-12:
        raise RuntimeError(
            "Jacobian magnitude is approaching zero - coordinate system near singular or over-shaped. "
            f"min(|J|)={J_abs_min}"
        )

    # Compute differential volume element (positive by construction if checks passed)
    surfvol = J * dx_vals * dy_vals * dz_vals

    return {
        'J': J,
        'surfvol': surfvol
    }



def compute_magnetic_field(x, geom, q_xy, B0, R0, basis):
    """
    Compute magnetic field components Bxy and Bpxy.
    
    Parameters:
        x: Radial coordinate array (nx  )
        geom: dict from compute_geometry()
        q_xy: Safety factor array (nx, ny, 1)
        B0: Toroidal field at R0
        R0: Major radius
        basis: dict from compute_basis_vectors()
    
    Returns:
        dict with:
            Bxy: Total magnetic field magnitude |B| (nx, ny, nz)
            Bpxy: Physical poloidal field B_theta (nx, ny, nz)
            Bphi: Toroidal field component (nx, ny, nz)
            Btheta: Poloidal field component (nx, ny, nz)
            bX, bY, bZ: Unit magnetic field vector components
            b_r, b_theta, b_phi: Unit field in coordinate basis
            etheta_mag: Magnitude of e_theta basis vector
    
    NOTE FOR BOUT++ 5.x:
      Bxy  = |B|, the full magnetic-field magnitude.
    
      Bpxy = physical poloidal magnetic-field strength B_theta,
             computed from the shaped geometry using:
                 B_theta = (B_phi / q) * (|e_phi| / |e_theta|)
    
      This is not the old BOUT++ 4.x covariant or projected
      poloidal component. It is the true physical poloidal field
      used consistently with metrics, curvature, and field-line pitch.
    
      This ensures compatibility with BOUT++ 5.x mesh loaders
      and prevents misinterpretation by older analysis scripts
      that expect the legacy 4.x Bpxy definition.
    
    GS-CONSISTENT PHYSICAL POLOIDAL FIELD Bθ:
    From Grad-Shafranov large-aspect ratio equilibrium:
      B_theta(r) = r * B_phi(R=r) / (q(r) * R0)
    
    but we must apply shaping correction because physical
    Bθ lives along e_theta, not simple circular basis.
    """

    R_vals = geom['R_vals']
    r_mesh = geom['r_mesh']
    q_r = q_xy  # Already (nx, ny, 1)

    etheta_X = basis['etheta_X']
    etheta_Y = basis['etheta_Y']
    etheta_Z = basis['etheta_Z']
    ephi_X = basis['ephi_X']
    ephi_Y = basis['ephi_Y']
    ephi_Z = basis['ephi_Z']

    nx = len(x)

    #Toroidal field: Bphi = B0 * R0 / R
    R_3d = np.broadcast_to(R_vals, (nx, R_vals.shape[1], ephi_X.shape[2]))
    Bphi = B0 * (R0 / R_3d)

    #Compute unshaped GS poloidal field
    Btheta_unshaped = (x[:, None, None] * Bphi) / (q_r * R0)

    #Magnitudes of coordinate basis vectors
    etheta_mag = np.sqrt(etheta_X**2 + etheta_Y**2 + etheta_Z**2)

    #Normalize Bθ along the shaped e_theta direction
    Btheta = Btheta_unshaped * (1.0 / etheta_mag)

    #True |B| magnitude
    Bmag = np.sqrt(Bphi**2 + (Btheta * etheta_mag)**2)

    #B-field sanity checks
    if np.any(~np.isfinite(Bmag)):
        raise RuntimeError("Magnetic field magnitude contains NaN/Inf — invalid geometry or q-profile.")

    if np.min(Bmag) <= 0:
        raise RuntimeError("Magnetic field magnitude has vanished or gone negative — nonphysical configuration.")

    #Unit magnetic field vector components
    bX = (Bphi * ephi_X + Btheta * etheta_X) / Bmag
    bY = (Bphi * ephi_Y + Btheta * etheta_Y) / Bmag
    bZ = (Bphi * ephi_Z + Btheta * etheta_Z) / Bmag

    #Project magnetic unit vector b onto coordinate basis
    er_X = basis['er_X']
    er_Y = basis['er_Y']
    er_Z = basis['er_Z']

    b_r = bX * er_X + bY * er_Y + bZ * er_Z
    b_theta = bX * etheta_X + bY * etheta_Y + bZ * etheta_Z
    b_phi = bX * ephi_X + bY * ephi_Y + bZ * ephi_Z

    return {
        'Bxy': Bmag,
        'Bpxy': Btheta,
        'Bphi': Bphi,
        'Btheta': Btheta,
        'bX': bX, 'bY': bY, 'bZ': bZ,
        'b_r': b_r, 'b_theta': b_theta, 'b_phi': b_phi,
        'etheta_mag': etheta_mag
    }


def compute_shift_angle(q_xy, theta, nx, ny, nz):

    """
    Legacy/non-canonical accumulated 1/q shift used only by the synthetic
    toroidal-ripple forensic path.

    This is intentionally NOT written as canonical BOUT++ ShiftAngle for
    normal v2 grids. Canonical axisymmetric generation uses
    compute_bout_field_aligned_system(), which writes zShift(x,y) and
    ShiftAngle(x) using the BOUT++ field-line pitch definition.

    Compute legacy shift angle α(x,y,z).
    
    Parameters:
        q_xy: Safety factor (nx, ny, nz)
        theta: Poloidal coordinate array (ny,)
        nx, ny, nz: Grid dimensions
    
    Returns:
        shiftAngle: Field-aligned coordinate shift (nx, ny, nz)
    
    α(x,θ) = ∫₀^{θ} dθ' / q(x,θ')
    
    NOTE:
    Broadcast of shiftAngle must keep the full (nx, ny, nz) structure.
    Avoid squeezing dimensions; it silently destroys the z-axis and relies
    on implicit NumPy broadcasting, which is fragile and confusing.
    """
    #1/q(x,θ) - already (nx,ny,nz) but symmetric in z
    inv_q = 1.0 / q_xy

    #Integrate along poloidal angle dimension (axis=1)
    alpha = cumulative_trapezoid(inv_q[:, :, 0], x=theta, axis=1, initial=0.0)

    #Force alpha into strict (nx, ny) before any reshaping
    alpha = np.asarray(alpha)
    alpha = alpha.reshape(nx, ny)

    #Broadcast alpha into full (nx, ny, nz)
    alpha_3d = alpha[:, :, None]  #(nx, ny, 1)
    alpha_3d = np.broadcast_to(alpha_3d, (nx, ny, nz))

    return alpha_3d


def compute_curvature(mode, bfield, basis, metric, geom, coords):
    """
    Compute magnetic curvature components G1 and G2.
    
    Parameters:
        mode: "exact", "simple", or "none"
        bfield: dict from compute_magnetic_field()
        basis: dict from compute_basis_vectors()
        metric: dict from compute_metric_tensor()
        geom: dict from compute_geometry()
        coords: dict from generate_coordinates()
    
    Returns:
        dict with:
            G1: Curvature component along e_theta (nx, ny, nz)
            G2: Curvature component along e_r (nx, ny, nz)
    
    Curvature calculation methods:
        "exact"  → full metric, Christoffel symbols, covariant derivatives
        "simple" → analytic tokamak curvature approximation
        "none"   → disable curvature calculation entirely
    """
    nx = basis['er_X'].shape[0]
    ny = basis['er_X'].shape[1]
    nz = basis['er_X'].shape[2]

    if mode == "none":
        #Zero curvature
        return {
            'G1': np.zeros((nx, ny, nz)),
            'G2': np.zeros((nx, ny, nz))
        }

    elif mode == "simple":
        #Analytic circular tokamak approximation:
        #G1 ≈ -cos(theta)/R , G2 ≈ -sin(theta)/R
        theta_mesh = geom['theta_mesh']
        R_vals = geom['R_vals']
        R_3d = np.broadcast_to(R_vals, (nx, ny, nz))

        theta_3d = np.broadcast_to(theta_mesh, (nx, ny, nz))
        G1 = -(np.cos(theta_3d) / R_3d)
        G2 = -(np.sin(theta_3d) / R_3d)

        return {'G1': G1, 'G2': G2}

    elif mode == "exact":
        #Full tensor curvature calculation

        bX = bfield['bX']
        bY = bfield['bY']
        bZ = bfield['bZ']
        b_r = bfield['b_r']
        b_theta = bfield['b_theta']
        b_phi = bfield['b_phi']

        er_X = basis['er_X']
        er_Y = basis['er_Y']
        er_Z = basis['er_Z']
        etheta_X = basis['etheta_X']
        etheta_Y = basis['etheta_Y']
        etheta_Z = basis['etheta_Z']
        ephi_X = basis['ephi_X']
        ephi_Y = basis['ephi_Y']
        ephi_Z = basis['ephi_Z']

        g_cov = metric['g_cov']
        g_contra = metric['g_contra']

        dr = coords['dr']
        dtheta = coords['dtheta']
        dphi = coords['dphi']

        #Compute ∂b/∂x^i (numerical partial derivatives)
        db_dr_X = np.gradient(bX, dr, axis=0)
        db_dr_Y = np.gradient(bY, dr, axis=0)
        db_dr_Z = np.gradient(bZ, dr, axis=0)

        db_dtheta_X = np.gradient(bX, dtheta, axis=1)
        db_dtheta_Y = np.gradient(bY, dtheta, axis=1)
        db_dtheta_Z = np.gradient(bZ, dtheta, axis=1)

        db_dphi_X = np.gradient(bX, dphi, axis=2)
        db_dphi_Y = np.gradient(bY, dphi, axis=2)
        db_dphi_Z = np.gradient(bZ, dphi, axis=2)

        #Metric derivatives
        dg_dr = np.gradient(g_cov, dr, axis=0)
        dg_dtheta = np.gradient(g_cov, dtheta, axis=1)
        dg_dphi = np.gradient(g_cov, dphi, axis=2)

        dg = np.stack([dg_dr, dg_dtheta, dg_dphi], axis=3)

        #Christoffel symbols
        Gamma = np.zeros((nx, ny, nz, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                term1 = dg[..., i, :, j]
                term2 = dg[..., j, :, i]
                term3 = dg[..., :, i, j]
                Gamma[..., :, i, j] = 0.5 * np.einsum(
                    "...km,...m->...k",
                    g_contra,
                    term1 + term2 - term3
                )

        #b in coordinate basis
        b_vec = np.stack([b_r, b_theta, b_phi], axis=-1)

        partial_b = np.stack([
            np.stack([db_dr_X, db_dr_Y, db_dr_Z], axis=-1),
            np.stack([db_dtheta_X, db_dtheta_Y, db_dtheta_Z], axis=-1),
            np.stack([db_dphi_X, db_dphi_Y, db_dphi_Z], axis=-1)
        ], axis=-2)

        basis_tensor = np.stack([
            np.stack([er_X, etheta_X, ephi_X], axis=-1),
            np.stack([er_Y, etheta_Y, ephi_Y], axis=-1),
            np.stack([er_Z, etheta_Z, ephi_Z], axis=-1)
        ], axis=-2)

        partial_b_coord = np.einsum("...ik,...kj->...ij", partial_b, basis_tensor)
        covDb = partial_b_coord + np.einsum("...kij,...k->...ij", Gamma, b_vec)

        kappa_coord = np.einsum("...i,...ij->...j", b_vec, covDb)

        kappa_X = kappa_coord[..., 0] * er_X + kappa_coord[..., 1] * etheta_X + kappa_coord[..., 2] * ephi_X
        kappa_Y = kappa_coord[..., 0] * er_Y + kappa_coord[..., 1] * etheta_Y + kappa_coord[..., 2] * ephi_Y
        kappa_Z = kappa_coord[..., 0] * er_Z + kappa_coord[..., 1] * etheta_Z + kappa_coord[..., 2] * ephi_Z

        G1 = kappa_X * etheta_X + kappa_Y * etheta_Y + kappa_Z * etheta_Z
        G2 = kappa_X * er_X + kappa_Y * er_Y + kappa_Z * er_Z

        return {'G1': G1, 'G2': G2}

    else:
        raise ValueError(f"Unknown curvature mode '{mode}'")


def compute_q_poloidal(x, q_vals, geom, B0, R0):
    """
    Compute TRUE GS-CONSISTENT q(x,y) from Bθ definition.
    
    Uses the large-aspect-ratio Grad-Shafranov relation:
      q = r * B_phi(R) / (R * B_theta(r))
    where B_theta is derived from the target q(r) profile.
    
    Parameters:
        x: Radial coordinate array (nx)
        q_vals: Radial q-profile (nx)
        geom: dict from compute_geometry()
        B0: Toroidal field at R0
        R0: Major radius
    
    Returns:
        q_xy: Poloidally varying safety factor (nx, ny, 1)
    """
    R_vals = geom['R_vals']
    r_mesh = geom['r_mesh']
    q_r = q_vals[:, None, None]  #(nx,1,1)

    #Toroidal field
    Bphi_q = B0 * (R0 / R_vals)

    #GS-consistent poloidal field:
    #   B_theta(r) = r * B_phi(R=r) / (q_r * R0)
    #NOTE: use R0 not R_vals; q-profile is radial, not poloidally varying.
    Btheta_q = (x[:, None, None] * B0) / (q_r * R0)

    #Final physical q(x,y):
    #q_xy = r Bφ / (R Bθ(r))
    q_xy = (x[:, None, None] * Bphi_q) / (R_vals * Btheta_q)

    return q_xy


def write_netcdf_grid(outfile, coords, geom, metric, jacobian, bfield, shift_data, curv, args, bout_contract=None):
    """
    Write all computed data to BOUT++ 5.x compatible netCDF grid file.
    
    Parameters:
        outfile: Output filename
        coords: dict from generate_coordinates()
        geom: dict from compute_geometry()
        metric: dict from compute_metric_tensor()
        jacobian: dict from compute_jacobian()
        bfield: dict from compute_magnetic_field()
        shift_data: canonical zShift/ShiftAngle data or legacy synthetic shift data
        curv: dict from compute_curvature()
        args: Command line arguments namespace
    
    Writes complete BOUT++ 5.x grid including:
        - Dimensions and coordinate arrays
        - Metric tensor (covariant and contravariant)
        - Jacobian and differential volume
        - Magnetic field components
        - Shift angle
        - Curvature
        - All required metadata
    
    METRIC DIMENSIONALITY:
        By default, writes 2D metrics (x, y) for axisymmetric tokamak.
        Use --metrics-3d flag to write 3D metrics (x, y, z) for non-axisymmetric geometry.
    """
    nx = args.nx
    ny = args.ny
    nz = args.nz
    precision = args.precision

    x = np.asarray((bout_contract or {}).get('psi', coords['x']), dtype=float)
    y = coords['y']
    z = coords['z']
    x3 = np.broadcast_to(x[:, None, None], (nx, ny, nz)) if (bout_contract or {}).get('canonical', False) else coords['x3']
    y3 = coords['y3']
    z3 = coords['z3']

    R_vals = geom['R_vals']
    Z_vals = geom['Z_vals']

    # Compute 3D spacing arrays
    dr = coords['dr']
    dtheta = coords['dtheta']
    dphi = coords['dphi']

    r_mesh = geom['r_mesh']
    theta_mesh = geom['theta_mesh']

    # IMPORTANT:
    # dx, dy, dz in BOUT++ are *coordinate* spacings, NOT physical arc-lengths.
    # Physical geometry factors (like r or R) belong in the metric tensor (g_ij),
    # which we already compute from the basis vectors. Writing dz = R*dphi makes
    # the toroidal period zlength(x,y)=Σ dz vary with x/y, and solvers/transforms
    # that require uniform zlength (PCR/SPT/ShiftedMetric) will abort.
    dx_1d = np.asarray((bout_contract or {}).get('dx_1d', np.full(nx, dr)), dtype=float)
    DX = np.broadcast_to(dx_1d[:, None, None], (nx, ny, nz)).copy()
    DY = np.full((nx, ny, nz), dtheta, dtype=float)
    DZ = np.full((nx, ny, nz), dphi, dtype=float)

    
    # Extract 2D slices for metrics (axisymmetric - no z variation)
    # All incoming arrays from metric, jacobian, etc. are already (nx, ny, nz)
    # but for axisymmetric tokamak they're constant in z
    DX_2d = DX[:, :, 0]
    DY_2d = DY[:, :, 0]
    DZ_2d = DZ[:, :, 0]
    
    # R_vals and Z_vals might be (nx, ny) or (nx, ny, 1) - squeeze to 2D
    R_2d = np.squeeze(R_vals) if R_vals.ndim > 2 else R_vals
    Z_2d = np.squeeze(Z_vals) if Z_vals.ndim > 2 else Z_vals
    
    g11_2d = metric['g11'][:, :, 0]
    g22_2d = metric['g22'][:, :, 0]
    g33_2d = metric['g33'][:, :, 0]
    g12_2d = metric['g12'][:, :, 0]
    g13_2d = metric['g13'][:, :, 0]
    g23_2d = metric['g23'][:, :, 0]
    
    g_11_2d = metric['g_11'][:, :, 0]
    g_22_2d = metric['g_22'][:, :, 0]
    g_33_2d = metric['g_33'][:, :, 0]
    g_12_2d = metric['g_12'][:, :, 0]
    g_13_2d = metric['g_13'][:, :, 0]
    g_23_2d = metric['g_23'][:, :, 0]
    
    J_2d = jacobian['J'][:, :, 0]
    surfvol_2d = jacobian['surfvol'][:, :, 0]
    zshift = shift_data.get('zShift')
    zshift_2d = zshift[:, :, 0] if zshift is not None and zshift.ndim == 3 else zshift
    shift_angle_1d = shift_data.get('ShiftAngle')
    shift_torsion = shift_data.get('ShiftTorsion')
    int_shift_torsion = shift_data.get('IntShiftTorsion')
    
    Bxy_2d = bfield['Bxy'][:, :, 0]
    Bpxy_2d = bfield['Bpxy'][:, :, 0]
    
    G1_2d = curv['G1'][:, :, 0]
    G2_2d = curv['G2'][:, :, 0]
    
    # Determine metric dimensionality from command line flag
    metrics_3d = getattr(args, 'metrics_3d', False)

    with Dataset(outfile, "w", format="NETCDF4") as nc:
        # -----------------------------
        # Dimensions + required metadata
        # -----------------------------
        nc.createDimension("x", nx)
        nc.createDimension("y", ny)
        nc.createDimension("z", nz)

        nc.setncattr("nx", nx)
        nc.setncattr("ny", ny)
        nc.setncattr("nz", nz)

        nx_var = nc.createVariable("nx", "i4")
        nx_var[:] = nx
        ny_var = nc.createVariable("ny", "i4")
        ny_var[:] = ny
        nz_var = nc.createVariable("nz", "i4")
        nz_var[:] = nz

        # Guard cells
        MXG = 2
        MYG = 2
        MZG = 0

        nc.setncattr("MXG", MXG)
        nc.setncattr("MYG", MYG)
        nc.setncattr("MZG", MZG)

        MXG_var = nc.createVariable("MXG", "i4")
        MXG_var[:] = MXG
        MYG_var = nc.createVariable("MYG", "i4")
        MYG_var[:] = MYG
        MZG_var = nc.createVariable("MZG", "i4")
        MZG_var[:] = MZG

        # -----------------------------
        # Coordinate arrays
        # -----------------------------
        nc.createVariable("x", precision, ("x",))[:] = x
        nc.createVariable("y", precision, ("y",))[:] = y
        nc.createVariable("z", precision, ("z",))[:] = z

        x_var = nc.variables["x"]
        y_var = nc.variables["y"]
        z_var = nc.variables["z"]

        x_var.units = "arb"
        y_var.units = "arb"
        z_var.units = "rad"

        q_actual = np.asarray((bout_contract or {}).get("q_actual", []), dtype=float)
        if q_actual.size == nx:
            q_var = nc.createVariable("q", precision, ("x",))
            q_var.coordinates = "x"
            q_var[:] = q_actual

        # Required coordinate map variables (always 3D)
        xcoord = nc.createVariable("xcoord", precision, ("x", "y", "z"))
        ycoord = nc.createVariable("ycoord", precision, ("x", "y", "z"))
        zcoord = nc.createVariable("zcoord", precision, ("x", "y", "z"))

        xcoord.coordinates = "x y z"
        ycoord.coordinates = "x y z"
        zcoord.coordinates = "x y z"

        xcoord[:] = x3
        ycoord[:] = y3
        zcoord[:] = z3

        # =====================================================================
        # METRIC TENSOR AND GEOMETRY VARIABLES
        # =====================================================================
        # For axisymmetric tokamak, metric tensors should be 2D (x, y) only.
        # 3D mode available via --metrics-3d flag for non-axisymmetric cases.
        #
        # BOUT++ expects 2D metrics for standard tokamak equilibria.
        # Only plasma fields (T, n, phi, etc.) are 3D.
        
        if metrics_3d:
            # 3D metrics mode (for toroidal ripple, non-axisymmetric geometry)
            metric_dims = ("x", "y", "z")
            print("[grid] Writing 3D metric tensors (x, y, z) - non-axisymmetric mode")
        else:
            # 2D metrics mode (standard axisymmetric tokamak)
            metric_dims = ("x", "y")
            print("[grid] Writing 2D metric tensors (x, y) - axisymmetric mode")

        # -----------------------------
        # Grid spacing
        # -----------------------------
        dx = nc.createVariable("dx", precision, metric_dims)
        dx.coordinates = "xcoord ycoord zcoord"
        
        dy = nc.createVariable("dy", precision, metric_dims)
        dy.coordinates = "xcoord ycoord zcoord"
        
        dz = nc.createVariable("dz", precision, metric_dims)
        dz.coordinates = "xcoord ycoord zcoord"
        
        if metrics_3d:
            # 3D: use full arrays
            dx[:] = DX
            dy[:] = DY
            dz[:] = DZ
        else:
            # 2D: use 2D slices
            dx[:] = DX_2d
            dy[:] = DY_2d
            dz[:] = DZ_2d

        # -----------------------------
        # Geometry R(x,y) and Z(x,y)
        # -----------------------------
        R = nc.createVariable("R", precision, metric_dims)
        R.coordinates = "xcoord ycoord zcoord"
        R.units = "m"
        
        Z = nc.createVariable("Z", precision, metric_dims)
        Z.coordinates = "xcoord ycoord zcoord"
        Z.units = "m"
        
        if metrics_3d:
            R[:] = np.broadcast_to(R_vals, (nx, ny, nz))
            Z[:] = np.broadcast_to(Z_vals, (nx, ny, nz))
        else:
            R[:] = R_2d
            Z[:] = Z_2d

        # -----------------------------
        # BOUT++ contravariant metric tensor g^{ij}
        # IMPORTANT: BOUT++ uses un-underscored names for contravariant terms.
        # Grid Suite's internal metric['g_..'] keys contain the inverse tensor.
        # -----------------------------
        g11 = nc.createVariable("g11", precision, metric_dims)
        g22 = nc.createVariable("g22", precision, metric_dims)
        g33 = nc.createVariable("g33", precision, metric_dims)
        g12 = nc.createVariable("g12", precision, metric_dims)
        g13 = nc.createVariable("g13", precision, metric_dims)
        g23 = nc.createVariable("g23", precision, metric_dims)

        g11.units = "1"
        g22.units = "1"
        g33.units = "1"
        g12.units = "1"
        g13.units = "1"
        g23.units = "1"

        for var in (g11, g22, g33, g12, g13, g23):
            var.coordinates = "xcoord ycoord zcoord"

        if metrics_3d:
            g11[:] = metric['g_11']
            g22[:] = metric['g_22']
            g33[:] = metric['g_33']
            g12[:] = metric['g_12']
            g13[:] = metric['g_13']
            g23[:] = metric['g_23']
        else:
            g11[:] = g_11_2d
            g22[:] = g_22_2d
            g33[:] = g_33_2d
            g12[:] = g_12_2d
            g13[:] = g_13_2d
            g23[:] = g_23_2d

        # -----------------------------
        # BOUT++ covariant metric tensor g_ij
        # IMPORTANT: BOUT++ uses underscored names for covariant terms.
        # -----------------------------
        g_11 = nc.createVariable("g_11", precision, metric_dims)
        g_22 = nc.createVariable("g_22", precision, metric_dims)
        g_33 = nc.createVariable("g_33", precision, metric_dims)
        g_12 = nc.createVariable("g_12", precision, metric_dims)
        g_13 = nc.createVariable("g_13", precision, metric_dims)
        g_23 = nc.createVariable("g_23", precision, metric_dims)

        g_11.units = "1"
        g_22.units = "1"
        g_33.units = "1"
        g_12.units = "1"
        g_13.units = "1"
        g_23.units = "1"

        for var in (g_11, g_22, g_33, g_12, g_13, g_23):
            var.coordinates = "xcoord ycoord zcoord"

        if metrics_3d:
            g_11[:] = metric['g11']
            g_22[:] = metric['g22']
            g_33[:] = metric['g33']
            g_12[:] = metric['g12']
            g_13[:] = metric['g13']
            g_23[:] = metric['g23']
        else:
            g_11[:] = g11_2d
            g_22[:] = g22_2d
            g_33[:] = g33_2d
            g_12[:] = g12_2d
            g_13[:] = g13_2d
            g_23[:] = g23_2d

        # -----------------------------
        # Jacobian and surfvol
        # -----------------------------
        J = nc.createVariable("J", precision, metric_dims)
        J.coordinates = "xcoord ycoord zcoord"
        J.units = "m2"

        surfvol = nc.createVariable("surfvol", precision, metric_dims)
        surfvol.coordinates = "xcoord ycoord zcoord"
        surfvol.units = "m3"

        if metrics_3d:
            J[:] = jacobian['J']
            surfvol[:] = jacobian['surfvol']
        else:
            J[:] = J_2d
            surfvol[:] = surfvol_2d

        # -----------------------------
        # Field-alignment quantities
        # -----------------------------
        # Canonical BOUT++ semantics:
        #   zShift(x,y)    = accumulated local field-line pitch
        #   ShiftAngle(x)  = total zShift change over one poloidal circuit
        if zshift_2d is not None:
            zshift_var = nc.createVariable("zShift", precision, ("x", "y"))
            zshift_var.coordinates = "x y"
            zshift_var.units = "rad"
            zshift_var[:] = zshift_2d

        if shift_angle_1d is not None:
            shift_var = nc.createVariable("ShiftAngle", precision, ("x",))
            shift_var.coordinates = "x"
            shift_var.units = "rad"
            shift_var[:] = shift_angle_1d

        if shift_torsion is not None:
            # BOUT++ Coordinates::readFromMesh() reads ShiftTorsion directly
            # and warns that vector derivatives may be incorrect when it is absent.
            st2d = shift_torsion[:, :, 0] if shift_torsion.ndim == 3 else shift_torsion
            st_var = nc.createVariable("ShiftTorsion", precision, ("x", "y"))
            st_var.coordinates = "x y"
            st_var[:] = st2d

        if int_shift_torsion is not None:
            # Integrated shear I = partial(zShift)/partial(x).
            I2d = int_shift_torsion[:, :, 0] if int_shift_torsion.ndim == 3 else int_shift_torsion
            it_var = nc.createVariable("IntShiftTorsion", precision, ("x", "y"))
            it_var.coordinates = "x y"
            it_var[:] = I2d

        # -----------------------------
        # Magnetic field
        # -----------------------------
        Bxy = nc.createVariable("Bxy", precision, metric_dims)
        Bxy.coordinates = "xcoord ycoord zcoord"
        Bxy.units = "T"

        Bpxy = nc.createVariable("Bpxy", precision, metric_dims)
        Bpxy.coordinates = "xcoord ycoord zcoord"
        Bpxy.units = "T"

        if metrics_3d:
            Bxy[:] = bfield['Bxy']
            Bpxy[:] = bfield['Bpxy']
        else:
            Bxy[:] = Bxy_2d
            Bpxy[:] = Bpxy_2d

        # -----------------------------
        # Curvature
        # -----------------------------
        G1 = nc.createVariable("G1", precision, metric_dims)
        G2 = nc.createVariable("G2", precision, metric_dims)
        G1.coordinates = "xcoord ycoord zcoord"
        G2.coordinates = "xcoord ycoord zcoord"
        G1.units = "1/m"
        G2.units = "1/m"

        if metrics_3d:
            G1[:] = curv['G1']
            G2[:] = curv['G2']
        else:
            G1[:] = G1_2d
            G2[:] = G2_2d

        # -----------------------------
        # Requested custom boundary provenance
        # -----------------------------
        if geom.get("geometry_model") == "custom_radial_boundary":
            # Boundary sampled at the grid's actual poloidal y locations.  This
            # gives diagnostics an exact requested-vs-generated fidelity contract.
            br = nc.createVariable("requested_boundary_R", precision, ("y",))
            bz = nc.createVariable("requested_boundary_Z", precision, ("y",))
            br.units = "m"; bz.units = "m"
            br[:] = np.asarray(geom["boundary_R"], dtype=float)
            bz[:] = np.asarray(geom["boundary_Z"], dtype=float)

        # -----------------------------
        # BOUT++ metadata
        # -----------------------------
        if metrics_3d:
            nc.mesh_type = "tokamak_3d"
        elif geom.get("geometry_model") == "custom_radial_boundary":
            nc.mesh_type = "tokamak_axisymmetric_custom"
        else:
            nc.mesh_type = "tokamak_axisymmetric_shaped"
            
        nc.description = "Chatwood Labs - Tokamak grid by bout_tokamak_grid_generator.py"
        nc.geometry_model = geom.get("geometry_model", "miller")
        nc.grid_contract = (bout_contract or {}).get("label", "LEGACY_UNSPECIFIED")
        nc.bout5_canonical = int(bool((bout_contract or {}).get("canonical", False)))
        nc.metric_naming = "BOUT++: g11..g23=contravariant; g_11..g_23=covariant"
        nc.toroidal_coordinate_sign = float((bout_contract or {}).get("toroidal_coordinate_sign", 1.0))
        if (bout_contract or {}).get("bout_J_max_rel_error") is not None:
            nc.bout_J_max_rel_error = float(bout_contract["bout_J_max_rel_error"])
        if (bout_contract or {}).get("bout_Bxy_max_rel_error") is not None:
            nc.bout_Bxy_max_rel_error = float(bout_contract["bout_Bxy_max_rel_error"])
        # Preserve geometry provenance.  For custom boundaries R0 is the actual
        # radial-mapping / magnetic-axis major radius; the four Miller values are
        # retained separately as descriptive equivalent-fit metadata.
        if geom.get("geometry_model") == "custom_radial_boundary":
            nc.R0 = float(geom["axis_R"])
            nc.axis_R = float(geom["axis_R"])
            nc.axis_Z = float(geom["axis_Z"])
            nc.miller_fit_R0 = args.R0
            nc.miller_fit_a = args.a
            nc.miller_fit_kappa = args.kappa
            nc.miller_fit_delta = args.delta
        else:
            nc.R0 = args.R0
            nc.axis_R = args.R0
            nc.axis_Z = 0.0
        nc.a = args.a
        nc.kappa = args.kappa
        nc.delta = args.delta
        nc.B0 = args.B0
        nc.q0 = args.q0
        nc.qa = args.qa
        nc.qform = args.qform
        nc.xmin_frac = args.xmin_frac

        nc.data_format = "BOUT++"
        nc.coord_system = "tridim"
        nc.toroidal_period_rad = 2 * np.pi
        nc.zperiod = 1
        nc.precision = precision

        # BOUT++ topology declaration.
        #
        # Grid Suite v2 Miller/custom-boundary output contains nested closed
        # flux surfaces only. It does not generate an X-point or an internal
        # separatrix. In BOUT++ the explicit convention ixseps1=ixseps2=nx
        # means the separatrix lies outside the represented radial domain, so
        # all grid points belong to the closed-core region.
        ixseps1_var = nc.createVariable("ixseps1", "i4")
        ixseps1_var[:] = nx
        ixseps2_var = nc.createVariable("ixseps2", "i4")
        ixseps2_var[:] = nx
        nc.topology_model = "closed_core_no_xpoint"

def main():
    t0 = time.time()

    #Parse command-line arguments
    args = parse_arguments()

    #Round nx to nearest power of 2 for PCR solver compatibility
    #PCR checks GlobalNxNoBoundaries = nx - 2*MXG, so we need (nx - 4) to be power of 2
    import math
    MXG = 2  # Guard cells (defined later, but needed here)
    
    if args.nx > 0:
        # Target: (nx - 2*MXG) must be power of 2
        nx_interior_target = args.nx - 2*MXG
        power = round(math.log2(max(1, nx_interior_target)))
        nx_interior_rounded = 2 ** power
        nx_rounded = nx_interior_rounded + 2*MXG
        
        if nx_rounded != args.nx:
            print(f"[WARNING] Rounding nx from {args.nx} to {nx_rounded} (interior={nx_interior_rounded}, power of 2 for PCR solver)")
            args.nx = nx_rounded

    # Validate analytic shaping only when the Miller model is actually used.
    # Custom boundaries are validated geometrically below and may have a poor
    # equivalent Miller fit without being invalid.
    if not args.boundary_file:
        validate_shaping_parameters(args.kappa, args.delta)
    if args.a <= 0:
        raise ValueError("a must be > 0")

    print(f"Writing grid to {args.outfile}")

    #Generate coordinate arrays
    xmin = args.xmin_frac * args.a
    coords = generate_coordinates(args.nx, args.ny, args.nz, xmin, args.a)

    #Compute safety factor profile
    q_vals = compute_q_profile(coords['x'], args.a, args.q0, args.qa, args.qform)

    # Compute tokamak geometry.  A supplied boundary is authoritative: the
    # visible/requested contour is reparameterised by poloidal angle and used
    # directly as the outer flux surface.  Miller parameters remain descriptive.
    if args.boundary_file:
        boundary = load_custom_boundary(args.boundary_file)
        geom = compute_custom_geometry(coords['x'], coords['y'], args.a, boundary)
        reference_R0 = float(geom['axis_R'])
        print(
            f"[grid] Custom boundary mode: axis R={geom['axis_R']:.4f} m, "
            f"Z={geom['axis_Z']:.4f} m, source points={len(boundary['R'])}"
        )
    else:
        boundary = None
        geom = compute_geometry(coords['x'], coords['y'], args.R0, args.kappa, args.delta)
        geom['geometry_model'] = 'miller'
        reference_R0 = float(args.R0)

    # Compute tokamak basis in the physical (r,theta,phi) geometry.
    phi_mesh = coords['z3'][:1, :1, :]  # (1,1,nz)

    ripple_cfg = {
        "enabled": bool(getattr(args, "toroidal_ripple", False)),
        "eps": float(getattr(args, "ripple_eps", 0.0)),
        "N": int(getattr(args, "ripple_n", 8)),
        "M": int(getattr(args, "ripple_m", 0)),
    }

    base_basis = compute_basis_vectors(
        geom,
        phi_mesh,
        theta_1d=coords['y'],
        ripple=ripple_cfg,
    )

    if ripple_cfg["enabled"]:
        # The ripple path is deliberately retained as a forensic/synthetic 3D
        # geometry mode.  It is not a canonical BOUT++ 5.x field-aligned
        # tokamak contract because a single axisymmetric psi/q transform is not
        # sufficient for the toroidally varying mapping.
        print(
            "[grid] WARNING: toroidal-ripple output is NON-CANONICAL for BOUT++ 5.x "
            "tokamak field-aligned use; intended for synthetic/forensic 3D geometry tests."
        )

        # Preserve the historical synthetic path but write explicit metadata so
        # diagnostics can distinguish mathematical consistency from BOUT++
        # canonical compatibility.
        er = np.stack([base_basis['er_X'], base_basis['er_Y'], base_basis['er_Z']], axis=-1)
        et = np.stack([base_basis['etheta_X'], base_basis['etheta_Y'], base_basis['etheta_Z']], axis=-1)
        ep = np.stack([base_basis['ephi_X'], base_basis['ephi_Y'], base_basis['ephi_Z']], axis=-1)
        J_raw = np.einsum('...i,...i', er, np.cross(et, ep))
        if float(np.nanmean(J_raw)) < 0.0:
            # Historical synthetic convention: reverse theta basis only.  This
            # mode is explicitly not presented as canonical BOUT++ 5.x output.
            base_basis['etheta_X'] = -base_basis['etheta_X']
            base_basis['etheta_Y'] = -base_basis['etheta_Y']
            base_basis['etheta_Z'] = -base_basis['etheta_Z']

        metric = compute_metric_tensor(base_basis)
        DX = np.full((args.nx, args.ny, args.nz), coords['dr'], dtype=float)
        DY = np.full((args.nx, args.ny, args.nz), coords['dtheta'], dtype=float)
        DZ = np.full((args.nx, args.ny, args.nz), coords['dphi'], dtype=float)
        jacobian = compute_jacobian(base_basis, DX, DY, DZ)
        q_xy = compute_q_poloidal(coords['x'], q_vals, geom, args.B0, reference_R0)
        bfield = compute_magnetic_field(coords['x'], geom, q_xy, args.B0, reference_R0, base_basis)
        legacy_shift = compute_shift_angle(q_xy, coords['y'], args.nx, args.ny, args.nz)
        shift_data = {
            'zShift': legacy_shift,
            'ShiftAngle': None,
            'ShiftTorsion': None,
            'IntShiftTorsion': None,
            'legacy_shiftAngle': legacy_shift,
        }
        curv = compute_curvature(args.curvature, bfield, base_basis, metric, geom, coords)
        bout_contract = {
            'canonical': False,
            'label': 'NONCANONICAL_SYNTHETIC_3D',
            'psi': coords['x'],
            'dx_1d': np.full(args.nx, coords['dr']),
            'q_actual': q_vals,
            'bout_J_max_rel_error': None,
            'bout_Bxy_max_rel_error': None,
            'toroidal_coordinate_sign': 1.0,
        }
    else:
        # Canonical BOUT++ 5.x axisymmetric field-aligned construction.
        bout = compute_bout_field_aligned_system(
            base_basis, geom, coords, q_vals, args.B0, reference_R0
        )
        metric = bout['metric']
        jacobian = {'J': bout['J'], 'surfvol': bout['surfvol']}
        bfield = bout['bfield']
        shift_data = {
            'zShift': bout['zShift'],
            'ShiftAngle': bout['ShiftAngle'],
            'ShiftTorsion': bout['ShiftTorsion'],
            'IntShiftTorsion': bout['IntShiftTorsion'],
            'legacy_shiftAngle': None,
        }
        curv_coords = dict(coords)
        # In canonical mode the radial coordinate used by BOUT++ is psi, not
        # the geometric minor-radius parameter r. Exact curvature derivatives
        # must therefore differentiate along psi.
        curv_coords['dr'] = bout['psi']
        curv = compute_curvature(args.curvature, bfield, bout['basis'], metric, geom, curv_coords)
        bout_contract = {
            'canonical': True,
            'label': 'BOUT5_FIELD_ALIGNED',
            'psi': bout['psi'],
            'dx_1d': bout['dx_1d'],
            'q_actual': bout['q_actual'],
            'bout_J_max_rel_error': bout['bout_J_max_rel_error'],
            'bout_Bxy_max_rel_error': bout['bout_Bxy_max_rel_error'],
            'toroidal_coordinate_sign': bout['toroidal_coordinate_sign'],
        }
        print(
            "[grid] BOUT++ 5.x canonical metric contract: "
            f"max rel dJ={bout['bout_J_max_rel_error']:.3e}, "
            f"max rel dBxy={bout['bout_Bxy_max_rel_error']:.3e}"
        )

    # Write the grid.  The writer maps the internal tensor representation onto
    # BOUT++'s fixed naming convention (g11=g^{11}, g_11=g_11).
    write_netcdf_grid(
        args.outfile, coords, geom, metric, jacobian, bfield,
        shift_data, curv, args, bout_contract=bout_contract
    )

    print("Grid generation time: %.2f seconds" % (time.time() - t0))
    print("Done.")


if __name__ == "__main__":
    main()


