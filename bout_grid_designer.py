#!/usr/bin/env python3
"""
bout_grid_designer.py  -  Chatwood Labs  v2.0.0
================================================
Interactive 2D poloidal cross-section designer and custom-boundary grid workflow for BOUT++ tokamak grids.

Companion scripts (default locations relative to this file):
    src/bout_tokamak_grid_generator.py
    src/bout_tokamak_grid_diagnostics.py

Directory layout created automatically:
    output/<run_id>/grid.nc
    reports/<run_id>/grid_report.html  (+ JSON, PNGs)
    saves/<name>.json

Usage:
    python3 bout_grid_designer.py [--generator PATH] [--diagnostics PATH]

Dependencies:
    python -m pip install -r requirements.txt
    Tk/Tkinter support from the local Python/platform installation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from scipy.interpolate import splprep, splev
import subprocess, threading, os, sys, argparse, math, json, secrets
try:
    from PIL import Image as _PILImage, ImageTk as _PILImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
from datetime import datetime
from pathlib import Path

__version__ = "2.0.0"

# ---- Paths ------
_HERE = Path(__file__).parent.resolve()
_DEFAULT_GENERATOR   = _HERE / "src" / "bout_tokamak_grid_generator.py"
_DEFAULT_DIAGNOSTICS = _HERE / "src" / "bout_tokamak_grid_diagnostics.py"
_DIR_OUTPUT  = _HERE / "output"
_DIR_REPORTS = _HERE / "reports"
_DIR_SAVES   = _HERE / "saves"

# ---- Colours ----
DARK_BG    = "#1e1e2e"
PANEL_BG   = "#2a2a3e"
ACCENT     = "#89b4fa"
ACCENT2    = "#a6e3a1"
WARN       = "#f38ba8"
WARN2      = "#fab387"
TEXT       = "#cdd6f4"
SUBTEXT    = "#6c7086"
CANVAS_BG  = "#11111b"
GRID_COL   = "#313244"
SHAPE_COL  = "#89b4fa"
SHAPE_FILL = "#1e3a5f"
POINT_COL  = "#fab387"
AXIS_COL   = "#585b70"
BTN_BG     = "#313244"

# ---- Preset canonical defaults ----
PRESET_DEFAULTS = {
    "circle":  dict(R0=6.2, a=2.0, kappa=1.0,  delta=0.0),
    "ellipse": dict(R0=6.2, a=2.0, kappa=1.7,  delta=0.0),
    "d_shape": dict(R0=6.2, a=2.0, kappa=1.7,  delta=0.33),
    "neg_d":   dict(R0=6.2, a=2.0, kappa=1.7,  delta=-0.33),
}


# ---- Run ID ----

def new_run_id() -> str:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    hex4 = secrets.token_hex(2).upper()
    return f"{ts}_{hex4}"


# ---- Geometry helpers ------

def _shoelace(R, Z):
    Rc, Zc = np.append(R, R[0]), np.append(Z, Z[0])
    cross = Rc[:-1]*Zc[1:] - Rc[1:]*Zc[:-1]
    area  = 0.5*np.sum(cross)
    if abs(area) < 1e-15:
        return area, float(np.mean(R)), float(np.mean(Z))
    return float(area), \
           float(np.sum((Rc[:-1]+Rc[1:])*cross)/(6*area)), \
           float(np.sum((Zc[:-1]+Zc[1:])*cross)/(6*area))


def check_convexity(R, Z):
    if len(R) < 3: return True, 1.0
    dR = np.diff(np.append(R, R[:2]))
    dZ = np.diff(np.append(Z, Z[:2]))
    cross = dR[:-1]*dZ[1:] - dZ[:-1]*dR[1:]
    np_, nn = int(np.sum(cross>0)), int(np.sum(cross<0))
    nt = np_+nn
    if nt == 0: return True, 1.0
    return (min(np_,nn)==0), float(max(np_,nn)/nt)


def extract_miller_params(R, Z):
    R, Z = np.asarray(R,float), np.asarray(Z,float)
    Rm,Rn,Zm,Zn = R.max(),R.min(),Z.max(),Z.min()
    Rc,Zc = 0.5*(Rm+Rn), 0.5*(Zm+Zn)
    _,Ra,Za = _shoelace(R,Z)
    aR,aZ = 0.5*(Rm-Rn), 0.5*(Zm-Zn)
    a     = aR
    kappa = aZ/aR if aR>1e-9 else 1.0
    idx   = np.argmax(Z)
    # Standard Miller triangularity: δ = (R0 - R_top) / a
    # This is the direct dimensionless definition, not arcsin of it.
    # The generator uses arcsin(δ) internally in the theta mapping,
    # but the parameter δ itself is the sine value, not the angle.
    sin_d = np.clip((Rc-R[idx])/a if a>1e-9 else 0.0, -1, 1)
    delta = float(sin_d)   # δ = sin_d directly, NOT arcsin(sin_d)
    is_conv, cfrac = check_convexity(R,Z)
    return dict(R0=float(Rc), a=float(a),
                kappa=float(np.clip(kappa,0.1,4.9)),
                delta=float(np.clip(delta,-0.85,0.85)),
                R_min=float(Rn), R_max=float(Rm),
                Z_min=float(Zn), Z_max=float(Zm),
                Z_centre=float(Zc), R_area=float(Ra), Z_area=float(Za),
                is_convex=is_conv, convexity_fraction=cfrac)


def smooth_contour(Rp, Zp, n=300, smooth_fraction=0.0):
    """Return a periodic spline through (or gently near) the supplied points.

    ``smooth_fraction=0`` is an interpolating spline and is used for normal
    editing so the visible boundary follows the user's control points exactly.
    The toolbar Smooth action supplies a small non-zero fraction and then commits
    the softened curve back into the control points.
    """
    R,Z = np.asarray(Rp,float), np.asarray(Zp,float)
    if len(R)<3: return R,Z
    try:
        scale = max(float(np.ptp(R)), float(np.ptp(Z)), 1e-9)
        # scipy splprep's smoothing factor is a sum of squared residuals.
        # Express the requested smoothing in geometry-relative terms so behaviour
        # is sensible for both compact and reactor-scale drawings.
        sf = max(0.0, float(smooth_fraction))
        s_val = len(R) * (sf * scale) ** 2
        tck,_ = splprep([np.append(R,R[0]),np.append(Z,Z[0])],
                        s=s_val, per=True, k=min(3, len(R)-1))
        Rs,Zs = splev(np.linspace(0,1,n,endpoint=False), tck)
        return np.asarray(Rs,float),np.asarray(Zs,float)
    except Exception:
        return R,Z


def _cross2(ax, ay, bx, by):
    return ax*by - ay*bx


def _segments_intersect(p1, p2, q1, q2, tol=1e-10):
    """Return True for a non-adjacent 2-D segment intersection."""
    p1=np.asarray(p1,float); p2=np.asarray(p2,float)
    q1=np.asarray(q1,float); q2=np.asarray(q2,float)
    r=p2-p1; s=q2-q1
    den=_cross2(r[0],r[1],s[0],s[1])
    qp=q1-p1
    if abs(den) <= tol:
        # Parallel/collinear.  Overlap of non-adjacent edges is invalid.
        if abs(_cross2(qp[0],qp[1],r[0],r[1])) > tol:
            return False
        rr=float(np.dot(r,r))
        if rr <= tol: return False
        t0=float(np.dot(q1-p1,r)/rr); t1=float(np.dot(q2-p1,r)/rr)
        lo=max(min(t0,t1),0.0); hi=min(max(t0,t1),1.0)
        return hi-lo > tol
    t=_cross2(qp[0],qp[1],s[0],s[1])/den
    u=_cross2(qp[0],qp[1],r[0],r[1])/den
    return (-tol <= t <= 1+tol) and (-tol <= u <= 1+tol)


def _point_in_polygon(px, py, R, Z):
    inside=False
    n=len(R)
    j=n-1
    for i in range(n):
        xi,yi=float(R[i]),float(Z[i]); xj,yj=float(R[j]),float(Z[j])
        if ((yi>py)!=(yj>py)):
            xhit=(xj-xi)*(py-yi)/(yj-yi)+xi
            if px < xhit: inside=not inside
        j=i
    return inside


def _ray_intersections(axis_R, axis_Z, theta, R, Z, tol=1e-9):
    """Positive distances where a ray from the axis meets the closed contour."""
    dx,dy=math.cos(theta),math.sin(theta)
    hits=[]
    n=len(R)
    for i in range(n):
        x1,y1=float(R[i]),float(Z[i])
        x2,y2=float(R[(i+1)%n]),float(Z[(i+1)%n])
        ex,ey=x2-x1,y2-y1
        den=_cross2(dx,dy,ex,ey)
        if abs(den) <= tol:
            continue
        qx,qy=x1-axis_R,y1-axis_Z
        t=_cross2(qx,qy,ex,ey)/den
        u=_cross2(qx,qy,dx,dy)/den
        if t > tol and -tol <= u <= 1+tol:
            hits.append(float(t))
    if not hits:
        return []
    hits.sort()
    unique=[hits[0]]
    merge_tol=max(tol, 1e-7*max(1.0,hits[-1]))
    for value in hits[1:]:
        if abs(value-unique[-1]) > merge_tol:
            unique.append(value)
    return unique


def validate_custom_boundary(R, Z, n_rays=192):
    """Validate that a custom contour can form a single-valued nested grid.

    Non-convexity alone is *not* an error.  The hard requirement for the radial
    mapping is that the boundary is simple and star-shaped about its chosen
    magnetic axis: every ray from that axis must meet the boundary exactly once.
    """
    R=np.asarray(R,float); Z=np.asarray(Z,float)
    if len(R) != len(Z) or len(R) < 8:
        return dict(valid=False, reason="At least 8 finite boundary points are required.")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
        return dict(valid=False, reason="Boundary contains NaN or infinite coordinates.")
    if np.any(R <= 0.0):
        return dict(valid=False, reason="Boundary reaches R <= 0, which cannot be revolved into a tokamak grid.")

    area,axis_R,axis_Z=_shoelace(R,Z)
    scale=max(float(np.ptp(R)),float(np.ptp(Z)),1.0)
    if abs(area) < 1e-8*scale*scale:
        return dict(valid=False, reason="Boundary has negligible enclosed area.")

    # Reject self-intersections.  Adjacent segments naturally share a vertex.
    n=len(R)
    seg_tol=1e-10*scale
    for i in range(n):
        p1=(R[i],Z[i]); p2=(R[(i+1)%n],Z[(i+1)%n])
        for j in range(i+1,n):
            if j==i or j==(i+1)%n or (j+1)%n==i:
                continue
            q1=(R[j],Z[j]); q2=(R[(j+1)%n],Z[(j+1)%n])
            if _segments_intersect(p1,p2,q1,q2,tol=seg_tol):
                return dict(valid=False, reason="Boundary self-intersects; redraw or smooth the contour.")

    if not _point_in_polygon(axis_R,axis_Z,R,Z):
        return dict(valid=False, reason="The inferred magnetic axis lies outside the boundary.")

    radii=[]
    bad=[]
    for theta in np.linspace(0.0,2*np.pi,int(n_rays),endpoint=False):
        hits=_ray_intersections(axis_R,axis_Z,float(theta),R,Z,tol=1e-10*scale)
        if len(hits)!=1:
            bad.append((float(theta),len(hits)))
            if len(bad)>=4: break
        else:
            radii.append(hits[0])
    if bad:
        return dict(
            valid=False, axis_R=float(axis_R), axis_Z=float(axis_Z),
            reason=("Boundary is not single-valued about the magnetic axis; some radial rays "
                    "intersect it zero or multiple times. Smooth or redraw the contour."),
            bad_rays=bad,
        )
    if not radii or min(radii) < 1e-4*scale:
        return dict(valid=False, axis_R=float(axis_R), axis_Z=float(axis_Z),
                    reason="Boundary approaches the magnetic axis too closely for a stable nested grid.")

    return dict(valid=True, axis_R=float(axis_R), axis_Z=float(axis_Z),
                min_radius=float(min(radii)), max_radius=float(max(radii)))


# ---- Preset generators ----

def preset_circle(R0,a,n=128):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    return R0+a*np.cos(t), a*np.sin(t)

def preset_ellipse(R0,a,kappa,n=128):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    return R0+a*np.cos(t), kappa*a*np.sin(t)

def preset_d_shape(R0,a,kappa,delta,n=256):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    ad=np.arcsin(np.clip(delta,-0.999,0.999))
    return R0+a*np.cos(t+ad*np.sin(t)), kappa*a*np.sin(t)

def preset_neg_d(R0,a,kappa,delta,n=256):
    return preset_d_shape(R0,a,kappa,-abs(delta),n)


# ---- Coordinate mapping ----

class CoordMap:
    def __init__(self,W,H,margin=40):
        self.W,self.H,self.M=W,H,margin
        self.set_view(0,12,-5,5)

    def set_view(self,Rl,Rh,Zl,Zh):
        self.R_lo,self.R_hi,self.Z_lo,self.Z_hi=Rl,Rh,Zl,Zh

    def fit_to_shape(self,R,Z,pad=0.25):
        if not len(R): return
        Rc=0.5*(R.max()+R.min()); Zc=0.5*(Z.max()+Z.min())
        Rs=R.max()-R.min(); Zs=Z.max()-Z.min()
        dW=self.W-2*self.M; dH=self.H-2*self.M
        sc=min(dW/(Rs*(1+pad)), dH/(Zs*(1+pad)))
        pw,ph=dW/sc, dH/sc
        self.R_lo,self.R_hi=Rc-pw/2,Rc+pw/2
        self.Z_lo,self.Z_hi=Zc-ph/2,Zc+ph/2

    def to_canvas(self,R,Z):
        fx=(R-self.R_lo)/(self.R_hi-self.R_lo)
        fz=(Z-self.Z_lo)/(self.Z_hi-self.Z_lo)
        return self.M+fx*(self.W-2*self.M), (self.H-self.M)-fz*(self.H-2*self.M)

    def to_physical(self,cx,cy):
        fx=(cx-self.M)/(self.W-2*self.M)
        fz=((self.H-self.M)-cy)/(self.H-2*self.M)
        return self.R_lo+fx*(self.R_hi-self.R_lo), self.Z_lo+fz*(self.Z_hi-self.Z_lo)


# ---- Application ------

class BoutGridDesigner(tk.Tk):

    def __init__(self, generator=None, diagnostics=None):
        super().__init__()
        self.title(f"BOUT++ Grid Studio  v{__version__}  -  Chatwood Labs")
        self.configure(bg=DARK_BG)
        self.minsize(1100, 780)

        self._gen_path  = str(generator  or _DEFAULT_GENERATOR)
        self._diag_path = str(diagnostics or _DEFAULT_DIAGNOSTICS)

        # ---- run state --------
        self._run_id      = new_run_id()
        self.draw_R       = []
        self.draw_Z       = []
        self.smooth_R     = np.array([])
        self.smooth_Z     = np.array([])
        self.miller       = {}
        self._geometry_mode = "custom"  # "miller" for untouched presets, "custom" for drawn/edited boundaries
        self._updating_miller_vars = False
        self._convex_warn = False
        self._convex_frac = 1.0
        self._drag_idx    = None
        self._pan_start   = None
        self.tool         = tk.StringVar(value="spline")
        self.cmap         = CoordMap(600, 560)

        self._build_ui()
        self._refresh_run_id_display()
        self._redraw()

    # ---- UI ------

    def _build_ui(self):
        # toolbar
        tb = tk.Frame(self, bg=PANEL_BG, pady=3)
        tb.pack(side=tk.TOP, fill=tk.X)
        self._build_toolbar(tb)

        # main area
        main = tk.Frame(self, bg=DARK_BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        # canvas
        cf = tk.Frame(main, bg=DARK_BG)
        cf.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        cf.rowconfigure(0, weight=1); cf.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(cf, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._bind_canvas()

        # right panel - uses notebook for tabs
        rp = tk.Frame(main, bg=PANEL_BG, width=340)
        rp.grid(row=0, column=1, sticky="nsew", padx=(0,8), pady=8)
        rp.pack_propagate(False)
        self._build_right_panel(rp)

        # ---- Log panel (collapsible) ------─
        log_outer = tk.Frame(self, bg=DARK_BG)
        log_outer.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0,8))

        # header bar with title + buttons
        log_hdr = tk.Frame(log_outer, bg=PANEL_BG)
        log_hdr.pack(fill=tk.X)
        tk.Label(log_hdr, text=" Run Log ", bg=PANEL_BG, fg=SUBTEXT,
                 font=("Courier",9)).pack(side=tk.LEFT, padx=4)
        self._log_visible = False
        self._log_hide_btn = tk.Button(
            log_hdr, text="▶ Show", command=self._toggle_log,
            bg=BTN_BG, fg=SUBTEXT, activebackground=BTN_BG, activeforeground=TEXT,
            relief=tk.FLAT, font=("Helvetica",8), padx=6, pady=1, cursor="hand2")
        self._log_hide_btn.pack(side=tk.RIGHT, padx=2, pady=2)
        tk.Button(log_hdr, text="🗑 Clear", command=self._log_clear,
                  bg=BTN_BG, fg=SUBTEXT, activebackground=BTN_BG, activeforeground=WARN,
                  relief=tk.FLAT, font=("Helvetica",8), padx=6, pady=1,
                  cursor="hand2").pack(side=tk.RIGHT, padx=2, pady=2)

        # collapsible body - starts hidden
        self._log_body = tk.Frame(log_outer, bg=DARK_BG)
        self.log = scrolledtext.ScrolledText(
            self._log_body, height=5, bg="#0d0d1a", fg=ACCENT2,
            font=("Courier",9), state=tk.DISABLED, wrap=tk.WORD)
        self.log.pack(fill=tk.X, padx=4, pady=4)

    def _build_toolbar(self, parent):
        bkw = dict(bg=BTN_BG, fg=TEXT, activebackground=ACCENT, activeforeground=DARK_BG,
                   relief=tk.FLAT, font=("Helvetica",10), padx=8, pady=3, cursor="hand2")

        tk.Label(parent, text="Presets:", bg=PANEL_BG, fg=SUBTEXT,
                 font=("Helvetica",9)).pack(side=tk.LEFT, padx=(8,2))
        for lbl,fn in [("● Circle",self._preset_circle),("⬭ Ellipse",self._preset_ellipse),
                        ("⊃ D-shape",self._preset_d),("⊂ Neg-D",self._preset_neg_d)]:
            tk.Button(parent, text=lbl, command=fn, **bkw).pack(side=tk.LEFT, padx=2)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Label(parent, text="Tool:", bg=PANEL_BG, fg=SUBTEXT,
                 font=("Helvetica",9)).pack(side=tk.LEFT, padx=(0,2))
        for lbl,val in [("✏ Draw","spline"),("✋ Pan","pan"),("✖ Erase","erase")]:
            tk.Radiobutton(parent, text=lbl, variable=self.tool, value=val,
                           bg=PANEL_BG, fg=TEXT, selectcolor=DARK_BG,
                           activebackground=PANEL_BG, font=("Helvetica",9)
                           ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        for lbl,fn in [("🔄 Smooth",self._do_smooth),("🗑 Clear",self._clear),
                        ("⤢ Fit",self._fit_view)]:
            tk.Button(parent, text=lbl, command=fn, **bkw).pack(side=tk.LEFT, padx=2)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Button(parent, text="🌐 3D View", command=self._open_3d_view,
                  bg=BTN_BG, fg=ACCENT2, activebackground=ACCENT2,
                  activeforeground=DARK_BG, relief=tk.FLAT,
                  font=("Helvetica",10), padx=8, pady=3,
                  cursor="hand2").pack(side=tk.LEFT, padx=2)

        ttk.Separator(parent, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        for lbl,fn in [("💾 Save",self._save_design),("📂 Load",self._load_design)]:
            tk.Button(parent, text=lbl, command=fn, **bkw).pack(side=tk.LEFT, padx=2)

    def _build_right_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---- Tab 1: Design ----
        t1 = tk.Frame(nb, bg=PANEL_BG)
        nb.add(t1, text="  Design  ")
        self._build_design_tab(t1)

        # ---- Tab 2: Report ----
        t2 = tk.Frame(nb, bg=PANEL_BG)
        nb.add(t2, text="  Report  ")
        self._build_report_tab(t2)
        self._nb = nb

    def _build_design_tab(self, parent):
        # Run ID bar
        rid_frame = tk.Frame(parent, bg=PANEL_BG)
        rid_frame.pack(fill=tk.X, padx=6, pady=(6,2))
        tk.Label(rid_frame, text="Run ID:", bg=PANEL_BG, fg=SUBTEXT,
                 font=("Helvetica",8)).pack(side=tk.LEFT)
        self._run_id_lbl = tk.Label(rid_frame, text="", bg=PANEL_BG, fg=ACCENT2,
                                     font=("Courier",8))
        self._run_id_lbl.pack(side=tk.LEFT, padx=4)
        tk.Button(rid_frame, text="⟳ New Run", command=self._new_run_id,
                  bg=BTN_BG, fg=WARN2, activebackground=WARN2, activeforeground=DARK_BG,
                  relief=tk.FLAT, font=("Helvetica",8), padx=6, pady=1,
                  cursor="hand2").pack(side=tk.RIGHT)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=4)

        # Miller params
        mf = tk.LabelFrame(parent, text=" Miller Parameters ",
                            bg=PANEL_BG, fg=ACCENT, font=("Helvetica",9,"bold"))
        mf.pack(fill=tk.X, padx=6, pady=(0,3))
        self._miller_frame = mf
        self._miller_vars = {}
        self._miller_entries = {}
        for row,(lbl,key,val) in enumerate([
            ("R₀ (m)","R0","6.20"),("a (m)","a","2.00"),
            ("κ","kappa","1.70"),("δ","delta","0.33")]):
            tk.Label(mf,text=lbl,width=8,anchor="w",bg=PANEL_BG,fg=TEXT,
                     font=("Helvetica",9)).grid(row=row,column=0,sticky="w",padx=4,pady=1)
            v=tk.StringVar(value=val); self._miller_vars[key]=v
            ent=tk.Entry(mf,textvariable=v,width=12,bg=DARK_BG,fg=ACCENT2,
                         disabledbackground=DARK_BG,disabledforeground=SUBTEXT,
                         insertbackground=ACCENT2,font=("Courier",9),relief=tk.FLAT)
            ent.grid(row=row,column=1,sticky="w",padx=4,pady=1)
            self._miller_entries[key]=ent
            v.trace_add("write", lambda *_args: self._on_miller_entry_change())

        # Grid params
        gf = tk.LabelFrame(parent, text=" Grid Parameters ",
                            bg=PANEL_BG, fg=ACCENT, font=("Helvetica",9,"bold"))
        gf.pack(fill=tk.X, padx=6, pady=3)
        self._grid_vars = {}
        for row,(lbl,key,val) in enumerate([
            ("B₀ (T)","B0","5.3"),("nx","nx","68"),("ny","ny","64"),
            ("nz","nz","128"),("q₀","q0","1.05"),("qₐ","qa","3.5"),
            ("xmin_frac","xmin_frac","0.1")]):
            tk.Label(gf,text=lbl,width=10,anchor="w",bg=PANEL_BG,fg=TEXT,
                     font=("Helvetica",9)).grid(row=row,column=0,sticky="w",padx=4,pady=1)
            v=tk.StringVar(value=val); self._grid_vars[key]=v
            tk.Entry(gf,textvariable=v,width=12,bg=DARK_BG,fg=ACCENT2,
                     insertbackground=ACCENT2,font=("Courier",9),relief=tk.FLAT
                     ).grid(row=row,column=1,sticky="w",padx=4,pady=1)
        for row,(lbl,attr,choices,dflt) in enumerate([
            ("q-form","_qform_var",["quadratic","linear","cubic"],"quadratic"),
            ("curvature","_curv_var",["exact","simple","none"],"exact"),
            ("precision","_prec_var",["f4","f8"],"f8")],start=7):
            tk.Label(gf,text=lbl,width=10,anchor="w",bg=PANEL_BG,fg=TEXT,
                     font=("Helvetica",9)).grid(row=row,column=0,sticky="w",padx=4,pady=1)
            v=tk.StringVar(value=dflt); setattr(self,attr,v)
            ttk.Combobox(gf,textvariable=v,values=choices,width=10,state="readonly"
                         ).grid(row=row,column=1,sticky="w",padx=4,pady=1)

        # Paths
        pf = tk.LabelFrame(parent, text=" Paths ",
                            bg=PANEL_BG, fg=ACCENT, font=("Helvetica",9,"bold"))
        pf.pack(fill=tk.X, padx=6, pady=3)
        self._genpath_var  = tk.StringVar(value=self._gen_path)
        self._diagpath_var = tk.StringVar(value=self._diag_path)
        for row,(lbl,var,cmd) in enumerate([
            ("Generator", self._genpath_var,  self._pick_generator),
            ("Diagnostics",self._diagpath_var, self._pick_diagnostics)]):
            tk.Label(pf,text=lbl,width=10,anchor="w",bg=PANEL_BG,fg=TEXT,
                     font=("Helvetica",9)).grid(row=row,column=0,sticky="w",padx=4,pady=1)
            tk.Entry(pf,textvariable=var,width=14,bg=DARK_BG,fg=SUBTEXT,
                     insertbackground=ACCENT2,font=("Courier",8),relief=tk.FLAT
                     ).grid(row=row,column=1,sticky="ew",padx=2,pady=1)
            tk.Button(pf,text="📂",command=cmd,bg=PANEL_BG,fg=TEXT,
                      relief=tk.FLAT,font=("Helvetica",9),cursor="hand2"
                      ).grid(row=row,column=2,padx=2,pady=1)
        pf.columnconfigure(1, weight=1)

        # Params summary
        self._params_lbl = tk.Label(parent, text="No shape loaded",
                                     bg=PANEL_BG, fg=ACCENT2,
                                     font=("Courier",8), justify=tk.LEFT, anchor="w")
        self._params_lbl.pack(fill=tk.X, padx=8, pady=(4,2))

        # Generate button
        self._run_btn = tk.Button(parent, text="▶  Generate Grid + Report",
                                   command=self._run_generator,
                                   bg=ACCENT, fg=DARK_BG, activebackground=ACCENT2,
                                   activeforeground=DARK_BG, relief=tk.FLAT,
                                   font=("Helvetica",11,"bold"), pady=8, cursor="hand2")
        self._run_btn.pack(fill=tk.X, padx=6, pady=6, side=tk.BOTTOM)

        # trace
        all_vars = list(self._miller_vars.values()) + list(self._grid_vars.values()) + \
                   [self._qform_var, self._curv_var, self._prec_var,
                    self._genpath_var, self._diagpath_var]
        for v in all_vars:
            v.trace_add("write", lambda *_: self._update_params_summary())
        self._update_params_summary()

    def _build_report_tab(self, parent):
        # Header row
        hf = tk.Frame(parent, bg=PANEL_BG)
        hf.pack(fill=tk.X, padx=6, pady=(6,2))
        tk.Label(hf, text="Diagnostics Report", bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica",10,"bold")).pack(side=tk.LEFT)
        tk.Button(hf, text="🌐 Open HTML", command=self._open_html_report,
                  bg=BTN_BG, fg=TEXT, relief=tk.FLAT, font=("Helvetica",8),
                  cursor="hand2").pack(side=tk.RIGHT)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=3)

        # Verdict banner
        self._verdict_lbl = tk.Label(parent, text="No report yet - generate a grid first",
                                      bg=PANEL_BG, fg=SUBTEXT,
                                      font=("Helvetica",10,"bold"), pady=4)
        self._verdict_lbl.pack(fill=tk.X, padx=6)

        # Inner notebook: Summary | Geometry | B-Field
        rnb = ttk.Notebook(parent)
        rnb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._report_nb = rnb

        # ---- Tab: Summary ----
        ts = tk.Frame(rnb, bg=PANEL_BG)
        rnb.add(ts, text=" Summary ")

        sf = tk.LabelFrame(ts, text=" Grid Stats ", bg=PANEL_BG, fg=ACCENT,
                            font=("Helvetica",9,"bold"))
        sf.pack(fill=tk.X, padx=6, pady=(6,3))
        self._stat_labels = {}
        for row,(lbl,key) in enumerate([
            ("nx × ny × nz","dims"),("Aspect ratio","aspect"),
            ("J range","j_range"),("B range","b_range"),
            ("|B| min","b_min"),("Conditioning","cond")]):
            tk.Label(sf,text=lbl,width=14,anchor="w",bg=PANEL_BG,fg=TEXT,
                     font=("Helvetica",9)).grid(row=row,column=0,sticky="w",padx=4,pady=1)
            v=tk.Label(sf,text="-",anchor="w",bg=PANEL_BG,fg=ACCENT2,font=("Courier",9))
            v.grid(row=row,column=1,sticky="w",padx=4,pady=1)
            self._stat_labels[key]=v

        cf2 = tk.LabelFrame(ts, text=" Check Results ", bg=PANEL_BG, fg=ACCENT,
                             font=("Helvetica",9,"bold"))
        cf2.pack(fill=tk.BOTH, expand=True, padx=6, pady=3)
        self._checks_text = scrolledtext.ScrolledText(
            cf2, height=8, bg="#0d0d1a", fg=TEXT,
            font=("Courier",8), state=tk.DISABLED, wrap=tk.NONE)
        self._checks_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._checks_text.tag_configure("pass",    foreground=ACCENT2)
        self._checks_text.tag_configure("fail",    foreground=WARN)
        self._checks_text.tag_configure("warn",    foreground=WARN2)
        self._checks_text.tag_configure("skip",    foreground=SUBTEXT)
        self._checks_text.tag_configure("heading", foreground=ACCENT,
                                         font=("Courier",8,"bold"))

        op = tk.LabelFrame(ts, text=" Output Paths ", bg=PANEL_BG, fg=ACCENT,
                            font=("Helvetica",9,"bold"))
        op.pack(fill=tk.X, padx=6, pady=(0,4))
        self._path_grid_lbl   = tk.Label(op,text="-",bg=PANEL_BG,fg=SUBTEXT,
                                          font=("Courier",7),anchor="w",wraplength=290)
        self._path_report_lbl = tk.Label(op,text="-",bg=PANEL_BG,fg=SUBTEXT,
                                          font=("Courier",7),anchor="w",wraplength=290)
        self._path_grid_lbl.pack(fill=tk.X,padx=4,pady=1)
        self._path_report_lbl.pack(fill=tk.X,padx=4,pady=1)

        # ---- Tab: Geometry ----
        tg = tk.Frame(rnb, bg=CANVAS_BG)
        rnb.add(tg, text=" Geometry ")
        self._geom_viewer  = ImageViewer(tg, label="Geometry")
        self._geom_viewer.pack(fill=tk.BOTH, expand=True)

        # ---- Tab: B-Field ----
        tb2 = tk.Frame(rnb, bg=CANVAS_BG)
        rnb.add(tb2, text=" B-Field ")
        self._bfield_viewer = ImageViewer(tb2, label="B-Field")
        self._bfield_viewer.pack(fill=tk.BOTH, expand=True)

        # keep PIL image references on the viewer objects themselves
        self._pil_geom   = None
        self._pil_bfield = None

        # legacy canvas refs used by _rescale_report_image (kept for compat)
        self._geom_canvas   = self._geom_viewer.canvas
        self._bfield_canvas = self._bfield_viewer.canvas

    # ---- Canvas bindings ----

    def _bind_canvas(self):
        for event,fn in [("<Button-1>",self._on_click),("<B1-Motion>",self._on_drag),
                          ("<ButtonRelease-1>",self._on_release),("<Button-3>",self._on_rclick),
                          ("<MouseWheel>",self._on_scroll),("<Button-4>",self._on_scroll),
                          ("<Button-5>",self._on_scroll),("<Configure>",self._on_resize)]:
            self.canvas.bind(event, fn)

    def _on_resize(self,e):
        self.cmap.W=e.width; self.cmap.H=e.height; self._redraw()

    def _on_click(self,e):
        t=self.tool.get()
        if t=="spline":
            idx=self._nearest(e.x,e.y)
            if idx is not None:
                self._drag_idx=idx
                self._geometry_mode="custom"
            else:
                R,Z=self.cmap.to_physical(e.x,e.y)
                self.draw_R.append(R); self.draw_Z.append(Z)
                self._geometry_mode="custom"
                self._update_shape()
        elif t=="pan":
            self._pan_start=(e.x,e.y,self.cmap.R_lo,self.cmap.R_hi,
                              self.cmap.Z_lo,self.cmap.Z_hi)
        elif t=="erase":
            idx=self._nearest(e.x,e.y,thr=14)
            if idx is not None:
                self.draw_R.pop(idx); self.draw_Z.pop(idx); self._geometry_mode="custom"; self._update_shape()

    def _on_drag(self,e):
        if self.tool.get()=="spline" and self._drag_idx is not None:
            R,Z=self.cmap.to_physical(e.x,e.y)
            self.draw_R[self._drag_idx]=R; self.draw_Z[self._drag_idx]=Z
            self._geometry_mode="custom"
            self._update_shape()
        elif self.tool.get()=="pan" and self._pan_start:
            x0,y0,Rl,Rh,Zl,Zh=self._pan_start
            dR=(x0-e.x)/(self.cmap.W-2*self.cmap.M)*(Rh-Rl)
            dZ=(e.y-y0)/(self.cmap.H-2*self.cmap.M)*(Zh-Zl)
            self.cmap.R_lo=Rl+dR; self.cmap.R_hi=Rh+dR
            self.cmap.Z_lo=Zl+dZ; self.cmap.Z_hi=Zh+dZ
            self._redraw()

    def _on_release(self,e): self._drag_idx=None; self._pan_start=None
    def _on_rclick(self,e):
        idx=self._nearest(e.x,e.y,thr=14)
        if idx is not None:
            self.draw_R.pop(idx); self.draw_Z.pop(idx); self._geometry_mode="custom"; self._update_shape()

    def _on_scroll(self,e):
        sc=0.85 if (e.num==4 or e.delta>0) else 1.15
        R0,Z0=self.cmap.to_physical(e.x,e.y)
        self.cmap.R_lo=R0+(self.cmap.R_lo-R0)*sc; self.cmap.R_hi=R0+(self.cmap.R_hi-R0)*sc
        self.cmap.Z_lo=Z0+(self.cmap.Z_lo-Z0)*sc; self.cmap.Z_hi=Z0+(self.cmap.Z_hi-Z0)*sc
        self._redraw()

    def _nearest(self,cx,cy,thr=10):
        best,bd=None,thr
        for i,(R,Z) in enumerate(zip(self.draw_R,self.draw_Z)):
            px,py=self.cmap.to_canvas(R,Z)
            d=math.hypot(cx-px,cy-py)
            if d<bd: bd=d; best=i
        return best

    # ---- Shape state --------

    def _on_miller_entry_change(self):
        """Keep untouched analytic presets WYSIWYG when parameters are edited."""
        if self._geometry_mode!="miller" or self._updating_miller_vars:
            return
        try:
            R0=float(self._miller_vars["R0"].get())
            a=float(self._miller_vars["a"].get())
            kappa=float(self._miller_vars["kappa"].get())
            delta=float(self._miller_vars["delta"].get())
        except (TypeError,ValueError):
            return  # transient state while the user is typing
        if R0<=0 or a<=0 or kappa<=0 or abs(delta)>=0.999:
            return
        R,Z=preset_d_shape(R0,a,kappa,delta,n=256)
        self.smooth_R=np.asarray(R,float); self.smooth_Z=np.asarray(Z,float)
        step=max(1,len(R)//32)
        self.draw_R=[float(v) for v in R[::step]]
        self.draw_Z=[float(v) for v in Z[::step]]
        self.miller=extract_miller_params(R,Z)
        self._convex_warn=False; self._convex_frac=1.0
        if hasattr(self,"canvas"):
            self._redraw(); self._update_params_summary()

    def _refresh_geometry_mode_ui(self):
        custom=(self._geometry_mode=="custom")
        if hasattr(self,"_miller_frame"):
            self._miller_frame.configure(text=(
                " Equivalent Miller Fit (read-only) " if custom else " Miller Parameters "
            ))
        for ent in getattr(self,"_miller_entries",{}).values():
            ent.configure(state=(tk.DISABLED if custom else tk.NORMAL))

    def _update_shape(self):
        if len(self.draw_R)>=3:
            self.smooth_R,self.smooth_Z=smooth_contour(self.draw_R,self.draw_Z)
            self.miller=extract_miller_params(self.smooth_R,self.smooth_Z)
            self._convex_warn=not self.miller.get("is_convex",True)
            self._convex_frac=self.miller.get("convexity_fraction",1.0)
            self._updating_miller_vars=True
            try:
                for key,var in self._miller_vars.items():
                    val=self.miller.get(key)
                    if val is not None: var.set(f"{val:.4f}")
            finally:
                self._updating_miller_vars=False
        else:
            self.smooth_R=np.array([]); self.smooth_Z=np.array([])
            self.miller={}; self._geometry_mode="custom"; self._convex_warn=False; self._convex_frac=1.0
        self._refresh_geometry_mode_ui()
        self._redraw(); self._update_params_summary()

    def _load_preset(self, R, Z, params):
        self._geometry_mode="miller"
        self.draw_R=list(R[::max(1,len(R)//32)])
        self.draw_Z=list(Z[::max(1,len(Z)//32)])
        self.smooth_R=R; self.smooth_Z=Z
        self.miller=extract_miller_params(R,Z)
        self._convex_warn=False; self._convex_frac=1.0
        self._updating_miller_vars=True
        try:
            for key in ("R0","a","kappa","delta"):
                self._miller_vars[key].set(f"{params[key]:.4f}")
        finally:
            self._updating_miller_vars=False
        self.cmap.fit_to_shape(R,Z)
        self._refresh_geometry_mode_ui()
        self._redraw(); self._update_params_summary()

    def _get_R0_a(self):
        try:
            R0=float(self._miller_vars["R0"].get())
            if R0<0.5: R0=6.2
        except: R0=6.2
        try:
            a=float(self._miller_vars["a"].get())
            if a<0.1 or a>20: a=2.0
        except: a=2.0
        return R0,a

    def _preset_circle(self):
        p=PRESET_DEFAULTS["circle"].copy()
        R,Z=preset_circle(p["R0"],p["a"]); self._load_preset(R,Z,p)

    def _preset_ellipse(self):
        p=PRESET_DEFAULTS["ellipse"].copy()
        R,Z=preset_ellipse(p["R0"],p["a"],p["kappa"]); self._load_preset(R,Z,p)

    def _preset_d(self):
        p=PRESET_DEFAULTS["d_shape"].copy()
        R,Z=preset_d_shape(p["R0"],p["a"],p["kappa"],p["delta"]); self._load_preset(R,Z,p)

    def _preset_neg_d(self):
        p=PRESET_DEFAULTS["neg_d"].copy()
        R,Z=preset_neg_d(p["R0"],p["a"],p["kappa"],p["delta"]); self._load_preset(R,Z,p)

    # ---- Toolbar actions ----

    def _do_smooth(self):
        if len(self.draw_R)>=3:
            # Apply a deliberately modest periodic smoothing pass, then commit the
            # result back into the editable control points.  This makes the Smooth
            # button materially change the requested boundary rather than merely
            # drawing the same interpolating spline at a higher sample count.
            Rs,Zs=smooth_contour(self.draw_R,self.draw_Z,n=384,smooth_fraction=0.015)
            keep=48 if len(Rs)>=48 else len(Rs)
            idx=np.linspace(0,len(Rs),keep,endpoint=False,dtype=int)
            self.draw_R=[float(Rs[i]) for i in idx]
            self.draw_Z=[float(Zs[i]) for i in idx]
            self._geometry_mode="custom"
            self._update_shape()
            self.cmap.fit_to_shape(self.smooth_R,self.smooth_Z)

    def _clear(self):
        self.draw_R=[]; self.draw_Z=[]; self.smooth_R=np.array([]); self.smooth_Z=np.array([])
        self.miller={}; self._geometry_mode="custom"; self._convex_warn=False; self._convex_frac=1.0
        self.cmap.set_view(0,12,-5,5); self._refresh_geometry_mode_ui(); self._redraw(); self._update_params_summary()

    def _fit_view(self):
        arr=self.smooth_R if len(self.smooth_R)>0 else np.array(self.draw_R)
        if len(arr)>0:
            self.cmap.fit_to_shape(arr, self.smooth_Z if len(self.smooth_R)>0
                                   else np.array(self.draw_Z))
        self._redraw()

    def _new_run_id(self):
        self._run_id=new_run_id()
        self._refresh_run_id_display()
        self._log(f"[New Run ID] {self._run_id}\n")

    def _refresh_run_id_display(self):
        self._run_id_lbl.configure(text=self._run_id)

    def _pick_generator(self):
        p=filedialog.askopenfilename(filetypes=[("Python","*.py"),("All","*.*")])
        if p: self._genpath_var.set(p)

    def _pick_diagnostics(self):
        p=filedialog.askopenfilename(filetypes=[("Python","*.py"),("All","*.*")])
        if p: self._diagpath_var.set(p)

    # ---- Save / Load --------

    def _save_design(self):
        _DIR_SAVES.mkdir(parents=True, exist_ok=True)
        path=filedialog.asksaveasfilename(
            initialdir=str(_DIR_SAVES),
            defaultextension=".json",
            filetypes=[("BOUT Grid Design","*.json"),("All","*.*")],
            title="Save Design")
        if not path: return
        state={
            "version": __version__,
            "run_id":  self._run_id,
            "draw_R":  self.draw_R,
            "draw_Z":  self.draw_Z,
            "geometry_mode": self._geometry_mode,
            "miller":  {k:v.get() for k,v in self._miller_vars.items()},
            "grid":    {k:v.get() for k,v in self._grid_vars.items()},
            "qform":   self._qform_var.get(),
            "curv":    self._curv_var.get(),
            "prec":    self._prec_var.get(),
            "gen_path":  self._genpath_var.get(),
            "diag_path": self._diagpath_var.get(),
        }
        try:
            Path(path).write_text(json.dumps(state, indent=2))
            self._log(f"[Saved] {path}\n")
        except Exception as ex:
            messagebox.showerror("Save failed", str(ex))

    def _load_design(self):
        _DIR_SAVES.mkdir(parents=True, exist_ok=True)
        path=filedialog.askopenfilename(
            initialdir=str(_DIR_SAVES),
            filetypes=[("BOUT Grid Design","*.json"),("All","*.*")],
            title="Load Design")
        if not path: return
        try:
            state=json.loads(Path(path).read_text())
        except Exception as ex:
            messagebox.showerror("Load failed", str(ex)); return

        self.draw_R=state.get("draw_R",[])
        self.draw_Z=state.get("draw_Z",[])
        self._geometry_mode=state.get("geometry_mode","custom")
        self._run_id=state.get("run_id", self._run_id)
        self._refresh_run_id_display()

        self._updating_miller_vars=True
        try:
            for k,v in state.get("miller",{}).items():
                if k in self._miller_vars: self._miller_vars[k].set(v)
        finally:
            self._updating_miller_vars=False
        for k,v in state.get("grid",{}).items():
            if k in self._grid_vars: self._grid_vars[k].set(v)

        if "qform" in state: self._qform_var.set(state["qform"])
        if "curv"  in state: self._curv_var.set(state["curv"])
        if "prec"  in state: self._prec_var.set(state["prec"])
        if "gen_path"  in state: self._genpath_var.set(state["gen_path"])
        if "diag_path" in state: self._diagpath_var.set(state["diag_path"])

        if len(self.draw_R)>=3:
            if self._geometry_mode=="miller":
                try:
                    R0=float(self._miller_vars["R0"].get())
                    a=float(self._miller_vars["a"].get())
                    k=float(self._miller_vars["kappa"].get())
                    d=float(self._miller_vars["delta"].get())
                    R,Z=preset_d_shape(R0,a,k,d,n=256)
                    self.smooth_R=np.asarray(R,float); self.smooth_Z=np.asarray(Z,float)
                    step=max(1,len(R)//32)
                    self.draw_R=[float(v) for v in R[::step]]
                    self.draw_Z=[float(v) for v in Z[::step]]
                except (TypeError,ValueError):
                    self.smooth_R,self.smooth_Z=smooth_contour(self.draw_R,self.draw_Z)
            else:
                self.smooth_R,self.smooth_Z=smooth_contour(self.draw_R,self.draw_Z)
            self.miller=extract_miller_params(self.smooth_R,self.smooth_Z)
            self.cmap.fit_to_shape(self.smooth_R,self.smooth_Z)
        else:
            self.smooth_R=np.array([]); self.smooth_Z=np.array([])

        self._refresh_geometry_mode_ui()
        self._redraw(); self._update_params_summary()
        self._log(f"[Loaded] {path}\n")

    # ---- Canvas drawing ----

    def _redraw(self):
        c=self.canvas; c.delete("all")
        self._draw_grid(); self._draw_axes(); self._draw_shape(); self._draw_points()

    def _draw_grid(self):
        c=self.canvas; W,H,M=self.cmap.W,self.cmap.H,self.cmap.M
        def nice(v):
            if v<=0: return 1.0
            mag=10**math.floor(math.log10(v))
            for s in [1,2,2.5,5,10]:
                if s*mag>=v: return s*mag
            return 10*mag
        step=nice((self.cmap.R_hi-self.cmap.R_lo)/6)
        r=math.ceil(self.cmap.R_lo/step)*step
        while r<=self.cmap.R_hi:
            x,_=self.cmap.to_canvas(r,0)
            c.create_line(x,M,x,H-M,fill=GRID_COL)
            c.create_text(x,H-M+12,text=f"{r:.1f}",fill=SUBTEXT,font=("Helvetica",7))
            r=round(r+step,10)
        step=nice((self.cmap.Z_hi-self.cmap.Z_lo)/6)
        z=math.ceil(self.cmap.Z_lo/step)*step
        while z<=self.cmap.Z_hi:
            _,y=self.cmap.to_canvas(0,z)
            c.create_line(M,y,W-M,y,fill=GRID_COL)
            c.create_text(M-22,y,text=f"{z:.1f}",fill=SUBTEXT,font=("Helvetica",7))
            z=round(z+step,10)
        c.create_text(W//2,H-6,text="R (m)",fill=SUBTEXT,font=("Helvetica",8))
        c.create_text(10,H//2,text="Z (m)",fill=SUBTEXT,font=("Helvetica",8),angle=90)

    def _draw_axes(self):
        c=self.canvas; _,y0=self.cmap.to_canvas(0,0)
        if self.cmap.M<=y0<=self.cmap.H-self.cmap.M:
            c.create_line(self.cmap.M,y0,self.cmap.W-self.cmap.M,y0,fill=AXIS_COL,dash=(4,4))

    def _draw_shape(self):
        if len(self.smooth_R)<3: return
        c=self.canvas
        outline=WARN if self._convex_warn else SHAPE_COL
        fill="#3d1a1a" if self._convex_warn else SHAPE_FILL
        pts=[]
        for R,Z in zip(self.smooth_R,self.smooth_Z):
            x,y=self.cmap.to_canvas(R,Z); pts+=[x,y]
        pts+=[pts[0],pts[1]]
        c.create_polygon(pts,outline=outline,fill=fill,
                         width=3 if self._convex_warn else 2,smooth=False)
        if self.miller:
            R0=self.miller["R0"]; a=self.miller["a"]
            k=self.miller["kappa"]; d=self.miller["delta"]
            Ra=self.miller.get("R_area",R0); Za=self.miller.get("Z_area",0)
            cx,cy=self.cmap.to_canvas(Ra,Za)
            c.create_line(cx-10,cy,cx+10,cy,fill=ACCENT,width=2)
            c.create_line(cx,cy-10,cx,cy+10,fill=ACCENT,width=2)
            c.create_oval(cx-3,cy-3,cx+3,cy+3,fill=ACCENT,outline="")
            mode_label="CUSTOM  |  equivalent fit" if self._geometry_mode=="custom" else "MILLER"
            c.create_text(self.cmap.W//2,18,
                          text=f"{mode_label}   R₀={R0:.2f}m  a={a:.2f}m  κ={k:.2f}  δ={d:.3f}",
                          fill=ACCENT,font=("Helvetica",9,"bold"))
            if self._convex_warn:
                pct=int(round((1-self._convex_frac)*100))
                c.create_rectangle(0,self.cmap.H-36,self.cmap.W,self.cmap.H-4,
                                   fill="#3d1a1a",outline=WARN)
                c.create_text(self.cmap.W//2,self.cmap.H-20,
                              text=f"⚠ NON-CONVEX ({pct}% concave vertices) - custom grid may be distorted; Smooth recommended",
                              fill=WARN,font=("Helvetica",8,"bold"))

    def _draw_points(self):
        c=self.canvas; r=5
        for i,(R,Z) in enumerate(zip(self.draw_R,self.draw_Z)):
            x,y=self.cmap.to_canvas(R,Z)
            c.create_oval(x-r,y-r,x+r,y+r,fill=POINT_COL,outline=DARK_BG,width=1)
            c.create_text(x+8,y-8,text=str(i+1),fill=SUBTEXT,font=("Helvetica",7))

    # ---- Params summary ----

    def _update_params_summary(self):
        mv,gv=self._miller_vars,self._grid_vars
        try:
            mode="Custom" if self._geometry_mode=="custom" else "Miller"
            txt=(f"{mode} | R0={mv['R0'].get()}m  a={mv['a'].get()}m  "
                 f"κ={mv['kappa'].get()}  δ={mv['delta'].get()}  "
                 f"nx={gv['nx'].get()}  ny={gv['ny'].get()}")
            self._params_lbl.configure(text=txt)
        except: pass

    # ---- Run pipeline ------

    def _snapshot_miller_generation_state(self):
        """
        Return the exact Miller parameters shown in the UI and rebuild the
        analytic contour from those values immediately before generation.

        This makes the generation contract explicit:
          displayed Miller fields == generator arguments == visible contour.
        """
        mv=self._miller_vars
        try:
            params={
                "R0":float(mv["R0"].get()),
                "a":float(mv["a"].get()),
                "kappa":float(mv["kappa"].get()),
                "delta":float(mv["delta"].get()),
            }
        except (TypeError,ValueError) as ex:
            raise ValueError("Miller parameters must be numeric.") from ex

        if params["R0"]<=0:
            raise ValueError("R0 must be greater than zero.")
        if params["a"]<=0:
            raise ValueError("Minor radius a must be greater than zero.")
        if params["kappa"]<=0:
            raise ValueError("Elongation kappa must be greater than zero.")
        if abs(params["delta"])>=0.999:
            raise ValueError("Triangularity delta must satisfy |delta| < 0.999.")

        R,Z=preset_d_shape(
            params["R0"],params["a"],params["kappa"],params["delta"],n=256
        )
        self.smooth_R=np.asarray(R,float)
        self.smooth_Z=np.asarray(Z,float)
        step=max(1,len(R)//32)
        self.draw_R=[float(v) for v in R[::step]]
        self.draw_Z=[float(v) for v in Z[::step]]
        self.miller=extract_miller_params(R,Z)
        self._convex_warn=False
        self._convex_frac=1.0

        if hasattr(self,"canvas"):
            self.cmap.fit_to_shape(self.smooth_R,self.smooth_Z)
            self._redraw()
            self._update_params_summary()

        return params

    def _build_gen_args(self, outfile, boundary_file=None, miller_snapshot=None):
        mv,gv=self._miller_vars,self._grid_vars
        def g(d,k):
            try: return d[k].get()
            except: return ""
        mp=miller_snapshot or {
            "R0":g(mv,"R0"),
            "a":g(mv,"a"),
            "kappa":g(mv,"kappa"),
            "delta":g(mv,"delta"),
        }
        args=[str(x) for x in [
            sys.executable, self._genpath_var.get(),
            "--R0",mp["R0"],"--a",mp["a"],
            "--kappa",mp["kappa"],"--delta",mp["delta"],
            "--B0",g(gv,"B0"),"--nx",g(gv,"nx"),"--ny",g(gv,"ny"),
            "--nz",g(gv,"nz"),"--q0",g(gv,"q0"),"--qa",g(gv,"qa"),
            "--qform",self._qform_var.get(),"--curvature",self._curv_var.get(),
            "--precision",self._prec_var.get(),"--xmin_frac",g(gv,"xmin_frac"),
            "--outfile",str(outfile),
        ]]
        if boundary_file is not None:
            args.extend(["--boundary-file",str(boundary_file)])
        return args

    def _run_generator(self):
        if len(self.smooth_R)<3 and len(self.draw_R)<3:
            messagebox.showwarning("No shape","Draw or load a cross-section first.")
            return
        custom_validation=None
        miller_snapshot=None
        if self._geometry_mode=="miller":
            try:
                miller_snapshot=self._snapshot_miller_generation_state()
            except ValueError as ex:
                messagebox.showerror("Invalid Miller Parameters",str(ex))
                return

        if self._geometry_mode=="custom":
            custom_validation=validate_custom_boundary(self.smooth_R,self.smooth_Z)
            if not custom_validation.get("valid",False):
                messagebox.showerror(
                    "Cannot Generate Custom Boundary",
                    custom_validation.get("reason","This boundary cannot form a valid nested grid.") +
                    "\n\nUse Smooth or redraw the contour, then try again."
                )
                return

            if self._convex_warn:
                pct=int(round((1-self._convex_frac)*100))
                if not messagebox.askyesno(
                    "Non-Convex Custom Boundary",
                    f"Shape has ~{pct}% concave vertices.\n\n"
                    "It is still single-valued about the magnetic axis, so Grid Suite can generate it, "
                    "but non-convex shaping may produce strongly distorted cells.\n\n"
                    "Smooth is recommended. Generate this custom boundary anyway?"
                ):
                    return
        gen=self._genpath_var.get()
        if not os.path.isfile(gen):
            messagebox.showerror("Generator not found",f"Cannot find:\n{gen}")
            return

        # create output dirs
        out_dir    = _DIR_OUTPUT  / self._run_id
        report_dir = _DIR_REPORTS / self._run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        grid_path = out_dir / "grid.nc"
        boundary_path=None
        if self._geometry_mode=="custom":
            boundary_path=out_dir / "requested_boundary.json"
            boundary_payload={
                "format":"chatwood-grid-boundary-v1",
                "mode":"custom",
                "R":[float(v) for v in self.smooth_R],
                "Z":[float(v) for v in self.smooth_Z],
                "axis_R":float(custom_validation["axis_R"]),
                "axis_Z":float(custom_validation["axis_Z"]),
                "equivalent_miller":{k:float(self.miller[k]) for k in ("R0","a","kappa","delta") if k in self.miller},
            }
            boundary_path.write_text(json.dumps(boundary_payload,indent=2),encoding="utf-8")

        args = self._build_gen_args(
            grid_path,
            boundary_file=boundary_path,
            miller_snapshot=miller_snapshot,
        )

        self._log_clear()
        self._log(f"Run ID: {self._run_id}\n")
        self._log(f"Grid  → {grid_path}\n")
        self._log(f"Report→ {report_dir}\n")
        self._log(f"Geometry mode: {'CUSTOM BOUNDARY' if self._geometry_mode=='custom' else 'MILLER ANALYTIC'}\n\n")
        self._log(f"$ {' '.join(args)}\n\n")
        self._run_btn.configure(state=tk.DISABLED, text="⏳ Running…")

        # update path labels
        self._path_grid_lbl.configure(text=str(grid_path))
        self._path_report_lbl.configure(text=str(report_dir))

        def worker():
            # Force UTF-8 for child Python processes so scientific symbols in
            # diagnostics output (e.g. kappa, delta, approx, check marks) are
            # decoded consistently on Windows as well as POSIX systems.
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env.setdefault("PYTHONUTF8", "1")

            # Step 1: generate grid
            try:
                proc=subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=child_env,
                )
                for line in proc.stdout: self._log(line)
                proc.wait()
                if proc.returncode!=0:
                    self._log(f"\n✖ Grid generation failed (code {proc.returncode})\n")
                    self.after(0, lambda code=proc.returncode: messagebox.showerror(
                        "Grid Cannot Be Generated",
                        f"The requested geometry could not produce a valid grid (generator exit code {code}).\n\n"
                        "Review the Run Log, then Smooth or redraw the contour and try again."
                    ))
                    return
                self._log(f"\n✔ Grid written: {grid_path}\n\n")
            except Exception as ex:
                self._log(f"\n✖ {ex}\n")
                self.after(0, lambda msg=str(ex): messagebox.showerror(
                    "Grid Cannot Be Generated",
                    f"Grid generation failed:\n\n{msg}"
                ))
                return

            # Step 2: run diagnostics
            diag=self._diagpath_var.get()
            if not os.path.isfile(diag):
                self._log(f"[WARN] Diagnostics not found: {diag}\n")
                return

            diag_args=[sys.executable, diag, str(grid_path),
                       "--outdir", str(report_dir)]
            self._log(f"$ {' '.join(diag_args)}\n\n")
            try:
                proc2=subprocess.Popen(
                    diag_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=child_env,
                )
                for line in proc2.stdout: self._log(line)
                proc2.wait()
                if proc2.returncode==0:
                    self._log(f"\n✔ Report written: {report_dir}\n")
                else:
                    self._log(f"\n⚠ Diagnostics exited with code {proc2.returncode}\n")
                # load and display report
                json_path = report_dir / "grid_report.json"
                if json_path.exists():
                    self.after(0, lambda: self._load_report(json_path))
            except Exception as ex:
                self._log(f"\n✖ Diagnostics error: {ex}\n")

        def finally_fn():
            self._run_btn.configure(state=tk.NORMAL,text="▶  Generate Grid + Report")

        def thread_fn():
            try: worker()
            finally: self.after(0, finally_fn)

        threading.Thread(target=thread_fn, daemon=True).start()

    # ---- Report display ----

    def _load_report(self, json_path: Path):
        """Parse diagnostics JSON and populate the Report tab."""
        try:
            data = json.loads(json_path.read_text())
        except Exception as ex:
            self._log(f"[WARN] Could not parse report JSON: {ex}\n")
            return

        # Switch to report tab, load images
        self._nb.select(1)
        self.after(100, lambda: self._load_report_images(_DIR_REPORTS / self._run_id))

        # Verdict
        crits = data.get("critical_failures", [])
        warns = data.get("warnings", [])
        if not crits:
            verdict = "✔  PASS - No critical failures"
            vcol    = ACCENT2
        else:
            verdict = f"✖  FAIL - {len(crits)} critical failure(s)"
            vcol    = WARN
        if warns and not crits:
            verdict += f"  ({len(warns)} warning(s))"
            vcol = WARN2
        self._verdict_lbl.configure(text=verdict, fg=vcol)

        # Stats
        nx=data.get("nx","?"); ny=data.get("ny","?"); nz=data.get("nz","?")
        self._stat_labels["dims"].configure(text=f"{nx} × {ny} × {nz}")
        ar=data.get("aspect_ratio")
        self._stat_labels["aspect"].configure(
            text=f"{ar:.3f}" if ar else "-")
        Jmn=data.get("J_min"); Jmx=data.get("J_max")
        self._stat_labels["j_range"].configure(
            text=f"{Jmn:.3f} → {Jmx:.3f}" if Jmn else "-")
        Bmn=data.get("B_min"); Bmx=data.get("B_max")
        self._stat_labels["b_range"].configure(
            text=f"{Bmn:.3f} → {Bmx:.3f} T" if Bmn else "-")
        self._stat_labels["b_min"].configure(
            text=f"{Bmn:.4f} T" if Bmn else "-")
        # conditioning from checks
        cond_val="-"
        for chk in data.get("checks",[]):
            if "conditioning" in chk.get("name",""):
                rv=chk.get("details",{}).get("ratio")
                if rv: cond_val=f"{rv:.1f}"
        self._stat_labels["cond"].configure(text=cond_val)

        # Checks list
        t=self._checks_text
        t.configure(state=tk.NORMAL)
        t.delete("1.0",tk.END)

        # Group by severity
        groups={"CRITICAL":[],"WARN":[],"INFO":[]}
        for chk in data.get("checks",[]):
            sev=chk.get("severity","INFO")
            groups.setdefault(sev,[]).append(chk)

        for sev,label in [("CRITICAL","CRITICAL"),("WARN","WARN"),("INFO","INFO")]:
            checks=groups.get(sev,[])
            if not checks: continue
            t.insert(tk.END,f"-- {label} --\n","heading")
            for chk in checks:
                status=chk.get("status","?")
                name=chk.get("name","")
                icon={"PASS":"✔","FAIL":"✖","SKIP":"–"}.get(status,"?")
                tag={"PASS":"pass","FAIL":"fail","SKIP":"skip"}.get(status,"skip")
                if status=="WARN": tag="warn"
                t.insert(tk.END,f"  {icon} {name}\n",tag)
        t.configure(state=tk.DISABLED)

    def _load_report_images(self, report_dir: Path):
        """Load PNG images from report dir into the zoomable ImageViewer tabs."""
        if not _PIL_OK:
            for v in (self._geom_viewer, self._bfield_viewer):
                v.set_message("Install Pillow to view images:\npip install Pillow")
            return

        render_path = report_dir / "grid_render.png"
        bfield_path = report_dir / "grid_bfield.png"

        if render_path.exists():
            self._pil_geom = _PILImage.open(render_path).copy()
            self._geom_viewer.set_image(self._pil_geom)
            self._report_nb.select(1)   # switch to Geometry tab
        else:
            self._pil_geom = None
            self._geom_viewer.set_message("grid_render.png not found")

        if bfield_path.exists():
            self._pil_bfield = _PILImage.open(bfield_path).copy()
            self._bfield_viewer.set_image(self._pil_bfield)
        else:
            self._pil_bfield = None
            self._bfield_viewer.set_message("grid_bfield.png not found")

    def _rescale_report_image(self, which: str):
        """Delegate to the ImageViewer widgets (kept for compatibility)."""
        if which == "geom" and self._pil_geom is not None:
            self._geom_viewer.set_image(self._pil_geom)
        elif which == "bfield" and self._pil_bfield is not None:
            self._bfield_viewer.set_image(self._pil_bfield)

    def _open_html_report(self):
        report_dir = _DIR_REPORTS / self._run_id
        html_files = list(report_dir.glob("*.html"))
        if not html_files:
            messagebox.showinfo("No report",
                f"No HTML report found in:\n{report_dir}\n\nGenerate a grid first.")
            return
        import webbrowser
        webbrowser.open(html_files[0].as_uri())

    # ---- Log ----

    def _log(self, text):
        def _do():
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, text)
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)
        self.after(0, _do)

    def _open_3d_view(self):
        """Open the 3D torus preview window."""
        if len(self.smooth_R) < 3:
            messagebox.showinfo("No shape",
                "Draw or load a poloidal cross-section first.")
            return
        TorusPreview(self, self.smooth_R, self.smooth_Z)

    def _toggle_log(self):
        """Show or hide the log body."""
        if self._log_visible:
            self._log_body.pack_forget()
            self._log_hide_btn.configure(text="▲ Show")
            self._log_visible = False
        else:
            self._log_body.pack(fill=tk.X)
            self._log_hide_btn.configure(text="▼ Hide")
            self._log_visible = True

    def _log_clear(self):
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0",tk.END)
        self.log.configure(state=tk.DISABLED)


# ---- Zoomable / Pannable Image Viewer --------

class ImageViewer(tk.Frame):
    """
    Embeddable widget that displays a PIL image with:
      - Fit-to-window on load / resize
      - Mouse-drag to pan
      - Scroll wheel to zoom (centred on cursor)
      - Double-click to open a fullscreen popup of the same image
      - "⤢ Pop out" button in the top-right corner
    """

    BG = "#0a0a14"

    def __init__(self, master, label="Image", **kw):
        super().__init__(master, bg=self.BG, **kw)
        self._label    = label
        self._pil_img  = None      # original PIL image (never modified)
        self._photo    = None      # current tkinter PhotoImage
        self._zoom     = 1.0       # current zoom factor
        self._offset   = [0, 0]    # pan offset in pixels
        self._drag_start = None
        self._fit_done = False

        # header bar
        hdr = tk.Frame(self, bg="#13131f")
        hdr.pack(fill=tk.X)
        self._title_lbl = tk.Label(hdr, text=label, bg="#13131f", fg=SUBTEXT,
                                    font=("Helvetica",8))
        self._title_lbl.pack(side=tk.LEFT, padx=6, pady=2)
        tk.Button(hdr, text="⤢ Pop out", command=self._popup,
                  bg=BTN_BG, fg=SUBTEXT, activebackground=ACCENT,
                  activeforeground=DARK_BG, relief=tk.FLAT,
                  font=("Helvetica",8), padx=6, pady=1,
                  cursor="hand2").pack(side=tk.RIGHT, padx=4, pady=2)
        tk.Button(hdr, text="⟳ Reset", command=self._reset_view,
                  bg=BTN_BG, fg=SUBTEXT, activebackground=ACCENT,
                  activeforeground=DARK_BG, relief=tk.FLAT,
                  font=("Helvetica",8), padx=6, pady=1,
                  cursor="hand2").pack(side=tk.RIGHT, padx=2, pady=2)

        # canvas
        self.canvas = tk.Canvas(self, bg=self.BG, highlightthickness=0,
                                cursor="fleur")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>",      self._on_configure)
        self.canvas.bind("<ButtonPress-1>",  self._on_press)
        self.canvas.bind("<B1-Motion>",      self._on_drag)
        self.canvas.bind("<ButtonRelease-1>",self._on_release)
        self.canvas.bind("<Double-Button-1>",lambda e: self._popup())
        self.canvas.bind("<MouseWheel>",     self._on_scroll)
        self.canvas.bind("<Button-4>",       self._on_scroll)
        self.canvas.bind("<Button-5>",       self._on_scroll)

    # ---- public API ----

    def set_image(self, pil_img):
        """Load a new PIL image, fit it to the current canvas."""
        self._pil_img  = pil_img
        self._fit_done = False
        self._fit()

    def set_message(self, text):
        """Show a plain text message (no image)."""
        self._pil_img = None
        self._photo   = None
        c = self.canvas
        c.delete("all")
        c.create_text(10, 10, text=text, fill=SUBTEXT,
                      font=("Helvetica",9), anchor="nw")

    # ---- internal ------

    def _fit(self):
        """Scale image to fit canvas, centred."""
        if self._pil_img is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 4 or ch < 4:
            return
        iw, ih = self._pil_img.size
        self._zoom = min(cw / iw, ch / ih) * 0.97
        self._offset = [(cw - iw*self._zoom)/2,
                        (ch - ih*self._zoom)/2]
        self._fit_done = True
        self._render()

    def _render(self):
        """Render the image at current zoom/offset."""
        if self._pil_img is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 4 or ch < 4:
            return

        iw, ih = self._pil_img.size
        nw = max(1, int(iw * self._zoom))
        nh = max(1, int(ih * self._zoom))

        if _PIL_OK:
            resized     = self._pil_img.resize((nw, nh), _PILImage.LANCZOS)
            self._photo = _PILImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        if self._photo:
            self.canvas.create_image(
                int(self._offset[0]), int(self._offset[1]),
                image=self._photo, anchor="nw")
        # zoom hint
        self.canvas.create_text(
            cw-4, ch-4,
            text=f"{self._zoom*100:.0f}%",
            fill=SUBTEXT, font=("Helvetica",7), anchor="se")

    def _reset_view(self):
        self._fit_done = False
        self._fit()

    def _on_configure(self, event):
        if not self._fit_done:
            self._fit()
        else:
            self._render()

    def _on_press(self, e):
        self._drag_start = (e.x, e.y, self._offset[0], self._offset[1])

    def _on_drag(self, e):
        if not self._drag_start:
            return
        x0, y0, ox, oy = self._drag_start
        self._offset[0] = ox + (e.x - x0)
        self._offset[1] = oy + (e.y - y0)
        self._render()

    def _on_release(self, e):
        self._drag_start = None

    def _on_scroll(self, e):
        if self._pil_img is None:
            return
        # zoom centred on cursor
        if hasattr(e,"delta") and e.delta != 0:
            factor = 1.12 if e.delta > 0 else 0.89
        elif e.num == 4:
            factor = 1.12
        elif e.num == 5:
            factor = 0.89
        else:
            return

        old_zoom = self._zoom
        self._zoom = max(0.05, min(20.0, self._zoom * factor))
        ratio = self._zoom / old_zoom
        # keep the pixel under the cursor stationary
        self._offset[0] = e.x - ratio*(e.x - self._offset[0])
        self._offset[1] = e.y - ratio*(e.y - self._offset[1])
        self._render()

    def _popup(self):
        """Open a full-size popup window with its own ImageViewer."""
        if self._pil_img is None:
            return
        win = tk.Toplevel(self)
        win.title(self._label)
        win.geometry("1100x700")
        win.configure(bg=self.BG)
        iv = ImageViewer(win, label=self._label)
        iv.pack(fill=tk.BOTH, expand=True)
        # slight delay so the window is drawn before we try to fit
        win.after(80, lambda: iv.set_image(self._pil_img))


# ---- 3D Torus Preview ----

class TorusPreview(tk.Toplevel):
    """
    Standalone 3D torus preview window.
    Revolves the 2D (R, Z) poloidal contour around the Z-axis.

    Projection model: orthographic with a zoom scale in px/m, plus a mild
    perspective depth-cue. This correctly handles physical metre-scale geometry
    (R0 ~ 5-10m) without the fov/distance confusion of a pure perspective model.

    Controls: left-drag rotates, scroll wheel zooms.
    """

    N_PHI = 64
    BG    = "#0a0a14"

    def __init__(self, master, R_contour, Z_contour):
        super().__init__(master)
        self.title("3D Torus Preview - drag to rotate")
        self.configure(bg=self.BG)
        self.geometry("760x620")
        self.minsize(400, 340)

        # ---- build 3D mesh (N_theta × N_phi × 3) -----─
        R   = np.asarray(R_contour, float)
        Z   = np.asarray(Z_contour, float)
        nt  = len(R)
        phi = np.linspace(0, 2*np.pi, self.N_PHI, endpoint=False)
        self._pts = np.zeros((nt, self.N_PHI, 3))
        for i in range(nt):
            self._pts[i, :, 0] = R[i] * np.cos(phi)
            self._pts[i, :, 1] = R[i] * np.sin(phi)
            self._pts[i, :, 2] = Z[i]

        # ---- auto-fit zoom: compute bounding radius of mesh ----
        # After rotation the widest the object can be is its 3D bounding radius
        bnd = max(np.abs(self._pts[:,:,0]).max(),
                  np.abs(self._pts[:,:,1]).max(),
                  np.abs(self._pts[:,:,2]).max())
        # Will be replaced on first draw once canvas size is known
        self._bnd        = bnd
        self._zoom       = 1.0      # px/m - set in _auto_zoom()
        self._zoom_inited = False

        # ---- camera state ----
        self._az  =  30.0
        self._el  =  25.0
        self._drag_start = None

        # ---- UI --------
        self._canvas = tk.Canvas(self, bg=self.BG, highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        tk.Label(self, text="Left-drag: rotate     Scroll: zoom",
                 bg=self.BG, fg="#585b70", font=("Helvetica",8)
                 ).pack(side=tk.BOTTOM, pady=3)

        self._canvas.bind("<ButtonPress-1>",  self._on_press)
        self._canvas.bind("<B1-Motion>",      self._on_motion)
        self._canvas.bind("<Configure>",      self._on_configure)
        # Windows uses MouseWheel with delta; Linux uses Button-4/5
        self._canvas.bind("<MouseWheel>",     self._on_zoom)
        self._canvas.bind("<Button-4>",       self._on_zoom)
        self._canvas.bind("<Button-5>",       self._on_zoom)
        # Also bind to the Toplevel so scroll works even when mouse is on border
        self.bind("<MouseWheel>", self._on_zoom)

    # ---- helpers --------

    def _auto_zoom(self, W, H):
        """Set zoom so the torus fills ~70% of the smaller canvas dimension."""
        target_px = 0.70 * min(W, H)
        self._zoom = target_px / (2.0 * self._bnd)

    def _rot_matrix(self):
        az  = math.radians(self._az)
        el  = math.radians(self._el)
        caz, saz = math.cos(az), math.sin(az)
        cel, sel = math.cos(el), math.sin(el)
        # Ry(az) rotates around vertical, Rx(el) tilts toward viewer
        Ry = np.array([[caz, 0, saz], [0, 1, 0], [-saz, 0, caz]])
        Rx = np.array([[1, 0, 0], [0, cel, sel], [0, -sel, cel]])
        return Rx @ Ry

    def _project(self, pts3d, cx, cy):
        """
        Orthographic projection with mild perspective depth-cue.
        p_screen = zoom * p_rotated_xy,  depth = p_rotated_z
        The depth-cue scales each point by (1 + depth_norm * 0.15) so
        near points appear very slightly larger - just enough to read depth.
        """
        rot   = self._rot_matrix()
        p     = pts3d @ rot.T              # (..., 3)
        px, py, pz = p[...,0], p[...,1], p[...,2]

        # Mild perspective: scale by 1 + k*pz/bnd  (k=0.2)
        depth_norm = pz / max(self._bnd, 1e-6)
        persp = 1.0 + 0.2 * depth_norm

        x2d = cx + px * self._zoom * persp
        y2d = cy - py * self._zoom * persp   # y flipped (screen y down)
        return x2d, y2d, pz

    # ---- drawing --------

    def _on_configure(self, event):
        W, H = event.width, event.height
        if not self._zoom_inited and W > 10 and H > 10:
            self._auto_zoom(W, H)
            self._zoom_inited = True
        self._draw()

    def _draw(self):
        c  = self._canvas
        c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10 or H < 10:
            return
        cx, cy = W // 2, H // 2

        nt, np_ = self._pts.shape[:2]
        x2d, y2d, depth = self._project(self._pts, cx, cy)

        d_min = float(depth.min())
        d_max = float(depth.max())
        d_rng = max(d_max - d_min, 1e-6)

        # Build quads with mean depth for painter's algorithm
        quads = []
        for i in range(nt):
            ni = (i + 1) % nt
            for j in range(np_):
                nj = (j + 1) % np_
                xs = (x2d[i,j], x2d[ni,j], x2d[ni,nj], x2d[i,nj])
                ys = (y2d[i,j], y2d[ni,j], y2d[ni,nj], y2d[i,nj])
                ds = (depth[i,j]+depth[ni,j]+depth[ni,nj]+depth[i,nj])*0.25
                quads.append((ds, xs, ys))

        quads.sort(key=lambda q: q[0])   # back-to-front

        for ds, xs, ys in quads:
            t = (ds - d_min) / d_rng      # 0=far  1=near
            # Colour: far = dark navy, near = bright cornflower blue
            r = int(0x1e + (0x89 - 0x1e)*t)
            g = int(0x3a + (0xb4 - 0x3a)*t)
            b = int(0x5f + (0xfa - 0x5f)*t)
            col = f"#{r:02x}{g:02x}{b:02x}"
            pts_flat = [v for pair in zip(xs, ys) for v in pair]
            c.create_polygon(pts_flat, fill=col, outline="", smooth=False)

        # Toroidal guide rings at several poloidal locations.
        for i_frac in [0, 0.25, 0.5, 0.75]:
            i = int(i_frac * nt) % nt
            xl = list(x2d[i, :]) + [x2d[i, 0]]
            yl = list(y2d[i, :]) + [y2d[i, 0]]
            c.create_line([v for pair in zip(xl,yl) for v in pair],
                          fill="#cdd6f4", width=1, smooth=True)

        # Meridional traces make the *actual current designer boundary* visible
        # in 3D, which is particularly useful for irregular custom contours.
        for j_frac in [0.0, 0.5]:
            j = int(j_frac * np_) % np_
            xl = list(x2d[:, j]) + [x2d[0, j]]
            yl = list(y2d[:, j]) + [y2d[0, j]]
            c.create_line([v for pair in zip(xl,yl) for v in pair],
                          fill="#f5c2e7", width=2, smooth=True)

        # HUD
        c.create_text(10, 10,
                      text=f"az={self._az:.0f}°  el={self._el:.0f}°  "
                           f"zoom={self._zoom:.1f}px/m",
                      fill="#6c7086", font=("Helvetica",8), anchor="nw")

    # ---- interaction ----

    def _on_press(self, e):
        self._drag_start = (e.x, e.y, self._az, self._el)

    def _on_motion(self, e):
        if not self._drag_start:
            return
        x0, y0, az0, el0 = self._drag_start
        self._az = (az0 + (e.x - x0) * 0.4) % 360
        self._el = max(-89, min(89, el0 - (e.y - y0) * 0.4))
        self._draw()

    def _on_zoom(self, e):
        # Windows: e.delta = ±120 per notch
        # Linux:   e.num   = 4 (up) or 5 (down)
        if hasattr(e, "delta") and e.delta != 0:
            factor = 1.12 if e.delta > 0 else 0.89
        elif e.num == 4:
            factor = 1.12
        elif e.num == 5:
            factor = 0.89
        else:
            factor = 1.0
        self._zoom = max(1.0, min(500.0, self._zoom * factor))
        self._draw()


# ---- Entry point ----

def main():
    p=argparse.ArgumentParser(description="BOUT++ Grid Designer GUI")
    p.add_argument("--generator",   default=None,
                   help=f"Path to generator script (default: {_DEFAULT_GENERATOR})")
    p.add_argument("--diagnostics", default=None,
                   help=f"Path to diagnostics script (default: {_DEFAULT_DIAGNOSTICS})")
    args=p.parse_args()
    BoutGridDesigner(generator=args.generator, diagnostics=args.diagnostics).mainloop()

if __name__=="__main__":
    main()
