# bowl_from_soundboard.py
# Build lute bowls from the soundboard geometry using a side-profile-driven
# top curve with per-control shaping (single strategy, no control-depth constraints).
#
# Usage:
#   lute = ManolLavta(); lute.draw()
#   class MyCurve(SideProfilePerControlTopCurve):
#       SHAPE_GAMMAS = {"neck_joint":1.10,"soundhole_center":1.25,"form_center":0.85,"bridge":1.05}
#       SHAPE_WIDTHS = {
#           "form_center": ("span_frac", 0.45),
#           "soundhole_center": ("span_frac", 0.35),
#           "bridge": ("span_frac", 0.30),
#           "neck_joint": ("span_frac", 0.25),
#       }
#       # Optional: set absolute amplitude instead of default
#       # AMPLITUDE_MODE = "units"; AMPLITUDE_UNITS = 1.70
#
#   sections, ribs = build_bowl_for_lute(lute, n_ribs=13, n_sections=100, top_curve=MyCurve)
#   plot_bowl(lute, sections, ribs)

from typing import List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def set_axes_equal_3d(ax, xs=None, ys=None, zs=None, use_ortho=True):
    """Force equal data scale on a 3D axes so circles look circular."""
    if xs is None or ys is None or zs is None:
        xmin, xmax = ax.get_xlim3d()
        ymin, ymax = ax.get_ylim3d()
        zmin, zmax = ax.get_zlim3d()
    else:
        xs = np.asarray(xs); ys = np.asarray(ys); zs = np.asarray(zs)
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        zmin, zmax = float(zs.min()), float(zs.max())
    xmid, ymid, zmid = (xmin+xmax)/2.0, (ymin+ymax)/2.0, (zmin+zmax)/2.0
    r = max(xmax-xmin, ymax-ymin, zmax-zmin, 1e-12)/2.0
    ax.set_xlim3d([xmid-r, xmid+r])
    ax.set_ylim3d([ymid-r, ymid+r])
    ax.set_zlim3d([zmid-r, zmid+r])
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass
    if use_ortho:
        try: ax.set_proj_type('ortho')
        except Exception: pass

# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def circle_through_three_points_2d(P1, P2, P3):
    """Circle through 3 points in the YZ-plane (inputs are 2D [Y,Z] coords)."""
    P1 = np.asarray(P1, float); P2 = np.asarray(P2, float); P3 = np.asarray(P3, float)
    mid12 = 0.5*(P1+P2); mid13 = 0.5*(P1+P3)
    d12 = P2-P1; d13 = P3-P1
    area2 = d12[0]*d13[1] - d12[1]*d13[0]
    if abs(area2) < 1e-12:
        raise ValueError("Collinear points, no unique circle.")
    n12 = np.array([-d12[1], d12[0]]); n13 = np.array([-d13[1], d13[0]])
    A = np.column_stack([n12, -n13]); b = mid13-mid12
    t, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    C = mid12 + t*n12; r = float(np.linalg.norm(C-P1))
    return C, r

def _sample_soundboard_outline(lute, samples_per_arc):
    pts = []
    for arc in getattr(lute, 'final_arcs', []):
        pts.append(arc.sample_points(samples_per_arc))
    for arc in getattr(lute, 'final_reflected_arcs', []):
        pts.append(arc.sample_points(samples_per_arc))
    if not pts: return np.empty((0,2))
    return np.vstack(pts)

def _intersections_with_vertical(outline_xy, x_const, tol=1e-9):
    ys, P = [], outline_xy
    for i in range(len(P)-1):
        x0, y0 = P[i]; x1, y1 = P[i+1]
        dx0, dx1 = x0-x_const, x1-x_const
        if dx0 == 0 and dx1 == 0: ys.extend([y0,y1]); continue
        if (dx0 == 0) ^ (dx1 == 0): ys.append(y0 if dx0==0 else y1); continue
        if (dx0<0 and dx1>0) or (dx0>0 and dx1<0):
            t = (x_const-x0)/(x1-x0); ys.append(y0 + t*(y1-y0))
    ys = sorted(ys)
    dedup = []
    for y in ys:
        if not dedup or abs(y-dedup[-1]) > tol:
            dedup.append(y)
    return dedup

def extract_side_points_at_X(lute, X, debug=False, min_width=1e-3, samples_per_arc=500):
    """Slice the soundboard at numeric X and return (P_left_xy, P_right_xy, X)."""
    if abs(float(X)-float(lute.form_top.x)) < 1e-12 or \
       abs(float(X)-float(lute.form_bottom.x)) < 1e-12:
        return None
    outline = _sample_soundboard_outline(lute, samples_per_arc=samples_per_arc)
    ys = _intersections_with_vertical(outline, float(X))
    if len(ys) >= 2:
        yL, yR = ys[0], ys[-1]
        if abs(yR-yL) >= min_width:
            return (np.array([float(X), yL]),
                    np.array([float(X), yR]),
                    float(X))
    if debug:
        fig, ax = plt.subplots(figsize=(8,6))
        if outline.size: ax.plot(outline[:,0], outline[:,1], color='0.4', lw=1.0)
        ax.axvline(float(X), color='r', ls='--')
        for y in ys: ax.plot(float(X), y, 'ro')
        ax.set_aspect('equal', adjustable='box'); plt.show()
    raise RuntimeError(f"Could not find two side intersections at X={float(X):.4f}")

def spine_point_at_X(lute, X: float):
    x0, y0 = float(lute.form_top.x), float(lute.form_top.y)
    x1, y1 = float(lute.form_bottom.x), float(lute.form_bottom.y)
    if abs(x1-x0) < 1e-12: return y0
    t = (float(X)-x0)/(x1-x0)
    return y0 + t*(y1-y0)

# ---------------------------------------------------------------------
# Top curve construction
# ---------------------------------------------------------------------

class TopCurve:
    name = "base"
    @classmethod
    def build(cls, lute):
        raise NotImplementedError

class SideProfilePerControlTopCurve(TopCurve):
    """
    Side-profile-driven top curve with per-control shaping.
    DEPTHS are only used to choose a default global amplitude; they do NOT constrain the curve.
    """
    name = "side_per_control"

    # Depth presets (× unit) only for default amplitude selection
    DEPTHS = {
        "neck_joint":       0.35,
        "soundhole_center": 1.50,
        "form_center":      1.70,
        "bridge":           1.65,
    }

    # Per-control shaping: gamma>1 flatter, <1 fuller, 1 unchanged
    SHAPE_GAMMAS = {
        # e.g. "form_center": 0.85, ...
    }

    # Optional widths per control (X units) or ("span_frac", f)
    SHAPE_WIDTHS = {
        # e.g. "form_center": ("span_frac", 0.45)
    }

    # Global amplitude policy
    AMPLITUDE_MODE  = "max_depth"  # "max_depth" | "units"
    AMPLITUDE_UNITS = None         # used when AMPLITUDE_MODE == "units"

    # Other shaping defaults
    WIDTH_FACTOR = 0.9
    SAMPLES      = 400
    GATE_N_START = 0.0
    GATE_N_FULL  = 0.0
    MAX_EXPONENT_DELTA = 0.8
    KERNEL = "cauchy"              # "cauchy" or "gauss"

    @classmethod
    def build(cls, lute):
        if not cls.SHAPE_GAMMAS:
            raise RuntimeError("SHAPE_GAMMAS is empty for this top-curve class.")

        # --- Resolve widths ---
        widths = None
        if cls.SHAPE_WIDTHS:
            span = abs(float(lute.form_bottom.x) - float(lute.form_top.x))
            widths = {}
            for key, val in cls.SHAPE_WIDTHS.items():
                if isinstance(val,(int,float)):
                    widths[key] = float(val)
                elif isinstance(val,(tuple,list)) and len(val)==2 and val[0]=="span_frac":
                    widths[key] = float(val[1]) * span

        # --- Resolve amplitude ---
        u = float(getattr(lute, "unit", 1.0))
        if cls.AMPLITUDE_MODE == "units" and cls.AMPLITUDE_UNITS is not None:
            amplitude = float(cls.AMPLITUDE_UNITS) * u
        else:
            mx = max(cls.DEPTHS.values()) if cls.DEPTHS else 1.0
            amplitude = float(mx) * u

        # === Begin inlined _make_top_curve_from_side_percontrol ===
        gammas      = cls.SHAPE_GAMMAS
        width_factor= cls.WIDTH_FACTOR
        n_samples   = cls.SAMPLES
        margin      = 1e-3
        gate_N_start= cls.GATE_N_START
        gate_N_full = cls.GATE_N_FULL
        max_exponent_delta = cls.MAX_EXPONENT_DELTA
        kernel      = cls.KERNEL

        # 1) Sample normalized half-width N(x) from side
        xL = float(lute.form_top.x); xR = float(lute.form_bottom.x)
        span = xR - xL; eps = abs(span)*float(margin)
        xs = np.linspace(xL + eps, xR - eps, int(n_samples))

        W = []
        for X in xs:
            hit = extract_side_points_at_X(lute, X)
            if hit is None:
                W.append(0.0); continue
            L, R, _ = hit
            y_spine = spine_point_at_X(lute, X)
            W.append(max(abs(float(L[1])-y_spine), abs(float(R[1])-y_spine)))
        W = np.asarray(W, float)
        if W.size == 0 or float(W.max()) < 1e-12:
            return lambda _x: 0.0
        W[0] = 0.0; W[-1] = 0.0
        N = W / float(W.max())

        # 2) Controls
        ctrl_x_all = {
            "neck_joint":       float(lute.point_neck_joint.x),
            "soundhole_center": float(lute._get_soundhole_center().x),
            "form_center":      float(lute.form_center.x),
            "bridge":           float(lute.bridge.x),
        }
        unknown = [k for k in gammas.keys() if k not in ctrl_x_all]
        if unknown:
            raise KeyError(f"Unknown SHAPE_GAMMAS keys: {unknown}. "
                           f"Use only {list(ctrl_x_all.keys())}.")
        keys = [k for k in ("neck_joint","soundhole_center","form_center","bridge") if k in gammas]
        if not keys:
            def z_top_lin(x): 
                return float(amplitude * np.interp(float(x), xs, N, left=0.0, right=0.0))
            return z_top_lin

        xc = np.array([ctrl_x_all[k] for k in keys], float)
        gc = np.array([float(gammas[k]) for k in keys], float)
        order = np.argsort(xc); xc = xc[order]; gc = gc[order]
        keys_sorted = [keys[i] for i in order]

        # 3) Widths
        sig = []
        for i, k in enumerate(keys_sorted):
            if widths and widths.get(k, None) and widths[k] > 0.0:
                sig.append(float(widths[k]))
            else:
                left_gap  = xc[i] - (xc[i-1] if i-1 >= 0 else xL)
                right_gap = (xc[i+1] if i+1 < len(xc) else xR) - xc[i]
                local = 0.5*(abs(left_gap)+abs(right_gap))
                sig.append(max(1e-6, float(width_factor)*local))
        sig = np.array(sig, float)

        # 4) Log-space blend
        Xdiff2 = (xs[:,None] - xc[None,:])**2
        if kernel == "gauss":
            w = np.exp(-Xdiff2 / (2.0*(sig[None,:]**2 + 1e-12)))
        else:  # cauchy
            w = 1.0 / (1.0 + (Xdiff2 / (sig[None,:]**2 + 1e-12)))
        Wnorm = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
        log_gammas = np.log(np.clip(gc[None,:], 1e-6, None))
        logE = np.sum(Wnorm * log_gammas, axis=1)
        E = np.clip(np.exp(logE), 1.0 - float(max_exponent_delta), 1.0 + float(max_exponent_delta))

        # 5) End gating
        if gate_N_start > 0.0 or gate_N_full > 0.0:
            a = float(gate_N_start); b = float(gate_N_full)
            if b <= a + 1e-9: b = min(1.0, a + 1e-3)
            t = (N - a) / (b - a); t = np.clip(t, 0.0, 1.0)
            gate = t*t*(3 - 2*t)  # smoothstep
            E = 1.0 + gate*(E - 1.0)

        # 6) Apply exponent + amplitude
        N_shaped = N**E
        def z_top(x):
            return float(amplitude * np.interp(float(x), xs, N_shaped, left=0.0, right=0.0))
        return z_top

class SimpleAmplitudeCurve(SideProfilePerControlTopCurve):
    # 1) Global depth
    AMPLITUDE_MODE  = "units"
    AMPLITUDE_UNITS = 1.75

    # 2) Local shaping (start subtle!)
    SHAPE_GAMMAS = {
        "neck_joint":       1.00,
        "soundhole_center": 1.00,
        "form_center":      1,
        "bridge":           1,
    }



class DeepBackCurve(SideProfilePerControlTopCurve):
    # 1) Global depth
    AMPLITUDE_MODE  = "units"
    AMPLITUDE_UNITS = 2

    # 2) Local shaping (start subtle!)
    SHAPE_GAMMAS = {
        "neck_joint":       1.00,
        "soundhole_center": 1.00,
        "form_center":      0.80,
        "bridge":           0.80,
    }

    # 3) Influence widths (broad, smooth)
    SHAPE_WIDTHS = {
        "form_center":      ("span_frac", 0.45),
        "soundhole_center": ("span_frac", 0.35),
        "bridge":           ("span_frac", 0.30),
        "neck_joint":       ("span_frac", 0.25),
    }


class MidCurve(SideProfilePerControlTopCurve):
    # 1) Global depth
    AMPLITUDE_MODE  = "units"
    AMPLITUDE_UNITS = 1.750

    # 2) Local shaping (start subtle!)
    SHAPE_GAMMAS = {
        "neck_joint":       0.85, # fuller
        "soundhole_center": 0.80,
        "form_center":      1.00,
        "bridge":           1.00,
    }

    # 3) Influence widths (broad, smooth)
    SHAPE_WIDTHS = {
        "form_center":      ("span_frac", 0.45),
        "soundhole_center": ("span_frac", 0.35),
        "bridge":           ("span_frac", 0.30),
        "neck_joint":       ("span_frac", 0.25),
    }

class FlatBackCurve(SideProfilePerControlTopCurve):
    # 1) Global depth
    AMPLITUDE_MODE  = "units"
    AMPLITUDE_UNITS = 1.45

    # 2) Local shaping (start subtle!)
    SHAPE_GAMMAS = {
        "neck_joint":       0.65, # fuller
        "soundhole_center": 0.70,
        "form_center":      1.20,
        "bridge":           1.20,
    }

    # 3) Influence widths (broad, smooth)
    SHAPE_WIDTHS = {
        "form_center":      ("span_frac", 0.45),
        "soundhole_center": ("span_frac", 0.35),
        "bridge":           ("span_frac", 0.30),
        "neck_joint":       ("span_frac", 0.25),
    }

# ---------------------------------------------------------------------
# Ribs
# ---------------------------------------------------------------------
def build_ribs_above_soundboard(sections, n_ribs=12):
    """
    Build ribs along the edge-to-edge arc of each section circle,
    choosing the arc that passes through the apex (bowl side).

    IMPORTANT: n_ribs now means the number of INTERVALS between ribs.
    - Actual number of rib curves produced = n_ribs + 1 (including both edges).

    sections: list of (X, C_YZ, r, apex)
      - X:     section's spine coordinate
      - C_YZ:  (C_Y, C_Z) circle center in YZ
      - r:     circle radius
      - apex:  (Y_apex, Z_apex) point on bowl/top-curve

    Returns:
      ribs: list (length n_ribs+1) of arrays of shape (len(sections), 3)
            ribs[i][j] == (X_j, Y_ij, Z_ij)
    """
    import numpy as np

    def wrap(a): return (a + 2*np.pi) % (2*np.pi)

    def edge_to_edge_angles(thetaL, thetaR, theta_apex, n_points):
        """
        Return n_points angles from L to R along the arc that contains theta_apex.
        Includes both endpoints (i.e., n_points = intervals + 1).
        """
        thetaL = wrap(thetaL); thetaR = wrap(thetaR); theta_apex = wrap(theta_apex)
        dLR    = wrap(thetaR - thetaL)        # arc length from L→R (ccw)
        t_ap   = wrap(theta_apex - thetaL)    # apex position from L (ccw)
        if t_ap <= dLR + 1e-12:               # apex lies on L→R arc
            start, span = thetaL, dLR
        else:
            start, span = thetaR, wrap(thetaL - thetaR)  # complementary arc
        ts = np.linspace(0.0, 1.0, int(n_points))
        return wrap(start + ts * span)

    # number of rib curves (including both edges)
    rib_count = int(n_ribs) + 1
    if rib_count < 2:
        rib_count = 2  # at least two ribs (the two edges)

    # Precompute per section angles
    per_section_data = []
    for (X, C_YZ, r, apex) in sections:
        if r <= 0:
            per_section_data.append((X, None, None, None, None, None, None))
            continue

        C_Y, C_Z = float(C_YZ[0]), float(C_YZ[1])
        Y_apex, Z_apex = float(apex[0]), float(apex[1])

        s = -C_Z / r
        if abs(s) > 1:
            per_section_data.append((X, None, None, None, None, None, None))
            continue

        theta_z = np.arcsin(s)
        cand = [theta_z, np.pi - theta_z]
        Ycands = [C_Y + r*np.cos(th) for th in cand]
        idx = np.argsort(Ycands)
        thetaL = float(cand[idx[0]])
        thetaR = float(cand[idx[1]])
        theta_apex = float(np.arctan2(Z_apex - C_Z, Y_apex - C_Y))

        per_section_data.append((X, C_Y, C_Z, r, thetaL, thetaR, theta_apex))

    # Sample the correct arc on each section with rib_count points (intervals = rib_count-1)
    ribs = [[] for _ in range(rib_count)]
    for (X, C_Y, C_Z, r, thetaL, thetaR, theta_apex) in per_section_data:
        if r is None or r <= 0:
            for i in range(rib_count):
                ribs[i].append((float(X), np.nan, np.nan))
            continue

        thetas = edge_to_edge_angles(thetaL, thetaR, theta_apex, rib_count)
        Y = C_Y + r * np.cos(thetas)
        Z = C_Z + r * np.sin(thetas)
        for i in range(rib_count):
            ribs[i].append((float(X), float(Y[i]), float(Z[i])))

    return [np.asarray(rib, dtype=float) for rib in ribs]

# ---------------------------------------------------------------------
# Build bowl
# ---------------------------------------------------------------------

def _resolve_top_curve(lute, top_curve):
    """
    Resolve to a callable z_top(x):
      - None: use SideProfilePerControlTopCurve defaults
      - class / instance: SideProfilePerControlTopCurve (or subclass)
      - callable: use as-is
    """
    if callable(top_curve) and not isinstance(top_curve, type):
        return top_curve
    try:
        if isinstance(top_curve, type) and issubclass(top_curve, TopCurve):
            return top_curve.build(lute)
        if isinstance(top_curve, TopCurve):
            return top_curve.build(lute)
    except Exception:
        pass
    return SideProfilePerControlTopCurve.build(lute)

def build_bowl_for_lute(lute, n_ribs=13, n_sections=None, margin=1e-3, debug=False, top_curve=None):
    """Build a 3D bowl from a lute soundboard and a chosen top curve."""
    z_top = _resolve_top_curve(lute, top_curve)

    # 1) Choose section X positions
    if n_sections is None:
        xs = [
            float(lute.point_neck_joint.x),
            float(lute._get_soundhole_center().x),
            float(lute.form_center.x),
            float(lute.bridge.x),
        ]
    else:
        span = float(lute.form_bottom.x - lute.form_top.x)
        eps  = margin * abs(span)
        x0   = float(lute.form_top.x) + eps
        x1   = float(lute.form_bottom.x) - eps
        xs   = np.linspace(x0, x1, n_sections)
        if debug:
            print("Section X positions (excluding ends):")
            for X in xs:
                print(f"  X={X:.4f}  Δ={float(X - lute.form_top.x):.6f}")

    # 2) Build interior sections
    sections: List[Tuple[float, np.ndarray, float, np.ndarray]] = []
    for X in xs:
        try:
            hit = extract_side_points_at_X(lute, X, debug=debug)
            if hit is None: continue
            L, R, Xs = hit
            Y_apex = spine_point_at_X(lute, Xs)
            Z_apex = float(z_top(Xs))
            # Skip sections with effectively flat apex; end-caps are added anyway.
            if abs(Z_apex) < 1e-6:
                continue
            apex   = np.array([Y_apex, Z_apex])
            C_YZ, r = circle_through_three_points_2d(
                np.array([float(L[1]), 0.0]),
                np.array([float(R[1]), 0.0]),
                apex
            )
            sections.append((Xs, C_YZ, float(r), apex))
        except Exception as e:
            if debug: print(f"Section FAILED at X={X:.4f}: {e}")

    # 3) Synthetic end sections (zero radius) at form_top & form_bottom
    X_ft = float(lute.form_top.x);    Y_ft = float(lute.form_top.y)
    X_fb = float(lute.form_bottom.x); Y_fb = float(lute.form_bottom.y)
    sections.insert(0, (X_ft, np.array([Y_ft, 0.0]), 0.0, np.array([Y_ft, 0.0])))
    sections.append(   (X_fb, np.array([Y_fb, 0.0]), 0.0, np.array([Y_fb, 0.0])))

    # 4) Build ribs
    ribs = build_ribs_above_soundboard(sections, n_ribs=n_ribs) if sections else []
    if ribs:
        for rib in ribs:
            rib[0]  = np.array([X_ft, Y_ft, 0.0], dtype=float)
            rib[-1] = np.array([X_fb, Y_fb, 0.0], dtype=float)

    return sections, ribs

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _soundboard_half_profile_curve(lute, samples_per_arc=500, rotate=True, y_axis=300):
    """Return 3D coords (X,Y,Z) for one half of the soundboard outline.
    If rotate=True, rotate -90° about the spine axis (X, Y=y_axis, Z=0)."""
    pts = []
    for arc in reversed(getattr(lute, 'final_arcs', [])):
        samples = arc.sample_points(samples_per_arc)
        pts.extend(samples)
    if not pts:
        return np.empty((0, 3))
    outline = np.vstack(pts)  # shape (N,2), (X,Y)

    X2d = outline[:,0]
    Y2d = outline[:,1]
    Z2d = np.zeros_like(X2d)

    if rotate:
        # Translate relative to spine axis
        Yt = Y2d - y_axis
        Zt = np.zeros_like(Yt)

        # Rotate –90° about X: (Yt,Zt) → (0, -Yt)
        Yrot = np.zeros_like(Yt)
        Zrot = -Yt

        # Translate back
        Y3d = Yrot + y_axis
        Z3d = Zrot
        X3d = X2d
    else:
        X3d = X2d
        Y3d = Y2d
        Z3d = Z2d

    return np.column_stack([X3d, Y3d, Z3d])

def plot_bowl(lute, sections, ribs, show_apexes=False, highlight_neck_joint=True):
    """
    Plot the 3D lute bowl: ribs + section circles + (optionally) apex points.
    Also (optionally) highlight the neck joint and the section circle nearest to it.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot ribs ---
    for rib in ribs:
        ax.plot(rib[:,0], rib[:,1], rib[:,2], 'b-', lw=1)

    # --- Plot section circles (skip r≈0 and the final synthetic at form_bottom) ---
    if len(sections) <= 80:
        fb_x = float(lute.form_bottom.x)
        for (x, C_YZ, r, _) in sections:
            if r <= 0 or abs(x - fb_x) < 1e-9:
                continue
            C_Y, C_Z = float(C_YZ[0]), float(C_YZ[1])
            phi = np.linspace(0, 2*np.pi, 200)
            Y = C_Y + r*np.cos(phi)
            Z = C_Z + r*np.sin(phi)
            X = np.full_like(Y, float(x))
            ax.plot(X, Y, Z, color='0.3', alpha=0.25)

    # --- Plot apex points & polyline ---
    if show_apexes and sections:
        apex_Xs, apex_Ys, apex_Zs = [], [], []
        for (X, _, _, apex) in sections:
            apex_Xs.append(float(X))
            apex_Ys.append(float(apex[0]))
            apex_Zs.append(float(apex[1]))
        ax.scatter(apex_Xs, apex_Ys, apex_Zs, color="red", s=50, label="apex (top curve)")
        ax.plot(apex_Xs, apex_Ys, apex_Zs, color="red", lw=2, alpha=0.7, label="top curve")

    # --- Mark form_top & form_bottom ---
    ax.scatter([float(lute.form_top.x)],    [float(lute.form_top.y)],    [0.0],
               color="orange", s=60, label="form_top")
    ax.scatter([float(lute.form_bottom.x)], [float(lute.form_bottom.y)], [0.0],
               color="green",  s=60, label="form_bottom")

    # --- Guideline along spine (Z=0) ---
    xs = [float(lute.form_top.x), float(lute.form_bottom.x)]
    ys = [float(lute.form_top.y), float(lute.form_bottom.y)]
    ax.plot(xs, ys, [0.0, 0.0], "k--", alpha=0.6, label="spine")

    # --- Highlight neck joint & nearest section circle ---
    if highlight_neck_joint and sections:
        x_nj = float(lute.point_neck_joint.x)
        y_nj = float(lute.point_neck_joint.y)
        # neck joint point on soundboard (Z=0)
        ax.scatter([x_nj], [y_nj], [0.0], color="#8a2be2", s=70, label="neck_joint")

        # find nearest NON-DEGENERATE section to neck joint in X
        candidates = [(i, abs(float(sec[0]) - x_nj)) for i, sec in enumerate(sections) if float(sec[2]) > 0.0]
        if candidates:
            i_near = min(candidates, key=lambda t: t[1])[0]
            x_sec, C_YZ, r_sec, _ = sections[i_near]
            C_Y, C_Z = float(C_YZ[0]), float(C_YZ[1])
            phi = np.linspace(0, 2*np.pi, 300)
            Yc = C_Y + r_sec*np.cos(phi)
            Zc = C_Z + r_sec*np.sin(phi)
            Xc = np.full_like(Yc, float(x_sec))
            # bold highlighted circle
            ax.plot(Xc, Yc, Zc, color="#8a2be2", lw=2.5, alpha=0.9,
                    label=f"section @ X≈{float(x_sec):.1f} (nearest neck)")

    # --- Overlay original soundboard over the top curve ---
    sb_profile = _soundboard_half_profile_curve(lute)
    if sb_profile.size:
        ax.plot(sb_profile[:,0], sb_profile[:,1], sb_profile[:,2],
                color="darkred", lw=2.0, alpha=0.9, label="Soundboard profile (offset)")

    # --- Labels, aspect ---
    ax.set_xlabel("X (spine)")
    ax.set_ylabel("Y (width)")
    ax.set_zlabel("Z (depth)")
    ax.set_title("Lute bowl from soundboard")
    ax.legend(loc="best")

    # Equalize scales so circles look like circles
    allX = np.concatenate([rib[:,0] for rib in ribs]) if ribs else np.array(xs)
    allY = np.concatenate([rib[:,1] for rib in ribs]) if ribs else np.array(ys)
    allZ = np.concatenate([rib[:,2] for rib in ribs]) if ribs else np.zeros_like(allX)
    set_axes_equal_3d(ax, allX, allY, allZ, use_ortho=True)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Example main (comment out if you import this module elsewhere)
# ---------------------------------------------------------------------

def main():
    try:
        import lutes
    except Exception:
        print("Demo skipped: TurkishOudComplexLowerBout not available."); return

    lute = lutes.ManolLavta(); lute.draw_all()

    sections, ribs = build_bowl_for_lute(
        lute,
        n_ribs=13,
        n_sections=500,
        top_curve=MidCurve
    )
    plot_bowl(lute, sections, ribs)


if __name__ == '__main__':
    main()