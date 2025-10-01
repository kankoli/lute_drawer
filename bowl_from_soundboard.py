# bowl_from_soundboard.py
# Build lute bowls from the soundboard geometry using a side-profile-driven
# top curve with per-control shaping (single strategy, no control-depth constraints).
#
# Usage:
#   lute = ManolLavta()
#   sections, ribs = build_bowl_for_lute(lute, n_ribs=13, n_sections=100)
#   plot_bowl(lute, sections, ribs)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import warnings

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Section(NamedTuple):
    """Geometry for a single bowl slice."""

    x: float
    center: np.ndarray
    radius: float
    apex: np.ndarray


@dataclass(frozen=True)
class SideProfileParameters:
    """Parameters that describe a side-profile-driven top curve."""

    gammas: dict[str, float]
    widths: dict[str, float | tuple]
    amplitude_mode: str = "max_depth"  # "max_depth" | "units"
    amplitude_units: float | None = None
    width_factor: float = 0.9
    samples: int = 400
    margin: float = 1e-3
    gate_start: float = 0.0
    gate_full: float = 0.0
    max_exponent_delta: float = 0.8
    kernel: str = "cauchy"  # "cauchy" | "gauss"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_axes_equal_3d(ax, xs=None, ys=None, zs=None, use_ortho=True):
    """Force equal data scale on a 3D axes so circles look circular."""
    if xs is None or ys is None or zs is None:
        xmin, xmax = ax.get_xlim3d()
        ymin, ymax = ax.get_ylim3d()
        zmin, zmax = ax.get_zlim3d()
    else:
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        zmin, zmax = float(zs.min()), float(zs.max())
    xmid, ymid, zmid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    r = max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-12) / 2.0
    ax.set_xlim3d([xmid - r, xmid + r])
    ax.set_ylim3d([ymid - r, ymid + r])
    ax.set_zlim3d([zmid - r, zmid + r])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # pragma: no cover
        pass
    if use_ortho:
        try:
            ax.set_proj_type("ortho")
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Soundboard sampling
# ---------------------------------------------------------------------------


def circle_through_three_points_2d(P1, P2, P3):
    """Circle through 3 points in the YZ-plane (inputs are 2D [Y,Z] coords)."""
    P1 = np.asarray(P1, float)
    P2 = np.asarray(P2, float)
    P3 = np.asarray(P3, float)
    mid12 = 0.5 * (P1 + P2)
    mid13 = 0.5 * (P1 + P3)
    d12 = P2 - P1
    d13 = P3 - P1
    area2 = d12[0] * d13[1] - d12[1] * d13[0]
    if abs(area2) < 1e-12:
        raise ValueError("Collinear points, no unique circle.")
    n12 = np.array([-d12[1], d12[0]])
    n13 = np.array([-d13[1], d13[0]])
    A = np.column_stack([n12, -n13])
    b = mid13 - mid12
    t, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    C = mid12 + t * n12
    r = float(np.linalg.norm(C - P1))
    return C, r


def _sample_soundboard_outline(lute, samples_per_arc):
    pts = []
    for arc in getattr(lute, "final_arcs", []):
        pts.append(arc.sample_points(samples_per_arc))
    for arc in getattr(lute, "final_reflected_arcs", []):
        pts.append(arc.sample_points(samples_per_arc))
    if not pts:
        return np.empty((0, 2))
    return np.vstack(pts)


def _intersections_with_vertical(outline_xy, x_const, tol=1e-9):
    ys, P = [], outline_xy
    for i in range(len(P) - 1):
        x0, y0 = P[i]
        x1, y1 = P[i + 1]
        dx0, dx1 = x0 - x_const, x1 - x_const
        if dx0 == 0 and dx1 == 0:
            ys.extend([y0, y1])
            continue
        if (dx0 == 0) ^ (dx1 == 0):
            ys.append(y0 if dx0 == 0 else y1)
            continue
        if (dx0 < 0 and dx1 > 0) or (dx0 > 0 and dx1 < 0):
            t = (x_const - x0) / (x1 - x0)
            ys.append(y0 + t * (y1 - y0))
    ys = sorted(ys)
    dedup = []
    for y in ys:
        if not dedup or abs(y - dedup[-1]) > tol:
            dedup.append(y)
    return dedup


def _extract_side_points_at_X(lute, X, *, debug=False, min_width=1e-3, samples_per_arc=500):
    if abs(float(X) - float(lute.form_top.x)) < 1e-12 or abs(float(X) - float(lute.form_bottom.x)) < 1e-12:
        return None
    outline = _sample_soundboard_outline(lute, samples_per_arc=samples_per_arc)
    ys = _intersections_with_vertical(outline, float(X))
    if len(ys) >= 2:
        yL, yR = ys[0], ys[-1]
        if abs(yR - yL) >= min_width:
            return (np.array([float(X), yL]), np.array([float(X), yR]), float(X))
    if debug:
        fig, ax = plt.subplots(figsize=(8, 6))
        if outline.size:
            ax.plot(outline[:, 0], outline[:, 1], color="0.4", lw=1.0)
        ax.axvline(float(X), color="r", ls="--")
        for y in ys:
            ax.plot(float(X), y, "ro")
        ax.set_aspect("equal", adjustable="box")
        plt.show()
    raise RuntimeError(f"Could not find two side intersections at X={float(X):.4f}")


def _spine_point_at_X(lute, X: float):
    x0, y0 = float(lute.form_top.x), float(lute.form_top.y)
    x1, y1 = float(lute.form_bottom.x), float(lute.form_bottom.y)
    if abs(x1 - x0) < 1e-12:
        return y0
    t = (float(X) - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _select_section_positions(lute, n_sections: int | None, margin: float, debug: bool) -> np.ndarray:
    if n_sections is None:
        xs: List[float] = [float(lute.point_neck_joint.x)]
        getter = getattr(lute, "_get_soundhole_center", None)
        if callable(getter):
            try:
                center = getter()
                if center is not None:
                    xs.append(float(center.x))
            except Exception:
                pass
        xs.append(float(lute.form_center.x))
        xs.append(float(lute.bridge.x))
        return np.array(xs, dtype=float)

    span = float(lute.form_bottom.x - lute.form_top.x)
    eps = margin * abs(span)
    x0 = float(lute.form_top.x) + eps
    x1 = float(lute.form_bottom.x) - eps
    xs = np.linspace(x0, x1, n_sections)
    if debug:
        print("Section X positions (excluding ends):")
        for X in xs:
            print(f"  X={X:.4f}  Δ={float(X - lute.form_top.x):.6f}")
    return xs


def _sample_section(lute, X: float, z_top: Callable[[float], float], *, debug: bool) -> Section | None:
    hit = _extract_side_points_at_X(lute, X, debug=debug)
    if hit is None:
        return None
    L, R, Xs = hit
    Y_apex = _spine_point_at_X(lute, Xs)
    Z_apex = float(z_top(Xs))
    if abs(Z_apex) < 1e-6:
        return None
    apex = np.array([Y_apex, Z_apex])
    C_YZ, r = circle_through_three_points_2d(
        np.array([float(L[1]), 0.0]),
        np.array([float(R[1]), 0.0]),
        apex,
    )
    if float(C_YZ[1]) >= 0.0:
        warnings.warn(
            (
                "Section circle center lies on or above the soundboard plane "
                f"(X={Xs:.4f}); resulting mold may trap the bowl."
            ),
            RuntimeWarning,
        )
    return Section(Xs, C_YZ, float(r), apex)


def _build_sections(lute, xs: Sequence[float], z_top: Callable[[float], float], *, debug: bool) -> List[Section]:
    sections: List[Section] = []
    for X in xs:
        try:
            section = _sample_section(lute, X, z_top, debug=debug)
        except Exception as exc:
            if debug:
                print(f"Section FAILED at X={X:.4f}: {exc}")
            continue
        if section is not None:
            sections.append(section)
    return sections


def _add_endcap_sections(lute, sections: Sequence[Section]) -> List[Section]:
    X_ft = float(lute.form_top.x)
    Y_ft = float(lute.form_top.y)
    X_fb = float(lute.form_bottom.x)
    Y_fb = float(lute.form_bottom.y)
    start = Section(X_ft, np.array([Y_ft, 0.0]), 0.0, np.array([Y_ft, 0.0]))
    end = Section(X_fb, np.array([Y_fb, 0.0]), 0.0, np.array([Y_fb, 0.0]))
    return [start, *sections, end]


# ---------------------------------------------------------------------------
# Ribs
# ---------------------------------------------------------------------------


def _edge_to_edge_angles(thetaL, thetaR, theta_apex, n_points):
    def wrap(a):
        return (a + 2 * np.pi) % (2 * np.pi)

    thetaL = wrap(thetaL)
    thetaR = wrap(thetaR)
    theta_apex = wrap(theta_apex)
    dLR = wrap(thetaR - thetaL)
    t_ap = wrap(theta_apex - thetaL)
    if t_ap <= dLR + 1e-12:
        start, span = thetaL, dLR
    else:
        start, span = thetaR, wrap(thetaL - thetaR)
    ts = np.linspace(0.0, 1.0, int(n_points))
    return wrap(start + ts * span)


def _build_ribs(sections: Sequence[Section], n_ribs: int) -> List[np.ndarray]:
    rib_count = int(n_ribs) + 1
    if rib_count < 2:
        rib_count = 2

    per_section_data = []
    for sec in sections:
        X, (C_Y, C_Z), r, apex = sec
        if r <= 0:
            per_section_data.append((X, None, None, None, None, None, None))
            continue
        Y_apex, Z_apex = float(apex[0]), float(apex[1])
        s = -C_Z / r
        if abs(s) > 1:
            per_section_data.append((X, None, None, None, None, None, None))
            continue
        theta_z = np.arcsin(s)
        cand = [theta_z, np.pi - theta_z]
        Ycands = [C_Y + r * np.cos(th) for th in cand]
        idx = np.argsort(Ycands)
        thetaL = float(cand[idx[0]])
        thetaR = float(cand[idx[1]])
        theta_apex = float(np.arctan2(Z_apex - C_Z, Y_apex - C_Y))
        per_section_data.append((X, C_Y, C_Z, r, thetaL, thetaR, theta_apex))

    ribs = [[] for _ in range(rib_count)]
    for X, C_Y, C_Z, r, thetaL, thetaR, theta_apex in per_section_data:
        if r is None or r <= 0:
            for rib in ribs:
                rib.append((float(X), np.nan, np.nan))
            continue
        thetas = _edge_to_edge_angles(thetaL, thetaR, theta_apex, rib_count)
        Y = C_Y + r * np.cos(thetas)
        Z = C_Z + r * np.sin(thetas)
        for i in range(rib_count):
            ribs[i].append((float(X), float(Y[i]), float(Z[i])))

    return [np.asarray(rib, dtype=float) for rib in ribs]


# ---------------------------------------------------------------------------
# Top curve construction
# ---------------------------------------------------------------------------

class TopCurve:
    name = "base"

    @classmethod
    def build(cls, lute):
        raise NotImplementedError


class SideProfilePerControlTopCurve(TopCurve):
    PARAMETERS = SideProfileParameters(gammas={}, widths={})

    @classmethod
    def _parameters(cls) -> SideProfileParameters:
        return cls.PARAMETERS

    @classmethod
    def build(cls, lute):
        params = cls._parameters()
        if not params.gammas:
            raise RuntimeError("SideProfilePerControlTopCurve requires non-empty SHAPE_GAMMAS")
        return _build_side_profile_top_curve(lute, params)


class SimpleAmplitudeCurve(SideProfilePerControlTopCurve):
    PARAMETERS = SideProfileParameters(
        gammas={
            "neck_joint": 1.00,
            "soundhole_center": 1.00,
            "form_center": 1.00,
            "bridge": 1.00,
        },
        widths={},
        amplitude_mode="units",
        amplitude_units=1.75,
    )


class DeepBackCurve(SideProfilePerControlTopCurve):
    PARAMETERS = SideProfileParameters(
        gammas={
            "neck_joint": 1.35,
            "soundhole_center": 1.20,
            "form_center": 0.85,
            "bridge": 0.85,
        },
        widths={
            "form_center": ("span_frac", 0.45),
            "soundhole_center": ("span_frac", 0.35),
            "bridge": ("span_frac", 0.30),
            "neck_joint": ("span_frac", 0.25),
        },
        amplitude_mode="units",
        amplitude_units=1.8,
    )


class MidCurve(SideProfilePerControlTopCurve):
    PARAMETERS = SideProfileParameters(
        gammas={
            "neck_joint": 1.20,
            "soundhole_center": 1.10,
            "form_center": 0.95,
            "bridge": 0.95,
        },
        widths={
            "form_center": ("span_frac", 0.45),
            "soundhole_center": ("span_frac", 0.35),
            "bridge": ("span_frac", 0.30),
            "neck_joint": ("span_frac", 0.25),
        },
        amplitude_mode="units",
        amplitude_units=1.75,
    )


class FlatBackCurve(SideProfilePerControlTopCurve):
    PARAMETERS = SideProfileParameters(
        gammas={
            "neck_joint": 1.35,
            "soundhole_center": 1.20,
            "form_center": 1.05,
            "bridge": 1.05,
        },
        widths={
            "form_center": ("span_frac", 0.45),
            "soundhole_center": ("span_frac", 0.35),
            "bridge": ("span_frac", 0.30),
            "neck_joint": ("span_frac", 0.25),
        },
        amplitude_mode="units",
        amplitude_units=1.45,
    )


def _resolve_top_curve(lute, top_curve):
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


def _build_side_profile_top_curve(lute, params: SideProfileParameters) -> Callable[[float], float]:
    margin = params.margin
    n_samples = params.samples
    xs = _select_section_positions(lute, n_samples, margin, debug=False)

    W = []
    for X in xs:
        hit = _extract_side_points_at_X(lute, X)
        if hit is None:
            W.append(0.0)
            continue
        L, R, _ = hit
        y_spine = _spine_point_at_X(lute, X)
        W.append(max(abs(float(L[1]) - y_spine), abs(float(R[1]) - y_spine)))
    W = np.asarray(W, float)
    if W.size == 0 or float(W.max()) < 1e-12:
        return lambda _x: 0.0
    W[0] = 0.0
    W[-1] = 0.0
    N = W / float(W.max())

    ctrl_x_all = {
        "neck_joint": float(lute.point_neck_joint.x),
        "form_center": float(lute.form_center.x),
        "bridge": float(lute.bridge.x),
    }

    getter = getattr(lute, "_get_soundhole_center", None)
    if callable(getter):
        try:
            center = getter()
            if center is not None:
                ctrl_x_all["soundhole_center"] = float(center.x)
        except Exception:
            pass

    gammas = {k: params.gammas[k] for k in params.gammas if k in ctrl_x_all}
    if not gammas:
        def z_top_lin(x):
            return float(_resolve_amplitude(lute, params) * np.interp(float(x), xs, N, left=0.0, right=0.0))

        return z_top_lin

    xc = np.array([ctrl_x_all[k] for k in gammas], float)
    gc = np.array([float(gammas[k]) for k in gammas], float)
    order = np.argsort(xc)
    xc = xc[order]
    gc = gc[order]

    widths = _resolve_widths(lute, params)
    sig = []
    for idx, key in enumerate(np.array(list(gammas.keys()))[order]):
        width = widths.get(key)
        if width and width > 0.0:
            sig.append(float(width))
        else:
            left_gap = xc[idx] - (xc[idx - 1] if idx - 1 >= 0 else float(lute.form_top.x))
            right_gap = (xc[idx + 1] if idx + 1 < len(xc) else float(lute.form_bottom.x)) - xc[idx]
            local = 0.5 * (abs(left_gap) + abs(right_gap))
            sig.append(max(1e-6, float(params.width_factor) * local))
    sig = np.array(sig, float)

    Xdiff2 = (xs[:, None] - xc[None, :]) ** 2
    if params.kernel == "gauss":
        w = np.exp(-Xdiff2 / (2.0 * (sig[None, :] ** 2 + 1e-12)))
    else:
        w = 1.0 / (1.0 + (Xdiff2 / (sig[None, :] ** 2 + 1e-12)))
    Wnorm = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
    log_gammas = np.log(np.clip(gc[None, :], 1e-6, None))
    logE = np.sum(Wnorm * log_gammas, axis=1)
    E = np.clip(np.exp(logE), 1.0 - float(params.max_exponent_delta), 1.0 + float(params.max_exponent_delta))

    if params.gate_start > 0.0 or params.gate_full > 0.0:
        a = float(params.gate_start)
        b = float(params.gate_full)
        if b <= a + 1e-9:
            b = min(1.0, a + 1e-3)
        t = (N - a) / (b - a)
        t = np.clip(t, 0.0, 1.0)
        gate = t * t * (3 - 2 * t)
        E = 1.0 + gate * (E - 1.0)

    N_shaped = N ** E
    amplitude = _resolve_amplitude(lute, params)

    def z_top(x):
        return float(amplitude * np.interp(float(x), xs, N_shaped, left=0.0, right=0.0))

    return z_top


def _resolve_widths(lute, params: SideProfileParameters) -> dict[str, float]:
    widths: dict[str, float] = {}
    if not params.widths:
        return widths
    span = abs(float(lute.form_bottom.x) - float(lute.form_top.x))
    for key, value in params.widths.items():
        if isinstance(value, (int, float)):
            widths[key] = float(value)
        elif isinstance(value, (tuple, list)) and len(value) == 2 and value[0] == "span_frac":
            widths[key] = float(value[1]) * span
    return widths


def _resolve_amplitude(lute, params: SideProfileParameters) -> float:
    u = float(getattr(lute, "unit", 1.0))
    if params.amplitude_mode == "units" and params.amplitude_units is not None:
        return float(params.amplitude_units) * u
    return float(max(params.gammas.values()) if params.gammas else 1.0) * u


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_bowl_for_lute(
    lute,
    n_ribs: int = 13,
    n_sections: int | None = None,
    margin: float = 1e-3,
    debug: bool = False,
    top_curve=None,
):
    """Build a 3D bowl from a lute soundboard and a chosen top curve."""
    lute.draw_all()
    z_top = _resolve_top_curve(lute, top_curve)

    xs = _select_section_positions(lute, n_sections, margin, debug)
    sections = _build_sections(lute, xs, z_top, debug=debug)
    sections = _add_endcap_sections(lute, sections)

    ribs = _build_ribs(sections, n_ribs) if sections else []
    if ribs:
        X_ft = float(lute.form_top.x)
        Y_ft = float(lute.form_top.y)
        X_fb = float(lute.form_bottom.x)
        Y_fb = float(lute.form_bottom.y)
        for rib in ribs:
            rib[0] = np.array([X_ft, Y_ft, 0.0], dtype=float)
            rib[-1] = np.array([X_fb, Y_fb, 0.0], dtype=float)

    return sections, ribs


def plot_bowl(lute, sections, ribs, show_apexes=False, highlight_neck_joint=True):
    import numpy as np

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for rib in ribs:
        ax.plot(rib[:, 0], rib[:, 1], rib[:, 2], "b-", lw=1)

    if len(sections) <= 80:
        fb_x = float(lute.form_bottom.x)
        for section in sections:
            x, center, r, _ = section
            if r <= 0 or abs(x - fb_x) < 1e-9:
                continue
            C_Y, C_Z = float(center[0]), float(center[1])
            phi = np.linspace(0, 2 * np.pi, 200)
            Y = C_Y + r * np.cos(phi)
            Z = C_Z + r * np.sin(phi)
            X = np.full_like(Y, float(x))
            ax.plot(X, Y, Z, color="0.3", alpha=0.25)

    if show_apexes and sections:
        apex_Xs, apex_Ys, apex_Zs = [], [], []
        for section in sections:
            apex_Xs.append(float(section.x))
            apex_Ys.append(float(section.apex[0]))
            apex_Zs.append(float(section.apex[1]))
        ax.scatter(apex_Xs, apex_Ys, apex_Zs, color="red", s=50, label="apex (top curve)")
        ax.plot(apex_Xs, apex_Ys, apex_Zs, color="red", lw=2, alpha=0.7, label="top curve")

    ax.scatter([float(lute.form_top.x)], [float(lute.form_top.y)], [0.0], color="orange", s=60, label="form_top")
    ax.scatter([float(lute.form_bottom.x)], [float(lute.form_bottom.y)], [0.0], color="green", s=60, label="form_bottom")

    xs = [float(lute.form_top.x), float(lute.form_bottom.x)]
    ys = [float(lute.form_top.y), float(lute.form_bottom.y)]
    ax.plot(xs, ys, [0.0, 0.0], "k--", alpha=0.6, label="spine")

    if highlight_neck_joint and sections:
        x_nj = float(lute.point_neck_joint.x)
        y_nj = float(lute.point_neck_joint.y)
        ax.scatter([x_nj], [y_nj], [0.0], color="purple", s=60, label="neck joint")

    set_axes_equal_3d(
        ax,
        xs=[rib[:, 0] for rib in ribs] if ribs else None,
        ys=[rib[:, 1] for rib in ribs] if ribs else None,
        zs=[rib[:, 2] for rib in ribs] if ribs else None,
    )

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Execution helper
# ---------------------------------------------------------------------------


def main():
    try:
        import lutes
    except Exception:
        print("Demo skipped: lute definitions not available.")
        return

    lute = lutes.ManolLavta()
    sections, ribs = build_bowl_for_lute(
        lute,
        n_ribs=13,
        n_sections=500,
        top_curve=MidCurve,
    )
    plot_bowl(lute, sections, ribs)


if __name__ == "__main__":
    main()
