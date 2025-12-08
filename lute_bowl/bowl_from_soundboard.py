# bowl_from_soundboard.py
# Build lute bowls from the soundboard geometry using a side-profile-driven
# top curve with per-control shaping (single strategy, no control-depth constraints).
#
# Usage:
#   lute = ManolLavta()
#   sections, ribs = build_bowl_ribs(lute, n_ribs=13)
#   from plotting.bowl import plot_bowl
#   plot_bowl(lute, sections, ribs)

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List, Sequence

import numpy as np
import warnings

from .top_curves import (
    DeepBackCurve,
    FlatBackCurve,
    MidCurve,
    SideProfileParameters,
    SideProfilePerControlTopCurve,
    SimpleAmplitudeCurve,
    TopCurve,
)
from .section_curve import BaseSectionCurve, CircularSectionCurve

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_EX = np.array([1.0, 0.0, 0.0], dtype=float)
_EPS = 1e-9


@dataclass(frozen=True)
class Section:
    """Geometry for a single bowl slice."""

    x: float
    curve: BaseSectionCurve

    @property
    def center(self) -> np.ndarray:
        return self.curve.center

    @property
    def radius(self) -> float:
        return self.curve.radius

    @property
    def apex(self) -> np.ndarray:
        return self.curve.apex

    def __iter__(self):
        yield self.x
        yield self.center
        yield self.radius
        yield self.apex

    def __len__(self):
        return 4

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.x
        if idx == 1:
            return self.center
        if idx == 2:
            return self.radius
        if idx == 3:
            return self.apex
        raise IndexError(idx)


# ---------------------------------------------------------------------------
# Soundboard sampling
# ---------------------------------------------------------------------------


def _normalize_angle_deg(value: float) -> float:
    """Wrap angles to [-180, 180] degrees."""
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _angle_on_arc(arc, x: float, y: float) -> float:
    cx = float(arc.center.x)
    cy = float(arc.center.y)
    angle = math.degrees(math.atan2(y - cy, x - cx))
    return _normalize_angle_deg(angle)


def _point_within_arc(arc, x: float, y: float, tol: float = 1e-7) -> bool:
    angle1 = float(arc.angle1)
    angle2 = float(arc.angle2)
    target = _angle_on_arc(arc, x, y)
    delta = _normalize_angle_deg(angle2 - angle1)
    diff = _normalize_angle_deg(target - angle1)
    if delta >= 0:
        return -tol <= diff <= delta + tol
    return delta - tol <= diff <= tol


def _arc_intersections_with_vertical(arc, x: float, tol: float = 1e-9) -> List[np.ndarray]:
    cx = float(arc.center.x)
    cy = float(arc.center.y)
    r = float(arc.radius)
    dx = x - cx
    if abs(dx) > r + tol:
        return []
    inside = r * r - dx * dx
    if inside < 0:
        if inside < -1e-6:
            return []
        inside = 0.0
    root = math.sqrt(inside)
    candidates = [cy + root, cy - root] if root > tol else [cy]
    points: List[np.ndarray] = []
    for y in candidates:
        if _point_within_arc(arc, x, y):
            points.append(np.array([x, y], dtype=float))
    return points


def _extract_side_points_at_X(lute, X, *, debug=False, min_width=1e-3, samples_per_arc=500):
    x_val = float(X)
    arcs = list(getattr(lute, "final_arcs", [])) + list(getattr(lute, "final_reflected_arcs", []))
    if not arcs:
        raise RuntimeError("Soundboard has no final arcs to intersect.")

    intersections: List[np.ndarray] = []
    for arc in arcs:
        intersections.extend(_arc_intersections_with_vertical(arc, x_val))

    if not intersections:
        raise RuntimeError(f"Could not find intersections at X={x_val:.4f}")

    intersections.sort(key=lambda pt: pt[1])
    dedup: List[np.ndarray] = []
    for pt in intersections:
        if not dedup or abs(pt[1] - dedup[-1][1]) > 1e-7:
            dedup.append(pt)

    if len(dedup) >= 2:
        yL = dedup[0][1]
        yR = dedup[-1][1]
        if abs(yR - yL) >= min_width:
            return (np.array([x_val, yL]), np.array([x_val, yR]), x_val)
        mid_y = 0.5 * (yL + yR)
        point = np.array([x_val, mid_y], dtype=float)
        return (point, point.copy(), x_val)

    if len(dedup) == 1:
        y = float(dedup[0][1])
        point = np.array([x_val, y], dtype=float)
        return (point, point.copy(), x_val)

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        for arc in arcs:
            pts = arc.sample_points(400)
            ax.plot(pts[:, 0], pts[:, 1], color="0.4", lw=1.0)
        ax.axvline(x_val, color="r", ls="--")
        for pt in dedup:
            ax.plot(pt[0], pt[1], "ro")
        ax.set_aspect("equal", adjustable="box")
        plt.show()

    raise RuntimeError(f"Could not find two side intersections at X={x_val:.4f}")


def _spine_point_at_X(lute, X: float):
    x0, y0 = float(lute.form_top.x), float(lute.form_top.y)
    x1, y1 = float(lute.form_bottom.x), float(lute.form_bottom.y)
    if abs(x1 - x0) < 1e-12:
        return y0
    t = (float(X) - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _spine_point_xyz(lute, x: float) -> np.ndarray:
    """Return the (X,Y,Z) point on the spine lying on the soundboard plane."""
    y = float(_spine_point_at_X(lute, x))
    return np.array([float(x), y, 0.0], dtype=float)


def _select_section_positions(lute, n_sections: int, margin: float, debug: bool) -> np.ndarray:
    x_top = float(lute.form_top.x)
    x_bottom = float(lute.form_bottom.x)
    span = x_bottom - x_top
    if span <= _EPS:
        raise ValueError("Soundboard span collapsed; cannot sample sections.")

    margin = max(0.0, float(margin))
    eps = margin * abs(span)
    if eps * 2.0 >= span:
        eps = 0.5 * span * (1.0 - 1e-6)

    x0 = x_top + eps
    x1 = x_bottom - eps
    xs = np.linspace(x0, x1, n_sections)
    if debug:
        print("Section X positions (excluding ends):")
        for X in xs:
            print(f"  X={X:.4f}  Δ={float(X - lute.form_top.x):.6f}")
    return xs


def _sample_section(
    lute,
    X: float,
    z_top: Callable[[float], float],
    *,
    curve_cls: type[BaseSectionCurve] = CircularSectionCurve,
    curve_kwargs: dict | None = None,
) -> Section | None:
    curve_kwargs = curve_kwargs or {}
    hit = _extract_side_points_at_X(lute, X)
    if hit is None:
        return None
    L, R, Xs = hit
    Y_apex = _spine_point_at_X(lute, Xs)
    Z_apex = float(z_top(Xs))
    width = abs(float(L[1]) - float(R[1]))
    if width <= 1e-6:
        apex = np.array([float(Y_apex), max(Z_apex, 0.0)], dtype=float)
        center = np.array([float(Y_apex), 0.0], dtype=float)
        curve = curve_cls.degenerate(center, apex, **curve_kwargs)
        return Section(Xs, curve)

    if abs(Z_apex) < 1e-6:
        dist_bottom = abs(float(lute.form_bottom.x) - float(Xs))
        if dist_bottom <= 1.0:
            width = abs(float(L[1]) - float(R[1]))
            if width > 0.0:
                Z_apex = max(width * 0.02, 0.5)
            else:
                return None
        else:
            return None
    apex = np.array([Y_apex, Z_apex], dtype=float)
    curve = curve_cls.from_span(
        np.array([float(L[1]), 0.0], dtype=float),
        np.array([float(R[1]), 0.0], dtype=float),
        apex,
        **curve_kwargs,
    )
    if float(curve.center[1]) >= 0.0:
        warnings.warn(
            (
                "Section circle center lies on or above the soundboard plane "
                f"(X={Xs:.4f}, Yc={float(curve.center[1]):.4f}); resulting mold may trap the bowl."
            ),
            RuntimeWarning,
        )
    return Section(Xs, curve)

# ---------------------------------------------------------------------------
# Ribs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RibPlane:
    normal: np.ndarray
    reference_yz: np.ndarray


def _derive_planar_ribs(
    lute,
    sections: Sequence[Section],
    n_ribs: int,
    x_start: float,
    x_end: float,
    *,
    skirt_span: float = 0.0,
    z_top: Callable[[float], float] | None = None,
    eye_x: float | None = None,
    division_mode: str = "angle",
    debug_rib_indices: list[int] | None = None,
    debug_logger: Callable[[str], None] | None = None,
    debug_plot: bool = False,
) -> List[np.ndarray]:
    skirt_span = max(0.0, float(skirt_span))
    span = x_end - x_start
    has_skirts = skirt_span > _EPS and skirt_span < span - _EPS
    baseline_start = np.array([x_start, float(_spine_point_at_X(lute, x_start)), 0.0], dtype=float)
    baseline_end = np.array([x_end, float(_spine_point_at_X(lute, x_end)), 0.0], dtype=float)
    if eye_x is not None and z_top is not None:
        baseline_end = np.array(
            [eye_x, float(_spine_point_at_X(lute, eye_x)), float(z_top(eye_x))],
            dtype=float,
        )

    if not has_skirts:
        return _derive_planar_ribs_base(
            lute,
            sections,
            n_ribs,
            x_start,
            x_end,
            division_mode=division_mode,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            debug_rib_indices=debug_rib_indices,
            debug_logger=debug_logger,
            debug_plot=debug_plot,
        )
    if z_top is None:
        raise ValueError("z_top callable required for skirt ribs.")

    eye_x = eye_x if eye_x is not None else (x_end - skirt_span)
    eye_x = min(max(eye_x, x_start), x_end)
    return _derive_planar_ribs_skirt(
        lute,
        sections,
        n_ribs,
        x_start,
        x_end,
        eye_x,
        z_top,
        division_mode=division_mode,
        debug_rib_indices=debug_rib_indices,
        debug_logger=debug_logger,
        debug_plot=debug_plot,
    )


def _derive_planar_ribs_base(
    lute,
    sections: Sequence[Section],
    n_ribs: int,
    x_start: float,
    x_end: float,
    *,
    division_mode: str = "angle",
    baseline_start: np.ndarray | None = None,
    baseline_end: np.ndarray | None = None,
    debug_rib_indices: list[int] | None = None,
    debug_logger: Callable[[str], None] | None = None,
    debug_plot: bool = False,
) -> List[np.ndarray]:
    """Original rib construction without skirts."""
    if n_ribs < 1:
        raise ValueError("n_ribs must be at least 1.")

    rib_count = int(n_ribs) + 1
    rib_count = max(rib_count, 2)

    radii = [float(section.radius) for section in sections]
    ref_idx = int(np.argmax(radii))
    if radii[ref_idx] <= _EPS:
        raise ValueError("Unable to locate a section with positive radius.")

    ref_section = sections[ref_idx]
    ref_samples = ref_section.curve.divide(rib_count, mode=division_mode)

    x_ref = float(ref_section.x)

    baseline_start = (
        np.asarray(baseline_start, float)
        if baseline_start is not None
        else _spine_point_xyz(lute, x_start)
    )
    baseline_end = (
        np.asarray(baseline_end, float)
        if baseline_end is not None
        else _spine_point_xyz(lute, x_end)
    )

    baseline_vector = baseline_end - baseline_start
    if np.linalg.norm(baseline_vector) <= _EPS:
        raise ValueError("Baseline vector collapsed after applying end blocks.")

    def _baseline_point(x: float) -> np.ndarray:
        if abs(baseline_vector[0]) > _EPS:
            t = (x - baseline_start[0]) / baseline_vector[0]
        else:
            t = (x - baseline_start[0]) / (baseline_vector[0] + _EPS)
        return baseline_start + t * baseline_vector

    spine_ref = _baseline_point(x_ref)

    rib_planes: List[RibPlane] = []
    ribs: List[List[np.ndarray]] = [[] for _ in range(rib_count)]
    debug_ribs = set(debug_rib_indices or [])
    log = debug_logger if debug_logger is not None else (lambda _msg: None)

    for y_ref, z_ref in ref_samples:
        reference_point = np.array([x_ref, y_ref, z_ref], dtype=float)

        plane_normal = np.cross(baseline_vector, reference_point - baseline_start)
        norm_len = np.linalg.norm(plane_normal)
        if norm_len <= _EPS:
            raise ValueError("Degenerate plane encountered while constructing rib.")
        plane_normal /= norm_len

        rib_planes.append(
            RibPlane(
                plane_normal,
                np.array([y_ref, z_ref], dtype=float),
            )
        )

    for section in sections:
        x = float(section.x)
        r = float(section.radius)
        base_point = _baseline_point(x)
        base_yz = np.array([float(base_point[1]), float(base_point[2])], dtype=float)

        if r <= _EPS:
            for trace in ribs:
                trace.append(np.array([x, base_yz[0], base_yz[1]], dtype=float))
            continue

        for idx_rib, (plane, trace) in enumerate(zip(rib_planes, ribs)):
            left_yz, right_yz = section.curve.intersect_with_plane(
                plane.normal,
                baseline_start,
                float(section.x),
            )

            candidates = [np.asarray(left_yz, float), np.asarray(right_yz, float)]
            # Choose the intersection closest to the previous point (or reference).
            target_yz = (
                np.array([trace[-1][1], trace[-1][2]], dtype=float)
                if trace
                else plane.reference_yz
            )
            yz_point = min(candidates, key=lambda p: float(np.linalg.norm(p - target_yz)))
            if (idx_rib + 1) in debug_ribs:
                log(
                    f"rib={idx_rib+1} x={x:.4f} "
                    f"candidates={[(float(p[0]), float(p[1])) for p in candidates]} "
                    f"target={tuple(float(v) for v in target_yz)} "
                    f"chosen={(float(yz_point[0]), float(yz_point[1]))}"
                )
            trace.append(np.array([x, yz_point[0], yz_point[1]], dtype=float))

    return [np.asarray(trace, dtype=float) for trace in ribs]


def _derive_planar_ribs_skirt(
    lute,
    sections: Sequence[Section],
    n_ribs: int,
    x_start: float,
    x_end: float,
    eye_x: float,
    z_top: Callable[[float], float],
    *,
    division_mode: str = "angle",
    debug_rib_indices: list[int] | None = None,
    debug_logger: Callable[[str], None] | None = None,
    debug_plot: bool = False,
) -> List[np.ndarray]:
    """Skirt ribs: interior ribs end at eye; skirts run to tail via straight segment."""
    rib_count = int(n_ribs) + 1
    rib_count = max(rib_count, 3)

    def _section_at_x(target: float) -> Section:
        for s in sections:
            if abs(float(s.x) - target) <= 1e-8:
                return s
        raise ValueError(f"No section sampled at X={target:.6f}")

    neck_section = _section_at_x(x_start)
    eye_section = _section_at_x(eye_x)
    if float(neck_section.radius) <= _EPS or float(eye_section.radius) <= _EPS:
        raise ValueError("Neck or eye section radius must be positive for skirt ribs.")

    eye_point = _spine_point_xyz(lute, eye_x)
    eye_point[2] = max(float(z_top(eye_x)), 0.0)
    tail_point = _spine_point_xyz(lute, x_end)
    skirt_indices = {0, rib_count - 1}

    debug_ribs = set(debug_rib_indices or [])
    log = debug_logger if debug_logger is not None else (lambda _msg: None)

    sections_to_eye = [s for s in sections if float(s.x) <= eye_x + 1e-9]
    ribs_to_eye = _derive_planar_ribs_base(
        lute,
        sections_to_eye,
        n_ribs,
        x_start,
        eye_x,
        division_mode=division_mode,
        baseline_end=eye_point,
        debug_rib_indices=debug_rib_indices,
        debug_logger=debug_logger,
        debug_plot=debug_plot,
    )

    anchors_yz: List[np.ndarray] = []
    for rib in ribs_to_eye:
        anchors_yz.append(np.array([float(rib[-1][1]), float(rib[-1][2])], dtype=float))

    ribs: List[List[np.ndarray]] = [list(map(np.array, rib.tolist())) for rib in ribs_to_eye]

    for section in sections:
        x = float(section.x)
        if x <= eye_x + 1e-9:
            continue

        t_line = (x - eye_x) / max(x_end - eye_x, _EPS)
        t_line = max(0.0, min(1.0, t_line))
        for idx in range(rib_count):
            if idx in skirt_indices:
                anchor = anchors_yz[idx]
                yz_end = np.array([tail_point[1], tail_point[2]], dtype=float)
                yz = anchor + t_line * (yz_end - anchor)
                ribs[idx].append(np.array([x, yz[0], yz[1]], dtype=float))
            else:
                anchor = anchors_yz[idx]
                ribs[idx].append(np.array([x, anchor[0], anchor[1]], dtype=float))
            if (idx + 1) in debug_ribs:
                log(
                    f"rib={idx+1} x={x:.4f} anchor={(float(anchors_yz[idx][0]), float(anchors_yz[idx][1]))} "
                    f"t_line={t_line:.3f}"
                )
                if debug_plot:
                    try:
                        import matplotlib.pyplot as plt
                    except Exception:
                        continue
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.plot([float(eye_x), x_end], [anchor[0], tail_point[1]], "k--", alpha=0.6, label="skirt line")
                    ax.scatter([x], [yz[0]], c="tab:red", s=40, label="rib point")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y on skirt")
                    ax.legend()
                    ax.set_title(f"Skirt rib {idx+1} at X={x:.3f}")
                    plt.show()

    # Replace outermost ribs with soundboard outlines (projected on Z=0) to lock boundaries.
    boundary_left: list[np.ndarray] = []
    boundary_right: list[np.ndarray] = []
    for section in sections:
        hit = _extract_side_points_at_X(lute, section.x)
        if hit is None:
            raise RuntimeError(f"Failed to locate soundboard intersections at X={float(section.x):.6f}")
        L, R, Xs = hit
        boundary_left.append(np.array([float(Xs), float(L[1]), 0.0], dtype=float))
        boundary_right.append(np.array([float(Xs), float(R[1]), 0.0], dtype=float))

    ribs[0] = boundary_left
    ribs[-1] = boundary_right

    return [np.asarray(trace, dtype=float) for trace in ribs]
