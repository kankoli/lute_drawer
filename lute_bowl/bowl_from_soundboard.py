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
from .section_curve import SectionCurve

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_EX = np.array([1.0, 0.0, 0.0], dtype=float)
_EPS = 1e-9


@dataclass(frozen=True)
class Section:
    """Geometry for a single bowl slice."""

    x: float
    curve: SectionCurve

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


def _sample_section(lute, X: float, z_top: Callable[[float], float]) -> Section | None:
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
        curve = SectionCurve.degenerate(center, apex)
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
    curve = SectionCurve.from_span(
        np.array([float(L[1]), 0.0], dtype=float),
        np.array([float(R[1]), 0.0], dtype=float),
        apex,
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
    direction: np.ndarray


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
) -> List[np.ndarray]:
    skirt_span = max(0.0, float(skirt_span))
    span = x_end - x_start
    has_skirts = skirt_span > _EPS and skirt_span < span - _EPS

    if not has_skirts:
        return _derive_planar_ribs_base(lute, sections, n_ribs, x_start, x_end, division_mode=division_mode)
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
    )


def _derive_planar_ribs_base(
    lute,
    sections: Sequence[Section],
    n_ribs: int,
    x_start: float,
    x_end: float,
    *,
    division_mode: str = "angle",
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
    spine_ref = _spine_point_xyz(lute, x_ref)

    spine_start = _spine_point_xyz(lute, x_start)
    spine_end = _spine_point_xyz(lute, x_end)
    spine_vector = spine_end - spine_start
    if np.linalg.norm(spine_vector) <= _EPS:
        raise ValueError("Spine vector collapsed after applying end blocks.")

    rib_planes: List[RibPlane] = []
    ribs: List[List[np.ndarray]] = [[] for _ in range(rib_count)]

    for y_ref, z_ref in ref_samples:
        reference_point = np.array([x_ref, y_ref, z_ref], dtype=float)

        plane_normal = np.cross(spine_vector, reference_point - spine_start)
        norm_len = np.linalg.norm(plane_normal)
        if norm_len <= _EPS:
            raise ValueError("Degenerate plane encountered while constructing rib.")
        plane_normal /= norm_len

        direction = np.cross(plane_normal, _EX)
        dir_len = np.sqrt(direction[1] ** 2 + direction[2] ** 2)
        if dir_len <= _EPS:
            raise ValueError("Failed to derive planar direction for rib.")

        delta_ref = reference_point - spine_ref
        s_ref = (delta_ref[1] * direction[1] + delta_ref[2] * direction[2]) / (dir_len**2)
        if s_ref < 0.0:
            plane_normal = -plane_normal
            direction = -direction

        rib_planes.append(RibPlane(plane_normal, direction))

    for section in sections:
        x = float(section.x)
        r = float(section.radius)
        spine_y = float(_spine_point_at_X(lute, x))
        base_yz = np.array([spine_y, 0.0], dtype=float)

        if r <= _EPS:
            for trace in ribs:
                trace.append(np.array([x, base_yz[0], base_yz[1]], dtype=float))
            continue

        for plane, trace in zip(rib_planes, ribs):
            dy, dz = float(plane.direction[1]), float(plane.direction[2])
            yz_point = section.curve.intersect_with_direction(base_yz, np.array([dy, dz], dtype=float))
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

    neck_points = []
    for y_neck, z_neck in neck_section.curve.divide(rib_count, mode=division_mode):
        neck_points.append(np.array([x_start, y_neck, z_neck], dtype=float))

    eye_point = _spine_point_xyz(lute, eye_x)
    eye_point[2] = max(float(z_top(eye_x)), 0.0)
    tail_point = _spine_point_xyz(lute, x_end)
    skirt_indices = {0, rib_count - 1}

    v1 = neck_points[1] - eye_point
    v2 = neck_points[-2] - eye_point
    eye_normal = np.cross(v1, v2)
    norm_eye = np.linalg.norm(eye_normal)
    if norm_eye <= _EPS:
        raise ValueError("Eye plane is degenerate; adjust skirt span or geometry.")
    eye_normal /= norm_eye

    ribs: List[List[np.ndarray]] = [[] for _ in range(rib_count)]
    anchors_yz: List[np.ndarray | None] = [None for _ in range(rib_count)]

    for section in sections:
        x = float(section.x)

        if x > eye_x + 1e-9:
            t_line = (x - eye_x) / max(x_end - eye_x, _EPS)
            t_line = max(0.0, min(1.0, t_line))
            for idx in range(rib_count):
                if idx in skirt_indices:
                    anchor = anchors_yz[idx] if anchors_yz[idx] is not None else np.array([eye_point[1], eye_point[2]])
                    yz_end = np.array([tail_point[1], tail_point[2]], dtype=float)
                    yz = anchor + t_line * (yz_end - anchor)
                    ribs[idx].append(np.array([x, yz[0], yz[1]], dtype=float))
                else:
                    anchor = anchors_yz[idx]
                    if anchor is None:
                        raise RuntimeError("Missing eye anchor for interior rib.")
                    ribs[idx].append(np.array([x, anchor[0], anchor[1]], dtype=float))
            continue

        if float(section.radius) <= _EPS:
            yz = np.array([float(_spine_point_at_X(lute, x)), 0.0], dtype=float)
            for idx in range(rib_count):
                ribs[idx].append(np.array([x, yz[0], yz[1]], dtype=float))
            continue

        left_yz, right_yz = section.curve.intersect_with_plane(eye_normal, eye_point, float(section.x))
        samples = section.curve.divide_between_points(left_yz, right_yz, rib_count, mode=division_mode)

        for idx, yz in enumerate(samples):
            pt = np.array([x, yz[0], yz[1]], dtype=float)
            ribs[idx].append(pt)
            if abs(x - eye_x) <= 1e-8:
                anchors_yz[idx] = np.array([yz[0], yz[1]], dtype=float)

    # Replace outermost ribs with soundboard outlines (projected on Z=0).
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
