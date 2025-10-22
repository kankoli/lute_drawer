# bowl_from_soundboard.py
# Build lute bowls from the soundboard geometry using a side-profile-driven
# top curve with per-control shaping (single strategy, no control-depth constraints).
#
# Usage:
#   lute = ManolLavta()
#   sections, ribs = build_planar_bowl_for_lute(lute, n_ribs=13, n_sections=100)
#   from plotting.bowl import plot_bowl
#   plot_bowl(lute, sections, ribs)

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, List, NamedTuple, Sequence

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

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_EX = np.array([1.0, 0.0, 0.0], dtype=float)
_EPS = 1e-9


class Section(NamedTuple):
    """Geometry for a single bowl slice."""

    x: float
    center: np.ndarray
    radius: float
    apex: np.ndarray


# ---------------------------------------------------------------------------
# Soundboard sampling
# ---------------------------------------------------------------------------


def _circle_through_three_points_2d(P1, P2, P3):
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
    neck_point = getattr(lute, "point_neck_joint", None)
    x_top = float(neck_point.x) if neck_point is not None else float(lute.form_top.x)
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


def _sample_section(lute, X: float, z_top: Callable[[float], float], *, debug: bool) -> Section | None:
    hit = _extract_side_points_at_X(lute, X, debug=debug)
    if hit is None:
        return None
    L, R, Xs = hit
    Y_apex = _spine_point_at_X(lute, Xs)
    Z_apex = float(z_top(Xs))
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
    apex = np.array([Y_apex, Z_apex])
    C_YZ, r = _circle_through_three_points_2d(
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


def _add_endcap_sections(lute, sections: Sequence[Section], x_start: float, x_end: float) -> List[Section]:
    y_start = float(_spine_point_at_X(lute, x_start))
    y_end = float(_spine_point_at_X(lute, x_end))
    start = Section(x_start, np.array([y_start, 0.0]), 0.0, np.array([y_start, 0.0]))
    end = Section(x_end, np.array([y_end, 0.0]), 0.0, np.array([y_end, 0.0]))
    return [start, *sections, end]


# ---------------------------------------------------------------------------
# Ribs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RibPlane:
    normal: np.ndarray
    direction: np.ndarray


def _section_angles(section: Section) -> tuple[float, float, float]:
    _, center, radius, apex = section
    r = float(radius)
    if r <= _EPS:
        raise ValueError("Section radius must be positive to derive angles.")

    cy, cz = map(float, center)
    ay, az = map(float, apex)

    s = -cz / r
    if abs(s) > 1.0:
        raise ValueError("Section apex is incompatible with fitted circle.")

    theta_z = float(np.arcsin(s))
    candidates = [theta_z, np.pi - theta_z]
    y_candidates = [cy + r * np.cos(theta) for theta in candidates]
    order = np.argsort(y_candidates)
    theta_left = candidates[int(order[0])]
    theta_right = candidates[int(order[1])]
    theta_apex = float(np.arctan2(az - cz, ay - cy))
    return theta_left, theta_right, theta_apex


def _derive_planar_ribs(
    lute,
    sections: Sequence[Section],
    n_ribs: int,
    x_start: float,
    x_end: float,
) -> List[np.ndarray]:
    """Return rib polylines that lie in fixed planes to avoid twist."""
    if n_ribs < 1:
        raise ValueError("n_ribs must be at least 1.")

    rib_count = int(n_ribs) + 1
    if rib_count < 2:
        rib_count = 2

    radii = [float(section.radius) for section in sections]
    ref_idx = int(np.argmax(radii))
    if radii[ref_idx] <= _EPS:
        raise ValueError("Unable to locate a section with positive radius.")

    ref_section = sections[ref_idx]
    theta_left, theta_right, _ = _section_angles(ref_section)
    thetas = np.linspace(theta_left, theta_right, rib_count)

    x_ref = float(ref_section.x)
    cy_ref, cz_ref = map(float, ref_section.center)
    r_ref = float(ref_section.radius)
    spine_ref = _spine_point_xyz(lute, x_ref)

    spine_start = _spine_point_xyz(lute, x_start)
    spine_end = _spine_point_xyz(lute, x_end)
    spine_vector = spine_end - spine_start
    if np.linalg.norm(spine_vector) <= _EPS:
        raise ValueError("Spine vector collapsed after applying end blocks.")

    rib_planes: List[RibPlane] = []
    ribs: List[List[np.ndarray]] = [[] for _ in range(rib_count)]

    for theta in thetas:
        y_ref = cy_ref + r_ref * np.cos(theta)
        z_ref = cz_ref + r_ref * np.sin(theta)
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
        cy, cz = map(float, section.center)
        r = float(section.radius)
        spine_y = float(_spine_point_at_X(lute, x))
        base_yz = np.array([spine_y, 0.0], dtype=float)

        if r <= _EPS:
            for trace in ribs:
                trace.append(np.array([x, base_yz[0], base_yz[1]], dtype=float))
            continue

        for plane, trace in zip(rib_planes, ribs):
            dy, dz = float(plane.direction[1]), float(plane.direction[2])
            coeff_a = dy * dy + dz * dz
            coeff_b = 2.0 * (dy * (base_yz[0] - cy) + dz * (base_yz[1] - cz))
            coeff_c = (base_yz[0] - cy) ** 2 + (base_yz[1] - cz) ** 2 - r * r
            discriminant = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c

            if discriminant < -1e-7:
                raise RuntimeError(f"Planar rib plane misses section circle at X={x:.6f}")

            discriminant = max(0.0, discriminant)
            sqrt_disc = np.sqrt(discriminant)

            s_candidates = [
                (-coeff_b - sqrt_disc) / (2.0 * coeff_a),
                (-coeff_b + sqrt_disc) / (2.0 * coeff_a),
            ]
            s_valid = [s for s in s_candidates if s >= -1e-9]
            if not s_valid:
                s = max(s_candidates, key=lambda val: val)
                if s < 0.0:
                    s = 0.0
            else:
                s = max(s_valid)
                if s < 0.0:
                    s = 0.0

            y = base_yz[0] + s * dy
            z = base_yz[1] + s * dz
            trace.append(np.array([x, y, z], dtype=float))

    return [np.asarray(trace, dtype=float) for trace in ribs]
