from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

EPS = 1e-9


@dataclass
class RibSidePlane:
    """Planar guide positioned along one side of a rib surface."""

    rib_index: int
    side: str
    point: np.ndarray
    normal: np.ndarray
    corners: np.ndarray
    long_direction: np.ndarray
    height_direction: np.ndarray


@dataclass
class RibPlaneDeviation:
    rib_index: int
    long_deltas: np.ndarray
    height_deltas: np.ndarray


def build_rib_surfaces(
    *,
    rib_outlines: Sequence[np.ndarray],
    rib_index: int | Sequence[int] | None = None,
) -> list[tuple[int, list[np.ndarray]]]:
    """Return rib surfaces from precomputed planar rib outlines."""

    if len(rib_outlines) < 2:
        raise ValueError("At least two rib outlines are required to build surfaces.")

    if rib_index is None:
        indices = list(range(len(rib_outlines) - 1))
    elif isinstance(rib_index, Iterable) and not isinstance(rib_index, (str, bytes)):
        indices = [int(idx) - 1 for idx in rib_index]
    else:
        indices = [int(rib_index) - 1]

    surfaces: list[tuple[int, list[np.ndarray]]] = []

    for idx in indices:
        if idx < 0 or idx >= len(rib_outlines) - 1:
            raise ValueError(f"Rib index {idx + 1} is outside available range 1..{len(rib_outlines) - 1}")
        rib_a = np.asarray(rib_outlines[idx], dtype=float)
        rib_b = np.asarray(rib_outlines[idx + 1], dtype=float)
        _validate_planar_pair(rib_a, rib_b)
        quads = _planar_quads_between_ribs(rib_a, rib_b)
        surfaces.append((idx + 1, quads))

    return surfaces


def _validate_planar_pair(rib_a: np.ndarray, rib_b: np.ndarray) -> None:
    if rib_a.shape != rib_b.shape or rib_a.ndim != 2 or rib_a.shape[1] != 3:
        raise ValueError("Planar rib outlines must share shape (N,3)")
    if rib_a.shape[0] < 2:
        raise ValueError("Planar rib outlines require at least two samples per rib.")

    xs_a = rib_a[:, 0]
    xs_b = rib_b[:, 0]
    if not np.allclose(xs_a, xs_b, atol=1e-8):
        raise ValueError("Planar rib outlines must share the same X samples.")


def _planar_quads_between_ribs(rib_a: np.ndarray, rib_b: np.ndarray) -> List[np.ndarray]:
    quads: List[np.ndarray] = []
    for p0, p1, q0, q1 in zip(rib_a[:-1], rib_a[1:], rib_b[:-1], rib_b[1:], strict=False):
        quad = np.vstack([p0, p1, q1, q0])
        quads.append(quad)
    return quads


def _normalize_quads(outline1: np.ndarray, outline2: np.ndarray, quads: list[np.ndarray]):
    outline1 = np.asarray(outline1, float)
    outline2 = np.asarray(outline2, float)
    if outline1.shape != outline2.shape or outline1.ndim != 2 or outline1.shape[1] != 3:
        raise ValueError("_normalize_quads expects two (N,3) outlines with same shape")

    connectors = outline2 - outline1
    idx = int(np.argmax(np.linalg.norm(connectors, axis=1)))
    across = connectors[idx] / (np.linalg.norm(connectors[idx]) + EPS)

    if idx < len(outline1) - 1:
        along = outline1[idx + 1] - outline1[idx]
    else:
        along = outline1[idx] - outline1[idx - 1]
    along /= (np.linalg.norm(along) + EPS)

    z_local = np.cross(along, across)
    nz = np.linalg.norm(z_local)
    if nz < EPS:
        return quads
    z_local /= nz
    y_local = across
    x_local = np.cross(y_local, z_local)
    x_local /= (np.linalg.norm(x_local) + EPS)

    R = np.vstack([x_local, y_local, z_local]).T
    R_inv = R.T

    def to_local(A):
        return (R_inv @ A.T).T

    q_local = [to_local(q) for q in quads]
    mid = 0.5 * (to_local(outline1)[idx] + to_local(outline2)[idx])
    return [q - mid for q in q_local]


def find_rib_side_planes(
    *,
    rib_outlines: Sequence[np.ndarray],
    rib_index: int,
    plane_gap_mm: float | None = 60.0,
    unit_scale: float = 1.0,
) -> tuple[RibSidePlane, RibSidePlane]:
    """Return two parallel planes hugging each side of a rib surface.

    When `plane_gap_mm` is provided, the planes are offset symmetrically from the
    rib centerline so that their separation matches the requested millimeter
    distance after accounting for the lute's unit-to-mm scale."""

    if rib_index <= 0 or rib_index >= len(rib_outlines):
        raise ValueError(f"rib_index must be within 1..{len(rib_outlines) - 1}")
    if unit_scale <= 0.0:
        raise ValueError("unit_scale must be positive")
    if plane_gap_mm is not None and plane_gap_mm <= 0.0:
        raise ValueError("plane_gap_mm must be positive when provided")

    outline_a = np.asarray(rib_outlines[rib_index - 1], dtype=float)
    outline_b = np.asarray(rib_outlines[rib_index], dtype=float)
    _validate_planar_pair(outline_a, outline_b)

    gap_units = None if plane_gap_mm is None else plane_gap_mm / unit_scale
    return _derive_side_planes(outline_a, outline_b, rib_index, gap_units)


def _derive_side_planes(
    outline_a: np.ndarray,
    outline_b: np.ndarray,
    rib_index: int,
    gap_units: float | None,
) -> tuple[RibSidePlane, RibSidePlane]:
    connectors = outline_b - outline_a
    distances = np.linalg.norm(connectors, axis=1)
    idx = int(np.argmax(distances))
    width_vec = connectors[idx]
    width_norm = np.linalg.norm(width_vec)
    if width_norm < EPS:
        raise ValueError("Unable to derive side planes from identical rib outlines")
    across_dir = width_vec / width_norm

    midline = 0.5 * (outline_a + outline_b)
    if len(midline) >= 2:
        long_vec = midline[-1] - midline[0]
    else:
        long_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    long_vec = _ensure_non_parallel(long_vec, across_dir)
    long_dir = long_vec / (np.linalg.norm(long_vec) + EPS)

    height_dir = np.cross(across_dir, long_dir)
    if np.linalg.norm(height_dir) < EPS:
        fallback = np.array([0.0, 0.0, 1.0], dtype=float)
        height_dir = np.cross(across_dir, fallback)
    if np.linalg.norm(height_dir) < EPS:
        fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        height_dir = np.cross(across_dir, fallback)
    height_dir = height_dir / (np.linalg.norm(height_dir) + EPS)

    all_points = np.vstack([outline_a, outline_b])
    long_coords = all_points @ long_dir
    height_coords = all_points @ height_dir
    long_min, long_max = float(long_coords.min()), float(long_coords.max())
    height_min, height_max = float(height_coords.min()), float(height_coords.max())
    if long_max - long_min < EPS:
        pad = 0.5
        long_min -= pad
        long_max += pad
    if height_max - height_min < EPS:
        pad = 0.5
        height_min -= pad
        height_max += pad

    long_center = 0.5 * (long_min + long_max)
    height_center = 0.5 * (height_min + height_max)
    long_half = 0.5 * (long_max - long_min)
    height_half = 0.5 * (height_max - height_min)
    # Enlarge the panel rectangle slightly so tips don’t run outside the plane plot.
    long_half *= 1.05
    height_half *= 1.05

    def _center_point(point: np.ndarray) -> np.ndarray:
        centered = point.astype(float).copy()
        centered += long_dir * (long_center - np.dot(centered, long_dir))
        centered += height_dir * (height_center - np.dot(centered, height_dir))
        return centered

    def _corners(center: np.ndarray) -> np.ndarray:
        offsets = [
            -long_half * long_dir - height_half * height_dir,
            long_half * long_dir - height_half * height_dir,
            long_half * long_dir + height_half * height_dir,
            -long_half * long_dir + height_half * height_dir,
        ]
        return np.array([center + offset for offset in offsets], dtype=float)

    mid_point = 0.5 * (outline_a[idx] + outline_b[idx])
    span = width_norm if gap_units is None else float(gap_units)
    if span <= EPS:
        span = width_norm
    half_span = 0.5 * span

    center_a = _center_point(mid_point - across_dir * half_span)
    center_b = _center_point(mid_point + across_dir * half_span)

    plane_a = RibSidePlane(
        rib_index=rib_index,
        side="negative",
        point=center_a,
        normal=across_dir,
        corners=_corners(center_a),
        long_direction=long_dir,
        height_direction=height_dir,
    )
    plane_b = RibSidePlane(
        rib_index=rib_index,
        side="positive",
        point=center_b,
        normal=-across_dir,
        corners=_corners(center_b),
        long_direction=long_dir,
        height_direction=height_dir,
    )
    return plane_a, plane_b


def _ensure_non_parallel(vec: np.ndarray, other: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    if np.linalg.norm(vec) < EPS:
        vec = np.array([1.0, 0.0, 0.0], dtype=float)
    projection = other * (np.dot(vec, other))
    adjusted = vec - projection
    if np.linalg.norm(adjusted) < EPS:
        adjusted = np.cross(other, np.array([0.0, 0.0, 1.0], dtype=float))
    if np.linalg.norm(adjusted) < EPS:
        adjusted = np.cross(other, np.array([0.0, 1.0, 0.0], dtype=float))
    return adjusted


def _outline_plane_distances(outline: np.ndarray, plane: RibSidePlane) -> np.ndarray:
    normal = plane.normal / (np.linalg.norm(plane.normal) + EPS)
    return np.abs((outline - plane.point) @ normal)


def project_points_to_plane(points: np.ndarray, plane: RibSidePlane) -> np.ndarray:
    normal = plane.normal / (np.linalg.norm(plane.normal) + EPS)
    offsets = (points - plane.point) @ normal
    return points - np.outer(offsets, normal)


def line_plane_intersection(p0: np.ndarray, p1: np.ndarray, plane: RibSidePlane) -> np.ndarray | None:
    """Return the intersection of the infinite line p0->p1 with plane, or None if parallel."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    n = plane.normal
    denom = float(np.dot(n, p1 - p0))
    if abs(denom) < EPS:
        return None
    t = float(np.dot(n, plane.point - p0) / denom)
    return p0 + t * (p1 - p0)


def all_rib_surfaces_convex(
    *,
    rib_outlines: Sequence[np.ndarray],
    plane_gap_mm: float | None = 60.0,
    unit_scale: float = 1.0,
    atol: float = 1e-6,
) -> bool:
    """Return True when every rib surface is convex relative to its side planes."""

    if len(rib_outlines) < 2:
        raise ValueError("At least two rib outlines required to evaluate convexity")
    if unit_scale <= 0.0:
        raise ValueError("unit_scale must be positive")
    if plane_gap_mm is not None and plane_gap_mm <= 0.0:
        raise ValueError("plane_gap_mm must be positive when provided")

    non_convex: list[int] = []

    for rib_idx in range(1, len(rib_outlines)):
        outline_a = np.asarray(rib_outlines[rib_idx - 1], dtype=float)
        outline_b = np.asarray(rib_outlines[rib_idx], dtype=float)
        _validate_planar_pair(outline_a, outline_b)

        connectors = outline_b - outline_a
        distances = np.linalg.norm(connectors, axis=1)
        max_idx = int(np.argmax(distances))

        planes = find_rib_side_planes(
            rib_outlines=rib_outlines,
            rib_index=rib_idx,
            plane_gap_mm=plane_gap_mm,
            unit_scale=unit_scale,
        )
        plane_map = {plane.side: plane for plane in planes}
        pairs: list[tuple[np.ndarray, RibSidePlane | None]] = [
            (outline_a, plane_map.get("negative")),
            (outline_b, plane_map.get("positive")),
        ]

        rib_convex = True
        for outline, plane in pairs:
            if plane is None:
                continue
            dists = _outline_plane_distances(outline, plane)
            center_dist = float(dists[max_idx])
            end_dist = float(min(dists[0], dists[-1]))
            if center_dist + atol >= end_dist:
                rib_convex = False
                break

        if not rib_convex:
            non_convex.append(rib_idx)

    if non_convex:
        ribs = ", ".join(str(idx) for idx in non_convex)
        print(f"Non-convex ribs: {ribs}")
        return False
    return True


def measure_rib_plane_deviation(
    *,
    rib_outlines: Sequence[np.ndarray],
    rib_index: int,
    plane_gap_mm: float | None = 60.0,
    unit_scale: float = 1.0,
) -> RibPlaneDeviation:
    """Return per-sample deviation between rib outlines projected onto the side planes."""

    if rib_index <= 0 or rib_index >= len(rib_outlines):
        raise ValueError(f"rib_index must be within 1..{len(rib_outlines) - 1}")

    outline_a = np.asarray(rib_outlines[rib_index - 1], dtype=float)
    outline_b = np.asarray(rib_outlines[rib_index], dtype=float)
    _validate_planar_pair(outline_a, outline_b)

    planes = find_rib_side_planes(
        rib_outlines=rib_outlines,
        rib_index=rib_index,
        plane_gap_mm=plane_gap_mm,
        unit_scale=unit_scale,
    )
    plane_map = {plane.side: plane for plane in planes}
    neg_plane = plane_map.get("negative")
    pos_plane = plane_map.get("positive")
    if neg_plane is None or pos_plane is None:
        raise RuntimeError("Failed to generate both side planes for deviation measurement")

    projected_neg = project_points_to_plane(outline_a, neg_plane)
    projected_pos = project_points_to_plane(outline_b, pos_plane)

    long_dir = neg_plane.long_direction
    height_dir = neg_plane.height_direction

    long_neg = projected_neg @ long_dir
    long_pos = projected_pos @ long_dir
    height_neg = projected_neg @ height_dir
    height_pos = projected_pos @ height_dir

    long_deltas = np.abs(long_pos - long_neg)
    height_deltas = np.abs(height_pos - height_neg)

    return RibPlaneDeviation(
        rib_index=rib_index,
        long_deltas=long_deltas,
        height_deltas=height_deltas,
    )


def compute_rib_blank_width(
    *,
    rib_outlines: Sequence[np.ndarray],
    rib_index: int,
) -> float:
    """Return the minimum flat blank width required for a rib.

    The blank is assumed straight and allowed to bend along its length (wide face)
    but not stretch across its width. The width direction is anchored to the
    widest connector between outlines and orthogonalized against the rib's
    overall long axis to mimic a flat board."""

    if rib_index <= 0 or rib_index >= len(rib_outlines):
        raise ValueError(f"rib_index must be within 1..{len(rib_outlines) - 1}")

    outline_a = np.asarray(rib_outlines[rib_index - 1], dtype=float)
    outline_b = np.asarray(rib_outlines[rib_index], dtype=float)
    _validate_planar_pair(outline_a, outline_b)

    connectors = outline_b - outline_a
    distances = np.linalg.norm(connectors, axis=1)
    idx = int(np.argmax(distances))
    width_vec = connectors[idx]
    width_norm = np.linalg.norm(width_vec)
    if width_norm < EPS:
        raise ValueError("Unable to derive blank width from identical rib outlines")
    across_dir = width_vec / width_norm

    midline = 0.5 * (outline_a + outline_b)
    if len(midline) >= 2:
        long_vec = midline[-1] - midline[0]
    else:
        long_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    long_vec = _ensure_non_parallel(long_vec, across_dir)
    long_dir = long_vec / (np.linalg.norm(long_vec) + EPS)

    width_dir = across_dir - np.dot(across_dir, long_dir) * long_dir
    if np.linalg.norm(width_dir) < EPS:
        width_dir = np.cross(long_dir, np.array([0.0, 0.0, 1.0], dtype=float))
    if np.linalg.norm(width_dir) < EPS:
        width_dir = np.cross(long_dir, np.array([0.0, 1.0, 0.0], dtype=float))
    width_dir = width_dir / (np.linalg.norm(width_dir) + EPS)

    all_points = np.vstack([outline_a, outline_b])
    width_coords = all_points @ width_dir
    return float(width_coords.max() - width_coords.min())


__all__ = [
    "build_rib_surfaces",
    "find_rib_side_planes",
    "all_rib_surfaces_convex",
    "measure_rib_plane_deviation",
    "project_points_to_plane",
    "compute_rib_blank_width",
    "RibSidePlane",
    "RibPlaneDeviation",
]
