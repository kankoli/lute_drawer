"""Ribbon-based bowl construction helpers (stage 1 scaffolding)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .bowl_from_soundboard import _spine_point_at_X
from utils.geo_dsl import sample_arcs

_EPS = 1e-9


@dataclass(frozen=True)
class Plane:
    point: np.ndarray
    normal: np.ndarray


@dataclass
class ChainedTerminalStrategy:
    """Stateful helper to choose terminal s values for chained ribs.

    Constraints enforced:
    - Symmetry: index sign is ignored (abs index).
    - Monotonic offsets: offset magnitude cannot shrink and sign cannot flip.
    - Ordering: s_top <= s_bottom (clamped if needed).
    Call in non-decreasing abs index order: 0, 1, 2, ...
    """

    top_offset_fn: Callable[[int], float]
    bottom_offset_fn: Callable[[int], float]
    base_top_s: float = 0.0
    base_bottom_s: float = 1.0
    min_s: float = 0.0
    max_s: float = 1.0
    _last_abs_index: int = field(init=False, default=-1)
    _last_top_offset: float = field(init=False, default=0.0)
    _last_bottom_offset: float = field(init=False, default=0.0)

    def s_for_index(self, index: int) -> tuple[float, float]:
        abs_idx = abs(int(index))
        if abs_idx < self._last_abs_index:
            raise ValueError("ChainedTerminalStrategy expects non-decreasing abs index order.")

        top_offset = float(self.top_offset_fn(abs_idx))
        bottom_offset = float(self.bottom_offset_fn(abs_idx))
        if self._last_abs_index >= 0 and abs_idx > 0:
            top_offset = _enforce_monotonic_offset(self._last_top_offset, top_offset)
            bottom_offset = _enforce_monotonic_offset(self._last_bottom_offset, bottom_offset)

        top_s = _clamp(self.base_top_s + top_offset, self.min_s, self.max_s)
        bottom_s = _clamp(self.base_bottom_s + bottom_offset, self.min_s, self.max_s)
        if top_s > bottom_s:
            top_s = bottom_s

        self._last_abs_index = abs_idx
        self._last_top_offset = top_s - self.base_top_s
        self._last_bottom_offset = bottom_s - self.base_bottom_s
        return top_s, bottom_s

    def reset(self) -> None:
        self._last_abs_index = -1
        self._last_top_offset = 0.0
        self._last_bottom_offset = 0.0


@dataclass(frozen=True)
class CenterRib:
    top_point: np.ndarray
    bottom_point: np.ndarray
    negative_plane: Plane
    positive_plane: Plane


@dataclass(frozen=True)
class RegularRib:
    index: int
    top_point: np.ndarray
    bottom_point: np.ndarray
    inner_plane: Plane
    outer_plane: Plane
    inner_source: str
    mirrors: Sequence[Plane]


@dataclass(frozen=True)
class RibbonSurface:
    """Developable ribbon surface defined by a coplanar centerline.

    Ribbon coordinates (s, t):
    - s in [0, 1] runs along the centerline (s=0 top, s=1 bottom), arc-length normalized.
    - t in [-W/2, W/2] runs across the blank width; t=0 is the centerline.
    - The surface is R(s, t) = C(s) + t * n, where n is the constant unit normal
      of the centerline plane (the width axis).
    """

    centerline: np.ndarray
    s_values: np.ndarray
    width_axis: np.ndarray
    rotation: np.ndarray
    yz_offset: np.ndarray
    z_sign: float

    def point_at(self, s: float, t: float = 0.0) -> np.ndarray:
        base = _interp_centerline(self.centerline, self.s_values, s)
        return base + float(t) * self.width_axis

    def centerline_points(self, s_samples: Iterable[float]) -> np.ndarray:
        s_arr = np.asarray(list(s_samples), dtype=float)
        return _interp_centerline_points(self.centerline, self.s_values, s_arr)

    def to_oriented(self, points: np.ndarray) -> np.ndarray:
        """Map points into the oriented ribbon frame (terminal line on y=z=0)."""
        return _apply_terminal_line_transform(points, self.rotation, self.yz_offset, self.z_sign)

    def from_oriented(self, points: np.ndarray) -> np.ndarray:
        """Map points from the oriented ribbon frame back to the source frame."""
        return _undo_terminal_line_transform(points, self.rotation, self.yz_offset, self.z_sign)

    @classmethod
    def from_outline(cls, lute, *, samples_per_arc: int = 200) -> "RibbonSurface":
        outline_pts = sample_outline_arcs(lute, samples_per_arc=samples_per_arc)
        centerline = np.column_stack(
            [outline_pts[:, 0], np.zeros(outline_pts.shape[0], dtype=float), outline_pts[:, 1]]
        )
        centerline, rotation, yz_offset, z_sign = _orient_centerline(centerline)
        s_values = _cumulative_s(centerline)
        width_axis = _plane_normal(centerline)
        return cls(
            centerline=centerline,
            s_values=s_values,
            width_axis=width_axis,
            rotation=rotation,
            yz_offset=yz_offset,
            z_sign=z_sign,
        )

    def edge_planes_for_terminal_line(
        self,
        top_point: np.ndarray,
        bottom_point: np.ndarray,
        width: float,
        *,
        ref_s: float | None = None,
        ref_offset: float | None = None,
        s_min: float | None = None,
        s_max: float | None = None,
        sample_count: int = 200,
        max_angle_deg: float = 89.0,
        angle_samples: int = 361,
    ) -> tuple[Plane, Plane]:
        line_dir = np.asarray(bottom_point, dtype=float) - np.asarray(top_point, dtype=float)
        line_norm = np.linalg.norm(line_dir)
        if line_norm <= _EPS:
            raise ValueError("Terminal points collapse; cannot define terminal line.")
        line_dir = line_dir / line_norm

        s_low = 0.0 if s_min is None else float(s_min)
        s_high = 1.0 if s_max is None else float(s_max)
        if s_low > s_high:
            s_low, s_high = s_high, s_low
        if s_high - s_low <= _EPS:
            raise ValueError("Terminal s range is degenerate.")

        if ref_s is None:
            ref_s = 0.5 * (s_low + s_high)
        ref_s = max(s_low, min(s_high, float(ref_s)))
        ref_offset = 0.0 if ref_offset is None else float(ref_offset)
        ref_point = self.point_at(ref_s, ref_offset)
        try:
            base_normal = _plane_normal_from_line_and_point(top_point, line_dir, ref_point)
        except ValueError:
            if abs(ref_offset) <= _EPS:
                ref_point = self.point_at(ref_s, float(width) * 0.5)
                base_normal = _plane_normal_from_line_and_point(top_point, line_dir, ref_point)
            else:
                raise

        base_normal = base_normal / (np.linalg.norm(base_normal) + _EPS)
        basis_perp = np.cross(line_dir, base_normal)
        basis_norm = np.linalg.norm(basis_perp)
        if basis_norm <= _EPS:
            raise ValueError("Unable to derive a stable rotation basis for the edge planes.")
        basis_perp = basis_perp / basis_norm

        max_angle = np.deg2rad(max_angle_deg)
        angles = np.linspace(-max_angle, max_angle, max(3, int(angle_samples)))
        s_samples = np.linspace(s_low, s_high, max(2, int(sample_count)))
        base = self.centerline_points(s_samples)
        half_width = float(width) * 0.5
        tol = 1e-7

        best_pos: tuple[float, np.ndarray] | None = None
        best_neg: tuple[float, np.ndarray] | None = None

        for angle in angles:
            normal = (np.cos(angle) * base_normal) + (np.sin(angle) * basis_perp)
            denom = float(np.dot(normal, self.width_axis))
            if abs(denom) <= _EPS:
                continue
            t_vals = -((base - top_point) @ normal) / denom
            t_min = float(t_vals.min())
            t_max = float(t_vals.max())

            if t_min >= -tol and t_max <= half_width + tol:
                if best_pos is None or t_max > best_pos[0]:
                    best_pos = (t_max, normal)
            if t_max <= tol and t_min >= -half_width - tol:
                if best_neg is None or t_min < best_neg[0]:
                    best_neg = (t_min, normal)

        if best_pos is None or best_neg is None:
            raise ValueError("Cannot derive edge planes within the ribbon blank.")

        return (
            Plane(point=np.asarray(top_point, float), normal=best_pos[1]),
            Plane(point=np.asarray(top_point, float), normal=best_neg[1]),
        )


def sample_outline_arcs(
    lute,
    samples_per_arc: int = 200,
    *,
    start_point: np.ndarray | None = None,
) -> np.ndarray:
    arcs = list(getattr(lute, "final_arcs", []))
    if not arcs:
        raise ValueError("Soundboard outline arcs are missing.")
    if samples_per_arc < 2:
        raise ValueError("samples_per_arc must be at least 2.")

    if start_point is None:
        start_point = getattr(lute, "form_top", None)
    points = sample_arcs(arcs, samples_per_arc=samples_per_arc, start_point=start_point)
    return normalize_outline_points(lute, points)


def edge_curve(
    surface: RibbonSurface,
    plane: Plane,
    *,
    sample_count: int = 200,
    s_min: float | None = None,
    s_max: float | None = None,
) -> np.ndarray:
    s_low = 0.0 if s_min is None else float(s_min)
    s_high = 1.0 if s_max is None else float(s_max)
    if s_low > s_high:
        s_low, s_high = s_high, s_low
    s_samples = np.linspace(s_low, s_high, max(2, int(sample_count)))
    base = surface.centerline_points(s_samples)
    denom = float(np.dot(plane.normal, surface.width_axis))
    if abs(denom) <= _EPS:
        raise ValueError("Plane is parallel to the ribbon width direction.")
    t_values = -((base - plane.point) @ plane.normal) / denom
    return base + t_values[:, None] * surface.width_axis


def ribbon_surface_grid(
    surface: RibbonSurface,
    width: float,
    *,
    s_samples: int = 200,
    t_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a grid for plotting the blank ribbon surface."""
    s_count = max(2, int(s_samples))
    t_count = max(2, int(t_samples))
    s_vals = np.linspace(0.0, 1.0, s_count)
    t_vals = np.linspace(-float(width) * 0.5, float(width) * 0.5, t_count)
    base = surface.centerline_points(s_vals)
    grid = base[None, :, :] + t_vals[:, None, None] * surface.width_axis
    return grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]


def default_terminal_points(surface: RibbonSurface) -> tuple[np.ndarray, np.ndarray]:
    """Return terminal points at the midpoints of each rib-blank end."""
    top_point = surface.point_at(0.0, 0.0)
    bottom_point = surface.point_at(1.0, 0.0)
    return top_point, bottom_point


def _interp_centerline(centerline: np.ndarray, s_values: np.ndarray, s: float) -> np.ndarray:
    return _interp_centerline_points(centerline, s_values, np.array([s], dtype=float))[0]


def _cumulative_s(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    total = float(seg.sum())
    if total <= _EPS:
        raise ValueError("Centerline length is too small.")
    s_vals = np.concatenate([[0.0], np.cumsum(seg)])
    return s_vals / total


def _centerline_length(points: np.ndarray) -> float:
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _interp_centerline_points(
    centerline: np.ndarray, s_values: np.ndarray, s_arr: np.ndarray
) -> np.ndarray:
    s_vals = np.asarray(s_values, dtype=float)
    queries = np.asarray(s_arr, dtype=float)
    total_len = _centerline_length(centerline)
    start = centerline[0]
    end = centerline[-1]
    start_dir = centerline[1] - centerline[0]
    end_dir = centerline[-1] - centerline[-2]
    start_dir /= np.linalg.norm(start_dir) + _EPS
    end_dir /= np.linalg.norm(end_dir) + _EPS

    result = np.empty((queries.size, 3), dtype=float)
    mid_mask = (queries >= 0.0) & (queries <= 1.0)
    low_mask = queries < 0.0
    high_mask = queries > 1.0

    if np.any(mid_mask):
        q = queries[mid_mask]
        xs = np.interp(q, s_vals, centerline[:, 0])
        ys = np.interp(q, s_vals, centerline[:, 1])
        zs = np.interp(q, s_vals, centerline[:, 2])
        result[mid_mask] = np.column_stack([xs, ys, zs])
    if np.any(low_mask):
        offsets = (queries[low_mask] * total_len)[:, None]
        result[low_mask] = start + offsets * start_dir
    if np.any(high_mask):
        offsets = ((queries[high_mask] - 1.0) * total_len)[:, None]
        result[high_mask] = end + offsets * end_dir

    return result


def _plane_normal(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError("Need at least three points to define a plane.")
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm <= _EPS:
        raise ValueError("Unable to derive a stable plane normal.")
    return normal / norm


def _plane_normal_from_line_and_point(
    line_point: np.ndarray, line_dir: np.ndarray, ref_point: np.ndarray
) -> np.ndarray:
    v = np.asarray(ref_point, dtype=float) - np.asarray(line_point, dtype=float)
    normal = np.cross(line_dir, v)
    norm = np.linalg.norm(normal)
    if norm <= _EPS:
        raise ValueError("Reference point is colinear with the terminal line.")
    return normal / norm


def normalize_outline_points(lute, points: np.ndarray) -> np.ndarray:
    """Normalize outline points so the spine lies on Y=0 in the soundboard plane."""
    pts = np.asarray(points, dtype=float)
    xs = pts[:, 0]
    spine_ys = np.array([_spine_point_at_X(lute, float(x)) for x in xs], dtype=float)
    normalized = pts.copy()
    normalized[:, 1] = normalized[:, 1] - spine_ys
    return normalized


def _rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    vec_from = np.asarray(vec_from, dtype=float)
    vec_to = np.asarray(vec_to, dtype=float)
    n_from = np.linalg.norm(vec_from)
    n_to = np.linalg.norm(vec_to)
    if n_from <= 1e-12 or n_to <= 1e-12:
        return np.eye(3)
    a = vec_from / n_from
    b = vec_to / n_to
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        v = np.cross(a, axis)
        v /= np.linalg.norm(v) + 1e-12
        vx = np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=float,
        )
        return np.eye(3) + 2.0 * vx @ vx
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def _apply_terminal_line_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    yz_offset: np.ndarray,
    z_sign: float,
) -> np.ndarray:
    """Rotate/translate/flip points into the oriented ribbon frame.

    Steps:
    1) Rotate so the centerline end-to-end line aligns with +X.
    2) Translate so the start point has y=0, z=0.
    3) Flip z so the ribbon sits in +Z.
    """
    rotated = points @ rotation.T
    rotated[:, 1] -= yz_offset[1]
    rotated[:, 2] -= yz_offset[2]
    rotated[:, 2] *= float(z_sign)
    return rotated


def _undo_terminal_line_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    yz_offset: np.ndarray,
    z_sign: float,
) -> np.ndarray:
    """Inverse of _apply_terminal_line_transform."""
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 2] *= float(z_sign)
    pts[:, 1] += yz_offset[1]
    pts[:, 2] += yz_offset[2]
    return pts @ rotation


def _orient_centerline(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if centerline.shape[0] < 2:
        raise ValueError("Centerline must have at least two points to orient.")
    start = centerline[0]
    end = centerline[-1]
    line_dir = end - start
    rotation = _rotation_matrix_from_vectors(line_dir, np.array([1.0, 0.0, 0.0], dtype=float))
    rotated = centerline @ rotation.T
    yz_offset = np.array([0.0, rotated[0, 1], rotated[0, 2]], dtype=float)
    rotated[:, 1] -= yz_offset[1]
    rotated[:, 2] -= yz_offset[2]
    z_sign = 1.0 if float(rotated[:, 2].mean()) >= 0.0 else -1.0
    rotated[:, 2] *= z_sign
    return rotated, rotation, yz_offset, z_sign


def _rotate_around_axis(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= _EPS:
        raise ValueError("Rotation axis is degenerate.")
    axis = axis / axis_norm
    vec = np.asarray(vec, dtype=float)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


def _max_separation(
    surface: RibbonSurface,
    normal_a: np.ndarray,
    normal_b: np.ndarray,
    plane_point: np.ndarray,
    sample_count: int,
) -> float:
    denom_a = float(np.dot(normal_a, surface.width_axis))
    denom_b = float(np.dot(normal_b, surface.width_axis))
    if abs(denom_a) <= _EPS or abs(denom_b) <= _EPS:
        return float("nan")

    s_samples = np.linspace(0.0, 1.0, max(2, int(sample_count)))
    base = surface.centerline_points(s_samples)
    t_a = -((base - plane_point) @ normal_a) / denom_a
    t_b = -((base - plane_point) @ normal_b) / denom_b
    return float(np.max(np.abs(t_b - t_a)))


def _enforce_monotonic_offset(prev_offset: float, new_offset: float) -> float:
    if abs(prev_offset) <= _EPS:
        prev_sign = 1.0 if new_offset >= 0.0 else -1.0
    else:
        prev_sign = 1.0 if prev_offset >= 0.0 else -1.0
    new_sign = 1.0 if new_offset >= 0.0 else -1.0
    if new_sign != prev_sign:
        new_offset = abs(new_offset) * prev_sign
    if abs(new_offset) < abs(prev_offset):
        new_offset = abs(prev_offset) * prev_sign
    return new_offset


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def reflect_points_across_plane(points: np.ndarray, plane: Plane) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    normal /= np.linalg.norm(normal) + _EPS
    delta = pts - np.asarray(plane.point, dtype=float)
    return pts - 2.0 * (delta @ normal)[:, None] * normal


def reflect_plane_across_plane(plane: Plane, mirror: Plane) -> Plane:
    normal = np.asarray(mirror.normal, dtype=float)
    normal /= np.linalg.norm(normal) + _EPS
    point_ref = reflect_points_across_plane(np.asarray(plane.point, dtype=float)[None, :], mirror)[0]
    plane_normal = np.asarray(plane.normal, dtype=float)
    normal_ref = plane_normal - 2.0 * np.dot(plane_normal, normal) * normal
    normal_ref /= np.linalg.norm(normal_ref) + _EPS
    return Plane(point=point_ref, normal=normal_ref)


def apply_reflections_to_points(points: np.ndarray, mirrors: Sequence[Plane]) -> np.ndarray:
    transformed = np.asarray(points, dtype=float)
    for mirror in mirrors:
        transformed = reflect_points_across_plane(transformed, mirror)
    return transformed


def _apply_reflections_to_plane(plane: Plane, mirrors: Sequence[Plane]) -> Plane:
    transformed = plane
    for mirror in mirrors:
        transformed = reflect_plane_across_plane(transformed, mirror)
    return transformed


def build_regular_rib_chain(
    surface: RibbonSurface,
    width: float,
    terminal_strategy: ChainedTerminalStrategy,
    *,
    pairs: int = 1,
    center_top_t: float = 0.0,
    center_bottom_t: float = 0.0,
) -> tuple[CenterRib, Sequence[RegularRib]]:
    """Build a symmetric chain of regular ribs by reflecting across edge planes.

    Adjacent ribs are built by reflection, so terminal s values must stay fixed
    (index offsets are not supported here yet).
    """
    terminal_strategy.reset()
    center_top_s, center_bottom_s = terminal_strategy.s_for_index(0)
    center_top = surface.point_at(center_top_s, center_top_t)
    center_bottom = surface.point_at(center_bottom_s, center_bottom_t)
    ref_s = 0.5 * (center_top_s + center_bottom_s)
    pos_plane, neg_plane = surface.edge_planes_for_terminal_line(
        center_top,
        center_bottom,
        width,
        ref_s=ref_s,
        s_min=center_top_s,
        s_max=center_bottom_s,
    )
    center_rib = CenterRib(
        top_point=center_top,
        bottom_point=center_bottom,
        negative_plane=neg_plane,
        positive_plane=pos_plane,
    )

    tol = 1e-6
    for idx in range(1, max(0, int(pairs)) + 1):
        top_s, bottom_s = terminal_strategy.s_for_index(idx)
        if abs(top_s - center_top_s) > tol or abs(bottom_s - center_bottom_s) > tol:
            raise ValueError("Chained ribs require constant terminal s values for now.")

    ribs: list[RegularRib] = []

    def _append_chain(start_plane: Plane, inner_source: str, indices: Sequence[int]) -> None:
        mirrors: list[Plane] = [start_plane]
        source_inner = inner_source
        for idx in indices:
            inner_plane = _apply_reflections_to_plane(
                pos_plane if source_inner == "pos" else neg_plane,
                mirrors,
            )
            outer_plane = _apply_reflections_to_plane(
                neg_plane if source_inner == "pos" else pos_plane,
                mirrors,
            )
            ribs.append(
                RegularRib(
                    index=idx,
                    top_point=center_top,
                    bottom_point=center_bottom,
                    inner_plane=inner_plane,
                    outer_plane=outer_plane,
                    inner_source=source_inner,
                    mirrors=tuple(mirrors),
                )
            )
            mirrors = mirrors + [outer_plane]
            source_inner = "neg" if source_inner == "pos" else "pos"

    if pairs > 0:
        _append_chain(pos_plane, "pos", range(1, max(0, int(pairs)) + 1))
        _append_chain(neg_plane, "neg", range(-1, -max(0, int(pairs)) - 1, -1))

    return center_rib, ribs


__all__ = [
    "ChainedTerminalStrategy",
    "CenterRib",
    "Plane",
    "RegularRib",
    "RibbonSurface",
    "build_regular_rib_chain",
    "reflect_plane_across_plane",
    "reflect_points_across_plane",
    "apply_reflections_to_points",
    "sample_outline_arcs",
    "edge_curve",
    "ribbon_surface_grid",
    "default_terminal_points",
    "normalize_outline_points",
]
