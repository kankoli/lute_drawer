"""Ribbon-based bowl construction helpers (stage 1 scaffolding)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .bowl_from_soundboard import _spine_point_at_X

_EPS = 1e-9


@dataclass(frozen=True)
class Plane:
    point: np.ndarray
    normal: np.ndarray


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
        xs = np.interp(s_arr, self.s_values, self.centerline[:, 0])
        ys = np.interp(s_arr, self.s_values, self.centerline[:, 1])
        zs = np.interp(s_arr, self.s_values, self.centerline[:, 2])
        return np.column_stack([xs, ys, zs])

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
        ref_s: float = 0.5,
        ref_offset: float | None = None,
        sample_count: int = 200,
        max_angle_deg: float = 89.0,
        angle_samples: int = 361,
    ) -> tuple[Plane, Plane]:
        line_dir = np.asarray(bottom_point, dtype=float) - np.asarray(top_point, dtype=float)
        line_norm = np.linalg.norm(line_dir)
        if line_norm <= _EPS:
            raise ValueError("Terminal points collapse; cannot define terminal line.")
        line_dir = line_dir / line_norm

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
        s_samples = np.linspace(0.0, 1.0, max(2, int(sample_count)))
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

    points: list[np.ndarray] = []
    last_point: np.ndarray | None = None
    if start_point is None:
        start_point = getattr(lute, "form_top", None)
    if start_point is not None:
        try:
            last_point = np.array([float(start_point.x), float(start_point.y)], dtype=float)
        except AttributeError:
            last_point = np.asarray(start_point, dtype=float)
        last_point = normalize_outline_points(lute, last_point[None, :])[0]
    for arc in arcs:
        sampled = np.asarray(arc.sample_points(samples_per_arc), dtype=float)
        if sampled.size == 0:
            continue
        sampled = normalize_outline_points(lute, sampled)
        if last_point is not None:
            dist_start = float(np.linalg.norm(sampled[0] - last_point))
            dist_end = float(np.linalg.norm(sampled[-1] - last_point))
            if dist_end < dist_start:
                sampled = sampled[::-1]
        if last_point is not None and np.linalg.norm(sampled[0] - last_point) < 1e-7:
            sampled = sampled[1:]
        if sampled.size == 0:
            continue
        points.append(sampled)
        last_point = sampled[-1]

    if not points:
        raise ValueError("Unable to sample a soundboard outline.")

    return np.vstack(points)


def edge_curve(surface: RibbonSurface, plane: Plane, *, sample_count: int = 200) -> np.ndarray:
    s_samples = np.linspace(0.0, 1.0, max(2, int(sample_count)))
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
    s_val = float(s)
    xs = np.interp(s_val, s_values, centerline[:, 0])
    ys = np.interp(s_val, s_values, centerline[:, 1])
    zs = np.interp(s_val, s_values, centerline[:, 2])
    return np.array([xs, ys, zs], dtype=float)


def _cumulative_s(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    total = float(seg.sum())
    if total <= _EPS:
        raise ValueError("Centerline length is too small.")
    s_vals = np.concatenate([[0.0], np.cumsum(seg)])
    return s_vals / total


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


__all__ = [
    "Plane",
    "RibbonSurface",
    "sample_outline_arcs",
    "edge_curve",
    "ribbon_surface_grid",
    "default_terminal_points",
    "normalize_outline_points",
]
