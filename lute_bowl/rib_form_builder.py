from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .bowl_from_soundboard import Section, build_bowl_for_lute
from plotting.bowl import plot_rib_surfaces
from .bowl_top_curves import MidCurve

EPS = 1e-9

def build_extended_rib_surfaces(
    lute,
    *,
    top_curve=MidCurve,
    n_ribs: int = 13,
    n_sections: int = 40,
    end_extension: int = 20,
    rib_index: int | None = None,
):
    """Return extended rib surfaces for the requested rib indices."""
    sections, rib_outlines = build_bowl_for_lute(
        lute,
        n_ribs=n_ribs,
        n_sections=n_sections,
        top_curve=top_curve,
    )

    neck_point = getattr(lute, "point_neck_joint", None)
    if neck_point is None:
        raise ValueError("Neck joint is required for rib molds but was not provided on the lute.")
    neck_x = float(neck_point.x)
    top_x = float(getattr(lute.form_top, "x", float("nan")))
    bottom_x = float(getattr(lute.form_bottom, "x", float("nan")))
    if not (np.isfinite(top_x) and np.isfinite(bottom_x) and top_x < neck_x < bottom_x):
        raise ValueError("Neck joint must lie between form_top and form_bottom to build rib molds.")

    if rib_index is None:
        raise ValueError("rib_index must be provided unless draw_all=True")
    target_indices = [rib_index - 1]

    surfaces = []
    outlines: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []
    for idx in target_indices:
        if idx < 0 or idx >= len(rib_outlines) - 1:
            raise ValueError("Rib index is outside range:", idx + 1)
        rib1 = np.asarray(rib_outlines[idx], dtype=float)
        rib2 = np.asarray(rib_outlines[idx + 1], dtype=float)
        quads, updated_pair = _rib_surface_extended(
            rib1,
            rib2,
            end_extension=end_extension,
            neck_joint_x=neck_x,
        )
        surfaces.append((idx + 1, quads))
        outlines.append((idx + 1, updated_pair))
    return sections, surfaces, outlines


def plot_lute_ribs(
    lute,
    *,
    top_curve=MidCurve,
    n_ribs: int = 13,
    n_sections: int = 40,
    rib: int = 7,
    end_extension: float = 10.0,
    title: str | None = None,
):
    _, surfaces, outlines = build_extended_rib_surfaces(
        lute,
        top_curve=top_curve,
        n_ribs=n_ribs,
        n_sections=n_sections,
        end_extension=end_extension,
        rib_index=rib,
    )
    plot_rib_surfaces(
        surfaces,
        outlines=outlines,
        title=title,
        lute_name=type(lute).__name__,
    )
    return surfaces


def _rib_surface_extended(
    outline1,
    outline2,
    *,
    end_extension: float,
    neck_joint_x: float,
    trend_window: int = 10,
):
    rib1 = np.array(outline1, float)
    rib2 = np.array(outline2, float)
    n = min(len(rib1), len(rib2))
    rib1, rib2 = rib1[:n], rib2[:n]

    rib1, rib2 = _extend_ribs_towards_top(rib1, rib2, neck_joint_x=neck_joint_x, trend_window=trend_window)
    n = min(len(rib1), len(rib2))
    rib1, rib2 = rib1[:n], rib2[:n]

    connectors = rib2 - rib1
    finite_mask = (
        np.all(np.isfinite(rib1), axis=1)
        & np.all(np.isfinite(rib2), axis=1)
        & np.all(np.isfinite(connectors), axis=1)
    )
    valid_indices = np.flatnonzero(finite_mask)
    if valid_indices.size < 2:
        return [], (rib1, rib2)

    connector_lengths = np.linalg.norm(connectors[valid_indices], axis=1)
    if connector_lengths.size == 0:
        return [], (rib1, rib2)
    max_span = float(np.nanmax(connector_lengths))
    if not np.isfinite(max_span) or max_span < EPS:
        return [], (rib1, rib2)

    width = 1.5 * max_span
    half_width = 0.5 * width

    def _safe_unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        vec = np.asarray(vec, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < EPS:
            if fallback is None:
                return np.array([1.0, 0.0, 0.0], dtype=float)
            return np.asarray(fallback, dtype=float)
        return vec / norm

    samples: list[tuple[float, np.ndarray, np.ndarray]] = []
    pending: list[int] = []
    prev_dir: np.ndarray | None = None

    def _append_sample(idx: int, direction: np.ndarray):
        center = 0.5 * (rib1[idx] + rib2[idx])
        samples.append((float(idx), center, direction))

    for idx in valid_indices:
        vec = connectors[idx]
        norm = float(np.linalg.norm(vec))
        if norm < EPS and prev_dir is None:
            pending.append(int(idx))
            continue
        if norm >= EPS:
            direction = vec / norm
            if prev_dir is not None and np.dot(direction, prev_dir) < 0.0:
                direction = -direction
            prev_dir = direction
        else:
            direction = prev_dir

        if direction is None:
            continue

        for pending_idx in pending:
            _append_sample(pending_idx, direction)
        pending.clear()

        _append_sample(int(idx), direction)

    if pending and prev_dir is not None:
        for pending_idx in pending:
            _append_sample(pending_idx, prev_dir)
        pending.clear()

    if len(samples) < 2:
        return [], (rib1, rib2)

    samples.sort(key=lambda entry: entry[0])

    directions = [
        _safe_unit(direction)
        for _, _, direction in samples
    ]

    centers = [center for _, center, _ in samples]
    left_edges = [center - half_width * direction for center, direction in zip(centers, directions)]
    right_edges = [center + half_width * direction for center, direction in zip(centers, directions)]

    quads: list[np.ndarray] = []
    for left0, right0, left1, right1 in zip(left_edges, right_edges, left_edges[1:], right_edges[1:]):
        quads.append(np.vstack([left0, left1, right1, right0]))

    return quads, (rib1, rib2)


def _extend_ribs_towards_top(
    rib1: np.ndarray,
    rib2: np.ndarray,
    *,
    neck_joint_x: float,
    trend_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    rib1 = np.array(rib1, dtype=float, copy=True)
    rib2 = np.array(rib2, dtype=float, copy=True)

    if rib1.shape != rib2.shape or rib1.ndim != 2 or rib1.shape[1] != 3:
        raise ValueError("_extend_ribs_towards_top expects rib outlines shaped (N,3)")

    xs1 = rib1[:, 0].astype(float)
    xs2 = rib2[:, 0].astype(float)
    if not np.allclose(xs1, xs2, atol=1e-9):
        xs = 0.5 * (xs1 + xs2)
    else:
        xs = xs1

    indices = np.arange(xs.size)
    neck_idx = int(np.searchsorted(xs, neck_joint_x, side="left"))

    finite_mask = (
        np.all(np.isfinite(rib1), axis=1)
        & np.all(np.isfinite(rib2), axis=1)
    )
    candidate_indices = indices[(indices >= neck_idx) & finite_mask]
    if candidate_indices.size < 2:
        raise ValueError("Insufficient rib samples at or below the neck joint to continue the trend.")

    base_idx = int(candidate_indices[0])
    prefix_len = base_idx
    if prefix_len == 0:
        return rib1, rib2

    slope_count = min(trend_window + 1, candidate_indices.size)
    slope_indices = candidate_indices[:slope_count]

    def _build_prefix(rib: np.ndarray) -> np.ndarray:
        x_use = rib[slope_indices, 0].astype(float)
        y_use = rib[slope_indices, 1].astype(float)
        z_use = rib[slope_indices, 2].astype(float)

        dx = np.diff(x_use)
        mask = np.abs(dx) > EPS
        if not np.any(mask):
            slope_y = 0.0
            slope_z = 0.0
        else:
            slope_y = float(np.mean(np.diff(y_use)[mask] / dx[mask]))
            slope_z = float(np.mean(np.diff(z_use)[mask] / dx[mask]))

        base_point = rib[base_idx]
        x_base = float(base_point[0])
        y_base = float(base_point[1])
        z_base = float(base_point[2])

        prefix = np.empty((prefix_len, 3), dtype=float)
        prefix[:, 0] = xs[:prefix_len]
        delta = prefix[:, 0] - x_base
        prefix[:, 1] = y_base + slope_y * delta
        prefix[:, 2] = z_base + slope_z * delta
        return prefix

    prefix1 = _build_prefix(rib1)
    prefix2 = _build_prefix(rib2)

    rib1_new = np.vstack([prefix1, rib1[base_idx:]])
    rib2_new = np.vstack([prefix2, rib2[base_idx:]])
    return rib1_new, rib2_new


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


__all__ = [
    "build_extended_rib_surfaces",
    "plot_lute_ribs",
]
