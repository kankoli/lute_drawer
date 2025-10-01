from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bowl_from_soundboard import Section, build_bowl_for_lute, set_axes_equal_3d, MidCurve

EPS = 1e-9


# ---------------------------------------------------------------------------
# Options & orchestrators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RibSurfaceOptions:
    plane_offset: float = 10.0
    allowance_left: float = 0.0
    allowance_right: float = 0.0
    end_extension: float = 10.0
    spacing: float = 200.0


def build_extended_rib_surfaces(
    lute,
    *,
    top_curve=MidCurve,
    n_ribs: int = 13,
    n_sections: int = 40,
    options: RibSurfaceOptions | None = None,
    rib_index: int | None = None,
    draw_all: bool = False,
):
    """Return extended rib surfaces for the requested rib indices."""
    opts = options or RibSurfaceOptions()
    sections, rib_outlines = build_bowl_for_lute(
        lute,
        n_ribs=n_ribs,
        n_sections=n_sections,
        top_curve=top_curve,
    )

    if draw_all:
        target_indices = range(len(rib_outlines) - 1)
    else:
        if rib_index is None:
            raise ValueError("rib_index must be provided unless draw_all=True")
        target_indices = [rib_index - 1]

    surfaces = []
    for idx in target_indices:
        if idx < 0 or idx >= len(rib_outlines) - 1:
            raise ValueError("Rib index is outside range:", idx + 1)
        rib1, rib2 = rib_outlines[idx], rib_outlines[idx + 1]
        quads = _rib_surface_extended(
            rib1,
            rib2,
            plane_offset=opts.plane_offset,
            allowance_left=opts.allowance_left,
            allowance_right=opts.allowance_right,
            end_extension=opts.end_extension,
        )
        surfaces.append((idx + 1, quads))
    return sections, surfaces, opts


def plot_rib_surfaces(surfaces: Sequence[tuple[int, List[np.ndarray]]], *, spacing: float, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for rib_idx, quads in surfaces:
        offset = (rib_idx - 1) * spacing
        for quad in quads:
            quad = np.asarray(quad) + np.array([0.0, offset, 0.0])
            poly = Poly3DCollection([quad], alpha=0.6)
            poly.set_facecolor((0.3, 0.6, 0.8, 0.4))
            poly.set_edgecolor("#204060")
            ax.add_collection3d(poly)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    set_axes_equal_3d(ax)
    plt.tight_layout()
    plt.show()


def plot_lute_ribs(
    lute,
    *,
    top_curve=MidCurve,
    n_ribs: int = 13,
    n_sections: int = 40,
    rib: int = 7,
    draw_all: bool = False,
    plane_offset: float = 10.0,
    allowance_left: float = 0.0,
    allowance_right: float = 0.0,
    end_extension: float = 10.0,
    spacing: float = 200.0,
    title: str = "Extended Rib Surfaces",
):
    options = RibSurfaceOptions(
        plane_offset=plane_offset,
        allowance_left=allowance_left,
        allowance_right=allowance_right,
        end_extension=end_extension,
        spacing=spacing,
    )
    _, surfaces, opts = build_extended_rib_surfaces(
        lute,
        top_curve=top_curve,
        n_ribs=n_ribs,
        n_sections=n_sections,
        options=options,
        rib_index=rib,
        draw_all=draw_all,
    )
    plot_rib_surfaces(surfaces, spacing=opts.spacing, title=title)


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged core logic)
# ---------------------------------------------------------------------------


def _safe_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _align_with_normal(vec, nrm):
    vec = np.asarray(vec, dtype=float)
    nrm = np.asarray(nrm, dtype=float)
    return vec if np.dot(vec, nrm) >= 0.0 else -vec


def _intersect_line_plane_pd(p, d, p0, nrm):
    d = np.asarray(d, dtype=float)
    nrm = np.asarray(nrm, dtype=float)
    denom = float(np.dot(nrm, d))
    if abs(denom) < EPS:
        return None
    t = float(np.dot(nrm, (p0 - p)) / denom)
    return p + t * d


def _intersect_line_plane_p1p2(p1, p2, p0, nrm):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return _intersect_line_plane_pd(p1, p2 - p1, p0, nrm)


def _rib_surface_extended(
    outline1,
    outline2,
    *,
    plane_offset: float,
    allowance_left: float,
    allowance_right: float,
    end_extension: float,
):
    rib1 = np.array(outline1, float)
    rib2 = np.array(outline2, float)
    n = min(len(rib1), len(rib2))
    rib1, rib2 = rib1[:n], rib2[:n]

    across = _safe_unit(rib2.mean(axis=0) - rib1.mean(axis=0))
    nrm = across.copy()

    p0_left = rib1.mean(axis=0) - (plane_offset + allowance_left) * nrm
    p0_right = rib2.mean(axis=0) + (plane_offset + allowance_right) * nrm

    strips = []

    tip_top = 0.5 * (rib1[0] + rib2[0])
    across0 = (rib2[1] - rib1[1]) if n > 1 else np.array([1.0, 0.0, 0.0])
    across0 = _align_with_normal(_safe_unit(across0), nrm)
    left_top_cap = _intersect_line_plane_pd(tip_top, across0, p0_left, nrm)
    right_top_cap = _intersect_line_plane_pd(tip_top, across0, p0_right, nrm)
    if left_top_cap is None:
        left_top_cap = tip_top - nrm * (plane_offset + allowance_left)
    if right_top_cap is None:
        right_top_cap = tip_top + nrm * (plane_offset + allowance_right)
    strips.append([left_top_cap, tip_top, tip_top, right_top_cap])

    tip_dir0 = rib1[1] - rib1[0] if n > 1 else np.array([0.0, 0.0, 1.0])
    tip_dir0 = _safe_unit(tip_dir0)
    tip_top_ext = tip_top - tip_dir0 * end_extension
    left_top_ext = _intersect_line_plane_pd(tip_top_ext, across0, p0_left, nrm)
    right_top_ext = _intersect_line_plane_pd(tip_top_ext, across0, p0_right, nrm)
    if left_top_ext is None:
        left_top_ext = tip_top_ext - nrm * (plane_offset + allowance_left)
    if right_top_ext is None:
        right_top_ext = tip_top_ext + nrm * (plane_offset + allowance_right)
    strips.insert(0, [left_top_ext, tip_top_ext, tip_top_ext, right_top_ext])

    for j in range(1, n - 1):
        a = rib1[j]
        b = rib2[j]
        left_pt = _intersect_line_plane_p1p2(a, b, p0_left, nrm)
        right_pt = _intersect_line_plane_p1p2(a, b, p0_right, nrm)
        if left_pt is not None and right_pt is not None:
            strips.append([left_pt, a, b, right_pt])

    tip_bot = 0.5 * (rib1[-1] + rib2[-1])
    across1 = (rib2[-2] - rib1[-2]) if n > 1 else np.array([1.0, 0.0, 0.0])
    across1 = _align_with_normal(_safe_unit(across1), nrm)
    left_bot_cap = _intersect_line_plane_pd(tip_bot, across1, p0_left, nrm)
    right_bot_cap = _intersect_line_plane_pd(tip_bot, across1, p0_right, nrm)
    if left_bot_cap is None:
        left_bot_cap = tip_bot - nrm * (plane_offset + allowance_left)
    if right_bot_cap is None:
        right_bot_cap = tip_bot + nrm * (plane_offset + allowance_right)
    strips.append([left_bot_cap, tip_bot, tip_bot, right_bot_cap])

    tip_dir1 = rib1[-1] - rib1[-2] if n > 1 else np.array([0.0, 0.0, 1.0])
    tip_dir1 = _safe_unit(tip_dir1)
    tip_bot_ext = tip_bot + tip_dir1 * end_extension
    left_bot_ext = _intersect_line_plane_pd(tip_bot_ext, across1, p0_left, nrm)
    right_bot_ext = _intersect_line_plane_pd(tip_bot_ext, across1, p0_right, nrm)
    if left_bot_ext is None:
        left_bot_ext = tip_bot_ext - nrm * (plane_offset + allowance_left)
    if right_bot_ext is None:
        right_bot_ext = tip_bot_ext + nrm * (plane_offset + allowance_right)
    strips.append([left_bot_ext, tip_bot_ext, tip_bot_ext, right_bot_ext])

    quads = []
    for j in range(len(strips) - 1):
        s1, s2 = strips[j], strips[j + 1]
        for k in range(4):
            p1, p2 = np.array(s1[k]), np.array(s1[(k + 1) % 4])
            q2, q1 = np.array(s2[(k + 1) % 4]), np.array(s2[k])
            quads.append(np.array([p1, p2, q2, q1]))

    quads.append(np.array(strips[0]))
    quads.append(np.array(strips[-1]))

    return _normalize_quads(rib1, rib2, quads)


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
    "RibSurfaceOptions",
    "build_extended_rib_surfaces",
    "plot_rib_surfaces",
    "plot_lute_ribs",
]
