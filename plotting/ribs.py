"""Rib-specific plotting and export helpers."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lute_bowl.rib_form_builder import (
    RibSidePlane,
    project_points_to_plane,
    line_plane_intersection,
)

from .bowl import set_axes_equal_3d

MM_PER_INCH = 25.4
GRID_MM = 50.0
MIN_PANEL_IN = 2.0
LABEL_CLEARANCE_MM = 2.0
LABEL_OFFSET_FRAC = 0.15


def plot_rib_surfaces(
    surfaces: Sequence[tuple[int, Sequence[np.ndarray]]],
    *,
    outlines: Sequence[tuple[int, Sequence[np.ndarray]]] | None = None,
    spacing: float = 200.0,
    title: str | None = None,
    lute_name: str | None = None,
):
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(211, projection="3d")

    outline_map: dict[int, Sequence[np.ndarray]] = {}
    if outlines:
        outline_map = {idx: tuple(map(np.asarray, pair)) for idx, pair in outlines}

    outline_labeled = False

    for rib_idx, quads in surfaces:
        offset = (rib_idx - 1) * spacing
        for quad in quads:
            quad = np.asarray(quad) + np.array([0.0, offset, 0.0])
            poly = Poly3DCollection([quad], alpha=0.6)
            poly.set_facecolor((0.3, 0.6, 0.8, 0.4))
            poly.set_edgecolor("#204060")
            ax.add_collection3d(poly)
        if outline_map:
            pair = outline_map.get(rib_idx)
            if pair:
                for outline in pair:
                    if outline.size == 0:
                        continue
                    arr = np.asarray(outline, dtype=float) + np.array([0.0, offset, 0.0])
                    label = "rib outlines" if not outline_labeled else None
                    outline_labeled = outline_labeled or label is not None
                    ax.plot(
                        arr[:, 0],
                        arr[:, 1],
                        arr[:, 2],
                        color="#303030",
                        lw=1.2,
                        ls="--",
                        label=label,
                    )

    ax.set_xlabel("X (offset by rib index)")
    ax.set_ylabel("Y (across)")
    ax.set_zlabel("Z (depth)")
    if title is None:
        title = f"{lute_name} Extended Rib Surfaces" if lute_name else "Extended Rib Surfaces"
    elif lute_name and lute_name not in title:
        title = f"{lute_name} — {title}"
    ax.set_title(title)
    set_axes_equal_3d(ax)
    plt.tight_layout()
    plt.show()


def plot_rib_surface_with_planes(
    rib_index: int,
    quads: Sequence[np.ndarray],
    outline_pair: Sequence[np.ndarray],
    planes: Sequence[RibSidePlane],
    *,
    title: str | None = None,
    lute_name: str | None = None,
    panel_projections: Sequence[PanelProjection | None] | None = None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for quad in quads:
        quad = np.asarray(quad, dtype=float)
        poly = Poly3DCollection([quad], alpha=0.7)
        poly.set_facecolor((0.3, 0.6, 0.8, 0.4))
        poly.set_edgecolor("#204060")
        ax.add_collection3d(poly)

    outline_pair = tuple(np.asarray(outline, dtype=float) for outline in outline_pair)
    for outline in outline_pair:
        if outline.size == 0:
            continue
        ax.plot(outline[:, 0], outline[:, 1], outline[:, 2], color="#303030", lw=1.2)

    plane_colors = {"negative": "#ee9a00", "positive": "#ffcc66"}
    plane_map = {plane.side: plane for plane in planes}
    neg_plane = plane_map.get("negative") or (planes[0] if planes else None)
    pos_plane = plane_map.get("positive") or (planes[-1] if planes else None)

    for plane in planes:
        corners = np.asarray(plane.corners, dtype=float)
        if corners.size == 0:
            continue
        color = plane_colors.get(plane.side, "#c48f00")
        patch = Poly3DCollection([corners], alpha=0.35)
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        ax.add_collection3d(patch)

    if panel_projections is None:
        panel_projections = build_panel_projections(outline_pair, planes, unit_scale=1.0)

    outline_planes: list[tuple[np.ndarray, RibSidePlane | None, str, np.ndarray]] = [
        (outline_pair[0], neg_plane, plane_colors.get("negative", "#c48f00"), None),
        (outline_pair[1], pos_plane, plane_colors.get("positive", "#c48f00"), None),
    ]

    projected_paths: list[np.ndarray] = []
    for outline_entry, proj_data in zip(outline_planes, panel_projections, strict=False):
        outline, plane, color, _ = outline_entry
        if proj_data is None or plane is None or outline.size == 0:
            projected_paths.append(np.empty((0, 3)))
            continue
        projected = proj_data.projected_3d
        projected_paths.append(projected)
        ax.plot(projected[:, 0], projected[:, 1], projected[:, 2], color=color, lw=1.2, ls="--")
        for point, proj in zip(outline, projected, strict=False):
            segment = np.vstack([point, proj])
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=0.8, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title is None:
        base = f"Rib {rib_index} Surface"
        title = f"{lute_name} — {base}" if lute_name else base
    ax.set_title(title)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for quad in quads:
        arr = np.asarray(quad, dtype=float)
        xs.append(arr[:, 0])
        ys.append(arr[:, 1])
        zs.append(arr[:, 2])
    for outline in outline_pair:
        arr = np.asarray(outline, dtype=float)
        xs.append(arr[:, 0])
        ys.append(arr[:, 1])
        zs.append(arr[:, 2])
    for proj in projected_paths:
        if proj.size == 0:
            continue
        xs.append(proj[:, 0])
        ys.append(proj[:, 1])
        zs.append(proj[:, 2])
    for plane in planes:
        arr = np.asarray(plane.corners, dtype=float)
        xs.append(arr[:, 0])
        ys.append(arr[:, 1])
        zs.append(arr[:, 2])

    def _flatten(points: list[np.ndarray]) -> np.ndarray | None:
        if not points:
            return None
        return np.concatenate(points)

    set_axes_equal_3d(ax, _flatten(xs), _flatten(ys), _flatten(zs))
    plt.tight_layout()
    plt.show()


@dataclass
class PanelProjection:
    plane: RibSidePlane
    outline_3d: np.ndarray
    projected_3d: np.ndarray
    plane_outline_2d: np.ndarray
    plane_corners_2d: np.ndarray
    bbox_2d: tuple[float, float, float, float]


@dataclass
class ProjectionMarker:
    point: tuple[float, float]
    label: str | None = None
    color: str = "#aa3344"
    size: float = 18.0


def build_panel_projections(
    outline_pair: Sequence[np.ndarray],
    planes: Sequence[RibSidePlane],
    unit_scale: float,
    *,
    verbose: bool = False,
) -> list[PanelProjection | None]:
    panel_projections: list[PanelProjection | None] = []
    plane_order = [p for p in planes[:2]]

    for idx, (plane, outline) in enumerate(zip(plane_order, outline_pair, strict=False)):
        if plane is None:
            panel_projections.append(None)
            continue
        outline_arr = np.asarray(outline, dtype=float)

        # Build an outline from intersections of rib connectors with this plane.
        intersections: list[np.ndarray] = []
        other = outline_pair[1 - idx]
        other_arr = np.asarray(other, float)
        if outline_arr.shape == other_arr.shape and outline_arr.shape[0] == other_arr.shape[0]:
            for p0, p1 in zip(outline_arr, other_arr, strict=False):
                hit = line_plane_intersection(p0, p1, plane)
                if hit is not None:
                    intersections.append(hit)
                else:
                    # Log and fall back to projecting the midpoint if nearly parallel.
                    mid = 0.5 * (p0 + p1)
                    fallback = project_points_to_plane(np.array([mid]), plane)[0]
                    intersections.append(fallback)
                    if verbose:
                        print(
                            f"[rib-plane] connector failed to intersect plane; using midpoint projection. "
                            f"plane_norm={plane.normal}, p0={p0}, p1={p1}"
                        )
        source_outline = np.asarray(intersections, float)

        projected_3d = project_points_to_plane(source_outline, plane)
        plane_outline = _project_to_plane_coords(source_outline, plane, unit_scale)
        plane_corners = _project_to_plane_coords(plane.corners, plane, unit_scale)
        combined = np.vstack([plane_outline, plane_corners])
        x_min, x_max = combined[:, 0].min(), combined[:, 0].max()
        y_min, y_max = combined[:, 1].min(), combined[:, 1].max()
        panel_projections.append(
            PanelProjection(
                plane=plane,
                outline_3d=outline_arr,
                projected_3d=projected_3d,
                plane_outline_2d=plane_outline,
                plane_corners_2d=plane_corners,
                bbox_2d=(x_min, x_max, y_min, y_max),
            )
        )
    return panel_projections


def compute_panel_frame(
    panel_projections: Sequence[PanelProjection | None],
    pad_mm: float,
    min_panel_in: float,
    mm_per_inch: float,
    override: tuple[float, float] | None,
) -> tuple[float, float]:
    if override is not None:
        return override

    max_w = 0.0
    max_h = 0.0
    for proj in panel_projections:
        if proj is None:
            continue
        x_min, x_max, y_min, y_max = proj.bbox_2d
        max_w = max(max_w, (x_max - x_min) + 2 * pad_mm)
        max_h = max(max_h, (y_max - y_min) + 2 * pad_mm)

    min_mm = min_panel_in * mm_per_inch
    max_w = max(max_w, min_mm)
    max_h = max(max_h, min_mm)
    return (max_w, max_h)


def _compute_frame_from_bbox(
    bbox: tuple[float, float, float, float],
    *,
    pad_mm: float,
    min_panel_in: float,
    mm_per_inch: float,
    override: tuple[float, float] | None,
) -> tuple[float, float]:
    if override is not None:
        return override
    x_min, x_max, y_min, y_max = bbox
    max_w = (x_max - x_min) + 2 * pad_mm
    max_h = (y_max - y_min) + 2 * pad_mm
    min_mm = min_panel_in * mm_per_inch
    max_w = max(max_w, min_mm)
    max_h = max(max_h, min_mm)
    return (max_w, max_h)


def _draw_projection_panel(
    output_path: Path,
    *,
    outline_2d: np.ndarray,
    plane_2d: np.ndarray | None,
    frame_size_mm: tuple[float, float],
    label: str | None,
    title: str | None,
    dpi: int,
    markers: Sequence[ProjectionMarker] | None = None,
    grid_mm: float = GRID_MM,
) -> None:
    width_in = frame_size_mm[0] / MM_PER_INCH
    height_in = frame_size_mm[1] / MM_PER_INCH
    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    if plane_2d is not None:
        poly = Polygon(
            plane_2d,
            closed=True,
            facecolor="none",
            edgecolor="#8a6b2f",
            linewidth=0.4,
        )
        ax.add_patch(poly)
    ax.plot(outline_2d[:, 0], outline_2d[:, 1], color="#224466", lw=0.35)

    xlim = (0.0, frame_size_mm[0])
    ylim = (0.0, frame_size_mm[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")

    x_ticks = np.arange(0.0, math.ceil(xlim[1] / grid_mm) * grid_mm + grid_mm, grid_mm)
    y_ticks = np.arange(0.0, math.ceil(ylim[1] / grid_mm) * grid_mm + grid_mm, grid_mm)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, which="major", linestyle="--", color="0.85", linewidth=0.4)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if label:
        span_x = xlim[1] - xlim[0]
        span_y = ylim[1] - ylim[0]
        outline_bb = (
            outline_2d[:, 0].min(),
            outline_2d[:, 0].max(),
            outline_2d[:, 1].min(),
            outline_2d[:, 1].max(),
        )

        def _in_outline(pt):
            x, y = pt
            return outline_bb[0] <= x <= outline_bb[1] and outline_bb[2] <= y <= outline_bb[3]

        candidates = [
            (xlim[0] + 0.08 * span_x, ylim[1] - 0.08 * span_y),
            (xlim[0] + 0.08 * span_x, ylim[0] + 0.08 * span_y),
            (xlim[0] + 0.5 * span_x, ylim[0] + 0.08 * span_y),
        ]
        label_pos = next((pt for pt in candidates if not _in_outline(pt)), candidates[-1])
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha="left",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 2.0},
        )

        for x0, x1 in zip(x_ticks[:-1], x_ticks[1:], strict=False):
            for y0, y1 in zip(y_ticks[:-1], y_ticks[1:], strict=False):
                span_x = x1 - x0
                span_y = y1 - y0
                if span_x < grid_mm - 1e-6 or span_y < grid_mm - 1e-6:
                    continue
                candidates_cell = [
                    (x0 + LABEL_OFFSET_FRAC * span_x, y0 + LABEL_OFFSET_FRAC * span_y),
                    (x0 + LABEL_OFFSET_FRAC * span_x, y1 - LABEL_OFFSET_FRAC * span_y),
                    (x1 - LABEL_OFFSET_FRAC * span_x, y0 + LABEL_OFFSET_FRAC * span_y),
                    (x1 - LABEL_OFFSET_FRAC * span_x, y1 - LABEL_OFFSET_FRAC * span_y),
                ]
                for pt in candidates_cell:
                    if _min_distance_to_polyline(pt, outline_2d) >= LABEL_CLEARANCE_MM:
                        ax.text(
                            pt[0],
                            pt[1],
                            label,
                            ha="left",
                            va="center",
                            fontsize=7,
                            color="0.25",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6, "pad": 1.5},
                        )
                        break

    if markers:
        for marker in markers:
            x, y = marker.point
            ax.scatter([x], [y], s=marker.size, color=marker.color, zorder=5)
            if marker.label:
                ax.text(
                    x + 2.0,
                    y + 2.0,
                    marker.label,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
                )

    if title:
        ax.set_title(f"{title} (grid {grid_mm:.0f}x{grid_mm:.0f}mm)", fontsize=11, pad=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def save_outline_projection_png(
    output_path: str | Path,
    outline_points: np.ndarray,
    *,
    unit_scale: float,
    dpi: int = 300,
    title: str | None = None,
    frame_size_mm: tuple[float, float] | None = None,
    pad_mm: float = 20.0,
    markers: Sequence[ProjectionMarker] | None = None,
) -> Path:
    coords = np.asarray(outline_points, dtype=float)
    if coords.shape[1] > 2:
        coords = coords[:, :2]
    coords_mm = coords * float(unit_scale)

    x_min = float(coords_mm[:, 0].min())
    x_max = float(coords_mm[:, 0].max())
    y_min = float(coords_mm[:, 1].min())
    y_max = float(coords_mm[:, 1].max())
    if markers:
        for marker in markers:
            x_min = min(x_min, marker.point[0] * unit_scale)
            x_max = max(x_max, marker.point[0] * unit_scale)
            y_min = min(y_min, marker.point[1] * unit_scale)
            y_max = max(y_max, marker.point[1] * unit_scale)

    bbox = (x_min, x_max, y_min, y_max)
    frame_size_mm = _compute_frame_from_bbox(
        bbox,
        pad_mm=pad_mm,
        min_panel_in=MIN_PANEL_IN,
        mm_per_inch=MM_PER_INCH,
        override=frame_size_mm,
    )
    shift_x = pad_mm - x_min
    shift_y = frame_size_mm[1] - pad_mm - y_max

    coords_mm = coords_mm + np.array([shift_x, shift_y])
    shifted_markers: list[ProjectionMarker] = []
    if markers:
        for marker in markers:
            shifted = (
                marker.point[0] * unit_scale + shift_x,
                marker.point[1] * unit_scale + shift_y,
            )
            shifted_markers.append(
                ProjectionMarker(
                    point=shifted,
                    label=marker.label,
                    color=marker.color,
                    size=marker.size,
                )
            )

    _draw_projection_panel(
        Path(output_path),
        outline_2d=coords_mm,
        plane_2d=None,
        frame_size_mm=frame_size_mm,
        label="Soundboard",
        title=title,
        dpi=dpi,
        markers=shifted_markers,
    )
    return Path(output_path)


def save_plane_projection_png(
    output_path: str | Path,
    outline_pair: Sequence[np.ndarray],
    planes: Sequence[RibSidePlane],
    *,
    unit_scale: float,
    dpi: int = 300,
    title: str | None = None,
    frame_size_mm: tuple[float, float] | None = None,
    pad_mm: float = 20.0,
    panel_projections: Sequence[PanelProjection | None] | None = None,
) -> tuple[Path, Path]:
    """Save each rib-side plane projection to its own PNG.

    Returns (left_path, right_path); the left panel corresponds to the first
    outline in `outline_pair`, which is the lower-index rib outline."""

    path = Path(output_path)

    def _side_path(base: Path, side: str) -> Path:
        stem = base.stem
        match = re.search(r"(index\d+)", stem)
        if match:
            tagged = stem.replace(match.group(1), f"{match.group(1)}_{side}", 1)
        else:
            tagged = f"{stem}_{side}"
        return base.with_name(f"{tagged}{base.suffix}")

    rib_idx = getattr(planes[0], "rib_index", None) if planes else None
    panel_labels = [
        f"Rib {rib_idx}, left" if rib_idx is not None else "Left",
        f"Rib {rib_idx}, right" if rib_idx is not None else "Right",
    ]

    if panel_projections is None:
        panel_projections = build_panel_projections(outline_pair, planes, unit_scale)

    paths = (_side_path(path, "left"), _side_path(path, "right"))

    frame_size_mm = compute_panel_frame(panel_projections, pad_mm, MIN_PANEL_IN, MM_PER_INCH, frame_size_mm)

    # Use a common translation so left/right outlines retain their relative positioning.
    global_x_min = float("inf")
    global_x_max = float("-inf")
    global_y_min = float("inf")
    global_y_max = float("-inf")
    for proj in panel_projections:
        if proj is None:
            continue
        x_min, x_max, y_min, y_max = proj.bbox_2d
        global_x_min = min(global_x_min, x_min)
        global_x_max = max(global_x_max, x_max)
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)

    shift_x = pad_mm - global_x_min
    shift_y = frame_size_mm[1] - pad_mm - global_y_max

    for proj, side_path, panel_label in zip(panel_projections, paths, panel_labels, strict=False):
        if proj is None:
            continue
        coords_outline = proj.plane_outline_2d + np.array([shift_x, shift_y])
        plane_coords = proj.plane_corners_2d + np.array([shift_x, shift_y])

        _draw_projection_panel(
            side_path,
            outline_2d=coords_outline,
            plane_2d=plane_coords,
            frame_size_mm=frame_size_mm,
            label=panel_label,
            title=title,
            dpi=dpi,
        )

    return paths


def _project_to_plane_coords(points: np.ndarray, plane: RibSidePlane, unit_scale: float) -> np.ndarray:
    projected = project_points_to_plane(np.asarray(points, dtype=float), plane)
    long_coords = projected @ plane.long_direction
    height_coords = projected @ plane.height_direction
    return np.column_stack((long_coords, height_coords)) * unit_scale


def _min_distance_to_polyline(point: tuple[float, float], polyline: np.ndarray) -> float:
    """Return minimum distance from a 2D point to a polyline."""

    if polyline.size == 0:
        return float("inf")

    p = np.asarray(point, dtype=float)
    line = np.asarray(polyline, dtype=float)
    min_dist = float("inf")

    for a, b in zip(line[:-1], line[1:], strict=False):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            dist = float(np.linalg.norm(p - a))
            min_dist = min(min_dist, dist)
            continue
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        dist = float(np.linalg.norm(p - closest))
        min_dist = min(min_dist, dist)

    return min_dist


__all__ = [
    "plot_rib_surfaces",
    "plot_rib_surface_with_planes",
    "save_plane_projection_png",
    "save_outline_projection_png",
    "ProjectionMarker",
    "build_panel_projections",
    "compute_panel_frame",
    "PanelProjection",
]
