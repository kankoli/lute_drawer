"""Rib-specific plotting and export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lute_bowl.rib_form_builder import RibSidePlane, project_points_to_plane

from .bowl import set_axes_equal_3d


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

    outline_planes: list[tuple[np.ndarray, RibSidePlane | None, str]] = [
        (outline_pair[0], neg_plane, plane_colors.get("negative", "#c48f00")),
        (outline_pair[1], pos_plane, plane_colors.get("positive", "#c48f00")),
    ]

    projected_paths: list[np.ndarray] = []
    for outline, plane, color in outline_planes:
        if plane is None or outline.size == 0:
            projected_paths.append(np.empty((0, 3)))
            continue
        projected = project_points_to_plane(outline, plane)
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


def save_plane_projection_png(
    output_path: str | Path,
    outline_pair: Sequence[np.ndarray],
    planes: Sequence[RibSidePlane],
    *,
    unit_scale: float,
    dpi: int = 300,
    title: str | None = None,
) -> Path:
    path = Path(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plane_order = [p for p in planes[:2]]
    labels = {"negative": "Plane A", "positive": "Plane B"}

    for ax, plane, outline in zip(axes, plane_order, outline_pair, strict=False):
        if plane is None:
            ax.axis("off")
            continue
        coords_outline = _project_to_plane_coords(outline, plane, unit_scale)
        plane_coords = _project_to_plane_coords(plane.corners, plane, unit_scale)
        poly = plt.Polygon(plane_coords, closed=True, facecolor="#f5deb3", edgecolor="#8a6b2f", alpha=0.35)
        ax.add_patch(poly)
        ax.plot(coords_outline[:, 0], coords_outline[:, 1], color="#224466", lw=1.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Long axis (mm)")
        ax.set_ylabel("Height (mm)")
        ax.set_title(labels.get(plane.side, "Plane"))
        _add_scale_bar(ax)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _project_to_plane_coords(points: np.ndarray, plane: RibSidePlane, unit_scale: float) -> np.ndarray:
    projected = project_points_to_plane(np.asarray(points, dtype=float), plane)
    long_coords = projected @ plane.long_direction
    height_coords = projected @ plane.height_direction
    return np.column_stack((long_coords, height_coords)) * unit_scale


def _add_scale_bar(ax, length_mm: float = 100.0) -> None:
    if length_mm <= 0:
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_start = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y_start = ylim[0] - 0.08 * (ylim[1] - ylim[0])
    ax.plot([x_start, x_start + length_mm], [y_start, y_start], color="k", lw=2.0)
    ax.text(
        x_start + length_mm / 2,
        y_start - 0.02 * (ylim[1] - ylim[0]),
        f"{int(length_mm)} mm",
        ha="center",
        va="top",
    )


__all__ = [
    "plot_rib_surfaces",
    "plot_rib_surface_with_planes",
    "save_plane_projection_png",
]
