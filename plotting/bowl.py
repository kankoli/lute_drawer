"""Plotting helpers for bowl geometry."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bowl_from_soundboard import Section
from bowl_mold import MoldSection


def set_axes_equal_3d(ax, xs=None, ys=None, zs=None, use_ortho=True):
    """Force equal data scale on a 3D axes so circles look circular."""
    if xs is None or ys is None or zs is None:
        xmin, xmax = ax.get_xlim3d()
        ymin, ymax = ax.get_ylim3d()
        zmin, zmax = ax.get_zlim3d()
    else:
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        zmin, zmax = float(zs.min()), float(zs.max())
    xmid, ymid, zmid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
    r = max(xmax - xmin, ymax - ymin, zmax - zmin, 1e-12) / 2.0
    ax.set_xlim3d([xmid - r, xmid + r])
    ax.set_ylim3d([ymid - r, ymid + r])
    ax.set_zlim3d([zmid - r, zmid + r])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # pragma: no cover
        pass
    if use_ortho:
        try:
            ax.set_proj_type("ortho")
        except Exception:  # pragma: no cover
            pass


def plot_bowl(
    lute,
    sections: Sequence[Section],
    ribs: Sequence[np.ndarray],
    *,
    show_ribs: bool = True,
    show_soundboard: bool = True,
    show_section_circles: bool = True,
    show_apexes: bool = False,
    show_top_curve: bool = False,
    show_form_points: bool = True,
    show_spine: bool = True,
    highlight_neck_joint: bool = True,
    mold_sections: Sequence[MoldSection] | None = None,
    show_mold_faces: bool = True,
    show_mold_rib_points: bool = True,
    show_soundboard_outline: bool = True,
    outline_offset_mm: float = 0.0,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs_bounds: list[np.ndarray] = []
    ys_bounds: list[np.ndarray] = []
    zs_bounds: list[np.ndarray] = []

    if show_ribs:
        for rib in ribs:
            ax.plot(rib[:, 0], rib[:, 1], rib[:, 2], "b-", lw=1, label="ribs" if rib is ribs[0] else "")
            xs_bounds.append(np.asarray(rib[:, 0], dtype=float))
            ys_bounds.append(np.asarray(rib[:, 1], dtype=float))
            zs_bounds.append(np.asarray(rib[:, 2], dtype=float))

    if show_soundboard and hasattr(lute, "final_arcs"):
        arcs = list(getattr(lute, "final_arcs", [])) + list(getattr(lute, "final_reflected_arcs", []))
        for idx_arc, arc in enumerate(arcs):
            pts = arc.sample_points(200)
            zs = np.zeros(pts.shape[0])
            ax.plot(pts[:, 0], pts[:, 1], zs, color="0.2", alpha=0.3, label="soundboard" if idx_arc == 0 else "")

    apex_Xs: list[float] = []
    apex_Ys: list[float] = []
    apex_Zs: list[float] = []
    if sections:
        for section in sections:
            apex_Xs.append(float(section.x))
            apex_Ys.append(float(section.apex[0]))
            apex_Zs.append(float(section.apex[1]))

    if show_soundboard_outline and hasattr(lute, "final_arcs"):
        def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
            axis_norm = np.linalg.norm(axis)
            if axis_norm <= 1e-12:
                axis = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                axis = axis / axis_norm
            x, y, z = axis
            theta = np.deg2rad(angle_deg)
            c = np.cos(theta)
            s = np.sin(theta)
            C = 1.0 - c
            return np.array(
                [
                    [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                    [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
                ],
                dtype=float,
            )

        scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
        z_plane = outline_offset_mm / scale if outline_offset_mm else 0.0

        spine_start = np.array([float(lute.form_top.x), float(lute.form_top.y), z_plane], dtype=float)
        spine_end = np.array([float(lute.form_bottom.x), float(lute.form_bottom.y), z_plane], dtype=float)
        rotation = _rotation_matrix(spine_end - spine_start, -90.0)

        outline_arcs = [arc for arc in getattr(lute, "final_arcs", [])]

        for idx_arc, arc in enumerate(outline_arcs):
            pts_2d = arc.sample_points(200)
            pts_3d = np.column_stack(
                [
                    pts_2d[:, 0].astype(float),
                    pts_2d[:, 1].astype(float),
                    np.full(pts_2d.shape[0], z_plane, dtype=float),
                ]
            )
            rotated = (pts_3d - spine_start) @ rotation.T + spine_start
            label = "soundboard outline" if idx_arc == 0 else ""
            ax.plot(
                rotated[:, 0],
                rotated[:, 1],
                rotated[:, 2],
                color="0.4",
                linestyle="--",
                alpha=0.6,
                label=label,
            )
            xs_bounds.append(np.asarray(rotated[:, 0], dtype=float))
            ys_bounds.append(np.asarray(rotated[:, 1], dtype=float))
            zs_bounds.append(np.asarray(rotated[:, 2], dtype=float))

    if show_section_circles and len(sections) <= 80:
        fb_x = float(lute.form_bottom.x)
        for section in sections:
            x, center, r, _ = section
            if r <= 0 or abs(x - fb_x) < 1e-9:
                continue
            C_Y, C_Z = float(center[0]), float(center[1])
            phi = np.linspace(0, 2 * np.pi, 200)
            Y = C_Y + r * np.cos(phi)
            Z = C_Z + r * np.sin(phi)
            X = np.full_like(Y, float(x))
            ax.plot(X, Y, Z, color="0.3", alpha=0.25)

    if apex_Xs:
        if show_apexes:
            ax.scatter(apex_Xs, apex_Ys, apex_Zs, color="red", s=50, label="apex")
        if show_top_curve:
            ax.plot(apex_Xs, apex_Ys, apex_Zs, color="red", lw=2, alpha=0.7, label="top curve")

    if show_form_points:
        ax.scatter([float(lute.form_top.x)], [float(lute.form_top.y)], [0.0], color="orange", s=60, label="form_top")
        ax.scatter([float(lute.form_bottom.x)], [float(lute.form_bottom.y)], [0.0], color="green", s=60, label="form_bottom")

    if show_spine:
        xs_spine = [float(lute.form_top.x), float(lute.form_bottom.x)]
        ys_spine = [float(lute.form_top.y), float(lute.form_bottom.y)]
        ax.plot(xs_spine, ys_spine, [0.0, 0.0], "k--", alpha=0.6, label="spine")

    if highlight_neck_joint and sections:
        x_nj = float(lute.point_neck_joint.x)
        y_nj = float(lute.point_neck_joint.y)
        ax.scatter([x_nj], [y_nj], [0.0], color="purple", s=60, label="neck joint")

    if mold_sections and show_mold_faces:
        for section in mold_sections:
            color = None
            for face in section.faces:
                x = np.full_like(face.y, face.x)
                if color is None:
                    (line,) = ax.plot(x, face.y, face.z, linewidth=1.5)
                    color = line.get_color()
                else:
                    ax.plot(x, face.y, face.z, linewidth=1.5, color=color)
                if show_mold_rib_points:
                    ax.scatter(np.full(face.y.shape, face.x), face.y, face.z, s=12, color=color)
                xs_bounds.append(x)
                ys_bounds.append(face.y)
                zs_bounds.append(face.z)

    def _flatten(bounds):
        if not bounds:
            return None
        arrays = [np.asarray(b, dtype=float).ravel() for b in bounds]
        return np.concatenate(arrays) if arrays else None

    set_axes_equal_3d(
        ax,
        xs=_flatten(xs_bounds),
        ys=_flatten(ys_bounds),
        zs=_flatten(zs_bounds),
    )

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_rib_surfaces(
    surfaces: Sequence[tuple[int, Sequence[np.ndarray]]],
    *,
    spacing: float = 200.0,
    title: str = "Extended Rib Surfaces",
):
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

__all__ = ["set_axes_equal_3d", "plot_bowl", "plot_rib_surfaces"]
