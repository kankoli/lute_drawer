"""Plotting helpers for bowl geometry."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bowl_from_soundboard import Section


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
    show_apexes: bool = False,
    highlight_neck_joint: bool = True,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for rib in ribs:
        ax.plot(rib[:, 0], rib[:, 1], rib[:, 2], "b-", lw=1)

    if len(sections) <= 80:
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

    if show_apexes and sections:
        apex_Xs, apex_Ys, apex_Zs = [], [], []
        for section in sections:
            apex_Xs.append(float(section.x))
            apex_Ys.append(float(section.apex[0]))
            apex_Zs.append(float(section.apex[1]))
        ax.scatter(apex_Xs, apex_Ys, apex_Zs, color="red", s=50, label="apex (top curve)")
        ax.plot(apex_Xs, apex_Ys, apex_Zs, color="red", lw=2, alpha=0.7, label="top curve")

    ax.scatter([float(lute.form_top.x)], [float(lute.form_top.y)], [0.0], color="orange", s=60, label="form_top")
    ax.scatter([float(lute.form_bottom.x)], [float(lute.form_bottom.y)], [0.0], color="green", s=60, label="form_bottom")

    xs = [float(lute.form_top.x), float(lute.form_bottom.x)]
    ys = [float(lute.form_top.y), float(lute.form_bottom.y)]
    ax.plot(xs, ys, [0.0, 0.0], "k--", alpha=0.6, label="spine")

    if highlight_neck_joint and sections:
        x_nj = float(lute.point_neck_joint.x)
        y_nj = float(lute.point_neck_joint.y)
        ax.scatter([x_nj], [y_nj], [0.0], color="purple", s=60, label="neck joint")

    set_axes_equal_3d(
        ax,
        xs=[rib[:, 0] for rib in ribs] if ribs else None,
        ys=[rib[:, 1] for rib in ribs] if ribs else None,
        zs=[rib[:, 2] for rib in ribs] if ribs else None,
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
