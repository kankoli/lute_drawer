"""Plotting utilities for mold sections."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from bowl_mold import MoldSection
from .bowl import set_axes_equal_3d


def plot_mold_sections_2d(
    sections: Sequence[MoldSection],
    *,
    ax=None,
    show_rib_points: bool = True,
    invert_z: bool = True,
):
    """2D visualisation for mold section faces."""

    if ax is None:
        _, ax = plt.subplots()

    for section in sections:
        color = None
        for face in section.faces:
            label = f"X={face.x:.1f}" if color is None else None
            if color is None:
                (line,) = ax.plot(face.y, face.z, label=label)
                color = line.get_color()
            else:
                ax.plot(face.y, face.z, label=label, color=color)
            if show_rib_points:
                ax.scatter(face.y, face.z, s=12, zorder=3, color=color)

    ax.set_xlabel("Y (soundboard across)")
    ax.set_ylabel("Z (depth)")
    if invert_z:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    if sections:
        ax.legend(loc="best", fontsize="small")
    return ax


def plot_mold_sections_3d(
    sections: Sequence[MoldSection],
    *,
    ax=None,
    show_rib_points: bool = True,
):
    """Render mold sections in 3D for debugging."""

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    xs_all = []
    ys_all = []
    zs_all = []

    for section in sections:
        section_color = None
        for face in section.faces:
            y = face.y
            z = face.z
            x = np.full_like(y, face.x)
            if section_color is None:
                (line,) = ax.plot(x, y, z)
                section_color = line.get_color()
            else:
                ax.plot(x, y, z, color=section_color)
            if show_rib_points:
                ax.scatter(np.full(face.y.shape, face.x), face.y, face.z, s=12, color=section_color)

            xs_all.append(x)
            ys_all.append(y)
            zs_all.append(z)

    ax.set_xlabel("X (along spine)")
    ax.set_ylabel("Y (across)")
    ax.set_zlabel("Z (depth)")
    if xs_all:
        set_axes_equal_3d(ax, xs=np.concatenate(xs_all), ys=np.concatenate(ys_all), zs=np.concatenate(zs_all))
    return ax


__all__ = ["plot_mold_sections_2d", "plot_mold_sections_3d"]
