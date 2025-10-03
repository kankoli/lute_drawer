"""Helpers for generating physical bowl mold sections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np

from bowl_from_soundboard import Section, build_bowl_for_lute
from bowl_top_curves import SimpleAmplitudeCurve


@dataclass(frozen=True)
class MoldSectionFace:
    """Single face of a physical mold board."""

    x: float
    y: np.ndarray  # ordered by Y across the soundboard
    z: np.ndarray


@dataclass(frozen=True)
class MoldSection:
    """Cross-sectional board characterised by its two faces."""

    center_x: float
    thickness: float
    faces: tuple[MoldSectionFace, MoldSectionFace]


def _select_section_indices(sections: Sequence[Section], n_stations: int) -> List[int]:
    if n_stations <= 0:
        raise ValueError("n_stations must be positive")

    if len(sections) <= 2:
        raise ValueError("Not enough sections to select interior stations")

    usable = np.arange(1, len(sections) - 1)
    if n_stations > usable.size:
        raise ValueError(
            f"Requested {n_stations} stations but only {usable.size} interior samples are available"
        )

    positions = np.linspace(0, usable.size - 1, n_stations)
    indices = np.unique(usable[positions.astype(int)])
    return indices.tolist()


def build_mold_sections(
    *,
    sections: Sequence[Section],
    ribs: Sequence[np.ndarray],
    n_stations: int,
    board_thickness: float,
) -> List[MoldSection]:
    """Generate mold boards that intersect all ribs at chosen spine locations."""

    if board_thickness <= 0:
        raise ValueError("board_thickness must be positive")
    selected_indices = _select_section_indices(sections, n_stations)

    mold_sections: List[MoldSection] = []
    tol = 1e-6

    xs = np.array([sec.x for sec in sections], dtype=float)
    center_y = np.array([sec.center[0] for sec in sections], dtype=float)
    center_z = np.array([sec.center[1] for sec in sections], dtype=float)
    radii = np.array([sec.radius for sec in sections], dtype=float)

    neck_limit = float(xs[0])
    tail_limit = float(xs[-1])

    half_thickness = board_thickness / 2.0

    for idx in selected_indices:
        center_x = float(sections[idx].x)
        face_positions = (
            max(neck_limit, center_x - half_thickness),
            min(tail_limit, center_x + half_thickness),
        )

        faces: List[MoldSectionFace] = []
        for x_face in face_positions:
            rib_points = []
            for rib in ribs:
                y_interp = np.interp(x_face, rib[:, 0], rib[:, 1])
                z_interp = np.interp(x_face, rib[:, 0], rib[:, 2])
                rib_points.append((y_interp, z_interp))

            rib_points_array = np.asarray(rib_points, dtype=float)
            order = np.argsort(rib_points_array[:, 0])
            ordered = rib_points_array[order]

            faces.append(
                MoldSectionFace(
                    x=x_face,
                    y=ordered[:, 0],
                    z=ordered[:, 1],
                )
            )

        mold_sections.append(
            MoldSection(
                center_x=center_x,
                thickness=board_thickness,
                faces=tuple(faces),
            )
        )

    return mold_sections


def plot_mold_sections_2d(
    sections: Sequence[MoldSection],
    *,
    ax=None,
    show_rib_points: bool = True,
    invert_z: bool = True,
):
    """2D visualisation for mold section faces."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    for idx, section in enumerate(sections):
        color = None
        for i, face in enumerate(section.faces):
            label = f"X={face.x:.1f}" if i == 0 else None
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

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

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
                ax.scatter(
                    np.full(face.y.shape, face.x),
                    face.y,
                    face.z,
                    s=12,
                    color=section_color,
                )

    ax.set_xlabel("X (along spine)")
    ax.set_ylabel("Y (across)")
    ax.set_zlabel("Z (depth)")
    return ax


__all__ = [
    "MoldSectionFace",
    "MoldSection",
    "build_mold_sections",
    "plot_mold_sections_2d",
    "plot_mold_sections_3d",
]
