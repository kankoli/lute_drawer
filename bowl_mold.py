"""Helpers for generating physical bowl mold sections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from bowl_from_soundboard import Section


@dataclass(frozen=True)
class MoldSectionFace:
    """Single face of a physical mold board."""

    x: float
    y: np.ndarray
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
    board_thickness_mm: float = 30,
    lute=None,
) -> List[MoldSection]:
    """Generate mold boards that intersect all ribs at chosen spine locations."""

    if board_thickness_mm <= 0:
        raise ValueError("board_thickness_mm must be positive")
    if lute is None:
        raise ValueError("lute required to convert millimetres")
    mm_per_coordinate = lute.unit_in_mm() / lute.unit
    if mm_per_coordinate <= 0:
        raise ValueError("mm_per_coordinate must be positive")
    board_thickness_units = board_thickness_mm / mm_per_coordinate
    
    selected_indices = _select_section_indices(sections, n_stations)

    xs = np.array([sec.x for sec in sections], dtype=float)
    center_y = np.array([sec.center[0] for sec in sections], dtype=float)
    center_z = np.array([sec.center[1] for sec in sections], dtype=float)
    radii = np.array([sec.radius for sec in sections], dtype=float)

    neck_limit = float(xs[0])
    tail_limit = float(xs[-1])
    half_thickness = board_thickness_units / 2.0

    mold_sections: List[MoldSection] = []

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

            ordered = np.asarray(rib_points, dtype=float)
            order = np.argsort(ordered[:, 0])
            ordered = ordered[order]

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
                thickness=board_thickness_units,
                faces=tuple(faces),
            )
        )

    return mold_sections


__all__ = ["MoldSectionFace", "MoldSection", "build_mold_sections"]
