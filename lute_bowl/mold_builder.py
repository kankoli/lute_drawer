"""Helpers for generating physical bowl mold sections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .bowl_from_soundboard import Section


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


def _select_section_indices(xs: np.ndarray, targets: np.ndarray) -> List[int]:
    selected: List[int] = []
    used: set[int] = set()

    for target in targets:
        order = np.argsort(np.abs(xs - target))
        chosen = None
        for idx in order:
            idx_int = int(idx)
            if idx_int not in used:
                chosen = idx_int
                break
        if chosen is None:
            raise ValueError("Unable to assign unique section index for target position")
        selected.append(chosen)
        used.add(chosen)

    return selected


def build_mold_sections(
    *,
    sections: Sequence[Section],
    ribs: Sequence[np.ndarray],
    n_stations: int,
    board_thickness_mm: float = 30.0,
    lute=None,
    neck_limit: float | None = None,
    tail_limit: float | None = None,
) -> List[MoldSection]:
    """Generate mold boards that intersect all ribs at chosen spine locations."""

    if n_stations <= 0:
        raise ValueError("n_stations must be positive")

    if board_thickness_mm <= 0:
        raise ValueError("board_thickness_mm must be positive")

    xs = np.array([sec.x for sec in sections], dtype=float)
    if len(xs) < n_stations + 2:
        raise ValueError("Increase n_sections when sampling the bowl to support these stations")


    scale = lute.unit_in_mm() / lute.unit

    board_thickness_units = board_thickness_mm / scale

    neck_default = float(xs[0])
    tail_default = float(xs[-1])

    neck_limit_units = neck_default + float(neck_limit) * lute.unit
    tail_limit_units = tail_default - float(tail_limit) * lute.unit

    if neck_limit_units >= tail_limit_units:
        raise ValueError("neck_limit must be less than tail_limit")

    half_thickness = board_thickness_units / 2.0
    effective_start = neck_limit_units + half_thickness
    effective_end = tail_limit_units - half_thickness

    if effective_end <= effective_start:
        raise ValueError("Thickness and limits leave no usable span for mold sections")

    targets = np.linspace(effective_start, effective_end, n_stations)
    _select_section_indices(xs, targets)

    mold_sections: List[MoldSection] = []
    domain_min = float(xs[0])
    domain_max = float(xs[-1])

    for i, target_center in enumerate(targets):

        if i == 0:
            desired_left = neck_limit_units
            desired_right = desired_left + board_thickness_units
        elif i == n_stations - 1:
            desired_right = tail_limit_units
            desired_left = desired_right - board_thickness_units
        else:
            desired_left = target_center - half_thickness
            desired_right = target_center + half_thickness

        faces: List[MoldSectionFace] = []
        x_left = float(desired_left)
        x_right = float(desired_right)

        if x_left < domain_min - 1e-9 or x_right > domain_max + 1e-9:
            raise ValueError("Mold faces fall outside the sampled rib domain; adjust limits or thickness.")
        if x_right <= x_left + 1e-9:
            raise ValueError("Mold thickness collapsed; adjust neck/tail limits or thickness.")

        for x_face in (x_left, x_right):
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
                center_x=float(0.5 * (x_left + x_right)),
                thickness=float(x_right - x_left),
                faces=tuple(faces),
            )
        )

    return mold_sections



__all__ = ["MoldSectionFace", "MoldSection", "build_mold_sections"]
