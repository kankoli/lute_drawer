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
    mm_per_coordinate: float | None = None,
    lute=None,
    neck_limit_mm: float | None = None,
    tail_limit_mm: float | None = None,
) -> List[MoldSection]:
    """Generate mold boards that intersect all ribs at chosen spine locations."""

    if n_stations <= 0:
        raise ValueError("n_stations must be positive")

    scale = mm_per_coordinate
    if scale is None and lute is not None:
        scale = lute.unit_in_mm() / lute.unit

    if board_thickness_mm <= 0:
        raise ValueError("board_thickness_mm must be positive")
    if scale is None or scale <= 0:
        raise ValueError("mm_per_coordinate or lute required to convert millimetres")

    board_thickness_units = board_thickness_mm / scale

    xs = np.array([sec.x for sec in sections], dtype=float)
    if len(xs) < n_stations + 2:
        raise ValueError("Increase n_sections when sampling the bowl to support these stations")

    neck_default = float(xs[0])
    tail_default = float(xs[-1])

    def _convert_from_top(mm_value: float | None, default: float) -> float:
        if mm_value is None:
            return default
        return default + (mm_value / scale)

    def _convert_from_bottom(mm_value: float | None, default: float) -> float:
        if mm_value is None:
            return default
        return default - (abs(mm_value) / scale)

    neck_limit_units = _convert_from_top(neck_limit_mm, neck_default)
    tail_limit_units = _convert_from_bottom(tail_limit_mm, tail_default)

    if neck_limit_units >= tail_limit_units:
        raise ValueError("neck_limit must be less than tail_limit")

    half_thickness = board_thickness_units / 2.0
    effective_start = neck_limit_units + half_thickness
    effective_end = tail_limit_units - half_thickness

    if effective_end <= effective_start:
        raise ValueError("Thickness and limits leave no usable span for mold sections")

    targets = np.linspace(effective_start, effective_end, n_stations)
    selected_indices = _select_section_indices(xs, targets)

    mold_sections: List[MoldSection] = []

    for i, (idx, target_center) in enumerate(zip(selected_indices, targets, strict=False)):
        center_x = float(sections[idx].x)

        if i == 0:
            desired_left = neck_limit_units
            desired_right = desired_left + board_thickness_units
        elif i == len(selected_indices) - 1:
            desired_right = tail_limit_units
            desired_left = desired_right - board_thickness_units
        else:
            desired_left = target_center - half_thickness
            desired_right = target_center + half_thickness

        left_idx = int(np.clip(np.searchsorted(xs, desired_left), 0, len(xs) - 1))
        if left_idx > 0 and abs(xs[left_idx - 1] - desired_left) < abs(xs[left_idx] - desired_left):
            left_idx -= 1

        right_idx = int(np.clip(np.searchsorted(xs, desired_right), 0, len(xs) - 1))
        if right_idx + 1 < len(xs) and abs(xs[right_idx + 1] - desired_right) < abs(xs[right_idx] - desired_right):
            right_idx += 1

        if left_idx == right_idx:
            raise ValueError(
                "Board thickness is too small relative to sampled sections; increase n_sections when sampling the bowl."
            )

        face_indices = [left_idx, right_idx]
        actual_thickness = float(abs(xs[right_idx] - xs[left_idx]))

        faces: List[MoldSectionFace] = []
        for face_idx in face_indices:
            x_face = float(xs[face_idx])
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
                thickness=actual_thickness,
                faces=tuple(faces),
            )
        )

    return mold_sections


__all__ = ["MoldSectionFace", "MoldSection", "build_mold_sections"]
