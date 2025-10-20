"""Planar rib bowl generator with configurable end blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .bowl_from_soundboard import (
    Section,
    _resolve_top_curve,
    _sample_section,
    _spine_point_at_X,
)
from .bowl_top_curves import SimpleAmplitudeCurve

_EX = np.array([1.0, 0.0, 0.0], dtype=float)
_EPS = 1e-9


@dataclass(frozen=True)
class PlanarRib:
    normal: np.ndarray
    direction: np.ndarray


def _spine_point_xyz(lute, x: float) -> np.ndarray:
    y = float(_spine_point_at_X(lute, x))
    return np.array([float(x), y, 0.0], dtype=float)


def _section_angles(section: Section):
    _, center, radius, apex = section
    r = float(radius)
    if r <= _EPS:
        raise ValueError("Section radius must be positive to derive angles.")

    cy, cz = map(float, center)
    ay, az = map(float, apex)

    s = -cz / r
    if abs(s) > 1.0:
        raise ValueError("Section apex is incompatible with fitted circle.")

    theta_z = float(np.arcsin(s))
    candidates = [theta_z, np.pi - theta_z]
    y_candidates = [cy + r * np.cos(theta) for theta in candidates]
    order = np.argsort(y_candidates)
    theta_left = candidates[int(order[0])]
    theta_right = candidates[int(order[1])]
    theta_apex = float(np.arctan2(az - cz, ay - cy))
    return theta_left, theta_right, theta_apex


def _derive_planar_ribs(
    lute,
    sections: Sequence[Section],
    n_ribs: int,
    x_start: float,
    x_end: float,
) -> List[np.ndarray]:
    if n_ribs < 1:
        raise ValueError("n_ribs must be at least 1.")

    # Reference section: choose the one with maximum radius for numerical stability.
    radii = [float(section.radius) for section in sections]
    ref_idx = int(np.argmax(radii))
    if radii[ref_idx] <= _EPS:
        raise ValueError("Unable to locate a section with positive radius.")

    ref_section = sections[ref_idx]
    theta_left, theta_right, _ = _section_angles(ref_section)
    rib_count = n_ribs + 1
    thetas = np.linspace(theta_left, theta_right, rib_count)

    x_ref = float(ref_section.x)
    cy_ref, cz_ref = map(float, ref_section.center)
    r_ref = float(ref_section.radius)
    spine_y_ref = float(_spine_point_at_X(lute, x_ref))
    spine_ref = np.array([x_ref, spine_y_ref, 0.0], dtype=float)

    spine_start = _spine_point_xyz(lute, x_start)
    spine_end = _spine_point_xyz(lute, x_end)
    spine_vector = spine_end - spine_start

    if np.linalg.norm(spine_vector) <= _EPS:
        raise ValueError("Spine vector collapsed after applying end blocks.")

    ribs = [[] for _ in range(rib_count)]
    rib_planes: List[PlanarRib] = []

    for theta in thetas:
        y_ref = cy_ref + r_ref * np.cos(theta)
        z_ref = cz_ref + r_ref * np.sin(theta)
        reference_point = np.array([x_ref, y_ref, z_ref], dtype=float)

        plane_normal = np.cross(spine_vector, reference_point - spine_start)
        norm_len = np.linalg.norm(plane_normal)
        if norm_len <= _EPS:
            raise ValueError("Degenerate plane encountered while constructing rib.")
        plane_normal /= norm_len

        direction = np.cross(plane_normal, _EX)
        dir_len = np.sqrt(direction[1] ** 2 + direction[2] ** 2)
        if dir_len <= _EPS:
            raise ValueError("Failed to derive planar direction for rib.")

        delta_ref = reference_point - spine_ref
        s_ref = (delta_ref[1] * direction[1] + delta_ref[2] * direction[2]) / (dir_len**2)
        if s_ref < 0.0:
            plane_normal = -plane_normal
            direction = -direction

        rib_planes.append(PlanarRib(plane_normal, direction))

    for section in sections:
        x = float(section.x)
        cy, cz = map(float, section.center)
        r = float(section.radius)
        spine_y = float(_spine_point_at_X(lute, x))
        base_yz = np.array([spine_y, 0.0], dtype=float)

        if r <= _EPS:
            for points in ribs:
                points.append(np.array([x, base_yz[0], base_yz[1]], dtype=float))
            continue

        for plane, points in zip(rib_planes, ribs):
            dy, dz = float(plane.direction[1]), float(plane.direction[2])
            coeff_a = dy * dy + dz * dz
            coeff_b = 2.0 * (dy * (base_yz[0] - cy) + dz * (base_yz[1] - cz))
            coeff_c = (base_yz[0] - cy) ** 2 + (base_yz[1] - cz) ** 2 - r * r
            discriminant = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c

            if discriminant < -1e-7:
                raise RuntimeError(f"Planar rib plane misses section circle at X={x:.6f}")

            discriminant = max(0.0, discriminant)
            sqrt_disc = np.sqrt(discriminant)

            s_candidates = [
                (-coeff_b - sqrt_disc) / (2.0 * coeff_a),
                (-coeff_b + sqrt_disc) / (2.0 * coeff_a),
            ]
            s_valid = [s for s in s_candidates if s >= -1e-9]
            if not s_valid:
                # fallback to the candidate that is furthest in the target direction
                s = max(s_candidates, key=lambda val: val)
                if s < 0.0:
                    s = 0.0
            else:
                s = max(s_valid)
                if s < 0.0:
                    s = 0.0

            y = base_yz[0] + s * dy
            z = base_yz[1] + s * dz
            points.append(np.array([x, y, z], dtype=float))

    return [np.asarray(points, dtype=float) for points in ribs]


def build_planar_bowl_for_lute(
    lute,
    *,
    n_ribs: int = 13,
    n_sections: int = 200,
    top_curve=None,
    upper_block_units: float = 1.0,
    lower_block_units: float = 0.1,
    debug: bool = False,
) -> tuple[list[Section], List[np.ndarray]]:
    """Build a bowl with planar ribs bounded by end blocks."""
    if n_sections < 2:
        raise ValueError("n_sections must be at least 2.")

    if top_curve is None:
        top_curve = SimpleAmplitudeCurve

    z_top = _resolve_top_curve(lute, top_curve)

    unit = float(getattr(lute, "unit", 1.0))
    start_x = float(lute.form_top.x) + float(upper_block_units) * unit
    end_x = float(lute.form_bottom.x) - float(lower_block_units) * unit

    if start_x >= end_x - _EPS:
        raise ValueError("End blocks overlap the bowl span; adjust block sizes.")

    xs = np.linspace(start_x, end_x, n_sections)
    sections: list[Section] = []

    for X in xs:
        try:
            section = _sample_section(lute, float(X), z_top, debug=debug)
        except Exception as exc:  # pragma: no cover - diagnostic aid
            raise RuntimeError(f"Failed to sample section at X={float(X):.6f}") from exc
        if section is None:
            raise RuntimeError(f"No valid section geometry at X={float(X):.6f}")
        sections.append(section)

    ribs = _derive_planar_ribs(lute, sections, n_ribs, start_x, end_x)

    return sections, ribs


__all__ = ["build_planar_bowl_for_lute"]
