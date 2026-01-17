import math

import numpy as np

from lute_bowl.bowl_from_soundboard import (
    Section,
    compute_bowl_inner_volume,
    compute_equivalent_flat_side_depth,
    compute_soundboard_outline_area,
)
from lute_bowl.section_curve import CircularSectionCurve


def test_compute_bowl_inner_volume_matches_half_cylinder():
    radius = 2.0
    left = np.array([-radius, 0.0])
    right = np.array([radius, 0.0])
    apex = np.array([0.0, radius])
    curve = CircularSectionCurve.from_span(left, right, apex)

    sections = [
        Section(3.0, curve),
        Section(0.0, curve),
    ]

    volume = compute_bowl_inner_volume(sections, samples_per_section=800)
    expected_area = 0.5 * math.pi * radius * radius
    expected_volume = expected_area * 3.0

    assert abs(volume - expected_volume) / expected_volume < 5e-3


class _HalfCircleArc:
    def __init__(self, radius: float, side: str):
        self.radius = float(radius)
        self.side = side

    def sample_points(self, n: int = 100) -> np.ndarray:
        if self.side == "right":
            angles = np.linspace(math.pi / 2.0, -math.pi / 2.0, n)
            x = self.radius * np.cos(angles)
            y = self.radius * np.sin(angles)
            return np.column_stack([x, y])
        if self.side == "left":
            angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, n)
            x = -self.radius * np.cos(angles)
            y = self.radius * np.sin(angles)
            return np.column_stack([x, y])
        raise ValueError(f"Unknown arc side: {self.side}")


class _DummyCircularLute:
    def __init__(self, radius: float):
        self.final_arcs = [_HalfCircleArc(radius, "right")]
        self.final_reflected_arcs = [_HalfCircleArc(radius, "left")]


def test_compute_soundboard_outline_area_matches_circle():
    radius = 2.5
    lute = _DummyCircularLute(radius)
    area = compute_soundboard_outline_area(lute, samples_per_arc=800)
    expected = math.pi * radius * radius
    assert abs(area - expected) / expected < 5e-3


def test_compute_equivalent_flat_side_depth_matches_half_cylinder():
    radius = 2.0
    length = 4.0
    curve = CircularSectionCurve.from_span(
        np.array([-radius, 0.0]),
        np.array([radius, 0.0]),
        np.array([0.0, radius]),
    )
    sections = [Section(0.0, curve), Section(length, curve)]
    lute = _DummyCircularLute(radius)
    depth = compute_equivalent_flat_side_depth(
        lute,
        sections,
        samples_per_section=800,
        samples_per_arc=800,
    )
    expected = 0.5 * length
    assert abs(depth - expected) / expected < 5e-3
