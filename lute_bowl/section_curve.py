from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_EPS = 1e-9


def _circle_through_three_points_2d(P1, P2, P3):
    """Circle through 3 points in the YZ-plane (inputs are 2D [Y,Z] coords)."""
    P1 = np.asarray(P1, float)
    P2 = np.asarray(P2, float)
    P3 = np.asarray(P3, float)
    mid12 = 0.5 * (P1 + P2)
    mid13 = 0.5 * (P1 + P3)
    d12 = P2 - P1
    d13 = P3 - P1
    area2 = d12[0] * d13[1] - d12[1] * d13[0]
    if abs(area2) < 1e-12:
        raise ValueError("Collinear points, no unique circle.")
    n12 = np.array([-d12[1], d12[0]])
    n13 = np.array([-d13[1], d13[0]])
    A = np.column_stack([n12, -n13])
    b = mid13 - mid12
    t, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    C = mid12 + t * n12
    r = float(np.linalg.norm(C - P1))
    return C, r


class BaseSectionCurve(ABC):
    @property
    def is_degenerate(self) -> bool:
        return float(self.radius) <= _EPS

    @classmethod
    @abstractmethod
    def from_span(cls, left_yz: np.ndarray, right_yz: np.ndarray, apex_yz: np.ndarray) -> "BaseSectionCurve":
        ...

    @classmethod
    @abstractmethod
    def degenerate(cls, center_yz: np.ndarray, apex_yz: np.ndarray) -> "BaseSectionCurve":
        ...

    @abstractmethod
    def angles(self) -> Tuple[float, float, float]:
        ...

    @abstractmethod
    def point_at_angle(self, theta: float) -> np.ndarray:
        ...

    @abstractmethod
    def angle_of_point(self, point_yz: np.ndarray) -> float:
        ...

    @abstractmethod
    def intersect_with_direction(self, origin_yz: np.ndarray, direction_yz: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def intersect_with_plane(
        self,
        plane_normal: np.ndarray,
        plane_point: np.ndarray,
        section_x: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def divide(self, count: int, *, mode: str = "angle") -> list[np.ndarray]:
        ...

    @abstractmethod
    def divide_between_points(
        self,
        start_yz: np.ndarray,
        end_yz: np.ndarray,
        count: int,
        *,
        mode: str = "angle",
    ) -> list[np.ndarray]:
        ...

    def sample_points(self, n: int = 200) -> np.ndarray:
        """Return sampled YZ points along the curve for plotting/debug."""
        pts = self.divide(max(2, int(n)), mode="angle")
        return np.asarray(pts, float)


@dataclass(frozen=True)
class CircularSectionCurve(BaseSectionCurve):
    center: np.ndarray
    radius: float
    apex: np.ndarray

    @classmethod
    def from_span(cls, left_yz: np.ndarray, right_yz: np.ndarray, apex_yz: np.ndarray) -> "CircularSectionCurve":
        """Build a circular curve through the left, right, and apex points."""
        center, radius = _circle_through_three_points_2d(left_yz, right_yz, apex_yz)
        return cls(np.asarray(center, float), float(radius), np.asarray(apex_yz, float))

    @classmethod
    def degenerate(cls, center_yz: np.ndarray, apex_yz: np.ndarray) -> "CircularSectionCurve":
        """Create a zero-radius section centred on the spine."""
        return cls(np.asarray(center_yz, float), 0.0, np.asarray(apex_yz, float))

    def angles(self) -> Tuple[float, float, float]:
        """Return (left, right, apex) angles in radians around the center."""
        r = float(self.radius)
        if r <= _EPS:
            raise ValueError("Section radius must be positive to derive angles.")

        cy, cz = map(float, self.center)
        ay, az = map(float, self.apex)

        s = -cz / r
        if abs(s) > 1.0:
            raise ValueError("Section apex is incompatible with fitted circle.")

        theta_z = float(np.arcsin(s))
        candidates = [theta_z, np.pi - theta_z]
        y_candidates = [cy + r * np.cos(theta) for theta in candidates]
        order = np.argsort(y_candidates)
        theta_left = candidates[int(order[0])]
        theta_right = candidates[int(order[1])]
        theta_apex = float(math.atan2(az - cz, ay - cy))
        return theta_left, theta_right, theta_apex

    def point_at_angle(self, theta: float) -> np.ndarray:
        cy, cz = map(float, self.center)
        r = float(self.radius)
        return np.array([cy + r * math.cos(theta), cz + r * math.sin(theta)], dtype=float)

    def angle_of_point(self, point_yz: np.ndarray) -> float:
        cy, cz = map(float, self.center)
        py, pz = map(float, point_yz)
        return float(math.atan2(pz - cz, py - cy))

    def intersect_with_direction(self, origin_yz: np.ndarray, direction_yz: np.ndarray) -> np.ndarray:
        """Return the furthest non-negative intersection along the given direction."""
        if self.is_degenerate:
            return np.asarray(origin_yz, float)

        cy, cz = map(float, self.center)
        r = float(self.radius)
        dy, dz = map(float, direction_yz)
        if abs(dy) < _EPS and abs(dz) < _EPS:
            raise ValueError("Direction vector is degenerate.")

        coeff_a = dy * dy + dz * dz
        coeff_b = 2.0 * (dy * (origin_yz[0] - cy) + dz * (origin_yz[1] - cz))
        coeff_c = (origin_yz[0] - cy) ** 2 + (origin_yz[1] - cz) ** 2 - r * r
        discriminant = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c

        if discriminant < -1e-7:
            raise RuntimeError("Planar rib plane misses section curve.")

        discriminant = max(0.0, discriminant)
        sqrt_disc = math.sqrt(discriminant)

        s_candidates = [
            (-coeff_b - sqrt_disc) / (2.0 * coeff_a),
            (-coeff_b + sqrt_disc) / (2.0 * coeff_a),
        ]
        s_valid = [s for s in s_candidates if s >= -1e-9]
        if not s_valid:
            s = max(s_candidates, key=lambda val: val)
            if s < 0.0:
                s = 0.0
        else:
            s = max(s_valid)
            if s < 0.0:
                s = 0.0

        return np.array([origin_yz[0] + s * dy, origin_yz[1] + s * dz], dtype=float)

    def intersect_with_plane(
        self,
        plane_normal: np.ndarray,
        plane_point: np.ndarray,
        section_x: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Intersect the section with a 3D plane normal; returns two YZ points sorted by Y."""
        if self.is_degenerate:
            raise RuntimeError("Cannot intersect a degenerate section with a plane.")

        cy, cz = map(float, self.center)
        r = float(self.radius)
        A = plane_normal[1] * r
        B = plane_normal[2] * r
        C = plane_normal[0] * (section_x - plane_point[0]) + plane_normal[1] * (cy - plane_point[1]) + plane_normal[2] * (
            cz - plane_point[2]
        )
        R = math.hypot(A, B)
        if R <= _EPS:
            raise RuntimeError("Plane is parallel to section curve.")
        arg = -C / R
        arg = max(-1.0, min(1.0, arg))
        base = math.atan2(B, A)
        delta = math.acos(arg)
        t1 = base + delta
        t2 = base - delta
        p1 = np.array([cy + r * math.cos(t1), cz + r * math.sin(t1)], dtype=float)
        p2 = np.array([cy + r * math.cos(t2), cz + r * math.sin(t2)], dtype=float)
        pts = sorted([p1, p2], key=lambda p: p[0])
        return pts[0], pts[1]

    def divide(self, count: int, *, mode: str = "angle") -> list[np.ndarray]:
        """Return evenly spaced YZ points along the curve."""
        if count < 2:
            raise ValueError("count must be at least 2")
        if self.is_degenerate:
            return [np.array(self.center, dtype=float) for _ in range(count)]
        if mode not in ("angle", "arc_length"):
            raise ValueError(f"Unsupported division mode {mode}")
        theta_left, theta_right, _ = self.angles()
        thetas = np.linspace(theta_left, theta_right, count)
        return [self.point_at_angle(theta) for theta in thetas]

    def divide_between_points(
        self,
        start_yz: np.ndarray,
        end_yz: np.ndarray,
        count: int,
        *,
        mode: str = "angle",
    ) -> list[np.ndarray]:
        """Return evenly spaced YZ points between two points on the curve."""
        if count < 2:
            raise ValueError("count must be at least 2")
        if self.is_degenerate:
            return [np.array(self.center, dtype=float) for _ in range(count)]

        if mode not in ("angle", "arc_length"):
            raise ValueError(f"Unsupported division mode {mode}")

        theta_start = self.angle_of_point(start_yz)
        theta_end = self.angle_of_point(end_yz)
        thetas = np.linspace(theta_start, theta_end, count)
        return [self.point_at_angle(theta) for theta in thetas]

    def sample_points(self, n: int = 200) -> np.ndarray:
        pts = self.divide(max(2, int(n)), mode="angle")
        return np.asarray(pts, float)


__all__ = ["BaseSectionCurve", "CircularSectionCurve"]
