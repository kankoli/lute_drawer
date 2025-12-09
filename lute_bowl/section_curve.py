from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import warnings

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


@dataclass(frozen=True)
class CubicBezierSectionCurve(BaseSectionCurve):
    """
    Symmetric cubic Bézier passing through left, apex (at t=0.5), and right.

    Control points are chosen so P(0)=left, P(1)=right, and P(0.5)=apex with
    C1=C2 to keep the curve smooth and centred.
    """

    left: np.ndarray
    apex: np.ndarray
    right: np.ndarray
    control: np.ndarray
    center: np.ndarray
    radius: float

    @classmethod
    def from_span(
        cls,
        left_yz: np.ndarray,
        right_yz: np.ndarray,
        apex_yz: np.ndarray,
        *,
        control_scale: float = 1.0,
    ) -> "CubicBezierSectionCurve":
        left = np.asarray(left_yz, float)
        right = np.asarray(right_yz, float)
        apex = np.asarray(apex_yz, float)
        control = (apex - 0.125 * (left + right)) / 0.75
        if abs(control_scale - 1.0) > 1e-8:
            warnings.warn(
                "control_scale is ignored for symmetric Bézier sections; "
                "the control point is uniquely determined by left/apex/right.",
                RuntimeWarning,
            )
        try:
            center, radius = _circle_through_three_points_2d(left, right, apex)
        except Exception:
            center = np.array([(left[0] + right[0]) * 0.5, 0.0], dtype=float)
            radius = float(np.linalg.norm(left - right)) * 0.5
        return cls(left, apex, right, control, np.asarray(center, float), float(radius))

    @classmethod
    def degenerate(cls, center_yz: np.ndarray, apex_yz: np.ndarray, **_ignored) -> "CubicBezierSectionCurve":
        center = np.asarray(center_yz, float)
        apex = np.asarray(apex_yz, float)
        return cls(center.copy(), apex, center.copy(), center.copy(), center.copy(), 0.0)

    def angles(self) -> Tuple[float, float, float]:
        return (0.0, 1.0, 0.5)

    def _coeffs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P0 = self.left
        P1 = self.control
        P2 = self.control
        P3 = self.right
        a = -P0 + 3 * P1 - 3 * P2 + P3
        b = 3 * P0 - 6 * P1 + 3 * P2
        c = -3 * P0 + 3 * P1
        d = P0
        return a, b, c, d

    def point_at_angle(self, theta: float) -> np.ndarray:
        t = float(theta)
        a, b, c, d = self._coeffs()
        return ((a * t + b) * t + c) * t + d

    def angle_of_point(self, point_yz: np.ndarray) -> float:
        target = np.asarray(point_yz, float)
        ts = np.linspace(0.0, 1.0, 600)
        pts = np.vstack([self.point_at_angle(t) for t in ts])
        idx = int(np.argmin(np.linalg.norm(pts - target[None, :], axis=1)))
        return float(ts[idx])

    def _line_roots(self, origin: np.ndarray, direction: np.ndarray) -> list[float]:
        dy, dz = direction
        y0, z0 = origin
        a, b, c, d = self._coeffs()
        # y(t) = ay t^3 + by t^2 + cy t + dy
        ay, by, cy, dy0 = a[0], b[0], c[0], d[0]
        az, bz, cz, dz0 = a[1], b[1], c[1], d[1]
        # Solve dy*(z(t)-z0) - dz*(y(t)-y0) = 0 => cubic
        coeff3 = dy * az - dz * ay
        coeff2 = dy * bz - dz * by
        coeff1 = dy * cz - dz * cy
        coeff0 = dy * (dz0 - z0) - dz * (dy0 - y0)
        roots = np.roots([coeff3, coeff2, coeff1, coeff0])
        out: list[float] = []
        for r in roots:
            if abs(r.imag) > 1e-6:
                continue
            t = float(r.real)
            if -1e-6 <= t <= 1.0 + 1e-6:
                out.append(t)
        return out

    def intersect_with_plane(
        self,
        plane_normal: np.ndarray,
        plane_point: np.ndarray,
        section_x: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = np.asarray(plane_normal, float)
        p0 = np.asarray(plane_point, float)
        a, b, c, d = self._coeffs()

        # Solve dot(p(t) - p0, n) = 0 as a cubic in t.
        n0, n1, n2 = float(n[0]), float(n[1]), float(n[2])
        coeff3 = n1 * float(a[0]) + n2 * float(a[1])
        coeff2 = n1 * float(b[0]) + n2 * float(b[1])
        coeff1 = n1 * float(c[0]) + n2 * float(c[1])
        coeff0 = (
            n0 * float(section_x)
            + n1 * float(d[0])
            + n2 * float(d[1])
            - float(np.dot(p0, n))
        )

        roots_raw = np.roots([coeff3, coeff2, coeff1, coeff0])
        ts: list[float] = []
        for r in roots_raw:
            if abs(r.imag) > 1e-8:
                continue
            t = float(r.real)
            if -1e-9 <= t <= 1.0 + 1e-9:
                ts.append(min(1.0, max(0.0, t)))

        ts.sort()
        ts_unique: list[float] = []
        for t in ts:
            if not ts_unique or abs(t - ts_unique[-1]) > 1e-6:
                ts_unique.append(t)

        if not ts_unique:
            warnings.warn(
                "CubicBezierSectionCurve.intersect_with_plane fell back to closest point; no real roots on [0,1].",
                RuntimeWarning,
            )
            sample_ts = np.linspace(0.0, 1.0, 400)
            fvals = coeff3 * sample_ts**3 + coeff2 * sample_ts**2 + coeff1 * sample_ts + coeff0
            t_closest = float(sample_ts[int(np.argmin(np.abs(fvals)))])
            ts_unique = [t_closest]

        if len(ts_unique) == 1:
            ts_unique = [ts_unique[0], ts_unique[0]]
        elif len(ts_unique) > 2:
            ts_unique = [ts_unique[0], ts_unique[-1]]

        pts = [self.point_at_angle(t) for t in ts_unique[:2]]
        pts_sorted = sorted(pts, key=lambda p: p[0])
        return pts_sorted[0], pts_sorted[1]

    def divide(self, count: int, *, mode: str = "angle") -> list[np.ndarray]:
        if count < 2:
            raise ValueError("count must be at least 2")
        ts = np.linspace(0.0, 1.0, count)
        return [self.point_at_angle(t) for t in ts]

    def divide_between_points(
        self,
        start_yz: np.ndarray,
        end_yz: np.ndarray,
        count: int,
        *,
        mode: str = "angle",
    ) -> list[np.ndarray]:
        if count < 2:
            raise ValueError("count must be at least 2")
        t_start = self.angle_of_point(start_yz)
        t_end = self.angle_of_point(end_yz)
        ts = np.linspace(t_start, t_end, count)
        return [self.point_at_angle(t) for t in ts]

    def sample_points(self, n: int = 1000) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, max(2, int(n)))
        pts = [self.point_at_angle(t) for t in ts]
        return np.asarray(pts, float)


@dataclass(frozen=True)
class CosineArchSectionCurve(BaseSectionCurve):
    """
    Boxy “inverted tupperware” arch with rounded corners.

    Y moves linearly between the sides with a bump to hit the apex Y,
    Z follows a superellipse profile to stay 0 at the sides and flatten near the top.
    Higher shape_power => squarer top with tighter corner radii.
    """

    left: np.ndarray
    right: np.ndarray
    apex: np.ndarray
    shape_power: float
    center: np.ndarray
    radius: float

    @classmethod
    def from_span(
        cls,
        left_yz: np.ndarray,
        right_yz: np.ndarray,
        apex_yz: np.ndarray,
        *,
        shape_power: float = 2.2,
        cos_power: float | None = None,
    ) -> "CosineArchSectionCurve":
        left = np.asarray(left_yz, float)
        right = np.asarray(right_yz, float)
        apex = np.asarray(apex_yz, float)
        if left[0] > right[0]:
            left, right = right, left
        power = float(shape_power if cos_power is None else cos_power)
        power = max(power, 1e-6)
        center = np.array([0.5 * (left[0] + right[0]), 0.0], dtype=float)
        radius = abs(right[0] - left[0]) * 0.5
        return cls(left, right, apex, power, center, radius)

    @classmethod
    def degenerate(cls, center_yz: np.ndarray, apex_yz: np.ndarray, **_ignored) -> "CosineArchSectionCurve":
        center = np.asarray(center_yz, float)
        apex = np.asarray(apex_yz, float)
        return cls(center.copy(), center.copy(), apex, 1.0, center.copy(), 0.0)

    def angles(self) -> Tuple[float, float, float]:
        return (0.0, 1.0, 0.5)

    def _bump(self, t: float) -> float:
        """Parabolic bump peaking at t=0.5, zero at ends."""
        return 4.0 * t * (1.0 - t)

    def _yz_at(self, t: float) -> np.ndarray:
        t_clamped = min(1.0, max(0.0, float(t)))
        y_span = self.right[0] - self.left[0]
        y_base = self.left[0] + y_span * t_clamped
        mid_y = 0.5 * (self.left[0] + self.right[0])
        y = y_base + (self.apex[0] - mid_y) * self._bump(t_clamped)

        x_rel = abs(2.0 * t_clamped - 1.0)  # 0 at center, 1 at sides
        base = max(0.0, 1.0 - (x_rel ** self.shape_power))
        z = float(self.apex[1]) * (base ** (1.0 / self.shape_power))
        return np.array([y, z], dtype=float)

    def point_at_angle(self, theta: float) -> np.ndarray:
        return self._yz_at(theta)

    def angle_of_point(self, point_yz: np.ndarray) -> float:
        target = np.asarray(point_yz, float)
        ts = np.linspace(0.0, 1.0, 800)
        pts = np.vstack([self._yz_at(t) for t in ts])
        idx = int(np.argmin(np.linalg.norm(pts - target[None, :], axis=1)))
        return float(ts[idx])

    def _find_roots(self, func, *, samples: int = 800, tol: float = 1e-9) -> list[float]:
        us = np.linspace(0.0, 1.0, samples + 1)
        vals = [float(func(u)) for u in us]
        roots: list[float] = []
        for i in range(len(us) - 1):
            u0, u1 = us[i], us[i + 1]
            v0, v1 = vals[i], vals[i + 1]
            if abs(v0) <= tol:
                roots.append(u0)
            if v0 == 0.0:
                continue
            if v0 * v1 <= 0.0:
                a, b = u0, u1
                fa, fb = v0, v1
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    fm = float(func(mid))
                    if abs(fm) < tol or abs(b - a) < tol:
                        a = b = mid
                        break
                    if fa * fm <= 0.0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                roots.append(0.5 * (a + b))
        if abs(vals[-1]) <= tol:
            roots.append(us[-1])
        roots.sort()
        dedup: list[float] = []
        for r in roots:
            if not dedup or abs(r - dedup[-1]) > 1e-5:
                dedup.append(r)
        return dedup

    def intersect_with_plane(
        self,
        plane_normal: np.ndarray,
        plane_point: np.ndarray,
        section_x: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = np.asarray(plane_normal, float)
        p0 = np.asarray(plane_point, float)
        section_term = n[0] * float(section_x)
        offset = float(np.dot(p0, n))

        def _f(u: float) -> float:
            y, z = self._yz_at(u)
            return section_term + n[1] * y + n[2] * z - offset

        roots = self._find_roots(_f)
        if not roots:
            # Fall back to closest approach to avoid crashing the build.
            samples = np.linspace(0.0, 1.0, 800)
            vals = [abs(_f(u)) for u in samples]
            t_best = float(samples[int(np.argmin(vals))])
            warnings.warn(
                "CosineArchSectionCurve.intersect_with_plane found no roots; using closest sampled point.",
                RuntimeWarning,
            )
            roots = [t_best, t_best]
        elif len(roots) == 1:
            roots = [roots[0], roots[0]]
        elif len(roots) > 2:
            roots = [roots[0], roots[-1]]

        pts = [self._yz_at(t) for t in roots[:2]]
        pts_sorted = sorted(pts, key=lambda p: p[0])
        return pts_sorted[0], pts_sorted[1]

    def divide(self, count: int, *, mode: str = "angle") -> list[np.ndarray]:
        if count < 2:
            raise ValueError("count must be at least 2")
        ts = np.linspace(0.0, 1.0, count)
        return [self._yz_at(t) for t in ts]

    def divide_between_points(
        self,
        start_yz: np.ndarray,
        end_yz: np.ndarray,
        count: int,
        *,
        mode: str = "angle",
    ) -> list[np.ndarray]:
        if count < 2:
            raise ValueError("count must be at least 2")
        t_start = self.angle_of_point(start_yz)
        t_end = self.angle_of_point(end_yz)
        ts = np.linspace(t_start, t_end, count)
        return [self._yz_at(t) for t in ts]

    def sample_points(self, n: int = 800) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, max(2, int(n)))
        return np.asarray([self._yz_at(t) for t in ts], float)


__all__ = ["BaseSectionCurve", "CircularSectionCurve", "CubicBezierSectionCurve", "CosineArchSectionCurve"]
