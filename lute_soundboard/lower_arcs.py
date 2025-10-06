"""Lower-arc construction helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TangentParameter:
    geo: GeoDSL
    previous_circle: object
    point: object
    radius: float
    closest_point: object

    def calculate(self):
        line = self.geo.line(self.previous_circle.center, self.point)
        prev_circle = (
            self.previous_circle.tangent_circle
            if isinstance(self.previous_circle, TangentParameter)
            else self.previous_circle
        )
        circle, point = self.geo.get_tangent_circle(prev_circle, line, self.radius, self.closest_point, True)
        return circle, point


class LowerArcBuilder(ABC):
    @abstractmethod
    def tangent_parameters(self, lute) -> List[TangentParameter]:
        """Return tangent configuration for the intermediate arcs."""

    @abstractmethod
    def blender_radius(self, lute) -> float:
        """Radius used for the final blend circle."""

    def build(self, lute) -> None:
        tangent_circles = [lute.top_arc_circle]
        tangent_points = [lute.form_top]
        current_circle = lute.top_arc_circle

        for param in self.tangent_parameters(lute):
            circle, point = param.calculate()
            tangent_circles.append(circle)
            tangent_points.append(point)
            current_circle = circle

        blender_radius = self._resolve_blender_radius(lute, current_circle, lute.bottom_arc_circle)
        blender_circle, p1, p2 = lute.geo.blend_two_circles(
            blender_radius,
            current_circle,
            lute.bottom_arc_circle,
        )
        tangent_circles.extend([blender_circle, lute.bottom_arc_circle])
        tangent_points.extend([p1, p2, lute.form_bottom])

        lute.tangent_circles = tangent_circles
        lute.tangent_points = tangent_points
        lute.arc_params = [
            [tangent_circles[i].center, tangent_points[i + 1], tangent_points[i]]
            for i in range(len(tangent_circles))
        ]

    def _resolve_blender_radius(self, lute, circle_a, circle_b) -> float:
        desired = float(self.blender_radius(lute))
        if desired <= 0.0:
            desired = min(float(circle_a.radius), float(circle_b.radius)) * 0.5 or 1.0

        safe_cap = self._max_feasible_blender(circle_a, circle_b)
        radius = min(desired, safe_cap)
        if radius <= 0.0:
            radius = max(safe_cap * 0.5, 1e-3)
        elif safe_cap - radius < 1e-6:
            radius = max(safe_cap - 1e-6, safe_cap * 0.5)
        return radius

    @staticmethod
    def _max_feasible_blender(circle_a, circle_b) -> float:
        r1 = float(circle_a.radius)
        r2 = float(circle_b.radius)
        radial_cap = max(1e-6, min(r1, r2))
        centre_distance = float(circle_a.center.distance(circle_b.center))
        distance_cap = (r1 + r2 - centre_distance) / 2.0
        if distance_cap <= 1e-6:
            return radial_cap * 0.5
        return max(1e-6, min(radial_cap, distance_cap))


class SimpleBlend(LowerArcBuilder):
    def tangent_parameters(self, lute):
        return []

    def blender_radius(self, lute) -> float:
        return lute.unit


class SimpleBlendScaled(SimpleBlend):
    def __init__(self, scale: float):
        self.scale = scale

    def blender_radius(self, lute) -> float:
        return self.scale * lute.unit


class SimpleBlendDynamic(SimpleBlend):
    def __init__(self, radius_getter: Callable[["LuteSoundboard"], float]):
        self.radius_getter = radius_getter

    def blender_radius(self, lute) -> float:
        return self.radius_getter(lute)


class StepCircleBuilder(LowerArcBuilder):
    def __init__(self, step_scale: float, blend_scale: float):
        self.step_scale = step_scale
        self.blend_scale = blend_scale

    def tangent_parameters(self, lute):
        step_circle = lute.geo.circle_by_center_and_radius(lute.form_side, self.step_scale * lute.unit)
        intersections = step_circle.intersection(lute.top_arc_circle)
        finish = lute.geo.pick_point_furthest_from(lute.form_top, intersections)
        connector = lute.geo.line(finish, lute.top_arc_center)
        centre = connector.intersection(lute.spine)[0]
        radius = centre.distance(finish)
        return [TangentParameter(lute.geo, lute.top_arc_circle, centre, radius, lute.form_bottom)]

    def blender_radius(self, lute) -> float:
        return self.blend_scale * lute.unit


class SideCircleTangentBuilder(LowerArcBuilder):
    """Anchor a tangent circle at ``form_side`` and blend with a configurable radius."""

    def __init__(self, tangent_scale: float, blend_strategy: Callable[["LuteSoundboard"], float]):
        self.tangent_scale = tangent_scale
        self.blend_strategy = blend_strategy

    def tangent_parameters(self, lute):
        return [
            TangentParameter(
                lute.geo,
                lute.top_arc_circle,
                lute.form_center,
                self.tangent_scale * lute.unit,
                lute.form_side,
            )
        ]

    def blender_radius(self, lute) -> float:
        value = self.blend_strategy(lute)
        if not isinstance(value, (int, float)):
            value = float(value)
        return float(value)


class SideCircleTangentScaled(SideCircleTangentBuilder):
    """Convenience wrapper for unit-scaled blend radii."""

    def __init__(self, tangent_scale: float, blend_scale: float):
        super().__init__(tangent_scale, lambda lute: blend_scale * lute.unit)


class VerticalUnitBlend(LowerArcBuilder):
    """Blend using a radius derived from ``lute.vertical_unit``."""

    def __init__(self, tangent_scale: float = 3.0):
        self.tangent_scale = tangent_scale

    def tangent_parameters(self, lute):
        return [
            TangentParameter(
                lute.geo,
                lute.top_arc_circle,
                lute.form_center,
                self.tangent_scale * lute.unit,
                lute.form_center,
            )
        ]

    def blender_radius(self, lute) -> float:
        return getattr(lute, "vertical_unit", lute.unit)


__all__ = [
    "TangentParameter",
    "LowerArcBuilder",
    "SimpleBlend",
    "SimpleBlendScaled",
    "SimpleBlendDynamic",
    "StepCircleBuilder",
    "SideCircleTangentBuilder",
    "SideCircleTangentScaled",
    "VerticalUnitBlend"
]