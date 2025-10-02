"""Lower-arc construction helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TangentParameter:
    def __init__(self, geo, previous_circle, point, radius, closest_point):
        self.geo = geo
        self.previous_circle = previous_circle
        self.point = point
        self.radius = radius
        self.closest_point = closest_point
        self.tangent_circle = None
        self.tangent_point = None

    def calculate(self):
        line = self.geo.line(self.previous_circle.center, self.point)
        prev_circle = self.previous_circle.tangent_circle if isinstance(self.previous_circle, TangentParameter) else self.previous_circle
        tangent_circle, tangent_point = self.geo.get_tangent_circle(prev_circle, line, self.radius, self.closest_point, True)
        self.tangent_circle = tangent_circle
        self.tangent_point = tangent_point
        return tangent_circle, tangent_point


class LowerArcBuilder(ABC):
    def build_lower_arcs(self, lute):
        current_circle = lute.top_arc_circle
        tangent_circles = [current_circle]
        tangent_points = [lute.form_top]

        for parameter in self.tangent_parameters(lute):
            circle, point = parameter.calculate()
            tangent_circles.append(circle)
            tangent_points.append(point)
            current_circle = circle

        blender_circle, p1, p2 = lute.geo.blend_two_circles(
            self.blender_radius(lute),
            current_circle,
            lute.bottom_arc_circle,
        )
        tangent_circles.extend([blender_circle, lute.bottom_arc_circle])
        tangent_points.extend([p1, p2, lute.form_bottom])

        lute.tangent_circles = tangent_circles
        lute.tangent_points = tangent_points
        lute.arc_params = [
            [tangent_circles[i].center, tangent_points[i + 1], tangent_points[i]]
            for i in range(len(tangent_circles) - 1)
        ]

    @abstractmethod
    def tangent_parameters(self, lute):
        """Yield TangentParameter instances."""

    @abstractmethod
    def blender_radius(self, lute) -> float:
        """Radius for final blender circle."""


class SimpleBlend(LowerArcBuilder):
    def tangent_parameters(self, lute):
        return []


class SimpleBlendWithRadius(SimpleBlend):
    def __init__(self, radius_multiplier: float):
        self.radius_multiplier = radius_multiplier

    def blender_radius(self, lute) -> float:
        return self.radius_multiplier * lute.unit


class SideCircle(LowerArcBuilder):
    def __init__(self, step_radius_multiplier: float, blender_radius_multiplier: float):
        self.step_radius_multiplier = step_radius_multiplier
        self.blender_radius_multiplier = blender_radius_multiplier

    def tangent_parameters(self, lute):
        params = []
        step_circle = lute.geo.circle_by_center_and_radius(lute.form_side, self.step_radius_multiplier * lute.unit)
        step_intersections = step_circle.intersection(lute.top_arc_circle)
        top_finish = lute.geo.pick_point_furthest_from(lute.form_top, step_intersections)
        connector = lute.geo.line(top_finish, lute.top_arc_center)
        connector_intersections = connector.intersection(lute.spine)
        second_center = connector_intersections[0]
        second_radius = second_center.distance(top_finish)
        params.append(TangentParameter(lute.geo, lute.top_arc_circle, second_center, second_radius, lute.form_bottom))
        return params

    def blender_radius(self, lute) -> float:
        return self.blender_radius_multiplier * lute.unit


__all__ = [
    "TangentParameter",
    "LowerArcBuilder",
    "SimpleBlend",
    "SimpleBlendWithRadius",
    "SideCircle",
]
