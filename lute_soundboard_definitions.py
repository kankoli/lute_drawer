"""Soundboard geometry definitions for lute instruments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import sympy

from geo_dsl import GeoDSL


# ---------------------------------------------------------------------------
# Top arc profiles
# ---------------------------------------------------------------------------


class TopArc(ABC):
    @abstractmethod
    def radius(self, lute) -> float:
        """Return top-arc radius in geometry units."""

    def configure(self, lute) -> None:
        r = self.radius(lute)
        lute.top_arc_radius = r
        lute.top_arc_center = lute.geo.translate_y(lute.form_side, r)
        lute.top_arc_circle = lute.geo.circle_by_center_and_radius(lute.top_arc_center, r)


class TopArcType1(TopArc):
    def radius(self, lute) -> float:
        return 4 * lute.unit


class TopArcType2(TopArc):
    def radius(self, lute) -> float:
        return 5 * lute.unit


class TopArcType3(TopArc):
    def radius(self, lute) -> float:
        return 6 * lute.unit


class TopArcType4(TopArc):
    def radius(self, lute) -> float:
        return 7 * lute.unit


class TopArcType10(TopArc):
    def radius(self, lute) -> float:
        return 13 * lute.unit


# ---------------------------------------------------------------------------
# Neck placement strategies
# ---------------------------------------------------------------------------


class NeckProfile(ABC):
    @abstractmethod
    def make_neck_joint(self, lute) -> None:
        """Populate ``lute.point_neck_joint``."""


class NeckThroughTop2(NeckProfile):
    def make_neck_joint(self, lute) -> None:
        lute.ensure_top_2_point()
        helper_line = lute.geo.line(lute.top_2, lute.top_arc_center)
        helper_point = lute.geo.pick_point_closest_to(lute.form_top, helper_line.intersection(lute.top_arc_circle))
        helper_circle = lute.geo.circle_by_center_and_point(lute.top_2, helper_point)
        lute.point_neck_joint = lute.geo.pick_point_closest_to(
            lute.form_top,
            helper_circle.intersection(lute.spine),
        )


class NeckDoubleGolden(NeckProfile):
    def make_neck_joint(self, lute) -> None:
        lute.ensure_top_2_point()
        first = lute.geo.golden_ratio_divider(lute.top_2, lute.form_top)
        lute.point_neck_joint = lute.geo.golden_ratio_divider(lute.form_top, first)


class NeckQuartered(NeckProfile):
    def make_neck_joint(self, lute) -> None:
        lute.ensure_top_2_point()
        lute.point_neck_joint = lute.geo.translate_x(lute.form_top, lute.unit / 4)


# ---------------------------------------------------------------------------
# Soundhole placement and sizing
# ---------------------------------------------------------------------------


class SoundholePlacement(ABC):
    @abstractmethod
    def center(self, lute):
        """Return the soundhole centre point."""


class SoundholeAtNeckBridgeMidpoint(SoundholePlacement):
    def center(self, lute):
        return lute.point_neck_joint.midpoint(lute.bridge)


class SoundholeSizing(ABC):
    @abstractmethod
    def radius(self, lute):
        """Return the soundhole radius."""


class SoundholeOneThird(SoundholeSizing):
    def radius(self, lute):
        center = lute.soundhole_center
        perp = lute.spine.perpendicular_line(center)
        intersection = lute.geo.pick_point_closest_to(lute.spine, lute.top_arc_circle.intersection(perp))
        return center.distance(intersection) / 3


class SoundholeGoldenRatio(SoundholeSizing):
    def radius(self, lute):
        center = lute.soundhole_center
        perp = lute.spine.perpendicular_line(center)
        intersection = lute.geo.pick_point_closest_to(lute.spine, lute.top_arc_circle.intersection(perp))
        length = center.distance(intersection)
        return length - length / sympy.S.GoldenRatio


class SoundholeFixed(SoundholeSizing):
    def __init__(self, fraction: float):
        self.fraction = fraction

    def radius(self, lute):
        return self.fraction * lute.unit


class SoundholeNone(SoundholeSizing):
    def radius(self, lute):  # type: ignore[override]
        return None


# ---------------------------------------------------------------------------
# Small soundhole helpers
# ---------------------------------------------------------------------------


class SmallSoundholeStrategy(ABC):
    @abstractmethod
    def build(self, lute) -> Sequence:
        """Return iterable of small soundhole circles."""


class NoSmallSoundholes(SmallSoundholeStrategy):
    def build(self, lute) -> Sequence:
        return ()


class SmallSoundholesTurkish(SmallSoundholeStrategy):
    def build(self, lute) -> Sequence:
        center = lute.soundhole_center
        axis = center.midpoint(lute.form_center)
        line = lute.spine.perpendicular_line(axis)
        locator = lute.geo.circle_by_center_and_point(axis, lute.form_center)
        centres = locator.intersection(line)
        return [lute.geo.circle_by_center_and_radius(c, lute.soundhole_radius / 2) for c in centres]


class SmallSoundholesBrussels0164(SmallSoundholeStrategy):
    def build(self, lute) -> Sequence:
        radius = lute.soundhole_radius / 3
        axis = lute.geo.translate_x(lute.soundhole_center, 4 * radius)
        line = lute.spine.perpendicular_line(axis)
        locator = lute.geo.circle_by_center_and_radius(axis, 3 * radius)
        centres = locator.intersection(line)
        return [lute.geo.circle_by_center_and_radius(c, lute.soundhole_radius / 4) for c in centres]


# ---------------------------------------------------------------------------
# Lower-arc helpers
# ---------------------------------------------------------------------------


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


class SideCircleBuilder(LowerArcBuilder):
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


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


@dataclass
class SoundboardMeasurement:
    label: str
    value: float
    unit_in_mm: float

    @property
    def value_in_mm(self) -> float:
        return self.value * self.unit_in_mm


# ---------------------------------------------------------------------------
# Core soundboard base class
# ---------------------------------------------------------------------------


class LuteSoundboard(ABC):
    top_arc: TopArc
    neck_profile: NeckProfile
    soundhole_placement: SoundholePlacement
    soundhole_sizing: SoundholeSizing
    small_soundhole_strategy: SmallSoundholeStrategy
    lower_arc_builder: LowerArcBuilder

    def __init__(self, geo: GeoDSL | None = None, display_size: int = 100):
        self.geo = geo if geo is not None else GeoDSL(display_size)
        self.unit = self.geo.display_size
        self.top_arc = self.top_arc if hasattr(self, "top_arc") else TopArcType3()
        self.neck_profile = getattr(self, "neck_profile", NeckThroughTop2())
        self.soundhole_placement = getattr(self, "soundhole_placement", SoundholeAtNeckBridgeMidpoint())
        self.soundhole_sizing = getattr(self, "soundhole_sizing", SoundholeFixed(0.5))
        self.small_soundhole_strategy = getattr(self, "small_soundhole_strategy", NoSmallSoundholes())
        self.lower_arc_builder = getattr(self, "lower_arc_builder", SimpleBlend())
        self._build_geometry()

    # --- high level pipeline -------------------------------------------------

    def _build_geometry(self) -> None:
        self._base_construction()
        self._make_spine_points()
        self._make_bottom_arc_circle()
        self.top_arc.configure(self)
        self.neck_profile.make_neck_joint(self)
        self._configure_soundhole()
        self.lower_arc_builder.build(self)
        self._finalise_arcs()

    # --- base construction ---------------------------------------------------

    def _base_construction(self) -> None:
        self.form_center = self.geo.point(self.geo.display_size * 5.5, self.geo.display_size * 3)
        self.A = self.geo.point(self.form_center.x - self.unit, self.form_center.y - self.unit)
        self.B = self.geo.point(self.A.x + self.unit, self.A.y)
        self.waist_2 = self.geo.translate_y(self.form_center, -self.unit)
        self.form_side = self.geo.translate_y(self.form_center, -2 * self.unit)
        self.centerline = self.geo.line(self.form_side, self.form_center)
        self.spine = self.centerline.perpendicular_line(self.form_center)
        intersections = self.geo.circle_by_center_and_radius(self.form_side, self.unit).intersection(self.spine)
        self.form_top = self.geo.pick_west_point(*intersections)

    @abstractmethod
    def _make_spine_points(self) -> None:
        """Populate form_bottom, bridge, and any additional geometry helpers."""

    def ensure_top_2_point(self) -> None:
        if not hasattr(self, "top_2"):
            self._make_top_2_point()

    def _make_top_2_point(self) -> None:
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)

    def _make_bottom_arc_circle(self) -> None:
        self.bottom_arc_circle = self.geo.circle_by_center_and_point(self.form_top, self.form_bottom)

    def _configure_soundhole(self) -> None:
        center = self.soundhole_center()
        if center is not None:
            radius = self.soundhole_sizing.radius(self)
            self.soundhole_radius = radius
            self.soundhole_circle = self.geo.circle_by_center_and_radius(center, radius)
            self.small_soundhole_circles = list(self.small_soundhole_strategy.build(self))
        else:
            self.soundhole_radius = None
            self.soundhole_circle = None
            self.small_soundhole_circles = []

    def soundhole_center(self):
        try:
            return self.soundhole_placement.center(self)
        except AttributeError:
            return None

    def _finalise_arcs(self) -> None:
        self.final_arcs = [self.geo.arc_by_center_and_two_points(*params) for params in self.arc_params]
        self.final_reflected_arcs = [self.geo.reflect(arc, self.spine) for arc in self.final_arcs]

    # --- measurement helpers -------------------------------------------------

    def measurements(self) -> List[SoundboardMeasurement]:
        unit_mm = self.unit_in_mm()
        return [
            SoundboardMeasurement(label, value, unit_mm / self.unit)
            for label, value in self.measurement_pairs()
        ]

    def unit_in_mm(self) -> float:
        return getattr(self, "unit_mm", 100.0)

    # --- abstract extension hooks -------------------------------------------

    def _make_top_2_point(self) -> None:  # default already defined
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)


# Derived classes and concrete designs would follow here...
