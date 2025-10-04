"""Soundboard geometry definitions for lute instruments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import sympy

from utils.geo_dsl import GeoDSL


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


class NeckManual(NeckProfile):
    """No-op neck placement; instrument supplies the joint."""

    def make_neck_joint(self, lute) -> None:  # noqa: D401 - intentionally empty
        return None


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


class SoundholeNonePlacement(SoundholePlacement):
    def center(self, lute):  # type: ignore[override]
        return None


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
        radius = float(lute.soundhole_radius) / 3.0
        axis = lute.geo.simple_point(lute.geo.translate_x(lute.soundhole_center, 4.0 * radius))
        line = lute.spine.perpendicular_line(axis)
        locator = lute.geo.circle_by_center_and_radius(axis, 3.0 * radius)
        centres = [lute.geo.simple_point(p) for p in locator.intersection(line)]
        return [lute.geo.circle_by_center_and_radius(c, float(lute.soundhole_radius) / 4.0) for c in centres]


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
        self.ensure_top_2_point()
        self.neck_profile.make_neck_joint(self)
        self._make_spine_points()
        self._make_bottom_arc_circle()
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
        self.top_arc.configure(self)
        self.centerline = self.geo.line(self.top_arc_center, self.form_side)
        self.spine = self.centerline.perpendicular_line(self.form_center)
        intersections = self.top_arc_circle.intersection(self.spine)
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
            self.soundhole_center = center
            radius = self.soundhole_sizing.radius(self)
            self.soundhole_radius = radius
            self.soundhole_circle = self.geo.circle_by_center_and_radius(center, radius)
            self.small_soundhole_circles = list(self.small_soundhole_strategy.build(self))
            self.small_soundhole_centers = [circle.center for circle in self.small_soundhole_circles]
        else:
            self.soundhole_center = None
            self.soundhole_radius = None
            self.soundhole_circle = None
            self.small_soundhole_circles = []
            self.small_soundhole_centers = []

    def soundhole_center(self):
        if hasattr(self, "_soundhole_center"):
            return self._soundhole_center
        try:
            self._soundhole_center = self.soundhole_placement.center(self)
        except AttributeError:
            self._soundhole_center = None
        return self._soundhole_center

    def _finalise_arcs(self) -> None:
        self.final_arcs = [self.geo.arc_by_center_and_two_points(*params) for params in self.arc_params]
        self.final_reflected_arcs = [self.geo.reflect(arc, self.spine) for arc in self.final_arcs]

    # --- measurement helpers -------------------------------------------------

    def measurement_pairs(self) -> List[Tuple[str, float]]:
        return [
            ("Unit", self.unit),
            ("Form Width", self._form_width()),
            ("Form Length", self.form_bottom.distance(self.form_top)),
            ("Neck to Bottom", self.form_bottom.distance(self.point_neck_joint)),
            ("Neck to Bridge", self.point_neck_joint.distance(self.bridge)),
            ("(1/3-Neck) Scale", (3 / 2) * self.point_neck_joint.distance(self.bridge)),
            ("(1/3-Neck) Neck length", (1 / 2) * self.point_neck_joint.distance(self.bridge)),
            ("(Half-Neck) Scale", 2 * self.point_neck_joint.distance(self.bridge)),
            ("(Half-Neck) Neck", self.point_neck_joint.distance(self.bridge)),
            ("Neck-joint width", self.form_width_at_point(self.point_neck_joint)),
        ]

    def form_width_at_point(self, point) -> float:
        perpendicular_line = self.spine.perpendicular_line(point)
        intersection = self.geo.pick_point_closest_to(
            self.spine,
            self.top_arc_circle.intersection(perpendicular_line),
        )
        return 2 * point.distance(intersection)

    def _form_width(self) -> float:
        return max(2 * arc[1].distance(self.spine) for arc in self.arc_params)

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


class ManolLavta(LuteSoundboard):
    top_arc = TopArcType4()
    neck_profile = NeckThroughTop2()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    soundhole_sizing = SoundholeOneThird()
    small_soundhole_strategy = NoSmallSoundholes()
    lower_arc_builder = VerticalUnitBlend()
    unit_mm = 285 / 4

    def _make_bottom_arc_circle(self) -> None:
        bottom_arc_center = self.geo.translate_x(self.form_top, -2 * self.unit)
        self.bottom_arc_circle = self.geo.circle_by_center_and_point(bottom_arc_center, self.form_bottom)

    def _make_spine_points(self) -> None:
        self.form_bottom = self.geo.reflect(
            self.geo.divide_distance(self.form_top, self.form_center, 3)[-1],
            self.form_center,
        )
        self.vertical_unit = self.point_neck_joint.distance(self.form_bottom) / 5
        self.bridge = self.geo.translate_x(self.form_bottom, -self.vertical_unit)

    def measurement_pairs(self) -> List[Tuple[str, float]]:
        pairs = super().measurement_pairs()
        if self.soundhole_radius:
            pairs.append(("Soundhole radius", self.soundhole_radius))
        pairs.append(("Vertical Unit", getattr(self, "vertical_unit", 0.0)))
        return pairs


class ManolLavta_Type3(ManolLavta):
    top_arc = TopArcType3()


class ManolLavta_Athens(ManolLavta):
    soundhole_sizing = SoundholeGoldenRatio()


class Brussels0164(LuteSoundboard):
    top_arc = TopArcType1()
    neck_profile = NeckThroughTop2()
    soundhole_sizing = SoundholeOneThird()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    small_soundhole_strategy = SmallSoundholesBrussels0164()
    lower_arc_builder = SimpleBlendDynamic(
        lambda lute: lute.soundhole_center.distance(lute.form_center)
        if getattr(lute, "soundhole_center", None) is not None
        else lute.unit,
    )

    def _make_top_2_point(self) -> None:
        self.top_2, self.top_3, self.top_4 = self.geo.divide_distance(self.form_top, self.form_center, 4)

    def _make_spine_points(self) -> None:
        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.translate_x(self.form_bottom, -self.unit)
        center = self.geo.divide_distance(self.top_3, self.form_center, 3)[0]
        self._soundhole_center = center
        self.point_neck_joint = self.geo.reflect(self.bridge, center)

    def _configure_soundhole(self) -> None:
        self._soundhole_center = getattr(self, "_soundhole_center", None) or self.geo.divide_distance(self.top_3, self.form_center, 3)[0]
        super()._configure_soundhole()

    def measurement_pairs(self) -> List[Tuple[str, float]]:
        pairs = super().measurement_pairs()
        if self.soundhole_radius:
            pairs.append(("Soundhole radius", self.soundhole_radius))
        return pairs


class Brussels0404(LuteSoundboard):
    top_arc = TopArcType3()
    neck_profile = NeckManual()
    soundhole_sizing = SoundholeFixed(0.5)
    lower_arc_builder = SimpleBlendScaled(2.0)

    def _make_spine_points(self) -> None:
        half_vesica = self.geo.circle_by_center_and_radius(self.waist_2, 2 * self.unit)
        intersections = self.spine.intersection(half_vesica)
        self._vesica_piscis_intersections = intersections
        self.form_bottom = self.geo.pick_point_furthest_from(self.form_top, intersections)
        self.bridge = self.geo.translate_x(self.form_bottom, -self.unit)
        self._soundhole_center = self.geo.pick_point_closest_to(self.form_top, intersections)
        self.point_neck_joint = self.geo.reflect(self.bridge, self._soundhole_center)


class VesicaPiscesOud(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckDoubleGolden()
    soundhole_sizing = SoundholeFixed(0.5)
    small_soundhole_strategy = SmallSoundholesTurkish()
    lower_arc_builder = SimpleBlend()
    unit_mm = 366 / 4

    def _make_top_2_point(self) -> None:
        self.top_2, self.top_3, self.top_4 = self.geo.divide_distance(self.form_top, self.form_center, 4)

    def _make_spine_points(self) -> None:
        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.reflect(self.top_4, self.form_center)
        self._soundhole_center = self.top_3.midpoint(self.top_4)


class TurkishOud2(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckQuartered()
    soundhole_sizing = SoundholeFixed(0.5)
    small_soundhole_strategy = SmallSoundholesTurkish()
    lower_arc_builder = SimpleBlend()
    unit_mm = 366 / 4

    def _make_top_2_point(self) -> None:
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)
        self.top_3 = self.geo.translate_x(self.form_top, 2 * self.unit)
        self.top_4 = self.geo.translate_x(self.form_top, 3 * self.unit)

    def _make_spine_points(self) -> None:
        soundhole_center = self.geo.golden_ratio_divider(self.top_4, self.top_3)
        self._soundhole_center = soundhole_center
        vertical_x = self.point_neck_joint.distance(soundhole_center)
        self.bridge = self.geo.translate_x(soundhole_center, vertical_x)
        self.form_bottom = self.geo.translate_x(self.bridge, vertical_x / 2)
        self.vertical_unit = vertical_x


class TurkishOud2_1(TurkishOud2):
    lower_arc_builder = SimpleBlendDynamic(
        lambda lute: lute.small_soundhole_centers[0].distance(lute.soundhole_center)
        if getattr(lute, "small_soundhole_centers", [])
        else lute.unit,
    )


class TurkishOud2_2(TurkishOud2):
    lower_arc_builder = SideCircleTangentScaled(2.0, 1.0)


class TurkishOud2_3(TurkishOud2):
    lower_arc_builder = StepCircleBuilder(0.25, 1.0)


class TurkishOud2_4(TurkishOud2):
    lower_arc_builder = SimpleBlend()


class TurkishOud(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckQuartered()
    soundhole_sizing = SoundholeFixed(0.5)
    small_soundhole_strategy = NoSmallSoundholes()
    lower_arc_builder = SimpleBlend()
    unit_mm = 366 / 4

    def _make_top_2_point(self) -> None:
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)
        self.top_3 = self.geo.translate_x(self.form_top, 2 * self.unit)
        self.top_4 = self.geo.translate_x(self.form_top, 3 * self.unit)

    def _make_spine_points(self) -> None:
        self.vertical_unit = 16 * self.unit / 15
        soundhole_center = self.geo.translate_x(self.point_neck_joint, 2 * self.vertical_unit)
        self._soundhole_center = soundhole_center
        self.bridge = self.geo.translate_x(soundhole_center, 2 * self.vertical_unit)
        self.form_bottom = self.geo.translate_x(self.bridge, self.vertical_unit)


class TurkishOudSingleMiddleArc(TurkishOud):
    small_soundhole_strategy = SmallSoundholesTurkish()
    lower_arc_builder = SimpleBlendDynamic(
        lambda lute: 0.75
        * lute.small_soundhole_centers[0].distance(lute.small_soundhole_centers[1])
        if len(getattr(lute, "small_soundhole_centers", [])) >= 2
        else lute.unit,
    )


class TurkishOudDoubleMiddleArcs(TurkishOud):
    small_soundhole_strategy = SmallSoundholesTurkish()
    lower_arc_builder = SideCircleTangentScaled(3.0, 1.5)


class TurkishOudComplexLowerBout(TurkishOud):
    small_soundhole_strategy = SmallSoundholesTurkish()
    lower_arc_builder = StepCircleBuilder(0.25, 1.0)


class TurkishOudSoundholeThird(TurkishOud):
    small_soundhole_strategy = SmallSoundholesTurkish()
    soundhole_sizing = SoundholeOneThird()
    lower_arc_builder = SimpleBlendDynamic(
        lambda lute: 0.75
        * lute.small_soundhole_centers[0].distance(lute.small_soundhole_centers[1])
        if len(getattr(lute, "small_soundhole_centers", [])) >= 2
        else lute.unit,
    )


class IstanbulLavta(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckThroughTop2()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    soundhole_sizing = SoundholeOneThird()
    lower_arc_builder = StepCircleBuilder(0.5, 1.25)
    unit_mm = 300 / 4

    def _make_top_2_point(self) -> None:
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)
        self.top_3 = self.geo.translate_x(self.form_top, 2 * self.unit)
        self.top_4 = self.geo.translate_x(self.form_top, 3 * self.unit)

    def _make_spine_points(self) -> None:
        self.form_bottom = self.geo.translate_x(self.form_top, 6 * self.unit)
        self.vertical_unit = self.point_neck_joint.distance(self.form_bottom) / 4
        self.bridge = self.geo.translate_x(self.form_bottom, -self.vertical_unit)


class IkwanAlSafaOud(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckQuartered()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    soundhole_sizing = SoundholeFixed(0.5)
    lower_arc_builder = SimpleBlendScaled(2.0)

    def _make_top_2_point(self) -> None:
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)
        self.top_3 = self.geo.translate_x(self.form_top, 2 * self.unit)
        self.top_4 = self.geo.translate_x(self.form_top, 3 * self.unit)

    def _make_spine_points(self) -> None:
        self.bridge = self.geo.translate_x(self.form_center, 3 * self.unit / 4)
        self.form_bottom = self.geo.translate_x(self.form_center, 2 * self.unit)


class HannaNahatOud(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckQuartered()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    soundhole_sizing = SoundholeFixed(0.75)
    small_soundhole_strategy = NoSmallSoundholes()
    lower_arc_builder = SideCircleTangentScaled(3.0, 1.25)
    unit_mm = 365 / 4

    def _make_top_2_point(self) -> None:
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)
        self.top_3 = self.geo.translate_x(self.form_top, 2 * self.unit)
        self.top_4 = self.geo.translate_x(self.form_top, 3 * self.unit)

    def _make_spine_points(self) -> None:
        self.bridge = self.geo.translate_x(self.form_center, 3 * self.unit / 4)
        self.form_bottom = self.geo.translate_x(self.bridge, self.unit)


class HochLavta(LuteSoundboard):
    top_arc = TopArcType1()
    neck_profile = NeckThroughTop2()
    lower_arc_builder = SimpleBlendDynamic(
        lambda lute: lute.soundhole_center.distance(lute.waist_2)
        if getattr(lute, "soundhole_center", None) is not None
        else lute.unit,
    )

    def _make_top_2_point(self) -> None:
        self.top_2, self.top_3, self.top_4 = self.geo.divide_distance(self.form_top, self.form_center, 4)

    def _make_spine_points(self) -> None:
        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.reflect(self.top_4, self.form_center)
        self.neck_profile.make_neck_joint(self)
        center = self.top_3.midpoint(self.top_4.midpoint(self.form_center))
        self._soundhole_center = center

    def _configure_soundhole(self) -> None:
        center = self.top_3.midpoint(self.top_4.midpoint(self.form_center))
        self._soundhole_center = center
        radius = center.distance(self.top_3)
        self.soundhole_center = center
        self.soundhole_radius = radius
        self.soundhole_circle = self.geo.circle_by_center_and_radius(center, radius)
        self.small_soundhole_circles = []
        self.small_soundhole_centers = []


class LavtaSmallThreeCourse(LuteSoundboard):
    top_arc = TopArcType1()
    neck_profile = NeckThroughTop2()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    soundhole_sizing = SoundholeOneThird()
    lower_arc_builder = SimpleBlendScaled(1.0)

    def _make_top_2_point(self) -> None:
        self.top_2, self.top_3, self.top_4 = self.geo.divide_distance(self.form_top, self.form_center, 4)

    def _make_spine_points(self) -> None:
        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.reflect(self.top_4, self.form_center)
        self.neck_profile.make_neck_joint(self)

    def _configure_soundhole(self) -> None:
        self.neck_profile.make_neck_joint(self)
        super()._configure_soundhole()


class BaltaSaz(LuteSoundboard):
    top_arc = TopArcType10()
    neck_profile = NeckThroughTop2()
    soundhole_placement = SoundholeNonePlacement()
    soundhole_sizing = SoundholeNone()
    lower_arc_builder = SimpleBlendScaled(1.5)
    unit_mm = 200 / 4

    def _make_top_2_point(self) -> None:
        self.top_2 = self.geo.translate_x(self.form_top, self.unit)

    def _make_spine_points(self) -> None:
        self.bridge = self.geo.translate_x(self.form_center, self.unit)
        self.form_bottom = self.geo.reflect(self.form_center, self.bridge)
