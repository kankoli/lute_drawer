"""Soundboard geometry definitions for lute instruments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import sympy

from utils.geo_dsl import GeoDSL
from .top_arcs import *
from .neck_profiles import *
from .soundholes import *
from .lower_arcs import *


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
        self.soundhole_sizing = getattr(self, "soundhole_sizing", SoundholeFixedFraction(0.5))
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
        self.top_2, self.top_3, self.top_4 = self.geo.divide_distance(self.form_top, self.form_center, 4)

    def _make_bottom_arc_circle(self) -> None:
        self.bottom_arc_circle = self.geo.circle_by_center_and_point(self.form_top, self.form_bottom)

    def _configure_soundhole(self) -> None:
        center = self.soundhole_center()
        radius = self.soundhole_radius()
        if center is not None and radius is not None:
            self.soundhole_center = center
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

    def soundhole_radius(self):
        if hasattr(self, "_soundhole_radius"):
            return self._soundhole_radius
        try:
            self._soundhole_radius = self.soundhole_sizing.radius(self)
        except AttributeError:
            self._soundhole_radius = None
        return self._soundhole_radius

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
    small_soundhole_strategy = SmallSoundholesBrussels0164()
    lower_arc_builder = SimpleBlendDynamic(
        lambda lute: lute.soundhole_center.distance(lute.form_center)
        if getattr(lute, "soundhole_center", None) is not None
        else lute.unit,
    )

    def _make_spine_points(self) -> None:
        # Shortcut to a half-sized vesica pisces (2 * unit arc type 1 -> 1 * unit)
        # top_3 falls onto the upper intersection
        # Lower intersection
        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.translate_x(self.form_bottom, -self.unit)

        self._soundhole_center = self.geo.divide_distance(self.top_3, self.form_center, 3)[0]
        self._soundhole_radius = self.top_3.distance(self._soundhole_center)

        self.point_neck_joint = self.geo.reflect(self.bridge, self._soundhole_center)


    def measurement_pairs(self) -> List[Tuple[str, float]]:
        pairs = super().measurement_pairs()
        if self.soundhole_radius:
            pairs.append(("Soundhole radius", self.soundhole_radius))
        return pairs


class Brussels0404(LuteSoundboard):
    top_arc = TopArcType3()
    neck_profile = NeckManual()
    soundhole_sizing = SoundholeFixedFraction(0.5)
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
    """ Not a true Vesica Pisces construction, only form bottom is 2 units away from center """
    top_arc = TopArcType2()
    neck_profile = NeckDoubleGolden()
    soundhole_sizing = SoundholeFixedFraction(0.5)
    small_soundhole_strategy = SmallSoundholesTurkish()
    lower_arc_builder = SimpleBlend()
    unit_mm = 366 / 4

    def _make_spine_points(self) -> None:
        self.form_bottom = self.geo.translate_x(self.form_center, 2 * self.unit)
        self.bridge = self.geo.translate_x(self.form_center, self.unit)
        self._soundhole_center = self.top_3.midpoint(self.top_4)


class TurkishOud2(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckQuartered()
    soundhole_sizing = SoundholeFixedFraction(0.5)
    small_soundhole_strategy = SmallSoundholesTurkish()
    lower_arc_builder = SimpleBlend()
    unit_mm = 366 / 4

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
    soundhole_sizing = SoundholeFixedFraction(0.5)
    small_soundhole_strategy = NoSmallSoundholes()
    lower_arc_builder = SimpleBlend()
    unit_mm = 366 / 4

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

    def _make_spine_points(self) -> None:
        self.form_bottom = self.geo.translate_x(self.form_top, 6 * self.unit)
        self.vertical_unit = self.point_neck_joint.distance(self.form_bottom) / 4
        self.bridge = self.geo.translate_x(self.form_bottom, -self.vertical_unit)


class IkwanAlSafaOud(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckQuartered()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    soundhole_sizing = SoundholeFixedFraction(0.5)
    lower_arc_builder = SimpleBlendScaled(2.0)

    def _make_spine_points(self) -> None:
        self.bridge = self.geo.translate_x(self.form_center, 3 * self.unit / 4)
        self.form_bottom = self.geo.translate_x(self.form_center, 2 * self.unit)


class HannaNahatOud(LuteSoundboard):
    top_arc = TopArcType2()
    neck_profile = NeckQuartered()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    soundhole_sizing = SoundholeFixedFraction(0.75)
    small_soundhole_strategy = NoSmallSoundholes()
    lower_arc_builder = SideCircleTangentScaled(3.0, 1.25)
    unit_mm = 365 / 4

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

    def _make_spine_points(self) -> None:
        # Shortcut to a half-sized vesica pisces (2 * unit arc type 1 -> 1 * unit)
        # top_3 falls onto the upper intersection
        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.reflect(self.top_4, self.form_center)

        self._soundhole_center = self.top_3.midpoint(self.top_4.midpoint(self.form_center))
        self._soundhole_radius = self._soundhole_center.distance(self.top_3)


class LavtaSmallThreeCourse(LuteSoundboard):
    top_arc = TopArcType1()
    neck_profile = NeckThroughTop2()
    soundhole_placement = SoundholeAtNeckBridgeMidpoint()
    soundhole_sizing = SoundholeOneThird()
    lower_arc_builder = SimpleBlendScaled(1.0)

    def _make_spine_points(self) -> None:
        # Shortcut to a half-sized vesica pisces (2 * unit arc type 1 -> 1 * unit)
        # top_3 falls onto the upper intersection
        self.form_bottom = self.geo.reflect(self.top_3, self.form_center)
        self.bridge = self.geo.reflect(self.top_4, self.form_center)


class BaltaSaz(LuteSoundboard):
    top_arc = TopArcType10()
    neck_profile = NeckThroughTop2()
    soundhole_placement = SoundholeNonePlacement()
    soundhole_sizing = SoundholeNone()
    lower_arc_builder = SimpleBlendScaled(1.5)
    unit_mm = 200 / 4

    def _make_spine_points(self) -> None:
        self.bridge = self.geo.translate_x(self.form_center, self.unit)
        self.form_bottom = self.geo.reflect(self.form_center, self.bridge)
