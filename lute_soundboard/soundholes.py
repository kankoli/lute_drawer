"""Soundhole placement and sizing strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
import sympy


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
    """ Diameter is one-third of cross-section of soundboard at soundhole center """
    def radius(self, lute):
        center = lute.soundhole_center()
        perp = lute.spine.perpendicular_line(center)
        intersection = lute.geo.pick_point_closest_to(lute.spine, lute.top_arc_circle.intersection(perp))
        return center.distance(intersection) / 3


class SoundholeGoldenRatio(SoundholeSizing):
    """ Diameter is golden-ratio of cross-section of soundboard at soundhole center """
    def radius(self, lute):
        center = lute.soundhole_center()
        perp = lute.spine.perpendicular_line(center)
        intersection = lute.geo.pick_point_closest_to(lute.spine, lute.top_arc_circle.intersection(perp))
        length = center.distance(intersection)
        return length - length / sympy.S.GoldenRatio


class SoundholeFixedFraction(SoundholeSizing):
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


__all__ = [
    "SoundholePlacement",
    "SoundholeAtNeckBridgeMidpoint",
    "SoundholeNonePlacement",
    "SoundholeSizing",
    "SoundholeOneThird",
    "SoundholeGoldenRatio",
    "SoundholeFixedFraction",
    "SoundholeNone",

    "SmallSoundholeStrategy",
    "NoSmallSoundholes",
    "SmallSoundholesTurkish",
    "SmallSoundholesBrussels0164"
]
