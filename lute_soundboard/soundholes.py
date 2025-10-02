"""Soundhole placement and sizing strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
import sympy


class SoundholePlacement(ABC):
    @abstractmethod
    def center(self, lute):
        """Return the soundhole center point."""


class SoundholeAtNeckBridgeMidpoint(SoundholePlacement):
    def center(self, lute):
        return lute.point_neck_joint.midpoint(lute.bridge)


class SoundholeSizing(ABC):
    @abstractmethod
    def radius(self, lute):
        """Return the soundhole radius."""


class SoundholeOneThirdOfSegment(SoundholeSizing):
    def radius(self, lute):
        center = lute.soundhole_center
        perp = lute.spine.perpendicular_line(center)
        intersection = lute.geo.pick_point_closest_to(lute.spine, lute.top_arc_circle.intersection(perp))
        return center.distance(intersection) / 3


class SoundholeGoldenRatioOfSegment(SoundholeSizing):
    def radius(self, lute):
        center = lute.soundhole_center
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


__all__ = [
    "SoundholePlacement",
    "SoundholeAtNeckBridgeMidpoint",
    "SoundholeSizing",
    "SoundholeOneThirdOfSegment",
    "SoundholeGoldenRatioOfSegment",
    "SoundholeFixedFraction",
    "SoundholeNone",
]
