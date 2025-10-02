"""Top arc mixins for lute soundboards."""
from __future__ import annotations

from abc import ABC, abstractmethod


class TopArc(ABC):
    @abstractmethod
    def _get_top_arc_radius(self) -> float:
        """Return the radius of the top arc in units of self.unit."""

    def make_top_arc(self) -> None:
        radius = self._get_top_arc_radius()
        self.top_arc_radius = radius
        self.top_arc_center = self.geo.translate_y(self.form_side, radius)
        self.top_arc_circle = self.geo.circle_by_center_and_radius(self.top_arc_center, radius)


class TopArcType1(TopArc):
    def _get_top_arc_radius(self) -> float:
        return 4 * self.unit


class TopArcType2(TopArc):
    def _get_top_arc_radius(self) -> float:
        return 5 * self.unit


class TopArcType3(TopArc):
    def _get_top_arc_radius(self) -> float:
        return 6 * self.unit


class TopArcType4(TopArc):
    def _get_top_arc_radius(self) -> float:
        return 7 * self.unit


class TopArcType10(TopArc):
    def _get_top_arc_radius(self) -> float:
        return 13 * self.unit


__all__ = [
    "TopArc",
    "TopArcType1",
    "TopArcType2",
    "TopArcType3",
    "TopArcType4",
    "TopArcType10",
]
