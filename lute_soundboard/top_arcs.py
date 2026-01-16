"""Top arc mixins for lute soundboards."""
from __future__ import annotations

from abc import ABC, abstractmethod




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


class TopArcType5(TopArc):
    def radius(self, lute) -> float:
        return 8 * lute.unit


class TopArcType10(TopArc):
    def radius(self, lute) -> float:
        return 13 * lute.unit


__all__ = [
    "TopArc",
    "TopArcType1",
    "TopArcType2",
    "TopArcType3",
    "TopArcType4",
    "TopArcType5",
    "TopArcType10",
]
