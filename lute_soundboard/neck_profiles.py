"""Neck joint construction strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod



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


__all__ = [
    "NeckProfile",
    "NeckManual",
    "NeckThroughTop2",
    "NeckDoubleGolden",
    "NeckQuartered",
]
