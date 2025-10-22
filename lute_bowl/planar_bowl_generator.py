"""Planar rib bowl generator with configurable end blocks."""
from __future__ import annotations

from typing import List

import numpy as np

from .bowl_from_soundboard import (
    Section,
    _add_endcap_sections,
    _derive_planar_ribs,
    _sample_section,
)
from .bowl_top_curves import SimpleAmplitudeCurve, TopCurve

_EPS = 1e-9


def build_planar_bowl_for_lute(
    lute,
    *,
    n_ribs: int = 13,
    n_sections: int = 200,
    top_curve: type[TopCurve] = SimpleAmplitudeCurve,
    upper_block_units: float = 1.0,
    lower_block_units: float = 0.0,
    debug: bool = False,
) -> tuple[list[Section], List[np.ndarray]]:
    """Build a bowl with planar ribs bounded by end blocks."""
    if n_sections < 2:
        raise ValueError("n_sections must be at least 2.")

    if not isinstance(top_curve, type) or not issubclass(top_curve, TopCurve):
        raise TypeError("top_curve must be a TopCurve subclass")

    z_top = top_curve.build(lute)
    setattr(lute, "top_curve_label", top_curve.__name__)

    unit = float(getattr(lute, "unit", 1.0))

    neck_point = getattr(lute, "point_neck_joint", None)
    neck_x = float(neck_point.x) if neck_point is not None else float(lute.form_top.x)

    start_x = neck_x + float(upper_block_units) * unit
    bottom = float(lute.form_bottom.x)
    end_x = bottom - float(lower_block_units) * unit

    if start_x >= end_x - _EPS:
        raise ValueError("End blocks overlap the bowl span; adjust block sizes.")

    xs = np.linspace(start_x, end_x, n_sections)
    sections: list[Section] = []

    interior_xs = xs[1:-1] if len(xs) > 2 else []

    for X in interior_xs:
        try:
            section = _sample_section(lute, float(X), z_top, debug=debug)
        except Exception as exc:  # pragma: no cover - diagnostic aid
            raise RuntimeError(f"Failed to sample section at X={float(X):.6f}") from exc
        if section is None:
            raise RuntimeError(f"No valid section geometry at X={float(X):.6f}")
        sections.append(section)

    sections = _add_endcap_sections(lute, sections, start_x, end_x)

    ribs = _derive_planar_ribs(lute, sections, n_ribs, start_x, end_x)

    return sections, ribs


__all__ = ["build_planar_bowl_for_lute"]
