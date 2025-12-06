"""Helpers for sampling planar ribs and bowl sections."""
from __future__ import annotations

from typing import List

import numpy as np

from .bowl_from_soundboard import Section, _derive_planar_ribs, _sample_section
from .top_curves import SimpleAmplitudeCurve, TopCurve

_EPS = 1e-9


def build_bowl_ribs(
    lute,
    *,
    n_ribs: int = 13,
    n_sections: int = 300,
    top_curve: type[TopCurve] = SimpleAmplitudeCurve,
    skirt_span: float = 0.0,
    division_mode: str = "angle",
) -> tuple[list[Section], List[np.ndarray]]:
    """Return sampled sections and rib polylines between neck joint and tail."""
    if n_sections < 2:
        raise ValueError("n_sections must be at least 2.")

    if not isinstance(top_curve, type) or not issubclass(top_curve, TopCurve):
        raise TypeError("top_curve must be a TopCurve subclass")

    z_top = top_curve.build(lute)
    setattr(lute, "top_curve_label", top_curve.__name__)

    neck_point = getattr(lute, "point_neck_joint", None)
    start_x = float(neck_point.x) if neck_point is not None else float(lute.form_top.x)
    end_x = float(lute.form_bottom.x)

    if start_x >= end_x - _EPS:
        raise ValueError("End blocks overlap the bowl span; adjust block sizes.")

    skirt_span = max(0.0, float(skirt_span))
    span = end_x - start_x
    include_eye = skirt_span > _EPS and skirt_span < span - _EPS and n_sections >= 2
    eye_x = end_x - skirt_span if include_eye else None

    xs = np.linspace(start_x, end_x, n_sections)
    if include_eye and eye_x is not None and not np.any(np.isclose(xs, eye_x, atol=1e-8)):
        xs = np.append(xs, eye_x)
    xs = np.unique(xs)
    xs.sort()
    sections: list[Section] = []

    try:
        start_section = _sample_section(lute, float(start_x), z_top)
    except Exception as exc:  # pragma: no cover - diagnostic aid
        raise RuntimeError(f"Failed to sample section at X={float(start_x):.6f}") from exc
    if start_section is None:
        raise RuntimeError(f"No valid section geometry at X={float(start_x):.6f}")

    sections.append(start_section)

    interior_xs = xs[1:-1] if len(xs) > 2 else []

    for X in interior_xs:
        try:
            section = _sample_section(lute, float(X), z_top)
        except Exception as exc:  # pragma: no cover - diagnostic aid
            raise RuntimeError(f"Failed to sample section at X={float(X):.6f}") from exc
        if section is None:
            raise RuntimeError(f"No valid section geometry at X={float(X):.6f}")
        sections.append(section)

    try:
        end_section = _sample_section(lute, float(end_x), z_top)
    except Exception as exc:  # pragma: no cover - diagnostic aid
        raise RuntimeError(f"Failed to sample section at X={float(end_x):.6f}") from exc
    if end_section is None:
        raise RuntimeError(f"No valid section geometry at X={float(end_x):.6f}")
    sections.append(end_section)

    ribs = _derive_planar_ribs(
        lute,
        sections,
        n_ribs,
        start_x,
        end_x,
        skirt_span=skirt_span,
        z_top=z_top,
        eye_x=eye_x,
        division_mode=division_mode,
    )

    return sections, ribs


__all__ = ["build_bowl_ribs"]
