"""Helpers for sampling planar ribs and bowl sections."""
from __future__ import annotations

from typing import Callable, List, Mapping, Optional

import numpy as np

from .bowl_from_soundboard import Section, _derive_planar_ribs, _sample_section
from .top_curves import SimpleAmplitudeCurve, TopCurve
from .section_curve import BaseSectionCurve, CircularSectionCurve, CubicBezierSectionCurve

_EPS = 1e-9


def build_bowl_ribs(
    lute,
    *,
    n_ribs: Optional[int] = None,
    n_sections: Optional[int] = None,
    top_curve: type[TopCurve] | None = None,
    skirt_span: Optional[float] = None,
    section_curve_cls: type[BaseSectionCurve] | None = None,
    section_curve_kwargs: dict | None = None,
    division_mode: str | None = None,
    preset: Optional[Mapping] = None,
    debug_rib_indices: list[int] | None = None,
    debug_logger: Callable[[str], None] | None = None,
    debug_plot: bool = False,
) -> tuple[list[Section], List[np.ndarray]]:
    """Return sampled sections and rib polylines between neck joint and tail.

    debug_rib_indices is a 1-based list of ribs to log sampling decisions for,
    and debug_logger is a callable that receives formatted log strings.

    Presets can be supplied via lute.default_bowl (a dict) or the `preset`
    argument. Explicit keyword arguments take precedence over presets.
    """

    def _merge_cfg() -> dict:
        # Global defaults
        cfg = {
            "n_ribs": 13,
            "n_sections": 300,
            "top_curve": SimpleAmplitudeCurve,
            "skirt_span": 0.0,
            "section_curve_cls": CircularSectionCurve,
            "section_curve_kwargs": {},
            "division_mode": "angle",
        }
        for source in (getattr(lute, "default_bowl", None), preset):
            if source:
                cfg.update(source)
                if "section_curve_kwargs" in source and isinstance(source["section_curve_kwargs"], Mapping):
                    merged = dict(cfg.get("section_curve_kwargs", {}))
                    merged.update(source["section_curve_kwargs"])
                    cfg["section_curve_kwargs"] = merged

        explicit = {
            "n_ribs": n_ribs,
            "n_sections": n_sections,
            "top_curve": top_curve,
            "skirt_span": skirt_span,
            "section_curve_cls": section_curve_cls,
            "division_mode": division_mode,
        }
        for key, value in explicit.items():
            if value is not None:
                cfg[key] = value

        if section_curve_kwargs is not None:
            merged = dict(cfg.get("section_curve_kwargs", {}))
            merged.update(section_curve_kwargs)
            cfg["section_curve_kwargs"] = merged

        return cfg

    cfg = _merge_cfg()

    if not isinstance(cfg["top_curve"], type) or not issubclass(cfg["top_curve"], TopCurve):
        raise TypeError("top_curve must be a TopCurve subclass")

    curve_cls = cfg["section_curve_cls"]
    if not isinstance(curve_cls, type) or not issubclass(curve_cls, BaseSectionCurve):
        raise TypeError("section_curve_cls must be a BaseSectionCurve subclass")

    n_sections_val = int(cfg["n_sections"])
    if n_sections_val < 2:
        raise ValueError("n_sections must be at least 2.")
    n_ribs_val = int(cfg["n_ribs"])
    if n_ribs_val < 1:
        raise ValueError("n_ribs must be at least 1.")

    z_top = cfg["top_curve"].build(lute)
    curve_kwargs = cfg.get("section_curve_kwargs", {}) or {}
    setattr(lute, "top_curve_label", cfg["top_curve"].__name__)
    setattr(lute, "top_curve", z_top)

    neck_point = getattr(lute, "point_neck_joint", None)
    start_x = float(neck_point.x) if neck_point is not None else float(lute.form_top.x)
    end_x = float(lute.form_bottom.x)

    if start_x >= end_x - _EPS:
        raise ValueError("End blocks overlap the bowl span; adjust block sizes.")

    skirt_span_val = max(0.0, float(cfg["skirt_span"]))
    span = end_x - start_x
    include_eye = skirt_span_val > _EPS and skirt_span_val < span - _EPS and n_sections_val >= 2
    eye_x = end_x - skirt_span_val if include_eye else None

    xs = np.linspace(start_x, end_x, n_sections_val)
    if include_eye and eye_x is not None and not np.any(np.isclose(xs, eye_x, atol=1e-8)):
        xs = np.append(xs, eye_x)
    xs = np.unique(xs)
    xs.sort()
    sections: list[Section] = []

    try:
        start_section = _sample_section(lute, float(start_x), z_top, curve_cls=curve_cls, curve_kwargs=curve_kwargs)
    except Exception as exc:  # pragma: no cover - diagnostic aid
        raise RuntimeError(f"Failed to sample section at X={float(start_x):.6f}") from exc
    if start_section is None:
        raise RuntimeError(f"No valid section geometry at X={float(start_x):.6f}")

    sections.append(start_section)

    interior_xs = xs[1:-1] if len(xs) > 2 else []

    for X in interior_xs:
        try:
            section = _sample_section(lute, float(X), z_top, curve_cls=curve_cls, curve_kwargs=curve_kwargs)
        except Exception as exc:  # pragma: no cover - diagnostic aid
            raise RuntimeError(f"Failed to sample section at X={float(X):.6f}") from exc
        if section is None:
            raise RuntimeError(f"No valid section geometry at X={float(X):.6f}")
        sections.append(section)

    try:
        end_section = _sample_section(lute, float(end_x), z_top, curve_cls=curve_cls, curve_kwargs=curve_kwargs)
    except Exception as exc:  # pragma: no cover - diagnostic aid
        raise RuntimeError(f"Failed to sample section at X={float(end_x):.6f}") from exc
    if end_section is None:
        raise RuntimeError(f"No valid section geometry at X={float(end_x):.6f}")
    sections.append(end_section)

    ribs = _derive_planar_ribs(
        lute,
        sections,
        n_ribs_val,
        start_x,
        end_x,
        skirt_span=skirt_span_val,
        z_top=z_top,
        eye_x=eye_x,
        division_mode=cfg["division_mode"],
        debug_rib_indices=debug_rib_indices,
        debug_logger=debug_logger,
        debug_plot=debug_plot,
    )

    return sections, ribs


__all__ = ["build_bowl_ribs"]
