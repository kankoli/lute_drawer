"""Top-curve configurations for bowl construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SideProfileParameters:
    gammas: dict[str, float]
    widths: dict[str, float | tuple]
    amplitude_mode: str = "max_depth"
    amplitude_units: float | None = None
    width_factor: float = 0.9
    samples: int = 400
    margin: float = 1e-3
    gate_start: float = 0.0
    gate_full: float = 0.0
    max_exponent_delta: float = 0.8
    kernel: str = "cauchy"


class TopCurve:
    name = "base"

    @classmethod
    def build(cls, lute, builder: Callable[[object, SideProfileParameters], Callable[[float], float]]):
        raise NotImplementedError


class SideProfilePerControlTopCurve(TopCurve):
    PARAMETERS = SideProfileParameters(gammas={}, widths={})

    @classmethod
    def _parameters(cls) -> SideProfileParameters:
        return cls.PARAMETERS

    @classmethod
    def build(cls, lute, builder: Callable[[object, SideProfileParameters], Callable[[float], float]]):
        params = cls._parameters()
        if not params.gammas:
            raise RuntimeError("SideProfilePerControlTopCurve requires non-empty SHAPE_GAMMAS")
        return builder(lute, params)


class SimpleAmplitudeCurve(SideProfilePerControlTopCurve):
    PARAMETERS = SideProfileParameters(
        gammas={
            "neck_joint": 1.00,
            "soundhole_center": 1.00,
            "form_center": 1.00,
            "bridge": 1.00,
        },
        widths={},
        amplitude_mode="units",
        amplitude_units=1.75,
    )


class DeepBackCurve(SideProfilePerControlTopCurve):
    PARAMETERS = SideProfileParameters(
        gammas={
            "neck_joint": 1.35,
            "soundhole_center": 1.20,
            "form_center": 0.85,
            "bridge": 0.85,
        },
        widths={
            "form_center": ("span_frac", 0.45),
            "soundhole_center": ("span_frac", 0.35),
            "bridge": ("span_frac", 0.30),
            "neck_joint": ("span_frac", 0.25),
        },
        amplitude_mode="units",
        amplitude_units=1.8,
    )


class MidCurve(SideProfilePerControlTopCurve):
    PARAMETERS = SideProfileParameters(
        gammas={
            "neck_joint": 1.20,
            "soundhole_center": 1.10,
            "form_center": 0.95,
            "bridge": 0.95,
        },
        widths={
            "form_center": ("span_frac", 0.45),
            "soundhole_center": ("span_frac", 0.35),
            "bridge": ("span_frac", 0.30),
            "neck_joint": ("span_frac", 0.25),
        },
        amplitude_mode="units",
        amplitude_units=1.75,
    )


class FlatBackCurve(SideProfilePerControlTopCurve):
    PARAMETERS = SideProfileParameters(
        gammas={
            "neck_joint": 1.35,
            "soundhole_center": 1.20,
            "form_center": 1.05,
            "bridge": 1.05,
        },
        widths={
            "form_center": ("span_frac", 0.45),
            "soundhole_center": ("span_frac", 0.35),
            "bridge": ("span_frac", 0.30),
            "neck_joint": ("span_frac", 0.25),
        },
        amplitude_mode="units",
        amplitude_units=1.45,
    )


def resolve_top_curve(
    lute,
    top_curve,
    builder: Callable[[object, SideProfileParameters], Callable[[float], float]],
):
    if callable(top_curve) and not isinstance(top_curve, type):
        return top_curve
    try:
        if isinstance(top_curve, type) and issubclass(top_curve, TopCurve):
            return top_curve.build(lute, builder)
        if isinstance(top_curve, TopCurve):
            return top_curve.build(lute, builder)
    except Exception:
        pass
    return SideProfilePerControlTopCurve.build(lute, builder)


def resolve_widths(lute, params: SideProfileParameters) -> dict[str, float]:
    widths: dict[str, float] = {}
    if not params.widths:
        return widths
    span = abs(float(lute.form_bottom.x) - float(lute.form_top.x))
    for key, value in params.widths.items():
        if isinstance(value, (int, float)):
            widths[key] = float(value)
        elif isinstance(value, (tuple, list)) and len(value) == 2 and value[0] == "span_frac":
            widths[key] = float(value[1]) * span
    return widths


def resolve_amplitude(lute, params: SideProfileParameters) -> float:
    u = float(getattr(lute, "unit", 1.0))
    if params.amplitude_mode == "units" and params.amplitude_units is not None:
        return float(params.amplitude_units) * u
    return float(max(params.gammas.values()) if params.gammas else 1.0) * u


__all__ = [
    "SideProfileParameters",
    "TopCurve",
    "SideProfilePerControlTopCurve",
    "SimpleAmplitudeCurve",
    "DeepBackCurve",
    "MidCurve",
    "FlatBackCurve",
    "resolve_top_curve",
]
