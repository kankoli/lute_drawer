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
    def build(cls, lute):
        raise NotImplementedError


class SideProfilePerControlTopCurve(TopCurve):
    PARAMETERS = SideProfileParameters(gammas={}, widths={})

    @classmethod
    def _parameters(cls) -> SideProfileParameters:
        return cls.PARAMETERS

    @classmethod
    def build(cls, lute):  # type: ignore[override]
        params = cls._parameters()
        if not params.gammas:
            raise RuntimeError("SideProfilePerControlTopCurve requires non-empty SHAPE_GAMMAS")

        from . import bowl_from_soundboard as bfs

        xs = bfs._select_section_positions(lute, params.samples, params.margin, debug=False)

        widths = []
        for X in xs:
            try:
                hit = bfs._extract_side_points_at_X(lute, X)
            except RuntimeError:
                widths.append(0.0)
                continue
            if hit is None:
                widths.append(0.0)
                continue
            L, R, _ = hit
            y_spine = bfs._spine_point_at_X(lute, X)
            widths.append(max(abs(float(L[1]) - y_spine), abs(float(R[1]) - y_spine)))

        W = np.asarray(widths, float)
        if W.size == 0 or float(W.max()) < 1e-12:
            return lambda _x: 0.0

        W[0] = 0.0
        W[-1] = 0.0
        N = W / float(W.max())

        ctrl_x_all = {
            "neck_joint": float(lute.point_neck_joint.x),
            "form_center": float(lute.form_center.x),
            "bridge": float(lute.bridge.x),
        }

        getter = getattr(lute, "_get_soundhole_center", None)
        if callable(getter):
            try:
                center = getter()
                if center is not None:
                    ctrl_x_all["soundhole_center"] = float(center.x)
            except Exception:
                pass

        if "soundhole_center" not in ctrl_x_all and getattr(lute, "soundhole_center", None) is not None:
            try:
                ctrl_x_all["soundhole_center"] = float(lute.soundhole_center.x)
            except Exception:
                pass

        gammas = {k: params.gammas[k] for k in params.gammas if k in ctrl_x_all}
        if not gammas:
            amplitude = resolve_amplitude(lute, params)

            def z_top_lin(x):
                return float(amplitude * np.interp(float(x), xs, N, left=0.0, right=0.0))

            return z_top_lin

        xc = np.array([ctrl_x_all[k] for k in gammas], float)
        gc = np.array([float(gammas[k]) for k in gammas], float)
        order = np.argsort(xc)
        xc = xc[order]
        gc = gc[order]

        widths_map = resolve_widths(lute, params)
        sig = []
        keys_ordered = np.array(list(gammas.keys()))[order]
        for idx, key in enumerate(keys_ordered):
            width = widths_map.get(key)
            if width and width > 0.0:
                sig.append(float(width))
            else:
                left_gap = xc[idx] - (xc[idx - 1] if idx - 1 >= 0 else float(lute.form_top.x))
                right_gap = (xc[idx + 1] if idx + 1 < len(xc) else float(lute.form_bottom.x)) - xc[idx]
                local = 0.5 * (abs(left_gap) + abs(right_gap))
                sig.append(max(1e-6, float(params.width_factor) * local))
        sig = np.array(sig, float)

        Xdiff2 = (xs[:, None] - xc[None, :]) ** 2
        if params.kernel == "gauss":
            w = np.exp(-Xdiff2 / (2.0 * (sig[None, :] ** 2 + 1e-12)))
        else:
            w = 1.0 / (1.0 + (Xdiff2 / (sig[None, :] ** 2 + 1e-12)))
        Wnorm = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
        log_gammas = np.log(np.clip(gc[None, :], 1e-6, None))
        logE = np.sum(Wnorm * log_gammas, axis=1)
        E = np.clip(np.exp(logE), 1.0 - float(params.max_exponent_delta), 1.0 + float(params.max_exponent_delta))

        if params.gate_start > 0.0 or params.gate_full > 0.0:
            a = float(params.gate_start)
            b = float(params.gate_full)
            if b <= a + 1e-9:
                b = min(1.0, a + 1e-3)
            t = (N - a) / (b - a)
            t = np.clip(t, 0.0, 1.0)
            gate = t * t * (3 - 2 * t)
            E = 1.0 + gate * (E - 1.0)

        N_shaped = N ** E
        amplitude = resolve_amplitude(lute, params)

        def z_top(x):
            return float(amplitude * np.interp(float(x), xs, N_shaped, left=0.0, right=0.0))

        return z_top


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
            "neck_joint": 1.55,
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
        amplitude_units=2.0,
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
    "resolve_widths",
    "resolve_amplitude",
]
