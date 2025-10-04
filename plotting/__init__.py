"""Plotting utilities, lazily importing heavy matplotlib dependencies."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "plot_bowl",
    "plot_rib_surfaces",
    "set_axes_equal_3d",
    "plot_mold_sections_2d",
    "plot_mold_sections_3d",
    "SvgRenderer",
]


def __getattr__(name: str) -> Any:
    if name == "SvgRenderer":
        return import_module("plotting.svg").SvgRenderer
    if name in {"plot_bowl", "plot_rib_surfaces", "set_axes_equal_3d"}:
        module = import_module("plotting.bowl")
        return getattr(module, name)
    if name in {"plot_mold_sections_2d", "plot_mold_sections_3d"}:
        module = import_module("plotting.mold")
        return getattr(module, name)
    raise AttributeError(f"module 'plotting' has no attribute '{name}'")
