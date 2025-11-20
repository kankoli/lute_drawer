"""Plotting utilities, lazily importing heavy matplotlib dependencies."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "plot_bowl",
    "plot_rib_surfaces",
    "plot_rib_surface_with_planes",
    "set_axes_equal_3d",
    "plot_mold_sections_2d",
    "write_mold_sections_step",
    "SvgRenderer",
]


def __getattr__(name: str) -> Any:
    if name == "SvgRenderer":
        return import_module("plotting.svg").SvgRenderer
    if name in {"plot_bowl", "set_axes_equal_3d", "plot_mold_sections_2d"}:
        module = import_module("plotting.bowl")
        return getattr(module, name)
    if name in {"plot_rib_surfaces", "plot_rib_surface_with_planes"}:
        module = import_module("plotting.ribs")
        return getattr(module, name)
    if name == "write_mold_sections_step":
        return import_module("plotting.step_renderers").write_mold_sections_step
    raise AttributeError(f"module 'plotting' has no attribute '{name}'")
