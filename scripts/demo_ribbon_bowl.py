#!/usr/bin/env python3
"""Preview a single ribbon rib using the new ribbon-bowl scaffolding."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lute_bowl.ribbon_bowl import (
    RibbonSurface,
    default_terminal_points,
    edge_curve,
    normalize_outline_points,
    ribbon_surface_grid,
)
from plotting.bowl import set_axes_equal_3d

DEFAULT_LUTE = "lute_soundboard.ManolLavta"


def _resolve_class(path: str):
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Class path must include module name: {path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr}'") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lute", default=None, help=f"Fully qualified lute class (default: {DEFAULT_LUTE})")
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Rib blank width (units, default: 1/4 of lute.unit).",
    )
    parser.add_argument("--arc-samples", type=int, default=200, help="Samples per outline arc.")
    parser.add_argument("--edge-samples", type=int, default=250, help="Samples along each rib edge.")
    parser.add_argument("--surface-samples", type=int, default=200, help="Samples along the rib length.")
    parser.add_argument("--surface-width-samples", type=int, default=20, help="Samples across the rib width.")
    parser.add_argument("--show-edges", action="store_true", help="Overlay rib edge curves.")
    parser.add_argument(
        "--top-s",
        type=float,
        default=None,
        help="Top terminal point s (0-1, default: blank midline at s=0).",
    )
    parser.add_argument(
        "--top-t",
        type=float,
        default=None,
        help="Top terminal point t (width axis, default: blank midline at t=0).",
    )
    parser.add_argument(
        "--bottom-s",
        type=float,
        default=None,
        help="Bottom terminal point s (0-1, default: blank midline at s=1).",
    )
    parser.add_argument(
        "--bottom-t",
        type=float,
        default=None,
        help="Bottom terminal point t (width axis, default: blank midline at t=0).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _plot_soundboard_outline(ax, lute, samples_per_arc: int, surface) -> None:
    arcs = list(getattr(lute, "final_arcs", [])) + list(getattr(lute, "final_reflected_arcs", []))
    for idx, arc in enumerate(arcs):
        pts = np.asarray(arc.sample_points(samples_per_arc), dtype=float)
        if pts.size == 0:
            continue
        pts = normalize_outline_points(lute, pts)
        pts3 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0], dtype=float)])
        pts3 = surface.to_oriented(pts3)
        zs = pts3[:, 2]
        label = "soundboard" if idx == 0 else None
        ax.plot(pts3[:, 0], pts3[:, 1], zs, color="0.3", alpha=0.4, label=label)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lute_path = args.lute or DEFAULT_LUTE
    lute_cls = lute_path if isinstance(lute_path, type) else _resolve_class(lute_path)
    lute = lute_cls()

    width = float(args.width) if args.width is not None else float(lute.unit) / 4.0

    surface = RibbonSurface.from_outline(lute, samples_per_arc=args.arc_samples)
    top_point, bottom_point = default_terminal_points(surface)
    if any(value is not None for value in (args.top_s, args.top_t, args.bottom_s, args.bottom_t)):
        top_s = 0.0 if args.top_s is None else args.top_s
        top_t = 0.0 if args.top_t is None else args.top_t
        bottom_s = 1.0 if args.bottom_s is None else args.bottom_s
        bottom_t = 0.0 if args.bottom_t is None else args.bottom_t
        top_point = surface.point_at(top_s, top_t)
        bottom_point = surface.point_at(bottom_s, bottom_t)

    X, Y, Z = ribbon_surface_grid(
        surface,
        width,
        s_samples=args.surface_samples,
        t_samples=args.surface_width_samples,
    )

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.4], wspace=0.02)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    panel_ax = fig.add_subplot(grid[0, 1])
    panel_ax.set_axis_off()
    panel_ax.set_facecolor("0.96")
    surface_plot = ax.plot_surface(X, Y, Z, color="tab:blue", alpha=0.35, linewidth=0, antialiased=True)
    _plot_soundboard_outline(ax, lute, args.arc_samples, surface)

    ribs: list[np.ndarray] = []
    if args.show_edges:
        plane_a, plane_b = surface.edge_planes_for_terminal_line(top_point, bottom_point, width)
        ribs = [
            edge_curve(surface, plane_a, sample_count=args.edge_samples),
            edge_curve(surface, plane_b, sample_count=args.edge_samples),
        ]
        edge_lines = []
        for idx, rib in enumerate(ribs):
            label = "edges" if idx == 0 else None
            (line,) = ax.plot(rib[:, 0], rib[:, 1], rib[:, 2], color="tab:red", lw=1.4, label=label)
            edge_lines.append(line)
    else:
        edge_lines = []

    def _refresh_axes(x_grid, y_grid, z_grid, edge_curves: list[np.ndarray]) -> None:
        xs = [x_grid.ravel()]
        ys = [y_grid.ravel()]
        zs = [z_grid.ravel()]
        if edge_curves:
            xs.extend([curve[:, 0] for curve in edge_curves])
            ys.extend([curve[:, 1] for curve in edge_curves])
            zs.extend([curve[:, 2] for curve in edge_curves])
        set_axes_equal_3d(ax, xs=np.concatenate(xs), ys=np.concatenate(ys), zs=np.concatenate(zs))

    _refresh_axes(X, Y, Z, ribs)
    ax.set_xlabel("X (along spine)")
    ax.set_ylabel("Y (across soundboard)")
    ax.set_zlabel("Z (depth)")
    ax.legend(loc="best")
    ax.set_title(type(lute).__name__)

    panel_x = 0.08
    panel_w = 0.84
    value_y = 0.92
    slider_y = 0.82
    slider_h = 0.05
    slider_w = 0.58
    reset_w = 0.12
    gap_w = panel_w - slider_w - reset_w
    reset_y = slider_y
    reset_h = slider_h * 0.85
    slider_ax = panel_ax.inset_axes([panel_x, slider_y, slider_w, slider_h])
    width_min = float(lute.unit) * 0.15
    width_max = float(lute.unit) * 1.0
    width_min = min(width_min, width)
    width_max = max(width_max, width)
    unit_scale = (
        float(lute.unit_in_mm()) / float(lute.unit)
        if hasattr(lute, "unit_in_mm") and hasattr(lute, "unit")
        else 1.0
    )
    value_text = panel_ax.text(
        panel_x,
        value_y,
        "",
        transform=panel_ax.transAxes,
        ha="left",
        va="top",
    )
    width_slider = Slider(
        slider_ax,
        "",
        valmin=width_min,
        valmax=width_max,
        valinit=width,
        valfmt="%.1f",
    )
    default_width = width

    reset_ax = panel_ax.inset_axes([panel_x + slider_w + gap_w, reset_y, reset_w, reset_h])
    reset_button = Button(reset_ax, "↺")
    width_slider.label.set_visible(False)
    width_slider.valtext.set_visible(False)

    def _format_width(val: float) -> str:
        unit_val = float(val) / float(lute.unit)
        mm_val = float(val) * unit_scale
        return f"Rib Width | {unit_val:.2f} u / {mm_val:.2f} mm"

    def _update_width(val: float) -> None:
        nonlocal surface_plot, ribs
        new_width = float(val)
        x_grid, y_grid, z_grid = ribbon_surface_grid(
            surface,
            new_width,
            s_samples=args.surface_samples,
            t_samples=args.surface_width_samples,
        )
        surface_plot.remove()
        surface_plot = ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color="tab:blue",
            alpha=0.35,
            linewidth=0,
            antialiased=True,
        )
        new_ribs: list[np.ndarray] = []
        if args.show_edges:
            try:
                plane_a, plane_b = surface.edge_planes_for_terminal_line(top_point, bottom_point, new_width)
                new_ribs = [
                    edge_curve(surface, plane_a, sample_count=args.edge_samples),
                    edge_curve(surface, plane_b, sample_count=args.edge_samples),
                ]
            except ValueError:
                new_ribs = []
            if len(edge_lines) >= 2:
                for idx in range(2):
                    if idx < len(new_ribs):
                        edge_lines[idx].set_data(new_ribs[idx][:, 0], new_ribs[idx][:, 1])
                        edge_lines[idx].set_3d_properties(new_ribs[idx][:, 2])
                        edge_lines[idx].set_visible(True)
                    else:
                        edge_lines[idx].set_visible(False)
        ribs = new_ribs
        _refresh_axes(x_grid, y_grid, z_grid, ribs)
        value_text.set_text(_format_width(new_width))
        fig.canvas.draw_idle()

    width_slider.on_changed(_update_width)
    value_text.set_text(_format_width(width))
    reset_button.on_clicked(lambda _event: width_slider.set_val(default_width))
    separator_y = slider_y - 0.03
    panel_ax.plot(
        [panel_x, panel_x + panel_w],
        [separator_y, separator_y],
        transform=panel_ax.transAxes,
        color="0.8",
        lw=1.0,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
