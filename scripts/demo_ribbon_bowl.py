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
    ChainedTerminalStrategy,
    RibbonSurface,
    apply_reflections_to_points,
    build_regular_rib_chain,
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
    top_s = 0.0 if args.top_s is None else args.top_s
    top_t = 0.0 if args.top_t is None else args.top_t
    bottom_s = 1.0 if args.bottom_s is None else args.bottom_s
    bottom_t = 0.0 if args.bottom_t is None else args.bottom_t
    top_z_offset = 0.0
    bottom_z_offset = 0.0
    default_top_s = top_s
    default_bottom_s = bottom_s
    default_top_z_offset = top_z_offset
    default_bottom_z_offset = bottom_z_offset
    top_s_min = -0.25
    top_s_max = 0.25
    bottom_s_min = 0.75
    bottom_s_max = 1.25

    X_base, Y_base, Z_base = ribbon_surface_grid(
        surface,
        width,
        s_samples=args.surface_samples,
        t_samples=args.surface_width_samples,
    )

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider, TextBox

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.4], wspace=0.02)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    panel_ax = fig.add_subplot(grid[0, 1])
    panel_ax.set_axis_off()
    panel_ax.set_facecolor("0.96")
    X, Y, Z = X_base, Y_base, Z_base
    surface_plot = ax.plot_surface(X, Y, Z, color="tab:blue", alpha=0.35, linewidth=0, antialiased=True)
    _plot_soundboard_outline(ax, lute, args.arc_samples, surface)

    ribs: list[np.ndarray] = []
    edge_lines: list = []

    max_rib_count = 34
    rib_count = 3
    default_rib_count = rib_count
    rib_text_updating = False

    def _sanitize_rib_count(raw: str, current: int) -> int:
        try:
            value = int(float(raw))
        except (TypeError, ValueError):
            return current
        value = max(1, min(max_rib_count, value))
        if value % 2 == 0:
            value -= 1
        return max(1, value)

    def _rib_pairs(count: int) -> int:
        return max(0, (int(count) - 1) // 2)

    def _compute_edges(
        width_val: float,
        top_s_val: float,
        bottom_s_val: float,
        rib_count_val: int,
    ) -> list[np.ndarray]:
        if not args.show_edges:
            return []
        if float(top_s_val) >= float(bottom_s_val):
            return []
        min_s = min(top_s_min, bottom_s_min)
        max_s = max(top_s_max, bottom_s_max)
        terminal_strategy = ChainedTerminalStrategy(
            lambda _idx: 0.0,
            lambda _idx: 0.0,
            base_top_s=top_s_val,
            base_bottom_s=bottom_s_val,
            min_s=min_s,
            max_s=max_s,
        )
        try:
            center_rib, chained_ribs = build_regular_rib_chain(
                surface,
                width_val,
                terminal_strategy,
                pairs=_rib_pairs(rib_count_val),
                center_top_t=top_t,
                center_bottom_t=bottom_t,
            )
        except ValueError:
            return []
        base_pos = edge_curve(
            surface,
            center_rib.positive_plane,
            sample_count=args.edge_samples,
            s_min=top_s_val,
            s_max=bottom_s_val,
        )
        base_neg = edge_curve(
            surface,
            center_rib.negative_plane,
            sample_count=args.edge_samples,
            s_min=top_s_val,
            s_max=bottom_s_val,
        )
        curves = [base_pos, base_neg]
        for rib in chained_ribs:
            outer_source = base_neg if rib.inner_source == "pos" else base_pos
            curves.append(apply_reflections_to_points(outer_source, rib.mirrors))
        return curves

    def _set_edges(edge_curves: list[np.ndarray]) -> None:
        nonlocal ribs
        for line in edge_lines:
            line.remove()
        edge_lines.clear()
        for idx, curve in enumerate(edge_curves):
            label = "edges" if idx == 0 else None
            (line,) = ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color="tab:red", lw=1.4, label=label)
            edge_lines.append(line)
        ribs = edge_curves

    ribs = _compute_edges(width, top_s, bottom_s, rib_count)
    if args.show_edges:
        _set_edges(ribs)

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
    slider_h = 0.05
    slider_w = 0.58
    reset_w = 0.12
    gap_w = panel_w - slider_w - reset_w
    reset_h = slider_h * 0.85
    base_value_y = 0.92
    base_slider_y = 0.82
    control_gap = 0.14
    width_min = float(lute.unit) * 0.15
    width_max = float(lute.unit) * 1.0
    width_min = min(width_min, width)
    width_max = max(width_max, width)
    unit_scale = (
        float(lute.unit_in_mm()) / float(lute.unit)
        if hasattr(lute, "unit_in_mm") and hasattr(lute, "unit")
        else 1.0
    )
    def _format_s(label: str, val: float) -> str:
        return f"{label} | {val:.2f}"

    def _format_offset(label: str, val: float) -> str:
        unit_val = float(val) / float(lute.unit)
        mm_val = float(val) * unit_scale
        return f"{label} | {unit_val:.2f} u / {mm_val:.2f} mm"

    def _format_rib_count(val: int) -> str:
        return f"Ribs | {int(val)}"

    default_width = width

    def _format_width(val: float) -> str:
        unit_val = float(val) / float(lute.unit)
        mm_val = float(val) * unit_scale
        return f"Rib Width | {unit_val:.2f} u / {mm_val:.2f} mm"

    def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis) + 1e-12
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=float,
        )

    def _rotate_vector(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        return _rotation_matrix(axis, angle) @ vec

    def _solve_tilt_angle(
        top_ref: np.ndarray,
        bottom_ref: np.ndarray,
        delta_z: float,
        axis: np.ndarray,
        *,
        max_angle_deg: float = 60.0,
        samples: int = 721,
    ) -> float:
        if abs(delta_z) <= 1e-9:
            return 0.0
        axis = np.asarray(axis, dtype=float)
        v = np.asarray(bottom_ref, dtype=float) - np.asarray(top_ref, dtype=float)
        max_angle = np.deg2rad(max_angle_deg)
        angles = np.linspace(-max_angle, max_angle, max(3, int(samples)))
        best_angle = 0.0
        best_err = float("inf")
        for angle in angles:
            rotated = _rotate_vector(v, axis, angle)
            dz = float(rotated[2] - v[2])
            err = abs(dz - float(delta_z))
            if err < best_err:
                best_err = err
                best_angle = float(angle)
        return best_angle

    def _apply_z_transform(points: np.ndarray) -> np.ndarray:
        if abs(top_z_offset) <= 1e-9 and abs(bottom_z_offset) <= 1e-9:
            return np.asarray(points, dtype=float)
        top_ref = surface.point_at(top_s, top_t)
        bottom_ref = surface.point_at(bottom_s, bottom_t)
        axis = surface.width_axis
        z_shift = float(top_z_offset)
        delta_z = float(bottom_z_offset) - float(top_z_offset)
        angle = _solve_tilt_angle(top_ref, bottom_ref, delta_z, axis)
        pts = np.asarray(points, dtype=float)
        if abs(angle) <= 1e-12:
            rotated = pts.copy()
        else:
            rot = _rotation_matrix(axis, angle)
            rotated = (pts - top_ref) @ rot.T + top_ref
        rotated[:, 2] += z_shift
        return rotated

    def _depth_range(samples: int = 200) -> tuple[float, float]:
        s_vals = np.linspace(0.0, 1.0, max(2, int(samples)))
        z_vals = surface.centerline_points(s_vals)[:, 2]
        return float(z_vals.min()), float(z_vals.max())

    depth_min, depth_max = _depth_range()
    depth_span = max(1e-6, depth_max - depth_min)
    z_offset_span = 0.5 * depth_span
    top_z_min = -z_offset_span
    top_z_max = z_offset_span
    bottom_z_min = -z_offset_span
    bottom_z_max = z_offset_span

    def _add_control(
        row_idx: int,
        valmin: float,
        valmax: float,
        valinit: float,
    ) -> tuple[Slider, Button, any]:
        value_y = base_value_y - row_idx * control_gap
        slider_y = base_slider_y - row_idx * control_gap
        slider_ax = panel_ax.inset_axes([panel_x, slider_y, slider_w, slider_h])
        value_text = panel_ax.text(
            panel_x,
            value_y,
            "",
            transform=panel_ax.transAxes,
            ha="left",
            va="top",
        )
        slider = Slider(
            slider_ax,
            "",
            valmin=valmin,
            valmax=valmax,
            valinit=valinit,
            valfmt="%.2f",
        )
        slider.label.set_visible(False)
        slider.valtext.set_visible(False)
        reset_ax = panel_ax.inset_axes([panel_x + slider_w + gap_w, slider_y, reset_w, reset_h])
        reset_button = Button(reset_ax, "↺")
        separator_y = slider_y - 0.03
        panel_ax.plot(
            [panel_x, panel_x + panel_w],
            [separator_y, separator_y],
            transform=panel_ax.transAxes,
            color="0.8",
            lw=1.0,
        )
        return slider, reset_button, value_text

    def _transformed_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = np.column_stack([X_base.ravel(), Y_base.ravel(), Z_base.ravel()])
        pts = _apply_z_transform(pts)
        x_new = pts[:, 0].reshape(X_base.shape)
        y_new = pts[:, 1].reshape(Y_base.shape)
        z_new = pts[:, 2].reshape(Z_base.shape)
        return x_new, y_new, z_new

    def _update_surface_plot() -> None:
        nonlocal X, Y, Z, surface_plot
        X, Y, Z = _transformed_grid()
        surface_plot.remove()
        surface_plot = ax.plot_surface(
            X,
            Y,
            Z,
            color="tab:blue",
            alpha=0.35,
            linewidth=0,
            antialiased=True,
        )

    def _rebuild_surface(new_width: float) -> None:
        nonlocal X_base, Y_base, Z_base
        X_base, Y_base, Z_base = ribbon_surface_grid(
            surface,
            new_width,
            s_samples=args.surface_samples,
            t_samples=args.surface_width_samples,
        )
        _update_surface_plot()

    def _update_edges() -> None:
        new_ribs = _compute_edges(width, top_s, bottom_s, rib_count)
        if new_ribs:
            new_ribs = [_apply_z_transform(curve) for curve in new_ribs]
        _set_edges(new_ribs)
        _refresh_axes(X, Y, Z, new_ribs)

    def _update_width(val: float) -> None:
        nonlocal width
        width = float(val)
        _rebuild_surface(width)
        _update_edges()
        width_value_text.set_text(_format_width(width))
        fig.canvas.draw_idle()

    def _update_top_s(val: float) -> None:
        nonlocal top_s
        top_s = float(val)
        _update_surface_plot()
        _update_edges()
        top_value_text.set_text(_format_s("Top s", top_s))
        fig.canvas.draw_idle()

    def _update_bottom_s(val: float) -> None:
        nonlocal bottom_s
        bottom_s = float(val)
        _update_surface_plot()
        _update_edges()
        bottom_value_text.set_text(_format_s("Bottom s", bottom_s))
        fig.canvas.draw_idle()

    def _update_top_z(val: float) -> None:
        nonlocal top_z_offset
        top_z_offset = float(val)
        _update_surface_plot()
        _update_edges()
        top_z_value_text.set_text(_format_offset("Top z", top_z_offset))
        fig.canvas.draw_idle()

    def _update_bottom_z(val: float) -> None:
        nonlocal bottom_z_offset
        bottom_z_offset = float(val)
        _update_surface_plot()
        _update_edges()
        bottom_z_value_text.set_text(_format_offset("Bottom z", bottom_z_offset))
        fig.canvas.draw_idle()

    width_slider, width_reset, width_value_text = _add_control(0, width_min, width_max, width)
    top_slider, top_reset, top_value_text = _add_control(1, top_s_min, top_s_max, top_s)
    bottom_slider, bottom_reset, bottom_value_text = _add_control(2, bottom_s_min, bottom_s_max, bottom_s)
    top_z_slider, top_z_reset, top_z_value_text = _add_control(3, top_z_min, top_z_max, top_z_offset)
    bottom_z_slider, bottom_z_reset, bottom_z_value_text = _add_control(
        4, bottom_z_min, bottom_z_max, bottom_z_offset
    )
    ribs_text_ax = panel_ax.inset_axes([panel_x, base_slider_y - 5 * control_gap, slider_w, slider_h])
    ribs_text_box = TextBox(ribs_text_ax, "", initial=str(rib_count))
    ribs_text_box.label.set_visible(False)
    ribs_value_text = panel_ax.text(
        panel_x,
        base_value_y - 5 * control_gap,
        "",
        transform=panel_ax.transAxes,
        ha="left",
        va="top",
    )
    ribs_reset_ax = panel_ax.inset_axes(
        [panel_x + slider_w + gap_w, base_slider_y - 5 * control_gap, reset_w, reset_h]
    )
    ribs_reset = Button(ribs_reset_ax, "↺")
    separator_y = base_slider_y - 5 * control_gap - 0.03
    panel_ax.plot(
        [panel_x, panel_x + panel_w],
        [separator_y, separator_y],
        transform=panel_ax.transAxes,
        color="0.8",
        lw=1.0,
    )

    width_slider.on_changed(_update_width)
    top_slider.on_changed(_update_top_s)
    bottom_slider.on_changed(_update_bottom_s)
    top_z_slider.on_changed(_update_top_z)
    bottom_z_slider.on_changed(_update_bottom_z)

    def _update_rib_count(text: str) -> None:
        nonlocal rib_count, rib_text_updating
        if rib_text_updating:
            return
        rib_text_updating = True
        try:
            new_count = _sanitize_rib_count(text, rib_count)
            rib_count = new_count
            ribs_value_text.set_text(_format_rib_count(rib_count))
            if str(new_count) != str(text).strip():
                ribs_text_box.set_val(str(new_count))
            _update_edges()
            fig.canvas.draw_idle()
        finally:
            rib_text_updating = False

    ribs_text_box.on_submit(_update_rib_count)

    width_value_text.set_text(_format_width(width))
    top_value_text.set_text(_format_s("Top s", top_s))
    bottom_value_text.set_text(_format_s("Bottom s", bottom_s))
    top_z_value_text.set_text(_format_offset("Top z", top_z_offset))
    bottom_z_value_text.set_text(_format_offset("Bottom z", bottom_z_offset))
    ribs_value_text.set_text(_format_rib_count(rib_count))

    width_reset.on_clicked(lambda _event: width_slider.set_val(default_width))
    top_reset.on_clicked(lambda _event: top_slider.set_val(default_top_s))
    bottom_reset.on_clicked(lambda _event: bottom_slider.set_val(default_bottom_s))
    top_z_reset.on_clicked(lambda _event: top_z_slider.set_val(default_top_z_offset))
    bottom_z_reset.on_clicked(lambda _event: bottom_z_slider.set_val(default_bottom_z_offset))
    ribs_reset.on_clicked(lambda _event: ribs_text_box.set_val(str(default_rib_count)))
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
