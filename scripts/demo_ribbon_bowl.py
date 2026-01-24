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


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def _available_lutes() -> list[tuple[str, type]]:
    try:
        import lute_soundboard as ls
    except Exception:
        return []
    names = [n for n in getattr(ls, "__all__", []) if n not in ("LuteSoundboard", "SoundboardMeasurement")]
    choices: list[tuple[str, type]] = []
    for name in names:
        cls = getattr(ls, name, None)
        if isinstance(cls, type):
            choices.append((_class_path(cls), cls))
    return sorted(choices, key=lambda item: item[0])


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
    parser.add_argument(
        "--hide-neck-plane",
        action="store_false",
        default=True,
        dest="show_neck_plane",
        help="Hide the neck-joint plane and its intersection with the bowl surface.",
    )
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


def _plot_soundboard_outline(ax, lute, samples_per_arc: int, surface) -> list:
    arcs = list(getattr(lute, "final_arcs", [])) + list(getattr(lute, "final_reflected_arcs", []))
    lines = []
    for idx, arc in enumerate(arcs):
        pts = np.asarray(arc.sample_points(samples_per_arc), dtype=float)
        if pts.size == 0:
            continue
        pts = normalize_outline_points(lute, pts)
        pts3 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(pts.shape[0], dtype=float)])
        pts3 = surface.to_oriented(pts3)
        zs = pts3[:, 2]
        label = "soundboard" if idx == 0 else None
        (line,) = ax.plot(pts3[:, 0], pts3[:, 1], zs, color="0.3", alpha=0.4, label=label)
        lines.append(line)
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lute_path = args.lute or DEFAULT_LUTE
    lute_cls = lute_path if isinstance(lute_path, type) else _resolve_class(lute_path)
    lute = lute_cls()
    lute_choices = _available_lutes()
    selected_path = lute_path if isinstance(lute_path, str) else _class_path(lute_cls)
    lute_index = None
    for idx, (path, cls) in enumerate(lute_choices):
        if path == selected_path or cls is lute_cls:
            lute_index = idx
            break
    if lute_index is None:
        lute_choices.insert(0, (_class_path(lute_cls), lute_cls))
        lute_index = 0
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
    top_s_min = -0.10
    top_s_max = 0.10
    bottom_s_min = 0.90
    bottom_s_max = 1.10

    X_base, Y_base, Z_base = ribbon_surface_grid(
        surface,
        width,
        s_samples=args.surface_samples,
        t_samples=args.surface_width_samples,
    )

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.6, 1.4],
        height_ratios=[1.0, 0.12],
        wspace=0.02,
        hspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    panel_ax = fig.add_subplot(grid[0, 1])
    width_ax = fig.add_subplot(grid[1, :])
    panel_ax.set_axis_off()
    panel_ax.set_facecolor("0.96")
    width_ax.set_axis_off()
    width_ax.set_facecolor("0.96")
    X, Y, Z = X_base, Y_base, Z_base
    surface_plot = ax.plot_surface(X, Y, Z, color="tab:blue", alpha=0.35, linewidth=0, antialiased=True)
    soundboard_lines = _plot_soundboard_outline(ax, lute, args.arc_samples, surface)

    ribs: list[np.ndarray] = []
    edge_lines: list = []
    neck_plane = None
    neck_curve_line = None

    max_rib_count = 25
    rib_count = 19
    default_rib_count = rib_count
    rib_text_updating = False

    def _sanitize_rib_count(raw: str, current: int) -> int:
        try:
            value = int(float(raw))
        except (TypeError, ValueError):
            return current
        value = max(5, min(max_rib_count, value))
        if value % 2 == 0:
            value -= 1
        return max(5, value)

    def _rib_pairs(count: int) -> int:
        return max(0, (int(count) - 1) // 2)

    def _compute_edges(
        width_val: float,
        top_s_val: float,
        bottom_s_val: float,
        rib_count_val: int,
    ) -> list[np.ndarray]:
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
    _set_edges(ribs)

    def _neck_plane_x() -> float | None:
        neck_point = getattr(lute, "point_neck_joint", None)
        if neck_point is None:
            return None
        pt2 = np.array([[float(neck_point.x), float(neck_point.y)]], dtype=float)
        pt2 = normalize_outline_points(lute, pt2)
        pt3 = np.array([[pt2[0, 0], pt2[0, 1], 0.0]], dtype=float)
        pt3 = surface.to_oriented(pt3)[0]
        return float(pt3[0])

    neck_plane_x = _neck_plane_x()

    def _neck_profile_points(x_val: float, samples: int = 260) -> np.ndarray:
        s_vals = np.linspace(top_s, bottom_s, max(2, int(samples)))
        base = surface.centerline_points(s_vals)
        axis = surface.width_axis
        rot, pivot, z_shift, angle = _current_transform()
        if abs(angle) <= 1e-12:
            rot_base = base
            rot_axis = axis
        else:
            rot_base = (base - pivot) @ rot.T + pivot
            rot_axis = rot @ axis
        denom = float(rot_axis[0])
        if abs(denom) <= 1e-9:
            close = np.isclose(rot_base[:, 0], x_val, atol=1e-6)
            if not np.any(close):
                return np.empty((0, 3), dtype=float)
            pts = rot_base[close].copy()
            pts[:, 2] += z_shift
            return pts
        t_vals = (float(x_val) - rot_base[:, 0]) / denom
        half_width = 0.5 * float(width)
        mask = (t_vals >= -half_width) & (t_vals <= half_width)
        if not np.any(mask):
            return np.empty((0, 3), dtype=float)
        pts = rot_base[mask] + t_vals[mask][:, None] * rot_axis
        pts[:, 2] += z_shift
        return pts

    def _neck_plane_center(x_val: float, samples: int = 260) -> tuple[float, float]:
        s_vals = np.linspace(top_s, bottom_s, max(2, int(samples)))
        base = surface.centerline_points(s_vals)
        rot, pivot, z_shift, angle = _current_transform()
        if abs(angle) <= 1e-12:
            rot_base = base
        else:
            rot_base = (base - pivot) @ rot.T + pivot
        idx = int(np.argmin(np.abs(rot_base[:, 0] - float(x_val))))
        center = rot_base[idx].copy()
        center[2] += z_shift
        return float(center[1]), float(center[2])

    def _plot_neck_plane(x_val: float) -> None:
        nonlocal neck_plane, neck_curve_line
        if neck_plane is not None:
            neck_plane.remove()
        if neck_curve_line is not None:
            neck_curve_line.remove()
            neck_curve_line = None
        plane_size = 2.0 * float(lute.unit)
        half = 0.5 * plane_size
        y_center, z_center = _neck_plane_center(float(x_val))
        y_min, y_max = y_center - half, y_center + half
        z_min, z_max = z_center - half, z_center + half
        Yp, Zp = np.meshgrid([y_min, y_max], [z_min, z_max])
        Xp = np.full_like(Yp, float(x_val))
        neck_plane = ax.plot_surface(Xp, Yp, Zp, color="orange", alpha=0.15, linewidth=0)
        curve = _neck_profile_points(float(x_val))
        if curve.shape[0] >= 2:
            (neck_curve_line,) = ax.plot(
                curve[:, 0],
                curve[:, 1],
                curve[:, 2],
                color="orange",
                lw=1.6,
                label="neck profile",
            )

    def _clip_curve_by_neck_plane(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if args.show_neck_plane and neck_plane_x is not None:
            pts = pts[pts[:, 0] >= float(neck_plane_x)]
        pts = pts[pts[:, 2] >= 0.0]
        return pts

    def _append_bounds(store: list[np.ndarray], arr: np.ndarray) -> None:
        vals = np.asarray(arr, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            store.append(vals)

    def _refresh_axes(
        x_grid,
        y_grid,
        z_grid,
        edge_curves: list[np.ndarray],
        *,
        preserve_view: bool = True,
    ) -> None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        _append_bounds(xs, x_grid)
        _append_bounds(ys, y_grid)
        _append_bounds(zs, z_grid)
        if edge_curves:
            for curve in edge_curves:
                _append_bounds(xs, curve[:, 0])
                _append_bounds(ys, curve[:, 1])
                _append_bounds(zs, curve[:, 2])
        if not xs or not ys or not zs:
            return
        if preserve_view:
            return
        set_axes_equal_3d(ax, xs=np.concatenate(xs), ys=np.concatenate(ys), zs=np.concatenate(zs))

    _refresh_axes(X, Y, Z, ribs, preserve_view=False)
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
        return f"Rib Width | {unit_val:.3f} u / {mm_val:.3f} mm"

    def _format_lute_label() -> str:
        return f"Lute | {type(lute).__name__}"

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

    def _current_transform() -> tuple[np.ndarray, np.ndarray, float, float]:
        top_ref = surface.point_at(top_s, top_t)
        bottom_ref = surface.point_at(bottom_s, bottom_t)
        axis = surface.width_axis
        z_shift = float(top_z_offset)
        delta_z = float(bottom_z_offset) - float(top_z_offset)
        angle = _solve_tilt_angle(top_ref, bottom_ref, delta_z, axis)
        if abs(angle) <= 1e-12:
            return np.eye(3), top_ref, z_shift, 0.0
        return _rotation_matrix(axis, angle), top_ref, z_shift, angle

    def _apply_z_transform(points: np.ndarray) -> np.ndarray:
        if abs(top_z_offset) <= 1e-9 and abs(bottom_z_offset) <= 1e-9:
            return np.asarray(points, dtype=float)
        rot, pivot, z_shift, angle = _current_transform()
        pts = np.asarray(points, dtype=float)
        if abs(angle) <= 1e-12:
            rotated = pts.copy()
        else:
            rotated = (pts - pivot) @ rot.T + pivot
        rotated[:, 2] += z_shift
        return rotated

    top_z_min = -0.30 * float(lute.unit)
    top_z_max = 0.10 * float(lute.unit)
    bottom_z_min = -0.50 * float(lute.unit)
    bottom_z_max = 0.50 * float(lute.unit)

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
        if args.show_neck_plane and neck_plane_x is not None:
            mask = x_new < float(neck_plane_x)
            if np.any(mask):
                x_new = x_new.copy()
                y_new = y_new.copy()
                z_new = z_new.copy()
                x_new[mask] = np.nan
                y_new[mask] = np.nan
                z_new[mask] = np.nan
        mask = z_new < 0.0
        if np.any(mask):
            x_new = x_new.copy()
            y_new = y_new.copy()
            z_new = z_new.copy()
            x_new[mask] = np.nan
            y_new[mask] = np.nan
            z_new[mask] = np.nan
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
        if args.show_neck_plane and neck_plane_x is not None:
            _plot_neck_plane(neck_plane_x)

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
            new_ribs = [_clip_curve_by_neck_plane(_apply_z_transform(curve)) for curve in new_ribs]
        _set_edges(new_ribs)
        _refresh_axes(X, Y, Z, new_ribs, preserve_view=True)
        if args.show_neck_plane and neck_plane_x is not None:
            _plot_neck_plane(neck_plane_x)

    if args.show_neck_plane and neck_plane_x is not None:
        _update_surface_plot()
        _update_edges()

    def _update_width(val: float) -> None:
        nonlocal width
        width = float(val)
        _rebuild_surface(width)
        _update_edges()
        width_value_text.set_text(_format_width(width))
        fig.canvas.draw_idle()

    width_step = 0.001 * float(lute.unit)

    def _nudge_width(step: float) -> None:
        new_val = float(width_slider.val) + float(step)
        new_val = max(width_min, min(width_max, new_val))
        width_slider.set_val(new_val)

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

    width_value_text = width_ax.text(
        0.02,
        0.9,
        "",
        transform=width_ax.transAxes,
        ha="left",
        va="top",
    )
    width_slider_ax = width_ax.inset_axes([0.02, 0.2, 0.74, 0.55])
    width_slider = Slider(
        width_slider_ax,
        "",
        valmin=width_min,
        valmax=width_max,
        valinit=width,
        valfmt="%.3f",
    )
    width_slider.label.set_visible(False)
    width_slider.valtext.set_visible(False)
    width_button_h = 0.55
    width_button_y = 0.2
    width_button_w = 0.05
    width_minus_ax = width_ax.inset_axes([0.78, width_button_y, width_button_w, width_button_h])
    width_plus_ax = width_ax.inset_axes([0.84, width_button_y, width_button_w, width_button_h])
    width_reset_ax = width_ax.inset_axes([0.90, width_button_y, 0.07, width_button_h])
    width_minus = Button(width_minus_ax, "-")
    width_plus = Button(width_plus_ax, "+")
    width_reset = Button(width_reset_ax, "↺")

    lute_row = 0
    lute_value_text = panel_ax.text(
        panel_x,
        base_value_y - lute_row * control_gap,
        "",
        transform=panel_ax.transAxes,
        ha="left",
        va="top",
    )
    lute_button_y = base_slider_y - lute_row * control_gap
    lute_button_w = 0.08
    lute_button_gap = 0.02
    lute_prev_ax = panel_ax.inset_axes([panel_x + slider_w + 0.01, lute_button_y, lute_button_w, reset_h])
    lute_next_ax = panel_ax.inset_axes(
        [panel_x + slider_w + 0.01 + lute_button_w + lute_button_gap, lute_button_y, lute_button_w, reset_h]
    )
    lute_prev = Button(lute_prev_ax, "◀")
    lute_next = Button(lute_next_ax, "▶")
    lute_sep_y = lute_button_y - 0.03
    panel_ax.plot(
        [panel_x, panel_x + panel_w],
        [lute_sep_y, lute_sep_y],
        transform=panel_ax.transAxes,
        color="0.8",
        lw=1.0,
    )

    top_slider, top_reset, top_value_text = _add_control(1, top_s_min, top_s_max, top_s)
    bottom_slider, bottom_reset, bottom_value_text = _add_control(2, bottom_s_min, bottom_s_max, bottom_s)
    top_z_slider, top_z_reset, top_z_value_text = _add_control(3, top_z_min, top_z_max, top_z_offset)
    bottom_z_slider, bottom_z_reset, bottom_z_value_text = _add_control(
        4, bottom_z_min, bottom_z_max, bottom_z_offset
    )
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
    rib_button_w = 0.05
    rib_button_gap = 0.02
    rib_button_y = base_slider_y - 5 * control_gap
    rib_minus_ax = panel_ax.inset_axes([panel_x + slider_w + 0.01, rib_button_y, rib_button_w, reset_h])
    rib_plus_ax = panel_ax.inset_axes(
        [panel_x + slider_w + 0.01 + rib_button_w + rib_button_gap, rib_button_y, rib_button_w, reset_h]
    )
    rib_minus = Button(rib_minus_ax, "-")
    rib_plus = Button(rib_plus_ax, "+")
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
            _update_edges()
            fig.canvas.draw_idle()
        finally:
            rib_text_updating = False

    def _nudge_rib_count(step: int) -> None:
        _update_rib_count(str(rib_count + step))

    def _update_slider_bounds(slider: Slider, vmin: float, vmax: float) -> None:
        slider.valmin = vmin
        slider.valmax = vmax
        slider.ax.set_xlim(vmin, vmax)

    def _set_lute_by_index(idx: int) -> None:
        nonlocal lute_index
        nonlocal lute
        nonlocal surface
        nonlocal unit_scale
        nonlocal width_min
        nonlocal width_max
        nonlocal width_step
        nonlocal top_z_min
        nonlocal top_z_max
        nonlocal bottom_z_min
        nonlocal bottom_z_max
        nonlocal neck_plane_x
        nonlocal soundboard_lines
        nonlocal width
        nonlocal top_z_offset
        nonlocal bottom_z_offset
        if not lute_choices:
            return
        lute_index = idx % len(lute_choices)
        _, lute_cls_sel = lute_choices[lute_index]
        lute = lute_cls_sel()
        surface = RibbonSurface.from_outline(lute, samples_per_arc=args.arc_samples)
        unit_scale = (
            float(lute.unit_in_mm()) / float(lute.unit)
            if hasattr(lute, "unit_in_mm") and hasattr(lute, "unit")
            else 1.0
        )
        for line in soundboard_lines:
            line.remove()
        soundboard_lines = _plot_soundboard_outline(ax, lute, args.arc_samples, surface)
        neck_plane_x = _neck_plane_x()
        width_min = float(lute.unit) * 0.15
        width_max = float(lute.unit) * 1.0
        width_step = 0.001 * float(lute.unit)
        top_z_min = -0.30 * float(lute.unit)
        top_z_max = 0.10 * float(lute.unit)
        bottom_z_min = -0.50 * float(lute.unit)
        bottom_z_max = 0.50 * float(lute.unit)
        _update_slider_bounds(width_slider, width_min, width_max)
        _update_slider_bounds(top_z_slider, top_z_min, top_z_max)
        _update_slider_bounds(bottom_z_slider, bottom_z_min, bottom_z_max)

        width = max(width_min, min(width_max, float(width)))
        top_z_offset = max(top_z_min, min(top_z_max, float(top_z_offset)))
        bottom_z_offset = max(bottom_z_min, min(bottom_z_max, float(bottom_z_offset)))

        width_slider.set_val(width)
        top_z_slider.set_val(top_z_offset)
        bottom_z_slider.set_val(bottom_z_offset)
        ax.set_title(type(lute).__name__)
        lute_value_text.set_text(_format_lute_label())
        fig.canvas.draw_idle()

    def _select_prev_lute() -> None:
        _set_lute_by_index(lute_index - 1)

    def _select_next_lute() -> None:
        _set_lute_by_index(lute_index + 1)

    lute_prev.on_clicked(lambda _event: _select_prev_lute())
    lute_next.on_clicked(lambda _event: _select_next_lute())

    width_value_text.set_text(_format_width(width))
    top_value_text.set_text(_format_s("Top s", top_s))
    bottom_value_text.set_text(_format_s("Bottom s", bottom_s))
    top_z_value_text.set_text(_format_offset("Top z", top_z_offset))
    bottom_z_value_text.set_text(_format_offset("Bottom z", bottom_z_offset))
    ribs_value_text.set_text(_format_rib_count(rib_count))
    lute_value_text.set_text(_format_lute_label())

    width_reset.on_clicked(lambda _event: width_slider.set_val(default_width))
    width_minus.on_clicked(lambda _event: _nudge_width(-width_step))
    width_plus.on_clicked(lambda _event: _nudge_width(width_step))
    top_reset.on_clicked(lambda _event: top_slider.set_val(default_top_s))
    bottom_reset.on_clicked(lambda _event: bottom_slider.set_val(default_bottom_s))
    top_z_reset.on_clicked(lambda _event: top_z_slider.set_val(default_top_z_offset))
    bottom_z_reset.on_clicked(lambda _event: bottom_z_slider.set_val(default_bottom_z_offset))
    ribs_reset.on_clicked(lambda _event: _update_rib_count(str(default_rib_count)))
    rib_minus.on_clicked(lambda _event: _nudge_rib_count(-2))
    rib_plus.on_clicked(lambda _event: _nudge_rib_count(2))
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
