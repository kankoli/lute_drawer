#!/usr/bin/env python3
"""Visualise planar rib bowls constrained by end blocks."""
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

from lute_bowl.bowl_from_soundboard import compute_equivalent_flat_side_depth
from lute_bowl.rib_builder import build_bowl_ribs
from lute_bowl.rib_form_builder import (
    all_rib_surfaces_convex,
    build_rib_surfaces,
    compute_rib_blank_width,
    find_rib_side_planes,
    measure_rib_plane_deviation,
)
from lute_bowl.mold_builder import build_mold_sections
from lute_bowl.top_curves import TopCurve
from lute_bowl import section_curve as section_curve_mod
from plotting import plot_bowl, plot_mold_sections_2d
from plotting.step_renderers import write_mold_sections_step
from plotting.ribs import (
    plot_rib_surface_with_planes,
    save_plane_projection_png,
    build_panel_projections,
    compute_panel_frame,
)

DEFAULT_LUTE = "lute_soundboard.ManolLavta"
DEFAULT_CURVE = "lute_bowl.top_curves.FlatBackCurve"
DEFAULT_SECTION_CURVE = "lute_bowl.section_curve.CircularSectionCurve"
DEFAULT_SKIRT_SPAN = 1.0

PRESETS = {
    "mid-cosine-arch": {
        "top_curve": "lute_bowl.top_curves.MidCurve",
        "section_curve": "lute_bowl.section_curve.CosineArchSectionCurve",
        "section_curve_kwargs": {"shape_power": 2.2},
        "skirt_span": 1.5,
    },
    "turkish-oud-skirted": {
        "lute": "lute_soundboard.TurkishOud2_2",
        "n_ribs": 21,
        "top_curve": "lute_bowl.top_curves.DeepBackCurve",
        "section_curve": "lute_bowl.section_curve.CircularSectionCurve",
        "skirt_span": 1.2,
    },
    "lavta-skirted": {
        "lute": "lute_soundboard.ManolLavta",
        "n_ribs": 11,
        "top_curve": "lute_bowl.top_curves.FlatBackCurve",
        "section_curve": "lute_bowl.section_curve.CircularSectionCurve",
        "skirt_span": 1.2,
    },
}


def _resolve_class(path: str):
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Class path must include module name: {path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr}'") from exc


def _resolve_class_or_default(
    arg_value,
    preset_value,
    default_path: str,
    expected_base: type,
    flag_name: str,
):
    raw = arg_value if arg_value is not None else preset_value
    if raw is None:
        raw = default_path
    cls = raw if isinstance(raw, type) else _resolve_class(raw)
    if not isinstance(cls, type) or not issubclass(cls, expected_base):
        raise TypeError(f"{flag_name} must reference a {expected_base.__name__} subclass")
    return cls


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lute", default=None, help=f"Fully qualified lute class (default if none/preset: {DEFAULT_LUTE})")
    parser.add_argument(
        "--top-curve",
        default=None,
        help=f"Fully qualified top-curve class (default if none/preset: {DEFAULT_CURVE})",
    )
    parser.add_argument(
        "--section-curve",
        default=None,
        help=f"Fully qualified section-curve class (default if none/preset: {DEFAULT_SECTION_CURVE})",
    )
    parser.add_argument("--ribs", type=int, default=None, help="Number of rib intervals.")
    parser.add_argument(
        "--sections",
        type=int,
        default=None,
        help="Number of planar section samples (omit to use build_bowl_ribs default).",
    )
    parser.add_argument(
        "--skirt-span",
        type=float,
        default=None,
        help=f"Distance from tail along spine where skirt ribs begin (units, default if none/preset: {DEFAULT_SKIRT_SPAN}).",
    )
    parser.add_argument(
        "--show-section-circles",
        action="store_true",
        help="Draw section circle overlays when section count allows.",
    )
    parser.add_argument(
        "--show-top-curve",
        action="store_true",
        help="Highlight the sampled top-curve spine.",
    )
    parser.add_argument(
        "--build-molds",
        action="store_true",
        help="Generate mold sections alongside the bowl plot.",
    )
    parser.add_argument(
        "--stations",
        type=int,
        default=6,
        help="Number of mold stations (requires --build-molds).",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=20.0,
        help="Mold stations thickness along the spine in millimetres.",
    )
    parser.add_argument(
        "--neck-limit",
        type=float,
        default=1.0,
        help="Neck-side limit for mold faces in geometry units (default: 1).",
    )
    parser.add_argument(
        "--tail-limit",
        type=float,
        default=0.15,
        help="Tail-side limit for mold faces in geometry units (default: 0.15).",
    )
    parser.add_argument(
        "--plot2d",
        action="store_true",
        help="Plot mold sections in 2D instead of overlaying on the 3D bowl.",
    )
    parser.add_argument(
        "--plot2d-section",
        type=int,
        default=None,
        help="Optional mold section index for --plot2d; negative indices allowed.",
    )
    parser.add_argument(
        "--show-eye-plane",
        action="store_true",
        help="Visualize the eye/neck plane (skirt builds only).",
    )
    parser.add_argument(
        "--step-out",
        type=Path,
        default=None,
        help="Optional STEP (.stp) filepath for exported mold sections.",
    )
    parser.add_argument(
        "--step-support-extension",
        type=float,
        default=0.0,
        help="Additional tail support extension in millimetres for STEP output.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Optional named preset; explicit flags override preset values.",
    )
    parser.add_argument(
        "--print-flat-side-depth",
        action="store_true",
        help="Print equivalent flat-side depth (flat back) for matching bowl volume.",
    )
    parser.add_argument(
        "--plot-rib-planes",
        action="store_true",
        help="Plot rib surfaces with their side planes (former demo_rib_planes.py).",
    )
    parser.add_argument(
        "--rib-index",
        type=int,
        default=None,
        help="Rib index (1-based) to visualize; omit to process all ribs when plotting rib planes.",
    )
    parser.add_argument(
        "--plane-png",
        action="store_true",
        help="Save a PNG of each plane with the projected outlines (rib-planes mode).",
    )
    parser.add_argument(
        "--plane-gap-mm",
        type=float,
        default=18.0,
        help="Spacing between the two planes (mm) in rib-planes mode. Use zero/negative to match rib width.",
    )
    parser.add_argument(
        "--panel-pad-mm",
        type=float,
        default=20.0,
        help="Padding around panel projections in rib-planes mode (mm).",
    )
    parser.add_argument(
        "--min-panel-in",
        type=float,
        default=2.0,
        help="Minimum panel size in inches for rib-planes mode.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title override for rib-planes plots.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print rib-plane metrics (blank widths, projection deltas).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    preset_cfg = {}
    if args.preset:
        raw = dict(PRESETS[args.preset])
        if "top_curve" in raw:
            raw["top_curve"] = _resolve_class(raw["top_curve"])
        if "section_curve" in raw:
            raw["section_curve_cls"] = _resolve_class(raw.pop("section_curve"))
        if "lute" in raw:
            raw["lute_cls"] = _resolve_class(raw.pop("lute"))
        preset_cfg = raw

    lute_path = args.lute or preset_cfg.get("lute_cls") or DEFAULT_LUTE
    lute_cls = lute_path if isinstance(lute_path, type) else _resolve_class(lute_path)
    lute = lute_cls()

    top_curve_cls = _resolve_class_or_default(
        args.top_curve,
        preset_cfg.get("top_curve"),
        DEFAULT_CURVE,
        TopCurve,
        "--top-curve",
    )
    section_curve_cls = _resolve_class_or_default(
        args.section_curve,
        preset_cfg.get("section_curve_cls"),
        DEFAULT_SECTION_CURVE,
        section_curve_mod.BaseSectionCurve,
        "--section-curve",
    )

    build_kwargs = {
        "preset": preset_cfg or None,
        # "debug_rib_indices": [3, 4],
        # "debug_logger": print,
    }
    if args.ribs is not None:
        build_kwargs["n_ribs"] = args.ribs
    if args.sections is not None:
        build_kwargs["n_sections"] = args.sections
    build_kwargs["top_curve"] = top_curve_cls
    build_kwargs["section_curve_cls"] = section_curve_cls
    skirt_span = args.skirt_span
    if skirt_span is None:
        skirt_span = preset_cfg.get("skirt_span")
        if skirt_span is None:
            skirt_span = DEFAULT_SKIRT_SPAN
    build_kwargs["skirt_span"] = skirt_span
    # Optional section-curve kwargs for presets; explicit kwargs could be added via code edits if needed.
    if preset_cfg.get("section_curve_kwargs"):
        build_kwargs["section_curve_kwargs"] = preset_cfg["section_curve_kwargs"]

    sections, ribs = build_bowl_ribs(lute, **build_kwargs)

    if args.print_flat_side_depth:
        try:
            depth_units = compute_equivalent_flat_side_depth(lute, sections)
        except Exception as exc:
            print(f"Flat-side depth calculation failed: {exc}")
        else:
            unit_scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
            depth_mm = depth_units * unit_scale
            print(f"Equivalent flat-side depth: {depth_units:.4f} units ({depth_mm:.1f} mm)")

    if args.plot_rib_planes:
        _run_rib_planes_mode(lute, ribs, args)
    else:
        mold_sections = None
        if args.build_molds or args.step_out is not None or args.plot2d:
            mold_sections = build_mold_sections(
                sections=sections,
                ribs=ribs,
                n_stations=args.stations,
                board_thickness_mm=args.thickness,
                lute=lute,
                neck_limit=args.neck_limit,
                tail_limit=args.tail_limit,
            )

            if args.step_out is not None:
                scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
                step_path = write_mold_sections_step(
                    mold_sections,
                    args.step_out,
                    unit_scale=scale,
                    author="demo_planar_bowl",
                    organization="lute_drawer",
                )
                print(f"STEP export written to {step_path}")

            if args.plot2d or args.plot2d_section:
                plot_mold_sections_2d(
                    mold_sections,
                    section_index=args.plot2d_section,
                    form_top=lute.form_top,
                    form_bottom=lute.form_bottom,
                    lute_name=type(lute).__name__,
                )

        show_circles = False
        if args.show_section_circles:
            if len(sections) <= 80:
                show_circles = True
            else:
                print("Warning: Section circles not shown because more than 80 sections were sampled.")

        plot_bowl(
            lute,
            sections,
            ribs,
            show_section_circles=show_circles,
            show_top_curve=args.show_top_curve,
            top_curve_name=getattr(lute, "top_curve_label", None),
            show_eye_plane=args.show_eye_plane,
            mold_sections=mold_sections,
        )

        import matplotlib.pyplot as plt

        plt.show()
    return 0


def _run_rib_planes_mode(lute, rib_outlines, args: argparse.Namespace) -> None:
    """Rib-plane plotting/export (formerly demo_rib_planes.py)."""
    rib_targets: list[int] | None = None if args.rib_index is None else [args.rib_index]
    surfaces = build_rib_surfaces(rib_outlines=rib_outlines, rib_index=rib_targets)
    if not surfaces:
        raise RuntimeError("No surfaces returned for requested rib indices")

    unit_scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
    gap_value = args.plane_gap_mm if args.plane_gap_mm is not None and args.plane_gap_mm > 0 else None

    if all_rib_surfaces_convex(
        rib_outlines=rib_outlines,
        plane_gap_mm=gap_value,
        unit_scale=unit_scale,
        verbose=args.verbose,
    ):
        print("all convex")

    curve_label = getattr(lute, "top_curve_label", type(lute).__name__)
    saved_paths: list[tuple[Path, Path]] = []

    PAD_MM = args.panel_pad_mm
    MIN_PANEL_IN = args.min_panel_in
    MM_PER_INCH = 25.4

    rib_data: list[
        tuple[int, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, list]
    ] = []
    max_frame_w = 0.0
    max_frame_h = 0.0

    for rib_idx, quads in surfaces:
        outlines = (
            np.asarray(rib_outlines[rib_idx - 1], dtype=float),
            np.asarray(rib_outlines[rib_idx], dtype=float),
        )
        planes = find_rib_side_planes(
            rib_outlines=rib_outlines,
            rib_index=rib_idx,
            plane_gap_mm=gap_value,
            unit_scale=unit_scale,
        )

        panel_projections = build_panel_projections(outlines, planes, unit_scale, verbose=args.verbose)
        frame_w, frame_h = compute_panel_frame(
            panel_projections, PAD_MM, MIN_PANEL_IN, MM_PER_INCH, override=None
        )
        max_frame_w = max(max_frame_w, frame_w)
        max_frame_h = max(max_frame_h, frame_h)

        rib_data.append((rib_idx, outlines, planes, quads, panel_projections))

    frame_size_mm = (max_frame_w, max_frame_h)

    for rib_idx, outlines, planes, quads, panel_projections in rib_data:
        if args.plane_png:
            plane_dir = Path("output") / "rib_planes"
            plane_dir.mkdir(parents=True, exist_ok=True)
            name_parts = [type(lute).__name__, str(len(rib_outlines) - 1), f"index{rib_idx}", curve_label]
            plane_path = plane_dir / ("_".join(name_parts) + ".png")
            left_path, right_path = save_plane_projection_png(
                plane_path,
                outlines,
                planes,
                unit_scale=unit_scale,
                title=f"{type(lute).__name__} Rib {rib_idx} Planes",
                frame_size_mm=frame_size_mm,
                pad_mm=PAD_MM,
                panel_projections=panel_projections,
            )
            saved_paths.append((left_path, right_path))

        if args.verbose:
            deviation = measure_rib_plane_deviation(
                rib_outlines=rib_outlines,
                rib_index=rib_idx,
                plane_gap_mm=gap_value,
                unit_scale=unit_scale,
            )
            blank_width_units = compute_rib_blank_width(rib_outlines=rib_outlines, rib_index=rib_idx)
            blank_width_mm = blank_width_units * unit_scale
            print(f"Rib {rib_idx}: blank width {blank_width_units:.3f} units ({blank_width_mm:.1f} mm)")
            to_mm = unit_scale
            if deviation.long_deltas.size:
                long_stats = deviation.long_deltas * to_mm
                height_stats = deviation.height_deltas * to_mm
                print(
                    f"Rib {rib_idx} plane deviation — long axis max {long_stats.max():.3f} mm, "
                    f"height axis max {height_stats.max():.3f} mm"
                )

        if not args.plane_png:
            plot_rib_surface_with_planes(
                rib_idx,
                quads,
                outlines,
                planes,
                title=args.title,
                lute_name=type(lute).__name__,
                panel_projections=panel_projections,
            )

    if args.plane_png and saved_paths:
        if args.rib_index is None:
            folder = saved_paths[0][0].parent
            print(f"Plane projection PNGs saved under {folder}")
        else:
            left_path, right_path = saved_paths[0]
            print(f"Plane projection PNGs saved to {left_path} and {right_path}")

if __name__ == "__main__":
    raise SystemExit(main())
