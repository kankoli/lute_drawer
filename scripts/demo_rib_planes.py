#!/usr/bin/env python3
"""Quick demo that plots a rib surface together with its bounding planes."""
from __future__ import annotations

import argparse
import importlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lute_bowl.rib_builder as rib_builder
from lute_bowl.rib_form_builder import (
    all_rib_surfaces_convex,
    build_rib_surfaces,
    find_rib_side_planes,
    measure_rib_plane_deviation,
)
from lute_bowl.top_curves import TopCurve
from plotting.ribs import (
    plot_rib_surface_with_planes,
    save_plane_projection_png,
    build_panel_projections,
    compute_panel_frame,
)

DEFAULT_LUTE = "lute_soundboard.ManolLavta"
DEFAULT_CURVE = "lute_bowl.top_curves.MidCurve"


def _resolve_class(path: str):
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Class name must include module path: {path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr}'") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lute", default=DEFAULT_LUTE, help=f"Default: {DEFAULT_LUTE}")
    parser.add_argument("--curve", default=DEFAULT_CURVE, help=f"Default: {DEFAULT_CURVE}")
    parser.add_argument("--ribs", type=int, default=13, help="Number of rib intervals to sample")
    parser.add_argument(
        "--sections",
        type=int,
        default=None,
        help="Sections along the spine (omit to use build_bowl_ribs default).",
    )
    parser.add_argument(
        "--rib-index",
        type=int,
        default=None,
        help="Rib index (1-based) to visualize; omit to export all ribs.",
    )
    parser.add_argument("--title", default=None, help="Optional Matplotlib title override")
    parser.add_argument(
        "--plane-png",
        action="store_true",
        help="Save a PNG of each plane with the projected outlines.",
    )
    parser.add_argument(
        "--plane-gap-mm",
        type=float,
        default=60.0,
        help="Spacing between the two planes (mm). Use zero/negative to match the rib width.",
    )
    return parser.parse_args(argv)


def _build_output_path(
    kind: str,
    lute_name: str,
    rib_count: int,
    rib_index: int,
    curve_name: str,
    *,
    timestamp: str | None = None,
    stamp_in_filename: bool = True,
) -> Path:
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    safe_curve = curve_name.replace(".", "-")
    parts = []
    if stamp_in_filename:
        parts.append(timestamp)
    parts.extend([kind, lute_name, str(rib_count), f"index{rib_index}", safe_curve])
    name = "_".join(parts) + ".png"
    out_dir = Path("output") / "rib_planes"
    if not stamp_in_filename:
        out_dir = out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lute_cls = _resolve_class(args.lute)
    curve_cls = _resolve_class(args.curve)
    if not isinstance(curve_cls, type) or not issubclass(curve_cls, TopCurve):
        raise TypeError("curve must reference a TopCurve subclass")

    PAD_MM = 20.0
    MIN_PANEL_IN = 2.0
    MM_PER_INCH = 25.4

    lute = lute_cls()
    unit_scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
    build_kwargs = {"n_ribs": args.ribs, "top_curve": curve_cls}
    if args.sections is not None:
        build_kwargs["n_sections"] = args.sections
    _, rib_outlines = rib_builder.build_bowl_ribs(lute, **build_kwargs)

    rib_targets: list[int] | None = None if args.rib_index is None else [args.rib_index]
    surfaces = build_rib_surfaces(rib_outlines=rib_outlines, rib_index=rib_targets)
    if not surfaces:
        raise RuntimeError("No surfaces returned for requested rib indices")

    gap_value = None
    if args.plane_gap_mm is not None and args.plane_gap_mm > 0:
        gap_value = args.plane_gap_mm

    if all_rib_surfaces_convex(
        rib_outlines=rib_outlines,
        plane_gap_mm=gap_value,
        unit_scale=unit_scale,
    ):
        print("all convex")

    curve_label = f"{curve_cls.__name__}"
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M") if args.rib_index is None else None
    saved_paths: list[tuple[Path, Path]] = []

    # Precompute outlines/planes/projections and max frame size for consistent scaling.
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

        panel_projections = build_panel_projections(outlines, planes, unit_scale)
        frame_w, frame_h = compute_panel_frame(
            panel_projections, PAD_MM, MIN_PANEL_IN, MM_PER_INCH, override=None
        )
        max_frame_w = max(max_frame_w, frame_w)
        max_frame_h = max(max_frame_h, frame_h)

        rib_data.append((rib_idx, outlines, planes, quads, panel_projections))

    frame_size_mm = (max_frame_w, max_frame_h)

    for rib_idx, outlines, planes, quads, panel_projections in rib_data:
        if args.plane_png:
            plane_path = _build_output_path(
                "planes",
                type(lute).__name__,
                args.ribs,
                rib_idx,
                curve_label,
                timestamp=timestamp,
                stamp_in_filename=args.rib_index is not None,
            )
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

        deviation = measure_rib_plane_deviation(
            rib_outlines=rib_outlines,
            rib_index=rib_idx,
            plane_gap_mm=gap_value,
            unit_scale=unit_scale,
        )
        to_mm = unit_scale
        if deviation.long_deltas.size:
            long_stats = deviation.long_deltas * to_mm
            height_stats = deviation.height_deltas * to_mm
            print(
                f"Rib {rib_idx} plane deviation — long axis max {long_stats.max():.3f} mm, "
                f"height axis max {height_stats.max():.3f} mm"
            )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
