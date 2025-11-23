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
    parser.add_argument("--sections", type=int, default=160, help="Sections along the spine")
    parser.add_argument("--rib-index", type=int, default=7, help="Rib index (1-based) to visualize")
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


def _build_output_path(kind: str, lute_name: str, rib_count: int, rib_index: int, curve_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    safe_curve = curve_name.replace(".", "-")
    name = f"{timestamp}_{kind}_{lute_name}_{rib_count}_index{rib_index}_{safe_curve}.png"
    out_dir = Path("output") / "rib_planes"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lute_cls = _resolve_class(args.lute)
    curve_cls = _resolve_class(args.curve)
    if not isinstance(curve_cls, type) or not issubclass(curve_cls, TopCurve):
        raise TypeError("curve must reference a TopCurve subclass")

    lute = lute_cls()
    unit_scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
    _, rib_outlines = rib_builder.build_bowl_ribs(
        lute,
        n_ribs=args.ribs,
        n_sections=args.sections,
        top_curve=curve_cls,
    )
    surfaces = build_rib_surfaces(rib_outlines=rib_outlines, rib_index=args.rib_index)
    if not surfaces:
        raise RuntimeError(f"No surfaces returned for rib index {args.rib_index}")
    rib_idx, quads = surfaces[0]
    outlines = (
        np.asarray(rib_outlines[rib_idx - 1], dtype=float),
        np.asarray(rib_outlines[rib_idx], dtype=float),
    )
    gap_value = None
    if args.plane_gap_mm is not None and args.plane_gap_mm > 0:
        gap_value = args.plane_gap_mm
    planes = find_rib_side_planes(
        rib_outlines=rib_outlines,
        rib_index=rib_idx,
        plane_gap_mm=gap_value,
        unit_scale=unit_scale,
    )

    if all_rib_surfaces_convex(
        rib_outlines=rib_outlines,
        plane_gap_mm=gap_value,
        unit_scale=unit_scale,
    ):
        print("all convex")

    curve_label = f"{curve_cls.__module__}.{curve_cls.__name__}"
    if args.plane_png:
        plane_path = _build_output_path("planes", type(lute).__name__, args.ribs, rib_idx, curve_label)
        save_plane_projection_png(
            plane_path,
            outlines,
            planes,
            unit_scale=unit_scale,
            title=f"{type(lute).__name__} Rib {rib_idx} Planes",
        )
        print(f"Plane projection PNG saved to {plane_path}")

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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
