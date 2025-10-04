#!/usr/bin/env python3
"""Quick demo for generating extended rib surfaces."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lute_bowl.rib_form_builder import RibSurfaceOptions, build_extended_rib_surfaces
from plotting.bowl import plot_rib_surfaces
from plotting.step_renderers import write_rib_surfaces_step


def _resolve_class(path: str):
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Class name must include module path: {path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr}'") from exc



DEFAULT_LUTE = "lute_soundboard.ManolLavta"
DEFAULT_CURVE = "lute_bowl.bowl_top_curves.MidCurve"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lute",
        nargs="?",
        default=DEFAULT_LUTE,
        help=f"Fully qualified lute class (default: {DEFAULT_LUTE})",
    )
    parser.add_argument(
        "curve",
        nargs="?",
        default=DEFAULT_CURVE,
        help=f"Fully qualified top-curve class (default: {DEFAULT_CURVE})",
    )
    parser.add_argument("--ribs", type=int, default=13, help="Number of rib intervals")
    parser.add_argument("--sections", type=int, default=200, help="Number of sections to sample")
    parser.add_argument("--rib-index", type=int, default=7, help="Single rib index (1-based)")
    parser.add_argument("--all", action="store_true", help="Plot all ribs instead of a single one")
    parser.add_argument("--plane-offset", type=float, default=10.0)
    parser.add_argument("--allowance-left", type=float, default=0.0)
    parser.add_argument("--allowance-right", type=float, default=0.0)
    parser.add_argument("--end-extension", type=float, default=10.0)
    parser.add_argument("--spacing", type=float, default=200.0)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--step-out",
        type=Path,
        default=None,
        help="Optional STEP (.stp) file for exported rib solids",
    )
    parser.add_argument(
        "--step-base-thickness",
        type=float,
        default=25.0,
        help="Depth of the rectangular backing block beneath the rib (mm)",
    )
    parser.add_argument(
        "--step-support-extension",
        type=float,
        default=20.0,
        help="Additional distance to extend beyond the form bottom along the spine (mm)",
    )
    parser.add_argument(
        "--step-spacing",
        type=float,
        default=None,
        help="Optional separation between ribs in the STEP export (millimetres)",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    lute_cls = _resolve_class(args.lute)
    curve_cls = _resolve_class(args.curve)
    lute = lute_cls()

    options = RibSurfaceOptions(
        plane_offset=args.plane_offset,
        allowance_left=args.allowance_left,
        allowance_right=args.allowance_right,
        end_extension=args.end_extension,
        spacing=args.spacing,
    )

    _, surfaces, opts = build_extended_rib_surfaces(
        lute,
        top_curve=curve_cls,
        n_ribs=args.ribs,
        n_sections=args.sections,
        options=options,
        rib_index=args.rib_index,
        draw_all=args.all,
    )
    plot_rib_surfaces(
        surfaces,
        spacing=opts.spacing,
        title=args.title,
        lute_name=type(lute).__name__,
    )

    if args.step_out is not None:
        scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
        write_rib_surfaces_step(
            surfaces,
            args.step_out,
            unit_scale=scale,
            base_thickness_mm=args.step_base_thickness,
            support_extension_mm=args.step_support_extension,
            spacing_mm=args.step_spacing,
            author="demo_ribs",
            organization="lute_drawer",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
