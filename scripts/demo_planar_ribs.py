#!/usr/bin/env python3
"""Demo for planar rib surfaces based on end-block constrained bowls."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lute_bowl.planar_rib_form_builder import plot_planar_ribs
from plotting.step_renderers import write_rib_surfaces_step

DEFAULT_LUTE = "lute_soundboard.ManolLavta"
DEFAULT_CURVE = "lute_bowl.bowl_top_curves.MidCurve"


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
    parser.add_argument("--ribs", type=int, default=13, help="Number of rib intervals.")
    parser.add_argument("--sections", type=int, default=160, help="Number of planar section samples.")
    parser.add_argument("--rib-index", type=int, default=7, help="Single rib index (1-based).")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Optional fractional trim from both ends after applying end blocks.",
    )
    parser.add_argument(
        "--upper-block",
        type=float,
        default=1.0,
        help="Upper end-block height expressed in geometry units.",
    )
    parser.add_argument(
        "--lower-block",
        type=float,
        default=0.1,
        help="Lower end-block height expressed in geometry units.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--step-out",
        type=Path,
        default=None,
        help="Optional STEP (.stp) file for exported rib panels.",
    )
    parser.add_argument(
        "--step-support-extension",
        type=float,
        default=0.0,
        help="Extra distance to extend support beyond tail (mm) when exporting STEP.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lute_cls = _resolve_class(args.lute)
    curve_cls = _resolve_class(args.curve)
    lute = lute_cls()

    surfaces = plot_planar_ribs(
        lute,
        top_curve=curve_cls,
        n_ribs=args.ribs,
        n_sections=args.sections,
        margin=args.margin,
        upper_block_units=args.upper_block,
        lower_block_units=args.lower_block,
        rib=args.rib_index,
        title=args.title,
    )

    if args.step_out is not None:
        scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
        write_rib_surfaces_step(
            surfaces,
            args.step_out,
            unit_scale=scale,
            support_extension_mm=args.step_support_extension,
            author="demo_planar_ribs",
            organization="lute_drawer",
        )
        print(f"STEP export written to {args.step_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

