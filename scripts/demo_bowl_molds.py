#!/usr/bin/env python3
"""Demo script for visualising bowl mold sections."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lute_bowl.bowl_mold import build_mold_sections
from plotting.step_renderers import write_mold_sections_step
from plotting import plot_bowl, plot_mold_sections_2d
from lute_bowl.bowl_from_soundboard import build_bowl_for_lute
from lute_bowl.planar_bowl_generator import build_planar_bowl_for_lute
from lute_bowl.bowl_top_curves import SimpleAmplitudeCurve, FlatBackCurve


def _resolve_class(path: str):
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Class path must include module name: {path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr}'") from exc



DEFAULT_LUTE = "lute_soundboard.ManolLavta"
DEFAULT_CURVE = "lute_bowl.bowl_top_curves.FlatBackCurve"


def parse_args() -> argparse.Namespace:
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
        help=f"Top-curve class for bowl shaping (default: {DEFAULT_CURVE})",
    )
    parser.add_argument("--ribs", type=int, default=13, help="Number of rib intervals")
    parser.add_argument("--sections", type=int, default=120, help="Number of sections to sample")
    parser.add_argument(
        "--use-planar",
        action="store_true",
        help="Sample the bowl with the planar generator (includes end-block trimming).",
    )
    parser.add_argument(
        "--upper-block",
        type=float,
        default=0.0,
        help="Extra clearance past the neck joint in geometry units (planar generator).",
    )
    parser.add_argument(
        "--lower-block",
        type=float,
        default=0.0,
        help="Extra clearance above the tail block in geometry units (planar generator).",
    )

    parser.add_argument("--skip-mold-sections", action="store_true", help="Skip plotting mold sections")

    parser.add_argument("--stations", type=int, default=6, help="Number of mold sections to generate")
    parser.add_argument(
        "--thickness",
        type=float,
        default=30.0,
        help="Physical board thickness along the spine (millimetres)",
    )
    parser.add_argument(
        "--neck-limit",
        type=float,
        default=100,
        help="Optional neck-side limit for mold faces (millimetres from top)",
    )
    parser.add_argument(
        "--tail-limit",
        type=float,
        default=15,
        help="Optional tail-side limit for mold faces (millimetres from top)",
    )
    parser.add_argument(
        "--plot2d",
        action="store_true",
        help="Render mold sections in 2D instead of 3D",
    )
    parser.add_argument(
        "--step-out",
        type=Path,
        default=None,
        help="Optional STEP (.stp) filepath to export mold faces as polylines",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lute_cls = _resolve_class(args.lute)
    curve_cls = _resolve_class(args.curve)
    lute = lute_cls()

    if args.use_planar:
        sections, ribs = build_planar_bowl_for_lute(
            lute,
            n_ribs=args.ribs,
            n_sections=args.sections,
            top_curve=curve_cls,
            upper_block_units=args.upper_block,
            lower_block_units=args.lower_block,
        )
    else:
        sections, ribs = build_bowl_for_lute(
            lute,
            n_ribs=args.ribs,
            n_sections=args.sections,
            top_curve=curve_cls,
        )

    if not args.skip_mold_sections:
        mold_sections = build_mold_sections(
            sections=sections,
            ribs=ribs,
            n_stations=args.stations,
            board_thickness_mm=args.thickness,
            lute=lute,
            neck_limit_mm=args.neck_limit,
            tail_limit_mm=args.tail_limit,
        )

        if args.step_out is not None:
            scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
            step_path = write_mold_sections_step(
                mold_sections,
                args.step_out,
                unit_scale=scale,
                author="demo_bowl_molds",
                organization="lute_drawer",
            )
            print(f"STEP export written to {step_path}")

        if args.plot2d:
            plot_mold_sections_2d(
                mold_sections,
                form_top=lute.form_top,
                form_bottom=lute.form_bottom,
                lute_name=type(lute).__name__,
            )
        else:
            plot_bowl(
                lute,
                sections,
                ribs,
                show_section_circles=False,
                show_apexes=False,
                show_top_curve=False,
                show_soundboard=True,
                mold_sections=mold_sections,
            )
    else:
        plot_bowl(lute, sections, ribs)

    import matplotlib.pyplot as plt

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
