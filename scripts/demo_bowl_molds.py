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

from bowl_mold import build_mold_sections
from plotting import plot_mold_sections_2d, plot_mold_sections_3d
from bowl_from_soundboard import build_bowl_for_lute
from bowl_top_curves import SimpleAmplitudeCurve


def _resolve_class(path: str):
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Class path must include module name: {path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr}'") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lute",
        default="lute_soundboard.ManolLavta",
        help="Fully-qualified lute class to sample (default: lute_soundboard.ManolLavta)",
    )
    parser.add_argument(
        "--curve",
        default="bowl_top_curves.SimpleAmplitudeCurve",
        help="Top-curve class for bowl shaping (default: SimpleAmplitudeCurve)",
    )
    parser.add_argument("--stations", type=int, default=6, help="Number of mold sections to generate")
    parser.add_argument(
        "--thickness",
        type=float,
        default=30.0,
        help="Physical board thickness along the spine (millimetres)",
    )
    parser.add_argument(
        "--plot3d",
        action="store_true",
        help="Render mold sections in 3D instead of 2D",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lute_cls = _resolve_class(args.lute)
    curve_cls = _resolve_class(args.curve)
    lute = lute_cls()

    sections, ribs = build_bowl_for_lute(
        lute,
        top_curve=curve_cls,
    )

    mold_sections = build_mold_sections(
        sections=sections,
        ribs=ribs,
        n_stations=args.stations,
        board_thickness_mm=args.thickness,
        lute=lute,
    )

    if args.plot3d:
        plot_mold_sections_3d(mold_sections)
    else:
        plot_mold_sections_2d(mold_sections)

    import matplotlib.pyplot as plt

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
