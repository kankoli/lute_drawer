#!/usr/bin/env python3
"""Visualise planar rib bowls constrained by end blocks."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lute_bowl.planar_bowl_generator import build_planar_bowl_for_lute
from plotting import plot_bowl

DEFAULT_LUTE = "lute_soundboard.ManolLavta"
DEFAULT_CURVE = "lute_bowl.top_curves.FlatBackCurve"


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
    parser.add_argument(
        "lute",
        nargs="?",
        default=DEFAULT_LUTE,
        help=f"Fully qualified lute class (default: {DEFAULT_LUTE})",
    )
    parser.add_argument(
        "--top-curve",
        default=DEFAULT_CURVE,
        help=f"Fully qualified top-curve class or omit for default (default: {DEFAULT_CURVE})",
    )
    parser.add_argument("--ribs", type=int, default=13, help="Number of rib intervals.")
    parser.add_argument("--sections", type=int, default=160, help="Number of planar section samples.")
    parser.add_argument(
        "--upper-block",
        type=float,
        default=0.0,
        help="Extra clearance past the neck joint in geometry units (default: 0).",
    )
    parser.add_argument(
        "--lower-block",
        type=float,
        default=0.0,
        help="Extra clearance above the tail block in geometry units (default: 0).",
    )
    parser.add_argument(
        "--hide-section-circles",
        action="store_true",
        help="Hide section circle overlays in the plot.",
    )
    parser.add_argument(
        "--show-top-curve",
        action="store_true",
        help="Highlight the sampled top-curve spine.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose section sampling diagnostics.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    lute_cls = _resolve_class(args.lute)
    lute = lute_cls()

    top_curve = None
    if args.top_curve:
        top_curve = _resolve_class(args.top_curve)

    sections, ribs = build_planar_bowl_for_lute(
        lute,
        n_ribs=args.ribs,
        n_sections=args.sections,
        top_curve=top_curve,
        upper_block_units=args.upper_block,
        lower_block_units=args.lower_block,
        debug=args.debug,
    )

    plot_bowl(
        lute,
        sections,
        ribs,
        show_section_circles=not args.hide_section_circles,
        show_top_curve=args.show_top_curve,
        top_curve_name=getattr(lute, "top_curve_label", None),
    )

    import matplotlib.pyplot as plt

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
