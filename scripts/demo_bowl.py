#!/usr/bin/env python3
"""Quick demo for generating and plotting a lute bowl."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bowl_from_soundboard import build_bowl_for_lute
from plotting.bowl import plot_bowl
from bowl_top_curves import resolve_top_curve


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
DEFAULT_CURVE = "bowl_top_curves.MidCurve"


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
    parser.add_argument("--sections", type=int, default=120, help="Number of sections to sample")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    lute_cls = _resolve_class(args.lute)
    curve_cls = _resolve_class(args.curve)

    lute = lute_cls()
    sections, ribs = build_bowl_for_lute(
        lute,
        n_ribs=args.ribs,
        n_sections=args.sections,
        top_curve=curve_cls,
    )
    plot_bowl(lute, sections, ribs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
