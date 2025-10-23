#!/usr/bin/env python3
"""Demo for planar rib surfaces based on end-block constrained bowls."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lute_bowl.rib_builder as rib_builder
from lute_bowl.rib_form_builder import build_rib_surfaces
from lute_bowl.top_curves import TopCurve
from plotting.bowl import plot_rib_surfaces
from plotting.step_renderers import write_rib_surfaces_step

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
    if not isinstance(curve_cls, type) or not issubclass(curve_cls, TopCurve):
        raise TypeError("curve must reference a TopCurve subclass")

    lute = lute_cls()

    sections, rib_outlines = rib_builder.build_bowl_ribs(
        lute,
        n_ribs=args.ribs,
        n_sections=args.sections,
        top_curve=curve_cls,
    )
    surfaces = build_rib_surfaces(
        rib_outlines=rib_outlines,
        rib_index=args.rib_index,
    )

    outlines = [
        (
            rib_idx,
            (
                np.asarray(rib_outlines[rib_idx - 1], dtype=float),
                np.asarray(rib_outlines[rib_idx], dtype=float),
            ),
        )
        for rib_idx, _ in surfaces
    ]

    plot_rib_surfaces(
        surfaces,
        outlines=outlines,
        title=args.title,
        lute_name=type(lute).__name__,
    )

    if args.step_out is not None:
        scale = lute.unit_in_mm() / lute.unit if hasattr(lute, "unit") else 1.0
        write_rib_surfaces_step(
            surfaces,
            args.step_out,
            unit_scale=scale,
            support_extension_mm=args.step_support_extension,
            author="demo_ribs",
            organization="lute_drawer",
        )
        print(f"STEP export written to {args.step_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
