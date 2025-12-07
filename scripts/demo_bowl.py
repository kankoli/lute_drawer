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

from lute_bowl.rib_builder import build_bowl_ribs
from lute_bowl.mold_builder import build_mold_sections
from lute_bowl.top_curves import TopCurve
from lute_bowl import section_curve as section_curve_mod
from plotting import plot_bowl, plot_mold_sections_2d
from plotting.step_renderers import write_mold_sections_step

DEFAULT_LUTE = "lute_soundboard.ManolLavta"
DEFAULT_CURVE = "lute_bowl.top_curves.FlatBackCurve"
DEFAULT_SECTION_CURVE = "lute_bowl.section_curve.CircularSectionCurve"


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
    parser.add_argument("--lute", default=DEFAULT_LUTE, help=f"Default: {DEFAULT_LUTE}")
    parser.add_argument(
        "--top-curve",
        default=DEFAULT_CURVE,
        help=f"Fully qualified top-curve class or omit for default (default: {DEFAULT_CURVE})",
    )
    parser.add_argument(
        "--section-curve",
        default=DEFAULT_SECTION_CURVE,
        help=f"Fully qualified section-curve class or omit for default (default: {DEFAULT_SECTION_CURVE})",
    )
    parser.add_argument("--ribs", type=int, default=13, help="Number of rib intervals.")
    parser.add_argument(
        "--sections",
        type=int,
        default=None,
        help="Number of planar section samples (omit to use build_bowl_ribs default).",
    )
    parser.add_argument(
        "--skirt-span",
        type=float,
        default=0.0,
        help="Distance from tail along spine where skirt ribs begin (units).",
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
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    lute_cls = _resolve_class(args.lute)
    lute = lute_cls()

    top_curve_cls = _resolve_class(args.top_curve) if args.top_curve else None
    if top_curve_cls is None or not isinstance(top_curve_cls, type) or not issubclass(top_curve_cls, TopCurve):
        raise TypeError("--top-curve must reference a TopCurve subclass")

    section_curve_cls = _resolve_class(args.section_curve) if args.section_curve else None
    if section_curve_cls is None or not isinstance(section_curve_cls, type) or not issubclass(section_curve_cls, section_curve_mod.BaseSectionCurve):
        raise TypeError("--section-curve must reference a BaseSectionCurve subclass")

    build_kwargs = {"n_ribs": args.ribs, "top_curve": top_curve_cls}
    if args.sections is not None:
        build_kwargs["n_sections"] = args.sections
    if args.skirt_span is not None:
        build_kwargs["skirt_span"] = args.skirt_span
    build_kwargs["section_curve_cls"] = section_curve_cls
    sections, ribs = build_bowl_ribs(lute, **build_kwargs)

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


if __name__ == "__main__":
    raise SystemExit(main())
