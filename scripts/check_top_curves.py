#!/usr/bin/env python3
"""Diagnose bowl top-curve settings by checking section circle centers."""
import argparse
import inspect
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Matplotlib & renderer stubs (avoid cache writes in sandboxed runs)
# ---------------------------------------------------------------------------

def _ensure_matplotlib_stub() -> None:
    if "matplotlib" in sys.modules:
        return
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.__path__ = []
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.figure = lambda *args, **kwargs: None
    pyplot_stub.subplots = lambda *args, **kwargs: (None, None)
    pyplot_stub.show = lambda *args, **kwargs: None
    pyplot_stub.tight_layout = lambda *args, **kwargs: None
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)


_ensure_matplotlib_stub()


# ---------------------------------------------------------------------------
# Imports that rely on the stubs above
# ---------------------------------------------------------------------------

import lutes
from bowl_from_soundboard import (
    SideProfilePerControlTopCurve,
    build_bowl_for_lute,
)
from lute_renderers import SvgRenderer


@dataclass
class AnalysisResult:
    lute_name: str
    curve_name: str
    trapped_sections: List[Tuple[float, float]]
    warning_count: int
    min_center: float
    max_center: float
    max_apex: float
    section_count: int


def _iter_lute_classes(names: Iterable[str] | None) -> List[type]:
    lute_classes = []
    members = inspect.getmembers(lutes, inspect.isclass)
    for name, cls in members:
        if not issubclass(cls, lutes.Lute):
            continue
        if cls is lutes.Lute:
            continue
        if not getattr(cls, "__module__", "").startswith("lutes"):
            continue
        if getattr(cls, "__abstractmethods__", None):
            continue
        if names and name not in names:
            continue
        lute_classes.append((name, cls))
    if names:
        missing = sorted(set(names) - {name for name, _ in lute_classes})
        if missing:
            raise SystemExit(f"Unknown lute class(es): {', '.join(missing)}")
    return lute_classes

def _collect_top_curves() -> List[Tuple[str, type]]:
    import bowl_from_soundboard as bfs
    curves = []
    for name, cls in inspect.getmembers(bfs, inspect.isclass):
        if issubclass(cls, SideProfilePerControlTopCurve) and cls is not SideProfilePerControlTopCurve:
            curves.append((name, cls))
    return curves


def analyse_lute_curve(lute_cls: type, curve_cls: type, n_sections: int, n_ribs: int) -> AnalysisResult:
    lute = lute_cls()
    top_curve = curve_cls

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        sections, _ = build_bowl_for_lute(
            lute,
            n_ribs=n_ribs,
            n_sections=n_sections,
            top_curve=top_curve,
        )

    centers = [float(center[1]) for (_, center, radius, _) in sections if radius > 0]
    apex_z = [float(apex[1]) for (_, _, _, apex) in sections if apex is not None]

    trapped_sections = [
        (float(x), float(center[1]))
        for (x, center, radius, _) in sections
        if radius > 0 and float(center[1]) >= 0
    ]

    warning_count = sum(
        1
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "Section circle center" in str(w.message)
    )

    min_center = min(centers) if centers else float("nan")
    max_center = max(centers) if centers else float("nan")
    max_apex = max(apex_z) if apex_z else float("nan")

    return AnalysisResult(
        lute_name=lute_cls.__name__,
        curve_name=curve_cls.__name__,
        trapped_sections=trapped_sections,
        warning_count=warning_count,
        min_center=min_center,
        max_center=max_center,
        max_apex=max_apex,
        section_count=len(sections),
    )


def _format_trapped(entries: List[Tuple[float, float]]) -> str:
    if not entries:
        return "none"
    preview = entries[:5]
    sample = ", ".join(f"X={x:.1f}:Zc={z:.2f}" for x, z in preview)
    suffix = "..." if len(entries) > len(preview) else ""
    return f"count={len(entries)} [{sample}{suffix}]"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lute",
        action="append",
        help="Name of a lute class in lutes.py to analyse (default: all concrete lutes)",
    )
    parser.add_argument("--sections", type=int, default=120, help="Number of sections to sample")
    parser.add_argument("--ribs", type=int, default=13, help="Number of rib intervals")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Patch renderer to avoid filesystem writes
    original_draw = SvgRenderer.draw
    SvgRenderer.draw = lambda self, objs: None

    try:
        lute_classes = _iter_lute_classes(args.lute)
        curve_classes = _collect_top_curves()

        if not lute_classes:
            print("No lute classes found.")
            return 1
        if not curve_classes:
            print("No top-curve classes found.")
            return 1

        results: List[AnalysisResult] = []
        errors: List[str] = []

        for lute_name, lute_cls in lute_classes:
            for curve_name, curve_cls in curve_classes:
                try:
                    result = analyse_lute_curve(lute_cls, curve_cls, args.sections, args.ribs)
                    results.append(result)
                except Exception as exc:  # pragma: no cover - diagnostic tool
                    errors.append(f"{lute_name} + {curve_name}: {exc}")

        for result in results:
            trapped_str = _format_trapped(result.trapped_sections)
            print(
                f"{result.lute_name:<25} {result.curve_name:<24} "
                f"sections={result.section_count:>3} "
                f"min_center={result.min_center:>8.4f} max_center={result.max_center:>8.4f} "
                f"max_apex={result.max_apex:>8.4f} warnings={result.warning_count:<3} trapped={trapped_str}"
            )

        if errors:
            print("\nErrors:")
            for line in errors:
                print(f"  - {line}")
                
        return 0
    finally:
        SvgRenderer.draw = original_draw


if __name__ == "__main__":
    sys.exit(main())
