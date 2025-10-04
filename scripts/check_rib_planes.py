#!/usr/bin/env python3
"""Analyze rib outlines for planarity and concavity/convexity."""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.analysis_utils import ensure_matplotlib_stub, sandboxed_renderer
from lute_bowl.bowl_from_soundboard import build_bowl_for_lute

DEFAULT_LUTE = "lute_soundboard.ManolLavta"
DEFAULT_CURVE = "lute_bowl.bowl_top_curves.MidCurve"
CONCAVITY_TOL = 0.05


@dataclass
class RibPlaneSummary:
    label: str
    max_distance: float
    mean_distance: float
    concavity_ratio: float

    def concavity_label(self, tol: float = CONCAVITY_TOL) -> str:
        if self.max_distance < 1e-6:
            return "flat"
        if self.concavity_ratio > tol:
            return "convex"
        if self.concavity_ratio < -tol:
            return "concave"
        return "flat"

    def format(self) -> str:
        return (
            f"{self.label:<7}: max_dist={self.max_distance:>8.4f} "
            f"mean_dist={self.mean_distance:>8.4f} "
            f"concavity={self.concavity_label()} (ratio={self.concavity_ratio:+.2f})"
        )


def _resolve_class(path: str):
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Class name must include module path: {path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr}'") from exc


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    d = -np.dot(normal, centroid)
    return normal, d


def _distances(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return (points @ normal + d) / (np.linalg.norm(normal) + 1e-12)


def summarize_points(points: np.ndarray, label: str) -> RibPlaneSummary:
    normal, d = _fit_plane(points)
    distances = _distances(points, normal, d)
    max_dist = float(np.max(np.abs(distances)))
    mean_dist = float(np.mean(np.abs(distances)))
    above = np.sum(distances > 0)
    below = np.sum(distances < 0)
    total = max(above + below, 1)
    concavity = (above - below) / total
    return RibPlaneSummary(label, max_dist, mean_dist, concavity)


def analyze_ribs(ribs: List[np.ndarray]) -> List[RibPlaneSummary]:
    return [summarize_points(rib, f"rib {idx:>2}") for idx, rib in enumerate(ribs, start=1)]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--sections", type=int, default=120)
    parser.add_argument("--ribs", type=int, default=13)
    parser.add_argument("--margin", type=float, default=1e-3)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    lute_cls = _resolve_class(args.lute)
    curve_cls = _resolve_class(args.curve)

    ensure_matplotlib_stub()
    with sandboxed_renderer():
        lute = lute_cls()
        _, ribs = build_bowl_for_lute(
            lute,
            n_ribs=args.ribs,
            n_sections=args.sections,
            margin=args.margin,
            top_curve=curve_cls,
        )

    for summary in analyze_ribs(ribs):
        print(summary.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
