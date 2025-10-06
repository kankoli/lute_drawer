#!/usr/bin/env python3
"""Render SVG previews for every concrete lute soundboard."""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Type


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import lute_soundboard as lute_defs
from lute_soundboard import LuteSoundboard
from plotting.svg import SvgRenderer


def _iter_soundboard_classes(names: Iterable[str] | None = None) -> List[Tuple[str, Type[LuteSoundboard]]]:
    """Yield (name, class) pairs for every concrete soundboard definition."""

    selected = set(names or [])
    classes: List[Tuple[str, Type[LuteSoundboard]]] = []
    for name, cls in inspect.getmembers(lute_defs, inspect.isclass):
        if not issubclass(cls, LuteSoundboard):
            continue
        if cls is LuteSoundboard:
            continue
        if getattr(cls, "__abstractmethods__", None):
            continue
        if names and name not in selected:
            continue
        classes.append((name, cls))

    if names:
        missing = sorted(selected - {name for name, _ in classes})
        if missing:
            raise SystemExit(f"Unknown soundboard class(es): {', '.join(missing)}")

    return sorted(classes, key=lambda item: item[0])


def _core_objects(lute: LuteSoundboard) -> List[object]:
    """Return the default geometry primitives for rendering the outline."""

    objects: List[object] = [
        lute.A,
        lute.B,
        lute.form_top,
        lute.form_center,
        lute.form_bottom,
        lute.form_side,
        lute.point_neck_joint,
        lute.bridge,
        lute.spine,
        *lute.final_arcs,
        *lute.final_reflected_arcs,
    ]

    soundhole_circle = getattr(lute, "soundhole_circle", None)
    if soundhole_circle is not None:
        objects.append(soundhole_circle)

    soundhole_center = getattr(lute, "soundhole_center", None)
    if soundhole_center is not None:
        objects.append(soundhole_center)

    for circle in getattr(lute, "small_soundhole_circles", []):
        objects.append(circle)
    for center in getattr(lute, "small_soundhole_centers", []):
        objects.append(center)

    return objects


def _helper_objects(lute: LuteSoundboard) -> Sequence[object]:
    helpers: List[object] = []
    helpers.extend(getattr(lute, "tangent_circles", []))
    helpers.extend(getattr(lute, "tangent_points", []))
    return helpers


def _render_soundboard(
    name: str,
    cls: Type[LuteSoundboard],
    *,
    display_size: int,
    include_helpers: bool,
) -> Path:
    lute = cls(display_size=display_size)
    renderer = SvgRenderer(f"{name}_soundboard.svg", size=(lute.geo.display_size * 9, lute.geo.display_size * 6))
    objects = _core_objects(lute)
    if include_helpers:
        objects.extend(_helper_objects(lute))
    renderer.draw(objects)
    return Path(renderer.output_dir) / f"{name}_soundboard.svg"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lute",
        action="append",
        help="Render only the specified soundboard class (may be repeated).",
    )
    parser.add_argument(
        "--display-size",
        type=int,
        default=100,
        help="Display size used when instantiating each soundboard (default: 100).",
    )
    parser.add_argument(
        "--include-helpers",
        action="store_true",
        help="Include tangent circles and construction points in the SVG output.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first soundboard that raises an exception.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    classes = _iter_soundboard_classes(args.lute)

    if not classes:
        print("No soundboard classes found.")
        return 1

    failures: List[Tuple[str, Exception]] = []

    for name, cls in classes:
        try:
            output_path = _render_soundboard(
                name,
                cls,
                display_size=args.display_size,
                include_helpers=args.include_helpers,
            )
        except Exception as exc:  # pragma: no cover - diagnostic tool
            failures.append((name, exc))
            print(f"Failed {name}: {exc}")
            if args.fail_fast:
                break
        else:
            print(f"Rendered {name} -> {output_path}")

    if failures:
        print("\nErrors:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
