#!/usr/bin/env python3
"""Print measurement tables for every concrete soundboard class."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lute_soundboard as lute_defs
from lute_soundboard import LuteSoundboard


def iter_soundboards(names=None):
    for name, cls in inspect.getmembers(lute_defs, inspect.isclass):
        if not issubclass(cls, LuteSoundboard):
            continue
        if cls is LuteSoundboard:
            continue
        if getattr(cls, "__abstractmethods__", None):
            continue
        if names and name not in names:
            continue
        yield name, cls


def format_measurements(lute: LuteSoundboard) -> str:
    rows = []
    for measurement in lute.measurements():
        value = float(measurement.value)
        value_mm = float(measurement.value_in_mm)
        rows.append(
            f"  {measurement.label:<24} "
            f"{value:>10.3f} units  "
            f"({value_mm:>10.2f} mm)"
        )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name",
        nargs="*",
        help="Optional list of soundboard class names to print (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = list(iter_soundboards(args.name))
    if not entries:
        targets = ", ".join(args.name) if args.name else ""
        raise SystemExit(f"No matching soundboard classes found: {targets}")

    for name, cls in entries:
        lute = cls()
        print(f"{name}\n{'-' * len(name)}")
        print(format_measurements(lute))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
