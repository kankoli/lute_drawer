"""Shared helpers for geometry analysis scripts."""
from __future__ import annotations

import inspect
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Type

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lute_soundboard as lute_defs
from lute_soundboard import LuteSoundboard
from bowl_from_soundboard import SideProfilePerControlTopCurve
from plotting.svg import SvgRenderer


def ensure_matplotlib_stub() -> None:
    """Install a minimal matplotlib stub to avoid cache and GUI access."""
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


@contextmanager
def sandboxed_renderer() -> Iterator[None]:
    """Temporarily disable SVG writes by stubbing SvgRenderer.draw."""
    original_draw = SvgRenderer.draw
    SvgRenderer.draw = lambda self, objs: None
    try:
        yield
    finally:
        SvgRenderer.draw = original_draw


def iter_concrete_lute_classes(names: Iterable[str] | None = None) -> List[Tuple[str, Type[LuteSoundboard]]]:
    """Return (name, class) pairs for concrete lute subclasses."""
    lute_classes: List[Tuple[str, Type[LuteSoundboard]]] = []
    for name, cls in inspect.getmembers(lute_defs, inspect.isclass):
        if not issubclass(cls, LuteSoundboard):
            continue
        if cls is LuteSoundboard:
            continue
        if not getattr(cls, "__module__", "").startswith("lute_soundboard"):
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


def collect_top_curve_classes() -> List[Tuple[str, Type[SideProfilePerControlTopCurve]]]:
    """Discover concrete top-curve classes defined in bowl_from_soundboard."""
    import bowl_from_soundboard as bfs

    curves: List[Tuple[str, Type[SideProfilePerControlTopCurve]]] = []
    for name, cls in inspect.getmembers(bfs, inspect.isclass):
        if issubclass(cls, SideProfilePerControlTopCurve) and cls is not SideProfilePerControlTopCurve:
            curves.append((name, cls))
    return curves
