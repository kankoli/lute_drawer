from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .planar_bowl_generator import build_planar_bowl_for_lute
from .bowl_top_curves import MidCurve
from plotting.bowl import plot_rib_surfaces


def build_planar_rib_surfaces(
    lute,
    *,
    top_curve=MidCurve,
    n_ribs: int = 13,
    n_sections: int = 160,
    upper_block_units: float = 1.0,
    lower_block_units: float = 0.1,
    rib_index: int | Sequence[int] | None = None,
) -> tuple[list[object], list[tuple[int, list[np.ndarray]]], list[tuple[int, tuple[np.ndarray, np.ndarray]]]]:
    """Return rib surfaces built from planar rib outlines."""
    sections, rib_outlines = build_planar_bowl_for_lute(
        lute,
        n_ribs=n_ribs,
        n_sections=n_sections,
        top_curve=top_curve,
        upper_block_units=upper_block_units,
        lower_block_units=lower_block_units,
    )

    if rib_index is None:
        indices = list(range(len(rib_outlines) - 1))
    elif isinstance(rib_index, Iterable) and not isinstance(rib_index, (str, bytes)):
        indices = [idx - 1 for idx in rib_index]
    else:
        indices = [int(rib_index) - 1]

    surfaces: list[tuple[int, list[np.ndarray]]] = []
    outlines: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []

    for idx in indices:
        if idx < 0 or idx >= len(rib_outlines) - 1:
            raise ValueError(f"Rib index {idx + 1} is outside available range 1..{len(rib_outlines) - 1}")
        rib_a = np.asarray(rib_outlines[idx], dtype=float)
        rib_b = np.asarray(rib_outlines[idx + 1], dtype=float)
        _validate_planar_pair(rib_a, rib_b)
        quads = _planar_quads_between_ribs(rib_a, rib_b)
        surfaces.append((idx + 1, quads))
        outlines.append((idx + 1, (rib_a, rib_b)))

    return sections, surfaces, outlines


def plot_planar_ribs(
    lute,
    *,
    top_curve=MidCurve,
    n_ribs: int = 13,
    n_sections: int = 160,
    upper_block_units: float = 1.0,
    lower_block_units: float = 0.1,
    rib: int = 7,
    title: str | None = None,
):
    _, surfaces, outlines = build_planar_rib_surfaces(
        lute,
        top_curve=top_curve,
        n_ribs=n_ribs,
        n_sections=n_sections,
        upper_block_units=upper_block_units,
        lower_block_units=lower_block_units,
        rib_index=rib,
    )
    plot_rib_surfaces(
        surfaces,
        outlines=outlines,
        title=title,
        lute_name=type(lute).__name__,
    )
    return surfaces


def _validate_planar_pair(rib_a: np.ndarray, rib_b: np.ndarray) -> None:
    if rib_a.shape != rib_b.shape or rib_a.ndim != 2 or rib_a.shape[1] != 3:
        raise ValueError("Planar rib outlines must share shape (N,3)")
    if rib_a.shape[0] < 2:
        raise ValueError("Planar rib outlines require at least two samples per rib.")

    xs_a = rib_a[:, 0]
    xs_b = rib_b[:, 0]
    if not np.allclose(xs_a, xs_b, atol=1e-8):
        raise ValueError("Planar rib outlines must share the same X samples.")


def _planar_quads_between_ribs(rib_a: np.ndarray, rib_b: np.ndarray) -> List[np.ndarray]:
    quads: List[np.ndarray] = []
    for p0, p1, q0, q1 in zip(rib_a[:-1], rib_a[1:], rib_b[:-1], rib_b[1:], strict=False):
        quad = np.vstack([p0, p1, q1, q0])
        quads.append(quad)
    return quads


def _normalize_quads(outline1: np.ndarray, outline2: np.ndarray, quads: list[np.ndarray]):
    outline1 = np.asarray(outline1, float)
    outline2 = np.asarray(outline2, float)
    if outline1.shape != outline2.shape or outline1.ndim != 2 or outline1.shape[1] != 3:
        raise ValueError("_normalize_quads expects two (N,3) outlines with same shape")

    connectors = outline2 - outline1
    idx = int(np.argmax(np.linalg.norm(connectors, axis=1)))
    across = connectors[idx] / (np.linalg.norm(connectors[idx]) + EPS)

    if idx < len(outline1) - 1:
        along = outline1[idx + 1] - outline1[idx]
    else:
        along = outline1[idx] - outline1[idx - 1]
    along /= (np.linalg.norm(along) + EPS)

    z_local = np.cross(along, across)
    nz = np.linalg.norm(z_local)
    if nz < EPS:
        return quads
    z_local /= nz
    y_local = across
    x_local = np.cross(y_local, z_local)
    x_local /= (np.linalg.norm(x_local) + EPS)

    R = np.vstack([x_local, y_local, z_local]).T
    R_inv = R.T

    def to_local(A):
        return (R_inv @ A.T).T

    q_local = [to_local(q) for q in quads]
    mid = 0.5 * (to_local(outline1)[idx] + to_local(outline2)[idx])
    return [q - mid for q in q_local]


__all__ = [
    "build_planar_rib_surfaces",
    "plot_planar_ribs",
]
