"""Bowl construction helpers."""

from .bowl_from_soundboard import (
    Section,
    compute_bowl_inner_volume,
    compute_equivalent_flat_side_depth,
    compute_soundboard_outline_area,
)
from .top_curves import (
    DeepBackCurve,
    FlatBackCurve,
    MidCurve,
    SideProfileParameters,
    SideProfilePerControlTopCurve,
    SimpleAmplitudeCurve,
    TopCurve,
    resolve_widths,
    resolve_amplitude,
)
from .mold_builder import MoldSection, build_mold_sections
from .rib_builder import build_bowl_ribs
from .rib_form_builder import build_rib_surfaces

__all__ = [
    "Section",
    "compute_bowl_inner_volume",
    "compute_equivalent_flat_side_depth",
    "compute_soundboard_outline_area",
    "build_bowl_ribs",
    "DeepBackCurve",
    "FlatBackCurve",
    "MidCurve",
    "SideProfileParameters",
    "SideProfilePerControlTopCurve",
    "SimpleAmplitudeCurve",
    "TopCurve",
    "resolve_widths",
    "resolve_amplitude",
    "MoldSection",
    "build_mold_sections",
    "build_rib_surfaces",
]
