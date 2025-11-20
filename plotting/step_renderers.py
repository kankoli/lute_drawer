"""STEP export helpers for mold and rib geometry."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from lute_bowl.mold_builder import MoldSection


class _StepBuilder:
    def __init__(self) -> None:
        self.entities: list[str] = []

    def add(self, entity: str) -> int:
        idx = len(self.entities) + 1
        self.entities.append(f"#{idx}={entity};")
        return idx


def _format_number(value: float, unit_scale: float) -> str:
    scaled = float(value) * unit_scale
    if abs(scaled) < 1e-12:
        scaled = 0.0
    text = f"{scaled:.6f}"
    if "e" in text or "E" in text:
        return text
    text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_direction(vec: np.ndarray) -> str:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        vec = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    unit = vec / norm
    return "(%s,%s,%s)" % tuple(
        (f"{coord:.6f}".rstrip("0").rstrip(".") or "0") for coord in unit
    )


def _orient_triangles_outward(vertices: Sequence[np.ndarray], triangles: Sequence[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    verts = np.asarray(vertices, dtype=float)
    center = verts.mean(axis=0)
    oriented: list[tuple[int, int, int]] = []
    for tri in triangles:
        a, b, c = tri
        pa, pb, pc = verts[a], verts[b], verts[c]
        normal = np.cross(pb - pa, pc - pa)
        if np.linalg.norm(normal) < 1e-12:
            continue
        centroid = (pa + pb + pc) / 3.0
        if np.dot(normal, centroid - center) < 0.0:
            oriented.append((a, c, b))
        else:
            oriented.append(tri)
    return oriented


def _solid_triangles_to_step(
    builder: _StepBuilder,
    *,
    name: str,
    vertices: Sequence[np.ndarray],
    triangles: Sequence[tuple[int, int, int]],
    unit_scale: float,
) -> int:
    verts = [np.asarray(v, dtype=float) for v in vertices]
    triangles = _orient_triangles_outward(verts, triangles)
    if not triangles:
        raise ValueError("No triangles provided for STEP solid export")

    point_ids: list[int] = []
    vertex_ids: list[int] = []
    for v in verts:
        cp = builder.add(
            "CARTESIAN_POINT('',(%s,%s,%s))"
            % (
                _format_number(v[0], unit_scale),
                _format_number(v[1], unit_scale),
                _format_number(v[2], unit_scale),
            )
        )
        point_ids.append(cp)
        vertex_ids.append(builder.add(f"VERTEX_POINT('',#{cp})"))

    edge_cache: dict[tuple[int, int], int] = {}

    def _edge_curve(a: int, b: int) -> tuple[int, bool]:
        key = (a, b)
        if key in edge_cache:
            return edge_cache[key], True
        rev = (b, a)
        if rev in edge_cache:
            return edge_cache[rev], False
        start = verts[a]
        end = verts[b]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-12:
            raise ValueError("Degenerate edge encountered while tessellating solid")
        direction_id = builder.add(f"DIRECTION('',{_format_direction(delta)})")
        vector_id = builder.add(f"VECTOR('',#{direction_id},{_format_number(length, 1.0)})")
        line_id = builder.add(f"LINE('',#{point_ids[a]},#{vector_id})")
        edge_id = builder.add(f"EDGE_CURVE('',#{vertex_ids[a]},#{vertex_ids[b]},#{line_id},.T.)")
        edge_cache[key] = edge_id
        return edge_id, True

    face_ids: list[int] = []
    for tri in triangles:
        a, b, c = tri
        pa, pb, pc = verts[a], verts[b], verts[c]
        normal = np.cross(pb - pa, pc - pa)
        norm_len = float(np.linalg.norm(normal))
        if norm_len < 1e-12:
            continue
        normal_dir = builder.add(f"DIRECTION('',{_format_direction(normal)})")
        axis_ref = pb - pa
        axis_ref -= normal * (np.dot(axis_ref, normal) / (norm_len ** 2))
        if np.linalg.norm(axis_ref) < 1e-12:
            axis_ref = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis_ref) < 1e-12:
            axis_ref = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(axis_ref) < 1e-12:
            axis_ref = np.array([1.0, 0.0, 0.0])
        axis_dir = builder.add(f"DIRECTION('',{_format_direction(axis_ref)})")
        axis_id = builder.add(f"AXIS2_PLACEMENT_3D('',#{point_ids[a]},#{normal_dir},#{axis_dir})")
        plane_id = builder.add(f"PLANE('',#{axis_id})")

        oriented_edges: list[int] = []
        for start, end in ((a, b), (b, c), (c, a)):
            edge_id, same = _edge_curve(start, end)
            orient_flag = '.T.' if same else '.F.'
            oriented_edges.append(builder.add(f"ORIENTED_EDGE('',*,*,#{edge_id},{orient_flag})"))
        edge_loop = builder.add(
            "EDGE_LOOP('',(%s))" % ",".join(f"#{eid}" for eid in oriented_edges)
        )
        face_bound = builder.add(f"FACE_OUTER_BOUND('',#{edge_loop},.T.)")
        face_ids.append(builder.add(f"ADVANCED_FACE('',(#{face_bound}),#{plane_id},.T.)"))

    shell_id = builder.add(
        "CLOSED_SHELL('',(%s))" % ",".join(f"#{fid}" for fid in face_ids)
    )
    return builder.add(f"MANIFOLD_SOLID_BREP('{name}',#{shell_id})")


def _write_solids_step(
    solids: list[tuple[str, Sequence[np.ndarray], Sequence[tuple[int, int, int]]]],
    output_path: Path,
    *,
    unit_scale: float,
    author: str,
    organization: str,
    label: str,
) -> Path:
    builder = _StepBuilder()

    app_ctx = builder.add("APPLICATION_CONTEXT('mechanical design')")
    prod_ctx = builder.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')")
    prod_usage_ctx = builder.add(f"PRODUCT_CONTEXT('',#{app_ctx},'mechanical')")
    length_unit = builder.add("SI_UNIT($,$,SI_UNIT_NAME.METRE)")
    plane_unit = builder.add("SI_UNIT($,$,SI_UNIT_NAME.RADIAN)")
    solid_unit = builder.add("SI_UNIT($,$,SI_UNIT_NAME.STERADIAN)")
    geom_ctx = builder.add("GEOMETRIC_REPRESENTATION_CONTEXT('','',3)")
    builder.add(f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{plane_unit},#{solid_unit},#{length_unit}),#{geom_ctx})")
    product = builder.add(f"PRODUCT('{label}','{label}','',(#{prod_usage_ctx}))")
    formation = builder.add(f"PRODUCT_DEFINITION_FORMATION('', '', #{product})")
    prod_def = builder.add(f"PRODUCT_DEFINITION('design','',#{formation},#{prod_ctx})")
    prod_shape = builder.add(f"PRODUCT_DEFINITION_SHAPE('','',#{prod_def})")

    solid_ids: list[int] = []
    for name, vertices, triangles in solids:
        solid_ids.append(
            _solid_triangles_to_step(
                builder,
                name=name,
                vertices=vertices,
                triangles=triangles,
                unit_scale=unit_scale,
            )
        )

    shape_repr = builder.add(
        "SHAPE_REPRESENTATION('%s',(%s),#%d)"
        % (label, ",".join(f"#{sid}" for sid in solid_ids), geom_ctx)
    )
    builder.add(f"SHAPE_DEFINITION_REPRESENTATION(#{prod_shape},#{shape_repr})")

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('Generated by lute_drawer'),'2;1');\n"
        "FILE_NAME('%s','%s',('%s'),('%s'),'lute_drawer','lute_drawer','');\n"
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n"
        "ENDSEC;\n"
    ) % (output_path.name, timestamp, author, organization)

    data_section = "\n".join(builder.entities)
    body = f"DATA;\n{data_section}\nENDSEC;\nEND-ISO-10303-21;\n"
    output_path.write_text(header + body)
    return output_path


def write_mold_sections_step(
    mold_sections: Iterable[MoldSection],
    output_path: str | Path,
    *,
    unit_scale: float = 1.0,
    author: str = "auto",
    organization: str = "lute_drawer",
) -> Path:
    path = Path(output_path)
    if path.suffix.lower() not in {".step", ".stp"}:
        path = path.with_suffix(".stp")

    solids: list[tuple[str, Sequence[np.ndarray], Sequence[tuple[int, int, int]]]] = []

    for section_idx, section in enumerate(mold_sections, start=1):
        face_left, face_right = section.faces
        if face_left.y.size != face_right.y.size:
            raise ValueError("Mold faces must have matching sample counts")
        n_pts = face_left.y.size
        if n_pts < 3:
            raise ValueError("Mold section requires at least three samples")

        left_pts = [
            np.array([face_left.x, float(y), float(z)], dtype=float)
            for y, z in zip(face_left.y, face_left.z, strict=True)
        ]
        right_pts = [
            np.array([face_right.x, float(y), float(z)], dtype=float)
            for y, z in zip(face_right.y, face_right.z, strict=True)
        ]
        vertices = left_pts + right_pts

        left_indices = list(range(n_pts))
        right_indices = [idx + n_pts for idx in range(n_pts)]

        triangles: list[tuple[int, int, int]] = []
        for i in range(1, n_pts - 1):
            triangles.append((left_indices[0], left_indices[i], left_indices[i + 1]))
        for i in range(1, n_pts - 1):
            triangles.append((right_indices[0], right_indices[i + 1], right_indices[i]))
        for i in range(n_pts):
            j = (i + 1) % n_pts
            triangles.append((left_indices[i], left_indices[j], right_indices[j]))
            triangles.append((left_indices[i], right_indices[j], right_indices[i]))

        solids.append((f"MoldSection{section_idx}", vertices, triangles))

    if not solids:
        raise ValueError("No mold sections provided for STEP export")

    return _write_solids_step(
        solids,
        path,
        unit_scale=unit_scale,
        author=author,
        organization=organization,
        label="MoldSections",
    )


__all__ = ["write_mold_sections_step"]
