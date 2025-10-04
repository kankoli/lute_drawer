"""STEP export helpers for mold and rib geometry."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from lute_bowl.bowl_mold import MoldSection


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


def write_rib_surfaces_step(
    surfaces: Sequence[tuple[int, Sequence[np.ndarray]]],
    output_path: str | Path,
    *,
    unit_scale: float = 1.0,
    base_thickness_mm: float = 25.0,
    support_extension_mm: float = 20.0,
    spacing_mm: float | None = None,
    author: str = "auto",
    organization: str = "lute_drawer",
) -> Path:
    if unit_scale <= 0.0:
        raise ValueError("unit_scale must be positive")
    if base_thickness_mm <= 0.0:
        raise ValueError("base_thickness_mm must be positive")
    if support_extension_mm < 0.0:
        raise ValueError("support_extension_mm must be non-negative")

    path = Path(output_path)
    if path.suffix.lower() not in {".step", ".stp"}:
        path = path.with_suffix(".stp")

    thickness_units = base_thickness_mm / unit_scale
    support_extension_units = support_extension_mm / unit_scale
    spacing_units = None if spacing_mm is None else spacing_mm / unit_scale

    solids: list[tuple[str, Sequence[np.ndarray], Sequence[tuple[int, int, int]]]] = []

    for rib_idx, quads in surfaces:
        if not quads:
            continue
        vertices: list[np.ndarray] = []
        vertex_map: dict[tuple[int, int, int], int] = {}
        triangles: list[tuple[int, int, int]] = []

        def _add_vertex(coord: np.ndarray) -> int:
            key = tuple(np.round(coord, 9))
            if key in vertex_map:
                return vertex_map[key]
            idx = len(vertices)
            vertex_map[key] = idx
            vertices.append(np.asarray(coord, dtype=float))
            return idx

        for quad in quads:
            q = np.asarray(quad, dtype=float)
            if q.shape != (4, 3):
                raise ValueError("Each rib quad must be a (4,3) array")
            v0 = _add_vertex(q[0])
            v1 = _add_vertex(q[1])
            v2 = _add_vertex(q[2])
            v3 = _add_vertex(q[3])
            triangles.append((v0, v1, v2))
            triangles.append((v0, v2, v3))

        if not triangles:
            continue

        vertices_array = np.vstack(vertices)
        min_z = float(vertices_array[:, 2].min())
        base_z = min_z - thickness_units

        if spacing_units is not None:
            offset = np.array([0.0, (rib_idx - 1) * spacing_units, 0.0])
            vertices_array += offset
            for idx, arr in enumerate(vertices):
                vertices[idx] = vertices_array[idx]

        oriented_top: list[tuple[int, int, int]] = []
        edge_usage: dict[tuple[int, int], tuple[int, int] | None] = {}
        tail_threshold = float(vertices_array[:, 0].max()) - 1e-6
        tail_edges: set[tuple[int, int]] = set()

        for tri in triangles:
            a, b, c = tri
            pa, pb, pc = vertices[a], vertices[b], vertices[c]
            normal = np.cross(pb - pa, pc - pa)
            if np.linalg.norm(normal) < 1e-12:
                continue
            if normal[2] < 0:
                tri = (a, c, b)
            oriented_top.append(tri)
            for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = tuple(sorted(edge))
                if key in edge_usage:
                    edge_usage[key] = None
                else:
                    edge_usage[key] = edge
                    if max(vertices[edge[0]][0], vertices[edge[1]][0]) >= tail_threshold:
                        tail_edges.add(edge)

        if not oriented_top:
            continue

        bottom_vertices = [
            np.array([v[0], v[1], base_z], dtype=float)
            for v in vertices
        ]
        all_vertices = vertices + bottom_vertices
        n_top = len(vertices)

        solid_triangles: list[tuple[int, int, int]] = []
        solid_triangles.extend(oriented_top)
        solid_triangles.extend(
            (a + n_top, c + n_top, b + n_top)
            for (a, b, c) in oriented_top
        )

        for entry in edge_usage.values():
            if not entry:
                continue
            a, b = entry
            solid_triangles.append((a, b, b + n_top))
            solid_triangles.append((a, b + n_top, a + n_top))

        tail_vertices = sorted({idx for edge in tail_edges for idx in edge if vertices[idx][0] >= tail_threshold})
        if tail_vertices and support_extension_units > 1e-9:
            top_ext_map: dict[int, int] = {}
            bottom_ext_map: dict[int, int] = {}
            for idx_top in tail_vertices:
                v = vertices[idx_top]
                ext = np.array([v[0] + support_extension_units, v[1], v[2]], dtype=float)
                top_ext_map[idx_top] = len(all_vertices)
                all_vertices.append(ext)

                idx_bottom = idx_top + n_top
                vb = all_vertices[idx_bottom]
                ext_b = np.array([vb[0] + support_extension_units, vb[1], vb[2]], dtype=float)
                bottom_ext_map[idx_bottom] = len(all_vertices)
                all_vertices.append(ext_b)

            for edge in tail_edges:
                a, b = edge
                if max(vertices[a][0], vertices[b][0]) < tail_threshold:
                    continue
                a_ext_top = top_ext_map.get(a)
                b_ext_top = top_ext_map.get(b)
                if a_ext_top is None or b_ext_top is None:
                    continue
                a_bottom = a + n_top
                b_bottom = b + n_top
                a_ext_bottom = bottom_ext_map.get(a_bottom)
                b_ext_bottom = bottom_ext_map.get(b_bottom)
                if a_ext_bottom is None or b_ext_bottom is None:
                    continue

                solid_triangles.append((a, b, b_ext_top))
                solid_triangles.append((a, b_ext_top, a_ext_top))

                solid_triangles.append((a_ext_top, b_ext_top, b_ext_bottom))
                solid_triangles.append((a_ext_top, b_ext_bottom, a_ext_bottom))

                solid_triangles.append((a_bottom, a_ext_bottom, b_ext_bottom))
                solid_triangles.append((a_bottom, b_ext_bottom, b_bottom))

                solid_triangles.append((a, a_ext_top, a_ext_bottom))
                solid_triangles.append((a, a_ext_bottom, a_bottom))

        solid_triangles = _orient_triangles_outward(all_vertices, solid_triangles)
        solids.append((f"Rib{rib_idx}", all_vertices, solid_triangles))

    if not solids:
        raise ValueError("No rib surfaces provided for STEP export")

    return _write_solids_step(
        solids,
        path,
        unit_scale=unit_scale,
        author=author,
        organization=organization,
        label="RibSurfaces",
    )


__all__ = ["write_mold_sections_step", "write_rib_surfaces_step"]
