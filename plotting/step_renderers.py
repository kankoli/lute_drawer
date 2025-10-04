"""STEP export helpers for geometry visualisations."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from lute_bowl.bowl_mold import MoldSection


def write_mold_sections_step(
    mold_sections: Iterable[MoldSection],
    output_path: str | Path,
    *,
    unit_scale: float = 1.0,
    author: str = "auto",
    organization: str = "lute_drawer",
) -> Path:
    """Export mold boards as closed solids in a STEP (AP203) file."""

    path = Path(output_path)
    if path.suffix.lower() not in {".stp", ".step"}:
        path = path.with_suffix(".stp")

    sections = list(mold_sections)
    if not sections:
        raise ValueError("No mold sections provided for STEP export")

    def _format_number(value: float) -> str:
        scaled = float(value) * unit_scale
        if abs(scaled) < 1e-12:
            scaled = 0.0
        text = f"{scaled:.6f}"
        if "e" in text or "E" in text:
            return text
        text = text.rstrip("0").rstrip(".")
        return text or "0"

    def _format_direction(vec: np.ndarray) -> str:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            vec = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        unit = vec / norm
        return "(%s,%s,%s)" % tuple(
            f"{coord:.6f}".rstrip("0").rstrip(".") or "0" for coord in unit
        )

    entities: list[str] = []

    def _add(entity: str) -> int:
        idx = len(entities) + 1
        entities.append(f"#{idx}={entity};")
        return idx

    app_ctx = _add("APPLICATION_CONTEXT('mechanical design')")
    prod_ctx = _add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx},'design')")
    prod_usage_ctx = _add(f"PRODUCT_CONTEXT('',#{app_ctx},'mechanical')")
    length_unit = _add("SI_UNIT($,$,SI_UNIT_NAME.METRE)")
    plane_unit = _add("SI_UNIT($,$,SI_UNIT_NAME.RADIAN)")
    solid_unit = _add("SI_UNIT($,$,SI_UNIT_NAME.STERADIAN)")
    geom_ctx = _add("GEOMETRIC_REPRESENTATION_CONTEXT('','',3)")
    _add(f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{plane_unit},#{solid_unit},#{length_unit}),#{geom_ctx})")
    product = _add(f"PRODUCT('MoldSections','MoldSections','',(#{prod_usage_ctx}))")
    formation = _add(f"PRODUCT_DEFINITION_FORMATION('', '', #{product})")
    prod_def = _add(f"PRODUCT_DEFINITION('design','',#{formation},#{prod_ctx})")
    prod_shape = _add(f"PRODUCT_DEFINITION_SHAPE('','',#{prod_def})")

    solid_ids: list[int] = []

    for section_idx, section in enumerate(sections, start=1):
        face_left, face_right = section.faces
        if face_left.y.size != face_right.y.size:
            raise ValueError("Mismatched point counts between mold faces")
        n_pts = face_left.y.size
        if n_pts < 3:
            raise ValueError("Need at least three sample ribs to form a solid mold section")

        left_pts = [
            np.array([face_left.x, float(y), float(z)], dtype=float)
            for y, z in zip(face_left.y, face_left.z, strict=True)
        ]
        right_pts = [
            np.array([face_right.x, float(y), float(z)], dtype=float)
            for y, z in zip(face_right.y, face_right.z, strict=True)
        ]
        vertices = left_pts + right_pts
        solid_center = np.mean(vertices, axis=0)

        left_indices = list(range(n_pts))
        right_indices = [idx + n_pts for idx in range(n_pts)]

        triangles: list[list[int]] = []
        for i in range(1, n_pts - 1):
            triangles.append([left_indices[0], left_indices[i], left_indices[i + 1]])
        for i in range(1, n_pts - 1):
            triangles.append([right_indices[0], right_indices[i + 1], right_indices[i]])
        for i in range(n_pts):
            j = (i + 1) % n_pts
            triangles.append([left_indices[i], left_indices[j], right_indices[j]])
            triangles.append([left_indices[i], right_indices[j], right_indices[i]])

        point_ids: list[int] = []
        vertex_ids: list[int] = []
        for pt in vertices:
            cp = _add(
                "CARTESIAN_POINT('',(%s,%s,%s))"
                % (
                    _format_number(pt[0]),
                    _format_number(pt[1]),
                    _format_number(pt[2]),
                )
            )
            vp = _add(f"VERTEX_POINT('',#{cp})")
            point_ids.append(cp)
            vertex_ids.append(vp)

        edge_cache: dict[tuple[int, int], int] = {}

        def _edge_curve(a: int, b: int) -> tuple[int, bool]:
            if (a, b) in edge_cache:
                return edge_cache[(a, b)], True
            if (b, a) in edge_cache:
                return edge_cache[(b, a)], False

            start = vertices[a]
            end = vertices[b]
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1e-12:
                raise ValueError("Degenerate edge encountered while tessellating mold section")
            direction_id = _add(f"DIRECTION('',{_format_direction(delta)})")
            vector_id = _add(f"VECTOR('',#{direction_id},{_format_number(length)})")
            line_id = _add(f"LINE('',#{point_ids[a]},#{vector_id})")
            edge_id = _add(f"EDGE_CURVE('',#{vertex_ids[a]},#{vertex_ids[b]},#{line_id},.T.)")
            edge_cache[(a, b)] = edge_id
            return edge_id, True

        face_ids: list[int] = []

        for tri in triangles:
            v1, v2, v3 = tri
            p1, p2, p3 = vertices[v1], vertices[v2], vertices[v3]
            normal = np.cross(p2 - p1, p3 - p1)
            norm_len = float(np.linalg.norm(normal))
            if norm_len < 1e-12:
                continue
            centroid = (p1 + p2 + p3) / 3.0
            if float(np.dot(normal, centroid - solid_center)) < 0.0:
                v2, v3 = v3, v2
                p2, p3 = p3, p2
                normal = np.cross(p2 - p1, p3 - p1)
                norm_len = float(np.linalg.norm(normal))
                if norm_len < 1e-12:
                    continue

            normal_dir = _add(f"DIRECTION('',{_format_direction(normal)})")
            axis_ref = p2 - p1
            if float(np.linalg.norm(axis_ref)) < 1e-12:
                axis_ref = p3 - p1
            if float(np.linalg.norm(axis_ref)) < 1e-12 or abs(float(np.dot(axis_ref, normal))) / (float(np.linalg.norm(axis_ref)) * norm_len) > 0.99:
                axis_ref = np.cross(normal, np.array([1.0, 0.0, 0.0]))
            if float(np.linalg.norm(axis_ref)) < 1e-12:
                axis_ref = np.cross(normal, np.array([0.0, 1.0, 0.0]))
            if float(np.linalg.norm(axis_ref)) < 1e-12:
                axis_ref = np.array([1.0, 0.0, 0.0])
            axis_dir = _format_direction(axis_ref)
            axis_dir_id = _add(f"DIRECTION('',{axis_dir})")
            axis_id = _add(f"AXIS2_PLACEMENT_3D('',#{point_ids[v1]},#{normal_dir},#{axis_dir_id})")
            plane_id = _add(f"PLANE('',#{axis_id})")

            oriented_edges: list[int] = []
            for start, end in ((v1, v2), (v2, v3), (v3, v1)):
                edge_id, same = _edge_curve(start, end)
                orient = '.T.' if same else '.F.'
                oriented_edges.append(_add(f"ORIENTED_EDGE('',*,*,#{edge_id},{orient})"))

            edge_loop = _add(
                "EDGE_LOOP('',(%s))"
                % ",".join(f"#{eid}" for eid in oriented_edges)
            )
            face_bound = _add(f"FACE_OUTER_BOUND('',#{edge_loop},.T.)")
            face_ids.append(
                _add(f"ADVANCED_FACE('',(#{face_bound}),#{plane_id},.T.)")
            )

        if not face_ids:
            raise ValueError("Unable to construct facets for mold section")

        shell_id = _add(
            "CLOSED_SHELL('',(%s))"
            % ",".join(f"#{fid}" for fid in face_ids)
        )
        solid_ids.append(
            _add(f"MANIFOLD_SOLID_BREP('MoldSection{section_idx}',#{shell_id})")
        )

    shape_repr = _add(
        "SHAPE_REPRESENTATION('MoldSections',(%s),#%d)"
        % (",".join(f"#{sid}" for sid in solid_ids), geom_ctx)
    )
    _add(f"SHAPE_DEFINITION_REPRESENTATION(#{prod_shape},#{shape_repr})")

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('Mold sections export'),'2;1');\n"
        "FILE_NAME('%s','%s',('%s'),('%s'),'lute_drawer','lute_drawer','');\n"
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n"
        "ENDSEC;\n"
    ) % (path.name, timestamp, author, organization)

    data_section = "\n".join(entities)
    body = f"DATA;\n{data_section}\nENDSEC;\nEND-ISO-10303-21;\n"

    path.write_text(header + body)
    return path


__all__ = ["write_mold_sections_step"]
