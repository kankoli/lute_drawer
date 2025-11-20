import io
import sys
import types
import unittest
from contextlib import redirect_stdout

import numpy as np

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

plotting_stub = types.ModuleType("plotting")
plotting_bowl_stub = types.ModuleType("plotting.bowl")
plotting_bowl_stub.plot_bowl = lambda *args, **kwargs: None
plotting_bowl_stub.plot_mold_sections_2d = lambda *args, **kwargs: None
plotting_ribs_stub = types.ModuleType("plotting.ribs")
plotting_ribs_stub.plot_rib_surfaces = lambda *args, **kwargs: None
plotting_ribs_stub.plot_rib_surface_with_planes = lambda *args, **kwargs: None
plotting_step_stub = types.ModuleType("plotting.step_renderers")
plotting_step_stub.write_mold_sections_step = lambda *args, **kwargs: None
plotting_stub.bowl = plotting_bowl_stub
plotting_stub.ribs = plotting_ribs_stub
plotting_stub.step_renderers = plotting_step_stub
sys.modules.setdefault("plotting", plotting_stub)
sys.modules.setdefault("plotting.bowl", plotting_bowl_stub)
sys.modules.setdefault("plotting.ribs", plotting_ribs_stub)
sys.modules.setdefault("plotting.step_renderers", plotting_step_stub)

import lute_soundboard as lutes
from lute_bowl.rib_builder import build_bowl_ribs
from lute_bowl.rib_form_builder import (
    all_rib_surfaces_convex,
    build_rib_surfaces,
    find_rib_side_planes,
    measure_rib_plane_deviation,
)


class PlanarRibFormBuilderTests(unittest.TestCase):
    def setUp(self):
        self.lute = lutes.ManolLavta()

    def test_planar_rib_surfaces_generate_quads(self):
        sections, rib_outlines = build_bowl_ribs(
            self.lute,
            n_ribs=4,
            n_sections=30,
        )
        surfaces = build_rib_surfaces(
            rib_outlines=rib_outlines,
            rib_index=2,
        )

        self.assertEqual(len(surfaces), 1)
        rib_idx, quads = surfaces[0]
        self.assertEqual(rib_idx, 2)
        self.assertEqual(len(quads), len(sections) - 1)

        rib_a = np.asarray(rib_outlines[rib_idx - 1], dtype=float)
        rib_b = np.asarray(rib_outlines[rib_idx], dtype=float)
        self.assertEqual(rib_a.shape, rib_b.shape)
        self.assertEqual(rib_a.shape[0], len(sections))

        for quad, p0, p1, q0, q1 in zip(
            quads,
            rib_a[:-1],
            rib_a[1:],
            rib_b[:-1],
            rib_b[1:],
            strict=False,
        ):
            self.assertEqual(quad.shape, (4, 3))
            self.assertTrue(np.allclose(quad[0], p0))
            self.assertTrue(np.allclose(quad[1], p1))
            self.assertTrue(np.allclose(quad[2], q1))
            self.assertTrue(np.allclose(quad[3], q0))

    def test_find_rib_side_planes_parallel_and_centered(self):
        _, rib_outlines = build_bowl_ribs(
            self.lute,
            n_ribs=4,
            n_sections=40,
        )
        unit_scale = self.lute.unit_in_mm() / self.lute.unit
        plane_a, plane_b = find_rib_side_planes(
            rib_outlines=rib_outlines,
            rib_index=2,
            plane_gap_mm=70.0,
            unit_scale=unit_scale,
        )

        outline_a = np.asarray(rib_outlines[1], dtype=float)
        outline_b = np.asarray(rib_outlines[2], dtype=float)
        connectors = outline_b - outline_a
        idx = int(np.argmax(np.linalg.norm(connectors, axis=1)))
        width_vec = connectors[idx]
        width_dir = width_vec / np.linalg.norm(width_vec)

        self.assertAlmostEqual(float(np.dot(width_dir, plane_a.normal)), 1.0, places=3)
        self.assertAlmostEqual(float(np.dot(width_dir, plane_b.normal)), -1.0, places=3)
        self.assertEqual(plane_a.corners.shape, (4, 3))
        self.assertEqual(plane_b.corners.shape, (4, 3))

        expected_gap_units = 70.0 / unit_scale
        actual_gap = abs(float(np.dot(plane_b.point - plane_a.point, width_dir)))
        self.assertAlmostEqual(actual_gap, expected_gap_units, places=6)

        auto_a, auto_b = find_rib_side_planes(
            rib_outlines=rib_outlines,
            rib_index=2,
            plane_gap_mm=None,
            unit_scale=unit_scale,
        )
        auto_gap = abs(float(np.dot(auto_b.point - auto_a.point, width_dir)))
        self.assertAlmostEqual(auto_gap, np.linalg.norm(width_vec), places=6)

    def test_all_rib_surfaces_convex_positive(self):
        outline_a = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
            ]
        )
        outline_b = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 5.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        rib_outlines = [outline_a, outline_b]
        self.assertTrue(
            all_rib_surfaces_convex(
                rib_outlines=rib_outlines,
                plane_gap_mm=None,
                unit_scale=1.0,
            )
        )

    def test_all_rib_surfaces_convex_detects_failures(self):
        outline_a = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        outline_b = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        rib_outlines = [outline_a, outline_b]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = all_rib_surfaces_convex(
                rib_outlines=rib_outlines,
                plane_gap_mm=None,
                unit_scale=1.0,
            )
        self.assertFalse(result)
        self.assertIn("Non-convex ribs: 1", buffer.getvalue())

    def test_measure_rib_plane_deviation(self):
        outline_a = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
            ]
        )
        outline_b = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 5.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        rib_outlines = [outline_a, outline_b]
        deviation = measure_rib_plane_deviation(
            rib_outlines=rib_outlines,
            rib_index=1,
            plane_gap_mm=None,
            unit_scale=1.0,
        )
        self.assertEqual(deviation.long_deltas.shape, (3,))
        self.assertEqual(deviation.height_deltas.shape, (3,))
        self.assertAlmostEqual(float(deviation.long_deltas[1]), 0.0, places=6)
        self.assertGreater(float(deviation.height_deltas[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
