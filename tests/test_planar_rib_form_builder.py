import sys
import types
import unittest

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
plotting_bowl_stub.plot_rib_surfaces = lambda *args, **kwargs: None
plotting_bowl_stub.plot_bowl = lambda *args, **kwargs: None
plotting_bowl_stub.plot_mold_sections_2d = lambda *args, **kwargs: None
plotting_step_stub = types.ModuleType("plotting.step_renderers")
plotting_step_stub.write_rib_surfaces_step = lambda *args, **kwargs: None
plotting_step_stub.write_mold_sections_step = lambda *args, **kwargs: None
plotting_stub.bowl = plotting_bowl_stub
plotting_stub.step_renderers = plotting_step_stub
sys.modules.setdefault("plotting", plotting_stub)
sys.modules.setdefault("plotting.bowl", plotting_bowl_stub)
sys.modules.setdefault("plotting.step_renderers", plotting_step_stub)

import lute_soundboard as lutes
from lute_bowl.planar_bowl_generator import build_planar_bowl_for_lute
from lute_bowl.planar_rib_form_builder import build_rib_surfaces


class PlanarRibFormBuilderTests(unittest.TestCase):
    def setUp(self):
        self.lute = lutes.ManolLavta()

    def test_planar_rib_surfaces_generate_quads(self):
        sections, rib_outlines = build_planar_bowl_for_lute(
            self.lute,
            n_ribs=4,
            n_sections=30,
        )
        surfaces, outlines = build_rib_surfaces(
            rib_outlines=rib_outlines,
            rib_index=2,
        )

        self.assertEqual(len(surfaces), 1)
        rib_idx, quads = surfaces[0]
        self.assertEqual(rib_idx, 2)
        self.assertEqual(len(quads), len(sections) - 1)

        _, (rib_a, rib_b) = outlines[0]
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


if __name__ == "__main__":
    unittest.main()
