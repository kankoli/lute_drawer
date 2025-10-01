import sys
import types
import unittest
from unittest import mock
import warnings

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

import lutes
from bowl_from_soundboard import _resolve_top_curve, build_bowl_for_lute
from bowl_top_curves import SimpleAmplitudeCurve


class BuildBowlForLuteTests(unittest.TestCase):
    def _make_lute(self):
        return lutes.ManolLavta()

    @mock.patch("lute_renderers.SvgRenderer.draw", autospec=True)
    def test_build_bowl_constant_curve_shapes(self, mock_draw):
        top_curve = lambda _: 1.0
        lute = self._make_lute()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sections, rib_outlines = build_bowl_for_lute(lute, n_ribs=4, n_sections=8, top_curve=top_curve)

        self.assertTrue(
            any(
                issubclass(w.category, RuntimeWarning)
                and "Section circle center" in str(w.message)
                for w in caught
            )
        )

        self.assertEqual(len(rib_outlines), 5)  # n+1 outlines for n ribs
        self.assertEqual(len(sections), 8 + 2)
        self.assertTrue(all(rib.shape == (len(sections), 3) for rib in rib_outlines))

        if len(sections) > 2:
            for (x_pos, _, _, apex) in sections[1:-1]:
                expected_z = top_curve(x_pos)
                self.assertAlmostEqual(float(apex[1]), expected_z, places=6)

        for rib in rib_outlines:
            self.assertAlmostEqual(rib[0, 2], 0.0)
            self.assertAlmostEqual(rib[-1, 2], 0.0)
            if rib.shape[0] > 2:
                interior_z = rib[1:-1, 2]
                self.assertTrue(np.all(np.isfinite(interior_z)))
                self.assertGreaterEqual(float(interior_z.min()), 0.0)

        self.assertAlmostEqual(sections[0][2], 0.0)
        self.assertAlmostEqual(sections[-1][2], 0.0)

    @mock.patch("lute_renderers.SvgRenderer.draw", autospec=True)
    def test_build_bowl_enforces_minimum_rib_count(self, mock_draw):
        lute = self._make_lute()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sections, ribs = build_bowl_for_lute(lute, n_ribs=0, n_sections=5, top_curve=lambda _: 1.0)

        self.assertEqual(len(ribs), 2)
        self.assertEqual(ribs[0].shape[0], len(sections))

    @mock.patch("lute_renderers.SvgRenderer.draw", autospec=True)
    def test_build_bowl_without_soundhole(self, mock_draw):
        lute = lutes.BaltaSaz()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sections, ribs = build_bowl_for_lute(lute, n_ribs=4, n_sections=6, top_curve=SimpleAmplitudeCurve)

        self.assertEqual(len(ribs), 5)
        self.assertGreater(len(sections), 2)
        self.assertFalse(
            any(
                issubclass(w.category, RuntimeWarning)
                and "Section circle center" in str(w.message)
                for w in caught
            )
        )

        for rib in ribs:
            self.assertTrue(np.all(np.isfinite(rib[:, 2])))
            self.assertGreaterEqual(float(rib[:, 2].min()), -1e-9)


class ResolveTopCurveTests(unittest.TestCase):
    def setUp(self):
        self.lute = lutes.ManolLavta()

    def test_resolve_top_curve_passthrough(self):
        sentinel = lambda _: 2.5
        resolved = _resolve_top_curve(self.lute, sentinel)
        self.assertIs(resolved, sentinel)

    def test_resolve_top_curve_from_class(self):
        resolved = _resolve_top_curve(self.lute, SimpleAmplitudeCurve)
        self.assertTrue(callable(resolved))
        sample_x = float(self.lute.form_center.x)
        self.assertGreater(resolved(sample_x), 0.0)

    def test_resolve_top_curve_from_instance(self):
        resolved = _resolve_top_curve(self.lute, SimpleAmplitudeCurve())
        self.assertTrue(callable(resolved))


if __name__ == "__main__":
    unittest.main()
