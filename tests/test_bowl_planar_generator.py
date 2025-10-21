import unittest

import numpy as np

import lute_soundboard as lutes
from lute_bowl.planar_bowl_generator import build_planar_bowl_for_lute


def _max_plane_distance(points: np.ndarray) -> float:
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] < 3:
        return 0.0
    base = arr[0]
    normal = None
    for idx in range(1, arr.shape[0] - 1):
        v1 = arr[idx] - base
        v2 = arr[idx + 1] - base
        candidate = np.cross(v1, v2)
        if np.linalg.norm(candidate) > 1e-9:
            normal = candidate
            break
    if normal is None:
        return 0.0
    normal = normal / np.linalg.norm(normal)
    distances = np.abs((arr - base) @ normal)
    return float(distances.max())


class PlanarBowlGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.lute = lutes.ManolLavta()
        self.unit = float(self.lute.unit)

    def test_planar_ribs_respect_bounds(self):
        sections, ribs = build_planar_bowl_for_lute(
            self.lute,
            n_ribs=6,
            n_sections=40,
            upper_block_units=0.0,
            lower_block_units=0.0,
        )

        self.assertEqual(len(ribs), 7)
        self.assertEqual(len(sections), 40)

        start_expected = float(self.lute.point_neck_joint.x)
        end_expected = float(self.lute.form_bottom.x)

        self.assertTrue(
            abs(float(sections[0].x) - start_expected) < 1e-6,
            "Sections should begin at the upper end-block boundary.",
        )
        self.assertTrue(
            abs(float(sections[-1].x) - end_expected) < 1e-6,
            "Sections should end at the lower end-block boundary.",
        )

        tolerance = 1e-5 * max(1.0, self.unit)
        for rib in ribs:
            self.assertEqual(rib.shape[0], len(sections))
            min_x = float(rib[:, 0].min())
            max_x = float(rib[:, 0].max())
            self.assertGreaterEqual(min_x + 1e-6, start_expected)
            self.assertLessEqual(max_x - 1e-6, end_expected)
            max_distance = _max_plane_distance(rib)
            self.assertLess(
                max_distance,
                tolerance,
                f"Rib deviates from planarity by {max_distance}",
            )


if __name__ == "__main__":
    unittest.main()
