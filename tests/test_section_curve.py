import numpy as np

from lute_bowl.section_curve import CircularSectionCurve, CosineArchSectionCurve


def test_divide_between_points_prefers_apex_when_span_reasonable():
    center = np.array([0.0, -1.0])
    radius = 2.0
    apex = np.array([0.0, 1.0])
    curve = CircularSectionCurve(center, radius, apex)

    start = curve.point_at_angle(-1.2)
    end = curve.point_at_angle(1.0)

    biased_samples = curve.divide_between_points(start, end, 5)
    biased_z = np.array(biased_samples)[:, 1]

    np.testing.assert_allclose(biased_samples[0], start)
    np.testing.assert_allclose(biased_samples[-1], end)
    assert biased_z.max() > curve.apex[1] * 0.95


def test_divide_between_points_avoids_full_wrap_when_apex_far():
    center = np.array([0.0, -1.0])
    radius = 2.0
    apex = np.array([0.0, 1.0])
    curve = CircularSectionCurve(center, radius, apex)

    start = curve.point_at_angle(2.8)
    end = curve.point_at_angle(-2.8)

    samples = curve.divide_between_points(start, end, 5)
    angles = np.unwrap([curve.angle_of_point(pt) for pt in samples])
    span = abs(angles[-1] - angles[0])

    np.testing.assert_allclose(samples[0], start)
    np.testing.assert_allclose(samples[-1], end)
    assert span < np.pi + 1e-6


def test_cosine_arch_hits_anchors_and_stays_above_plane():
    left = np.array([-150.0, 0.0])
    right = np.array([150.0, 0.0])
    apex = np.array([0.0, 50.0])
    curve = CosineArchSectionCurve.from_span(left, right, apex, cos_power=0.6)

    np.testing.assert_allclose(curve.point_at_angle(0.0), left)
    np.testing.assert_allclose(curve.point_at_angle(1.0), right)
    apex_pt = curve.point_at_angle(0.5)
    np.testing.assert_allclose(apex_pt[0], apex[0])
    assert apex_pt[1] >= 0.99 * apex[1]

    samples = curve.sample_points(200)
    assert samples[:, 1].min() >= -1e-6
