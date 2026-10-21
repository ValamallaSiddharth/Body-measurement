import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from bodymeasure.pipeline import (View, estimate_scale, normalize_views, ellipse_circumference,
                                  slice_width, metric, sanity_checks, detect_pose, CaptureError)


def view(name="front", factor=1, chin=True):
    points = np.zeros((33, 2))
    points[11] = (30*factor, 30*factor)
    points[12] = (50*factor, 30*factor)
    return View(name, points, np.ones(33), np.zeros((120*factor, 80*factor), dtype=np.uint8),
                10*factor, 110*factor, 23*factor if chin else None)


class GeometryTests(unittest.TestCase):
    def test_independent_scale_and_different_camera_distances(self):
        front, side = view(), view("side", 2)
        a, b = estimate_scale(front, "male", []), estimate_scale(side, "male", [])
        self.assertAlmostEqual(a[0], b[0])
        self.assertAlmostEqual(front.cm_per_pixel, 2*side.cm_per_pixel)
        height = normalize_views(front, side, [a[0], b[0]], [a[1], b[1]], [])
        self.assertAlmostEqual(front.height_px*front.cm_per_pixel, height)
        self.assertAlmostEqual(side.height_px*side.cm_per_pixel, height)

    def test_head_anchor_and_missing_face_prior(self):
        subject = view()
        height, anchored = estimate_scale(subject, "male", [])
        self.assertTrue(anchored)
        self.assertAlmostEqual(height, 100*23.2/13)
        subject.chin = None
        warnings = []
        height, anchored = estimate_scale(subject, "female", warnings)
        self.assertFalse(anchored)
        self.assertAlmostEqual(height, 21.8*7.5)
        self.assertTrue(warnings)

    def test_ellipse_units_and_unavailable(self):
        self.assertAlmostEqual(ellipse_circumference(10, 10), 10*math.pi)
        self.assertEqual(metric(25.4), {"cm": 25.4, "in": 10.0})
        self.assertEqual(metric(None), {"cm": None, "in": None})

    def test_slice_excludes_separate_arms(self):
        subject = view(); subject.cm_per_pixel = 1
        subject.mask[40:80, 5:10] = 1
        subject.mask[40:80, 25:55] = 1
        subject.mask[40:80, 65:70] = 1
        self.assertEqual(slice_width(subject, .5, 40, "waist", []), 30)
        self.assertEqual(slice_width(subject, .5, 7, "wrist", [], limb=True), 5)
        warnings = []
        self.assertIsNone(slice_width(subject, .5, 40, "wrist", warnings, limb=True))
        self.assertTrue(warnings)

    def test_implausible_values_are_soft_warnings(self):
        warnings = []
        sanity_checks(250, {"torso": {"waist": metric(150), "hips": metric(90)}}, {"chest": (20, 45)}, warnings)
        self.assertEqual(len(warnings), 3)

    def test_low_required_joint_visibility_errors(self):
        landmarks = [SimpleNamespace(x=.5, y=.5, visibility=.1) for _ in range(33)]
        with patch("bodymeasure.pipeline.mp.solutions.pose.Pose") as factory:
            factory.return_value.__enter__.return_value.process.return_value = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))
            with self.assertRaises(CaptureError) as caught:
                detect_pose(np.zeros((256, 256, 3), dtype=np.uint8), "front", [])
        self.assertEqual(caught.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
