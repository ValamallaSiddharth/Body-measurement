"""Regression tests for soft frame-boundary warnings, including real HTTP."""
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from uuid import uuid4

import cv2
import numpy as np
from werkzeug.serving import make_server
from app import create_app
from bodymeasure.pipeline import (View, edges_touched, boundary_confidence,
                                  prepare_view, segment_body, slice_width, detect_pose)


class BoundaryTests(unittest.TestCase):
    def test_edge_names_tolerance_and_empty_mask(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        self.assertEqual(edges_touched(mask), [])
        mask[2:18, 2:28] = 1
        self.assertEqual(edges_touched(mask), ["top", "bottom", "left", "right"])
        self.assertEqual(edges_touched(mask, tol=1), [])

    def test_both_views_warn_and_use_ankle_ground(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[:, 50:150] = 1
        points = np.full((33, 2), 100.0)
        points[[27, 28], 1] = [179, 184]
        for name in ("front", "side"):
            warnings = []
            with patch("bodymeasure.pipeline.detect_pose", return_value=(points, np.ones(33))), \
                 patch("bodymeasure.pipeline.segment_body", return_value=mask), \
                 patch("bodymeasure.pipeline.detect_chin", return_value=25):
                view = prepare_view(image, name, warnings)
            self.assertEqual(view.ground, 184)
            self.assertEqual(view.touched_edges, ("top", "bottom"))
            self.assertIn(f"{name.title()} silhouette is close to the image edge", warnings)
            self.assertIn(f"{name.title()} touched edges: top, bottom", warnings)
            self.assertIn("Feet may be cut off; height may be underestimated", warnings)
            self.assertIn("Head may be cut off; height and scale may be inaccurate", warnings)
            for before, after in (("high", "medium"), ("medium", "low"), ("low", "low")):
                self.assertEqual(boundary_confidence(before, (view, view)), after)
            view.touched_edges = ("left",)
            self.assertEqual(boundary_confidence("medium", (view,)), "medium")

    def test_landmarks_outside_image_are_clamped(self):
        from types import SimpleNamespace
        landmarks = [SimpleNamespace(x=-.2, y=1.2, visibility=.9) for _ in range(33)]
        with patch("bodymeasure.pipeline.mp.solutions.pose.Pose") as factory:
            factory.return_value.__enter__.return_value.process.return_value = SimpleNamespace(
                pose_landmarks=SimpleNamespace(landmark=landmarks))
            warnings = []
            points, _ = detect_pose(np.zeros((200, 150, 3), dtype=np.uint8), "front", warnings)
        np.testing.assert_array_equal(points[:, 0], 0)
        np.testing.assert_array_equal(points[:, 1], 199)
        self.assertTrue(warnings)

    def test_outside_slice_rows_and_columns_do_not_wrap_or_crash(self):
        mask = np.zeros((200, 100), dtype=np.uint8)
        mask[:, 20:80] = 1
        view = View("front", np.zeros((33, 2)), np.ones(33), mask, 0, 199, 20, 1)
        for y, x in ((-10, 50), (10, 50), (.5, -1000), (.5, 1000)):
            warnings = []
            self.assertIsNone(slice_width(view, y, x, "waist", warnings))
            self.assertTrue(warnings)
        self.assertEqual(slice_width(view, 0, 50, "waist", []), 60)
        self.assertEqual(slice_width(view, 1, 50, "waist", []), 60)

    def test_real_bottom_cropped_photo_over_http(self):
        root = Path(__file__).resolve().parents[1]
        image = cv2.imread(str(root / "uploads/front.jpg"))
        # Existing photo is 800 px tall; this cuts through the bottoms of shoes.
        front = image[:695, :].copy()
        self.assertIn("bottom", edges_touched(segment_body(front, "front")))
        encoded, png = cv2.imencode(".png", front)
        self.assertTrue(encoded)
        boundary = uuid4().hex
        chunks = [f'--{boundary}\r\nContent-Disposition: form-data; name="gender"\r\n\r\nmale\r\n'.encode()]
        for name, data, extension, mime in (
                ("front", png.tobytes(), "png", "image/png"),
                ("side", (root / "uploads/side.jpg").read_bytes(), "jpg", "image/jpeg")):
            chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}_image"; filename="{name}.{extension}"\r\nContent-Type: {mime}\r\n\r\n'.encode(), data, b"\r\n"]
        chunks.append(f"--{boundary}--\r\n".encode())
        app = create_app()
        server = make_server("127.0.0.1", 0, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(f"http://127.0.0.1:{server.server_port}/measure", data=b"".join(chunks),
                              headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with self.assertNoLogs(app.logger, level="ERROR"), urlopen(request, timeout=120) as response:
                self.assertEqual(response.status, 200)
                result = json.load(response)
            self.assertEqual(result["status"], "ok")
            self.assertGreater(result["height"]["cm"], 0)
            self.assertGreater(result["measurements"]["torso"]["chest"]["cm"], 0)
            self.assertIn("Front silhouette is close to the image edge", result["warnings"])
            self.assertIn("Front touched edges: bottom", result["warnings"])
            self.assertIn("Feet may be cut off; height may be underestimated", result["warnings"])
            self.assertEqual(result["confidence"], "low")
            print("\nCropped-front HTTP regression:", json.dumps({
                "http_status": 200, "status": result["status"], "height": result["height"],
                "confidence": result["confidence"], "measurement_groups": list(result["measurements"]),
                "boundary_warnings": [w for w in result["warnings"] if "edge" in w or "cut off" in w],
                "server_errors": 0}))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
