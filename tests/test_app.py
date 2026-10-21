import io
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage
from bodymeasure.web import MemoryRequest, create_app
from bodymeasure.validation import MAX_FILE_BYTES, validate_upload
from bodymeasure.pipeline import CaptureError


def photo(fmt="PNG", size=(256, 256)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=fmt)
    return buffer.getvalue()


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def payload(self, front=None, side=None):
        return {"gender": "male", "front_image": (io.BytesIO(front or photo()), "front.png"),
                "side_image": (io.BytesIO(side or photo()), "side.png")}

    def test_home_health_cors_and_json_http_errors(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"API_BASE_URL", home.data)
        self.assertNotIn(b"calibration.json", home.data)
        home.close()
        health = self.client.get("/health", headers={"Origin": "http://localhost:8080"})
        self.assertEqual(health.json["status"], "ok")
        self.assertEqual(health.headers["Access-Control-Allow-Origin"], "http://localhost:8080")
        for method, path, status in (("get", "/missing", 404), ("get", "/measure", 405)):
            response = getattr(self.client, method)(path)
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.json["status"], "error")
        preflight = self.client.options("/measure", headers={"Origin": "http://localhost:8080", "Access-Control-Request-Method": "POST"})
        self.assertEqual(preflight.status_code, 200)
        self.assertIn("POST", preflight.headers["Access-Control-Allow-Methods"])

    def test_bad_inputs(self):
        cases = [{}, {"gender": "unknown"}, {"gender": "male"}, self.payload(front=b"not an image")]
        for data in cases:
            with self.subTest(data=list(data)):
                response = self.client.post("/measure", data=data)
                self.assertEqual(response.status_code, 400)
                self.assertIsInstance(response.json["error"], str)
        data = self.payload()
        data["front_image"] = (io.BytesIO(photo()), "front.exe")
        self.assertEqual(self.client.post("/measure", data=data).status_code, 400)

    def test_real_no_person_returns_422(self):
        response = self.client.post("/measure", data=self.payload())
        self.assertEqual(response.status_code, 422)
        self.assertIn("No person", response.json["error"])

    def test_exception_returns_json_500_and_releases_slot(self):
        with patch("bodymeasure.pipeline.measure", side_effect=RuntimeError("test failure")), self.assertLogs(self.app.logger, level="ERROR"):
            response = self.client.post("/measure", data=self.payload())
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json["status"], "error")
        self.assertNotIn("test failure", response.json["error"])
        with patch("bodymeasure.pipeline.measure", return_value={"status": "ok"}):
            self.assertEqual(self.client.post("/measure", data=self.payload()).status_code, 200)

    def test_uploads_stay_in_memory(self):
        self.assertIsInstance(MemoryRequest({})._get_file_stream(2_000_000, "image/png"), io.BytesIO)
        rng = np.random.default_rng(1)
        buffer = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)).save(buffer, "PNG")
        self.assertGreater(len(buffer.getvalue()), 500_000)
        # Pass raw multipart bytes; Werkzeug's TEST CLIENT encoder otherwise
        # creates its own temporary request-body file before our app sees it.
        body = (b'--memory-test\r\nContent-Disposition: form-data; name="front_image"; filename="front.png"\r\n'
                b'Content-Type: image/png\r\n\r\n' + buffer.getvalue() + b'\r\n--memory-test--\r\n')
        with self.app.test_request_context("/measure", method="POST", data=body,
                                           content_type="multipart/form-data; boundary=memory-test"):
            from flask import request
            self.assertIsInstance(request.files["front_image"].stream, io.BytesIO)

    def test_limits_and_type_mismatch(self):
        too_big = FileStorage(stream=io.BytesIO(b"x"*(MAX_FILE_BYTES+1)), filename="large.jpg")
        with self.assertRaises(CaptureError) as caught:
            validate_upload(too_big, "front")
        self.assertEqual(caught.exception.status_code, 413)
        wrong = FileStorage(stream=io.BytesIO(photo()), filename="wrong.jpg")
        with self.assertRaises(CaptureError) as caught:
            validate_upload(wrong, "front")
        self.assertEqual(caught.exception.status_code, 400)
        response = self.client.post("/measure", data=b"x"*(25*1024*1024+1), content_type="multipart/form-data; boundary=test")
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.is_json)

    def test_all_formats_and_exif_orientation(self):
        for fmt, ext in (("JPEG", "jpg"), ("PNG", "png"), ("WEBP", "webp")):
            data = FileStorage(stream=io.BytesIO(photo(fmt)), filename=f"image.{ext}")
            self.assertEqual(validate_upload(data, "front").shape, (256, 256, 3))
        buffer = io.BytesIO()
        source = Image.new("RGB", (160, 240), "white")
        exif = source.getexif()
        exif[274] = 6
        source.save(buffer, "JPEG", exif=exif)
        image = validate_upload(FileStorage(stream=io.BytesIO(buffer.getvalue()), filename="rotated.jpg"), "front")
        self.assertEqual(image.shape[:2], (160, 240))


class RealPhotoIntegrationTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).resolve().parents[1]/"uploads/front.jpg").exists(), "Optional local sample photos not distributed")
    def test_existing_sample_pair_real_models(self):
        root = Path(__file__).resolve().parents[1]
        client = create_app().test_client()
        with (root/"uploads/front.jpg").open("rb") as front, (root/"uploads/side.jpg").open("rb") as side:
            response = client.post("/measure", data={"gender": "male", "front_image": (front, "front.jpg"), "side_image": (side, "side.jpg")})
        self.assertEqual(response.status_code, 200, response.json)
        result = response.json
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["height"]["cm"], 0)
        self.assertIn(result["confidence"], ("low", "medium"))
        self.assertEqual(set(result["measurements"]), {"head_neck", "torso", "arms", "legs"})
        for group in result["measurements"].values():
            for value in group.values():
                if value["cm"] is not None:
                    self.assertAlmostEqual(value["in"], value["cm"]/2.54, delta=0.08)
        self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
