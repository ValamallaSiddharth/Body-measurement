"""Real localhost HTTP smoke test, including real model inference.

Run from the repository root: python tests/verify_http.py
Starts/stops a server on an available port; never saves images or responses.
"""
import json
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from werkzeug.serving import make_server
from app import create_app


def multipart(gender, images=None):
    boundary = uuid4().hex
    chunks = [f'--{boundary}\r\nContent-Disposition: form-data; name="gender"\r\n\r\n{gender}\r\n'.encode()]
    for view in ("front", "side"):
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{view}_image"; filename="{view}.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode())
        chunks.append(images[view] if images and view in images else
                      (ROOT / "uploads" / f"{view}.jpg").read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def main():
    server = make_server("127.0.0.1", 0, create_app())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(origin, timeout=10) as response:
            assert response.status == 200 and b"measure-form" in response.read()
        with urlopen(origin + "/health", timeout=10) as response:
            assert json.load(response)["status"] == "ok"
        for gender in ("male", "female"):
            body, content_type = multipart(gender)
            request = Request(origin + "/measure", data=body, headers={"Content-Type": content_type, "Origin": "http://localhost:8080"})
            try:
                with urlopen(request, timeout=120) as response:
                    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:8080"
                    result = json.load(response)
            except HTTPError as exc:
                raise AssertionError(f"HTTP {exc.code}: {exc.read().decode()}") from exc
            assert result["status"] == "ok"
            assert result["height"]["cm"] > 0
            print(json.dumps({"gender": gender, **result}, indent=2))
        print("HTTP smoke test passed: homepage, health, CORS and both gender sample measurements.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
