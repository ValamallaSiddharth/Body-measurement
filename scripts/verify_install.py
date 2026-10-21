"""Verify an installed wheel outside its checkout, including real HTTP and CLI.

Run using the fresh venv Python: python scripts/verify_install.py --sample-dir uploads
Only temporary process logs are captured in memory. Never saves photo/results.
"""
import argparse
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    args = parser.parse_args()
    samples = args.sample_dir.resolve()
    scripts = Path(sys.executable).parent
    cli = scripts / ("bodymeasure.exe" if os.name == "nt" else "bodymeasure")
    server_command = scripts / ("bodymeasure-server.exe" if os.name == "nt" else "bodymeasure-server")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="bodymeasure-wheel-") as working:
        check = subprocess.run([sys.executable, "-I", "-c", "import bodymeasure; print(bodymeasure.__file__)"], cwd=working, env=env, capture_output=True, text=True, check=True)
        assert "site-packages" in check.stdout, check.stdout
        print("Installed package:", check.stdout.strip())
        command = [str(cli), "--front", str(samples/"front.jpg"), "--side", str(samples/"side.jpg"), "--gender", "male", "--height-cm", "175"]
        result = subprocess.run(command, cwd=working, env=env, capture_output=True, text=True, check=True, timeout=120)
        prediction = json.loads(result.stdout)
        assert prediction["height"]["cm"] == 175 and prediction["height"]["source"] == "measured"
        print("Installed CLI: real inference passed")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env["HOST"], env["PORT"] = "127.0.0.1", str(port)
        # Pipe server diagnostics into memory; avoid opening a console on Windows.
        server = subprocess.Popen([str(server_command)], cwd=working, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            origin = f"http://127.0.0.1:{port}"
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError(server.communicate()[0].decode(errors="replace"))
                try:
                    with urlopen(origin + "/health", timeout=1) as response:
                        assert json.load(response)["status"] == "ok"
                    break
                except (URLError, TimeoutError):
                    time.sleep(.2)
            else:
                raise TimeoutError("Installed server did not start")
            with urlopen(origin, timeout=10) as response:
                assert b'height_cm' in response.read()
            boundary = uuid4().hex
            body = io.BytesIO()
            for key, value in (("gender", "male"), ("height_cm", "175")):
                body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
            for view in ("front", "side"):
                body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{view}_image"; filename="{view}.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode())
                body.write((samples/f"{view}.jpg").read_bytes())
                body.write(b"\r\n")
            body.write(f'--{boundary}--\r\n'.encode())
            request = Request(origin+"/measure", data=body.getvalue(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urlopen(request, timeout=120) as response:
                assert json.load(response) == prediction
            print("Installed server: packaged page, health and real measurement match CLI")
        finally:
            server.terminate()
            try:
                server.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.communicate()


if __name__ == "__main__":
    main()
