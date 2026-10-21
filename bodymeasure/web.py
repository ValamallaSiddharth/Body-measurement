"""Installed Flask application and local development server."""
import io
import os
from pathlib import Path
from threading import BoundedSemaphore
from flask import Flask, Request, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from . import pipeline
from .errors import CaptureError
from .validation import validate_upload
ROOT = Path(__file__).resolve().parent
PROCESSING_SLOT = BoundedSemaphore(1)

class MemoryRequest(Request):
    # Werkzeug otherwise spools multipart files above ~500 KB to a temp file.
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def create_app():
    app = Flask(__name__)
    app.request_class = MemoryRequest
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.config["MAX_FORM_MEMORY_SIZE"] = 128 * 1024
    app.config["MAX_FORM_PARTS"] = 8
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

    @app.get("/")
    def index():
        return send_file(ROOT / "templates" / "index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", service="two-photo-body-measurement",
                       boundary_policy="warn")

    @app.post("/measure")
    def measure():
        gender = request.form.get("gender", "").strip().lower()
        if gender not in ("male", "female"):
            raise CaptureError("Select Male or Female.", 400)
        height = pipeline.validate_height(request.form.get("height_cm"))
        if not PROCESSING_SLOT.acquire(blocking=False):
            raise CaptureError("Another measurement is processing. Please retry shortly.", 503)
        try:
            images = [validate_upload(request.files.get(f"{view}_image"), view)
                      for view in ("front", "side")]
            return jsonify(pipeline.measure(*images, gender, {"height_cm": height}))
        finally:
            PROCESSING_SLOT.release()

    @app.errorhandler(CaptureError)
    def capture_error(exc):
        return jsonify(status="error", error=str(exc)), exc.status_code

    @app.errorhandler(HTTPException)
    def http_error(exc):
        message = ("Request exceeds 25 MB; each image must be at most 12 MB."
                   if exc.code == 413 else exc.description)
        response = exc.get_response()
        response.data = app.json.dumps({"status": "error", "error": message})
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def internal_error(exc):
        app.logger.exception("Measurement request failed")
        return jsonify(status="error", error="Internal processing error. Please retry; see the server log for details."), 500

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app


app = create_app()

def main():
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")), debug=False)
