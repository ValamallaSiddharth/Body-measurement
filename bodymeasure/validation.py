"""Shared in-memory image validation for HTTP, CLI and evaluation."""
import io
import warnings
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .errors import CaptureError
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 20_000_000

def validate_upload(upload, view):
    """Check encoded type, byte/pixel limits and decoding, entirely in RAM."""
    if upload is None or not upload.filename:
        raise CaptureError(f"Upload the {view} image.", 400)
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in formats:
        raise CaptureError(f"{view.title()} image must be JPG, PNG or WebP.", 400)
    data = upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise CaptureError(f"{view.title()} image exceeds the 12 MB limit.", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format != formats[suffix]:
                    raise CaptureError(f"{view.title()} image contents do not match its extension.", 400)
                if source.width * source.height > MAX_PIXELS:
                    raise CaptureError(f"{view.title()} image exceeds the 20 megapixel limit.", 413)
                if min(source.size) < 128:
                    raise CaptureError(f"{view.title()} image must be at least 128 pixels on each side.", 400)
                if getattr(source, "is_animated", False):
                    raise CaptureError(f"{view.title()} image must be a still photo.", 400)
                source.load()
                oriented = ImageOps.exif_transpose(source).convert("RGBA")
                background = Image.new("RGBA", oriented.size, "white")
                background.alpha_composite(oriented)
                background.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                rgb = np.asarray(background.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise CaptureError(f"{view.title()} image dimensions are too large.", 413) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CaptureError(f"{view.title()} image cannot be decoded. Upload a valid JPG, PNG or WebP.", 400) from exc


