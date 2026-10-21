"""Optional CLI uses the same validation and pipeline as the web endpoint."""
import argparse
import json
from pathlib import Path
from werkzeug.datastructures import FileStorage
from .pipeline import CaptureError, run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate adult measurements from two photos")
    parser.add_argument("--front", required=True)
    parser.add_argument("--side", required=True)
    parser.add_argument("--gender", required=True, choices=("male", "female"))
    args = parser.parse_args(argv)
    # Deferred import keeps the package pipeline independent of Flask startup.
    from app import validate_upload
    try:
        images = []
        for view, filename in (("front", args.front), ("side", args.side)):
            with open(filename, "rb") as stream:
                images.append(validate_upload(FileStorage(stream=stream, filename=Path(filename).name), view))
        print(json.dumps(run_pipeline(*images, args.gender), indent=2))
        return 0
    except (CaptureError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1
