"""Evaluate the six-subject archive. Only this offline command writes CSV reports."""
import argparse
import csv
import math
from pathlib import Path
import sys

# Allow `python scripts/evaluate.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from werkzeug.datastructures import FileStorage
from bodymeasure.pipeline import measure
from bodymeasure.errors import CaptureError
from bodymeasure.validation import validate_upload


FIELDS = {
    "height": "height", "arm_length": "arm_length_cm",
    "chest": "chest_circumference_cm", "waist": "waist_circumference_cm",
    "hips": "hips_circumference_cm", "thigh": "thigh_circumference_cm",
}


def numeric_truth(text):
    """Never strip review suffixes such as _tbr or invent missing values."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def error_summary(errors):
    n = len(errors)
    if not n:
        return {"n": 0, "mae_cm": "", "rmse_cm": "", "bias_cm": "", "within_2cm_pct": "", "within_5cm_pct": ""}
    return {"n": n, "mae_cm": sum(abs(e) for e in errors)/n,
            "rmse_cm": math.sqrt(sum(e*e for e in errors)/n), "bias_cm": sum(errors)/n,
            "within_2cm_pct": 100*sum(abs(e) <= 2 for e in errors)/n,
            "within_5cm_pct": 100*sum(abs(e) <= 5 for e in errors)/n}


def evaluate(archive, known_height=False):
    archive = Path(archive)
    with (archive / "Body Measurements Image Dataset.csv").open(newline="", encoding="utf-8-sig") as stream:
        subjects = list(csv.DictReader(stream))
    rows, failures = [], []
    for subject in subjects:
        ident = subject["set_id"]
        folder = (archive / ident).resolve()
        if folder.parent != archive.resolve():
            raise ValueError("Dataset set_id must name a direct archive subdirectory")
        try:
            images = []
            for view in ("front", "side"):
                filename = folder / f"{view}_img.jpg"
                with filename.open("rb") as stream:
                    images.append(validate_upload(FileStorage(stream=stream, filename=filename.name), view))
            height = numeric_truth(subject["height"]) if known_height else None
            if known_height and height is None:
                raise ValueError("Missing clean known-height ground truth")
            result = measure(*images, subject["gender"].strip().lower(), {"height_cm": height})
        except (CaptureError, OSError, ValueError) as exc:
            failures.append({"set_id": ident, "error": str(exc)})
            continue
        values = {name: value for group in result["measurements"].values() for name, value in group.items()}
        values["height"] = result["height"]
        for name, column in FIELDS.items():
            # Supplied ground-truth height is an input, not an accuracy prediction.
            if known_height and name == "height":
                continue
            truth, predicted = numeric_truth(subject[column]), values[name]["cm"]
            usable = truth is not None and predicted is not None
            rows.append({"set_id": ident, "measurement": name, "truth_cm": truth if truth is not None else "",
                         "raw_truth": subject[column], "predicted_cm": predicted,
                         "error_cm": predicted-truth if usable else "", "source": values[name]["source"],
                         "confidence": result["confidence"], "included": usable,
                         "reason": "" if usable else "missing, flagged or unavailable value",
                         "warnings": " | ".join(result["warnings"])})
    summary = []
    for name in FIELDS:
        if known_height and name == "height":
            continue
        errors = [row["error_cm"] for row in rows if row["measurement"] == name and row["included"]]
        summary.append({"measurement": name, **error_summary(errors), "subjects_total": len(subjects),
                        "captures_failed": len(failures), "mode": "known_height" if known_height else "uncalibrated"})
    return rows, summary, failures


def write_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("archive"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/uncalibrated"))
    parser.add_argument("--known-height", action="store_true")
    args = parser.parse_args()
    rows, summary, failures = evaluate(args.archive, args.known_height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", rows, ["set_id", "measurement", "truth_cm", "raw_truth", "predicted_cm", "error_cm", "source", "confidence", "included", "reason", "warnings"])
    write_csv(args.output_dir / "summary.csv", summary, list(summary[0]))
    write_csv(args.output_dir / "failures.csv", failures, ["set_id", "error"])
    print("measurement       n    MAE     RMSE     bias    +/-2cm%  +/-5cm%")
    for row in summary:
        values = [row[k] for k in ("mae_cm", "rmse_cm", "bias_cm", "within_2cm_pct", "within_5cm_pct")]
        print(f"{row['measurement']:16} {row['n']:2} " + " ".join(f"{v:8.2f}" if v != "" else "     n/a" for v in values))
    print(f"Capture failures: {len(failures)}/{summary[0]['subjects_total']}; see failures.csv. No photos or masks saved.")


if __name__ == "__main__":
    main()
