"""Regression comparison script for sample photos before and after fixes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from bodymeasure.pipeline import measure

BASELINES = {
    "male": {
        "height": 167.2,
        "chest": 99.2,
        "waist": 91.2,
        "hips": 114.0,
        "thigh": 64.4,
        "shoulder_width": 46.6,
        "inseam": 64.5,
        "neck": 70.9,
    },
    "female": {
        "height": 157.1,
        "chest": 93.2,
        "waist": 85.7,
        "hips": 107.1,
        "thigh": 60.6,
        "shoulder_width": 43.8,
        "inseam": 60.6,
        "neck": 66.7,
    }
}


def run_comparison():
    root = Path(__file__).resolve().parent.parent
    front_path = root / "uploads" / "front.jpg"
    side_path = root / "uploads" / "side.jpg"

    if not front_path.exists() or not side_path.exists():
        print(f"Sample images not found at {front_path} and {side_path}")
        return

    front_img = cv2.imread(str(front_path))
    side_img = cv2.imread(str(side_path))

    print("# Regression Comparison on Sample Images (uploads/front.jpg, uploads/side.jpg)\n")

    for gender in ("male", "female"):
        res = measure(front_img, side_img, gender)
        base = BASELINES[gender]
        m = res["measurements"]

        current = {
            "height": res["height"]["cm"],
            "chest": m["torso"]["chest"]["cm"],
            "waist": m["torso"]["waist"]["cm"],
            "hips": m["torso"]["hips"]["cm"],
            "thigh": m["legs"]["thigh"]["cm"],
            "shoulder_width": m["torso"]["shoulder_width"]["cm"],
            "inseam": m["legs"]["inseam"]["cm"],
            "neck": m["head_neck"]["neck"]["cm"],
        }

        print(f"### {gender.title()} Body Measurements (cm)")
        print(f"| Measurement | Before (Baseline) | After (Fixes 1-5) | Change | Source / Status |")
        print(f"|---|---|---|---|---|")

        keys = ["height", "chest", "waist", "hips", "thigh", "shoulder_width", "inseam", "neck"]
        for k in keys:
            before_val = base[k]
            after_val = current[k]
            diff = round(after_val - before_val, 1)
            diff_str = f"+{diff}" if diff > 0 else f"{diff}"
            if k == "height":
                src = f"{res['height']['source']} ({res['height'].get('uncertainty_band', '')})"
            elif k in ("chest", "waist", "hips", "shoulder_width"):
                src = m["torso"][k]["source"]
            elif k in ("thigh", "inseam"):
                src = m["legs"][k]["source"]
            else:
                src = m["head_neck"][k]["source"]

            note = ""
            if k == "neck" and before_val > 50:
                note = " [Guard fixed hair/shoulder contamination]"
            elif k == "inseam":
                note = " [Crotch-to-ankle segmented]"
            elif k == "thigh":
                note = " [Isolated from crotch]"

            print(f"| {k.replace('_', ' ').title()} | {before_val:.1f} cm | {after_val:.1f} cm | {diff_str} cm | {src}{note} |")

        print(f"\n**Scale cues used**: {res['height'].get('cues_used')}")
        print(f"**Confidence**: {res['confidence']}")
        print(f"**Key Warnings**: {len(res['warnings'])} warnings emitted\n")


if __name__ == "__main__":
    run_comparison()
