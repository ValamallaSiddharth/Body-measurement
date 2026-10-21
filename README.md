# Two-Photo Adult Body Measurement & Height Estimation

Python **3.11**, Flask, MediaPipe Solutions, OpenCV-headless and NumPy.
Estimates adult body measurements and stature from two uncalibrated photos (full-body front + 90-degree side profile) and gender (`male` or `female`), without requiring the user to provide their height.

All processing strictly stays in memory. Uploaded images, intermediate buffers, segmentation masks, and results are never written to disk during requests.

---

## Quickstart

### Installation & Server Execution
```powershell
# Create and activate Python 3.11 environment
py -3.11 -m venv .venv-311
.\.venv-311\Scripts\Activate.ps1

# Install package dependencies
python -m pip install ".[dev]"

# Run Flask server on localhost:5000
python -m flask --app bodymeasure.app run --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in your web browser.

### CLI Execution
```powershell
# Measure without supplying height (multi-cue fused scale)
python -m bodymeasure --front uploads/front.jpg --side uploads/side.jpg --gender male

# Optional known height override
python -m bodymeasure --front uploads/front.jpg --side uploads/side.jpg --gender female --height-cm 165
```

---

## Pipeline Architecture & Accuracy Fixes

The pipeline executes through sequential in-memory stages:

1. **Validation & Normalization (`validation.py`)**:
   - Enforces image formats (JPEG, PNG, WebP), size limits, EXIF orientation correction, and resizes to a maximum dimension of 1280 px.
2. **Pose Detection & Dual-Pass Smoothing (`detection.py`)**:
   - Runs MediaPipe Pose (`model_complexity=2`, `static_image_mode=True`) on both original and horizontally flipped images.
   - Averages bilateral landmarks weighted by visibility to stabilize keypoint coordinates.
3. **Silhouette Segmentation (`detection.py`)**:
   - Uses MediaPipe Selfie Segmentation with morphological closing and opening.
   - Extracts the primary human connected component.
4. **Arm vs. Torso Isolation (Fix 1, `pipeline.py`)**:
   - Constructs a torso-only mask by subtracting shoulder-elbow-wrist corridor lines and wrist hand discs.
   - Clips torso horizontal runs at chest, waist, and hips to a dynamic corridor between shoulder and hip lines.
   - Flags arm contamination and applies biacromial/deltoid proportion fallbacks with capture tips if arms are merged.
5. **Leg Separation & Inseam (Fix 2, `pipeline.py`)**:
   - Scans rows downward from the pelvis to detect the crotch point where the central silhouette splits into two legs.
   - Measures inseam directly from crotch to ankle.
   - Samples thigh circumference between crotch and knee; if legs touch continuously, splits along the pelvis midline, reports `"estimated"`, and warns `"legs merged"`.
6. **Side-View Profile Joint Recovery (Fix 3, `detection.py`)**:
   - Requires at least one visible joint ($\ge 0.5$) for side-view hip, knee, and ankle pairs.
   - Automatically recovers occluded joints using the opposite side landmark or leg proportion projections along the hip-ankle line.
7. **Multi-Cue Scale Fusion Without User Height (Fix 4, `pipeline.py`)**:
   - Replaces single head-prior scaling with an inverse-variance fused multi-cue model:
     - *Head Cue:* 3D Face Mesh chin to skull curve crown (clips hair volume) with head pitch compensation.
     - *Neck-Shoulder Cue:* Distance between ear and shoulder span.
     - *Torso Proportion:* Shoulder-to-hip skeletal distance.
     - *Leg Proportion:* Hip-to-ankle skeletal distance.
     - *Camera Tilt Correction:* Cosine adjustment from vertical body axis.
     - *Population Prior:* Gaussian regularizer (male: $175 \pm 7\text{ cm}$, female: $162 \pm 6.5\text{ cm}$).
   - Drops statistical outliers ($> 2.5\times\text{MAD}$); shrinks lone cues toward the population prior so extreme implied heights (e.g. $191\text{ cm}$) are never output directly.
   - Reports `uncertainty_band`, `uncertainty_cm`, and `cues_used`.
8. **Plausibility Guards & Re-estimation (Fix 5, `pipeline.py`)**:
   - Enforces physical screening bounds (e.g. neck $\le 50\text{ cm}$, waist $\le 110\%$ of hips, shoulder $\ge 35\text{ cm}$).
   - Automatically catches anomalies and re-estimates values using anatomical fallbacks, downgrading source labels to `"estimated"`.
9. **Circumference Calculation (`pipeline.py`)**:
   - Uses Ramanujan's high-precision first ellipse approximation combining front widths and side depths.
   - Formats all values in both centimeters and inches rounded to 1 decimal place.

---

## API Reference

### `GET /`
Serves the responsive single-page user interface packaged in `bodymeasure/templates/index.html`.

### `GET /health`
Returns JSON `{"status": "ok"}` indicating service availability.

### `POST /measure`
Multipart form-data endpoint for body measurement estimation.

**Parameters:**
- `front_image` *(file, required)*: Full-body front-facing photograph.
- `side_image` *(file, required)*: Full-body 90-degree profile photograph.
- `gender` *(string, required)*: `"male"` or `"female"`.
- `height_cm` *(float, optional)*: Known stature in cm ($120\text{--}220\text{ cm}$). If omitted, multi-cue scale fusion is used.

**Sample Request:**
```powershell
curl.exe -F "front_image=@uploads/front.jpg" `
         -F "side_image=@uploads/side.jpg" `
         -F "gender=male" `
         http://127.0.0.1:5000/measure
```

**Sample Response:**
```json
{
  "status": "ok",
  "gender": "male",
  "confidence": "low",
  "scale_anchor": "head_prior",
  "height": {
    "cm": 167.0,
    "in": 65.7,
    "source": "estimated",
    "uncertainty_cm": 8.0,
    "uncertainty_band": "167.0 cm +/- 8.0",
    "cues_used": [
      "head_prior",
      "neck_shoulder_cue",
      "torso_proportion",
      "population_regularizer"
    ]
  },
  "measurements": {
    "head_neck": {
      "neck": { "cm": 40.9, "in": 16.1, "source": "estimated" }
    },
    "torso": {
      "shoulder_width": { "cm": 46.8, "in": 18.4, "source": "estimated" },
      "chest": { "cm": 99.7, "in": 39.3, "source": "estimated" },
      "waist": { "cm": 91.6, "in": 36.1, "source": "estimated" },
      "hips": { "cm": 114.6, "in": 45.1, "source": "estimated" },
      "torso_length": { "cm": 49.3, "in": 19.4, "source": "estimated" }
    },
    "arms": {
      "arm_length": { "cm": 48.4, "in": 19.1, "source": "estimated" },
      "wrist": { "cm": 16.7, "in": 6.6, "source": "estimated" }
    },
    "legs": {
      "inseam": { "cm": 52.5, "in": 20.7, "source": "estimated" },
      "thigh": { "cm": 50.5, "in": 19.9, "source": "estimated" }
    }
  },
  "warnings": [
    "Height estimated via multi-cue fusion with population regularization; absolute size is uncalibrated.",
    "Tip: Keep arms slightly away from the body in the front and side photos for accurate torso measurements.",
    "Inseam measured from segmented crotch level to ankle."
  ]
}
```

---

## Capture Tips for Maximum Accuracy

1. **Posture:**
   - Stand straight looking forward into the camera lens at eye level.
   - **Arms:** Hold arms slightly away from the torso (A-pose, 5–10 cm gap) to ensure clean torso contours.
   - **Legs:** Stand with feet shoulder-width apart so the crotch and thighs separate cleanly.
2. **Clothing:**
   - Wear snug, form-fitting attire (compression wear, leggings, swimwear). Loose or baggy clothes inflate circumferences.
3. **Camera Placement:**
   - Position the camera level at **navel / mid-torso height**, approximately **2.5 to 3 meters** away.
   - Avoid high or low angles pointing up or down to prevent perspective foreshortening.
4. **Environment:**
   - Ensure the entire body from top of head to bottom of feet is within the frame.
   - Use even lighting against a plain, contrasting background without heavy floor shadows.

---

## Testing & Verification

```powershell
# Run the complete test suite (synthetic & endpoint tests)
pytest --basetemp=./.pytest_temp

# Run sample image regression comparison
python scripts/regression.py

# Run frontend UI verification
node tests/test_frontend.cjs

# Run HTTP endpoint integration test
python tests/verify_http.py
```

---

## Known Limitations

- **Monocular 2D Scale:** Without camera calibration markers or depth sensors (LiDAR), uncalibrated photos rely on statistical anatomical ratios. Entering an exact height (`height_cm`) eliminates scale uncertainty.
- **Clothing Silhouettes:** The algorithm segments visible outer contours. Loose fabric, hoodies, or thick footwear cannot be penetrated and will inflate measurements.
- **Extreme Poses / Joint Occlusion:** Heavy body rotation or crossed limbs degrade landmark visibility and trigger heuristic fallbacks.
