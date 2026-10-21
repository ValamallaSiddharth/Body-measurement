# Two-photo body measurement estimates

Flask app with a single HTML/vanilla-JavaScript page. Upload a **front photo**, a
**side photo**, and select **Male/Female**. There is no height field, calibration
file, calibration board, licensed body model, PyTorch, training step or model
download step. Images and multipart upload buffers stay in memory. The app does
not write photos, masks, landmarks or results to disk.

## Run

Use **64-bit Python 3.11** (verification environment: Python **3.11.9**, Windows
x64). Do not reuse the old Python 3.14 environment. MediaPipe 0.10.21 is pinned
because this implementation uses its bundled Pose, Selfie Segmentation and Face
Mesh Solutions APIs. Python 3.13/3.14 are not supported by this dependency set.

```powershell
py -3.11 -m venv .venv-311
.\.venv-311\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**. Do not open `index.html` directly with `file://`.
On macOS/Linux with a compatible Python 3.11 wheel, use `python3.11 -m venv .venv-311`
and `source .venv-311/bin/activate` instead. Only Windows x64 was exercised here.
Set `HOST` and `PORT` environment variables to change the local server address.

In this prepared workspace, the fresh verification environment is already at
`.venv-fresh`, backed by the project-local Python 3.11 runtime in `.tools`:

```powershell
.\.venv-fresh\Scripts\python.exe app.py
```

`API_BASE_URL` in `index.html` defaults to the same origin. To host the page
separately, set it to your backend origin (for example
`http://127.0.0.1:5000`). CORS is enabled without credentials. The browser timeout
is 120 seconds; aborting the browser request does not cancel inference already
running on the server. One request runs inference at a time; a concurrent request
receives JSON 503 and can retry.

The application uses OpenCV without GUI calls. The requested
`opencv-python-headless` is pinned, and MediaPipe additionally requires the
`opencv-contrib-python` distribution. Both are pinned to the same OpenCV release
to keep this upstream dependency resolvable. They share the `cv2` namespace:
**keep this environment isolated and do not uninstall or upgrade either OpenCV
package separately**. This is not a guaranteed GUI-library-free Linux install.
MediaPipe also brings JAX and plotting/audio dependencies even though this app
does not use them. No paid or gated model is required.

## Photo guidelines

- Photograph one adult, the **same person** in both photos, in the same fitted
  clothing. This app does not verify identity or count people reliably.
- Include the full body with a visible margin above the head and below the feet.
  Use good, even lighting and a plain, contrasting background.
- Stand straight with feet slightly separated, legs straight, head level, and
  arms slightly away from the body. Face the camera directly in the front photo.
- Turn exactly 90 degrees for the side view. Hold arms slightly forward or away
  from the torso so they do not merge with it. Avoid bending elbows sharply,
  holding objects, or raising hands above the head.
- Place the camera about hip height, keep it level, and use the same distance and
  zoom for both shots. Avoid wide-angle lenses. Separate per-view scales handle
  different distances approximately, but cannot remove perspective distortion.
- Avoid bulky clothing, hats, tall hairstyles and shoes with thick soles.
- JPEG, PNG and still WebP are accepted: at most **12 MB per image**, **25 MB per
  request**, **20 megapixels**, minimum **128 pixels on each side**. EXIF rotation
  is applied; transparent areas are composited onto white. Inference images are
  reduced to a maximum dimension of 1280 pixels.

## Pipeline and accuracy limits

Each stage is a separate function in `bodymeasure/pipeline.py`:

1. `detect_pose`: MediaPipe Pose's 33 image landmarks. Front shoulders, hips,
   knees and ankles require visibility >= 0.5 on both sides. A profile requires
   at least one visible side in each pair, because far-side joints are normally
   occluded. Visibility below 0.75 produces a warning. Missing persons or
   insufficient essential landmarks produce a clear 422 error.
2. `segment_body`: MediaPipe Selfie Segmentation, thresholding, morphological
   closing/opening and largest connected component. Silhouettes within 2 pixels
   of an edge produce soft warnings. Head/foot boundary contact reduces confidence
   one level (with a floor of low); bottom contact uses clamped visible ankle
   landmarks for ground level. Out-of-image indices are clamped, and unavailable
   slices return null. An unusable mask that misses interior essential joints
   is still rejected with 422.
3. `detect_chin`: Face Mesh landmark **152**, detected in a pose-guided head crop.
   The top of the head comes from the silhouette, never the nose landmark.
4. `estimate_scale`: vertex-to-chin length is assumed to be **23.2 cm male** or
   **21.8 cm female**, as specified for this implementation. These are fixed
   adult priors, not individually measured dimensions or universal standards.
   Each view gets its own cm-per-pixel scale. A missing or implausible chin falls
   back to **7.5 head lengths per adult**, with a warning. If both faces fail,
   estimated height is therefore exactly **174.0 cm male / 163.5 cm female**,
   rather than information recovered from the person's actual stature.
5. `normalize_views`: fuse independent stature estimates (weight 1 for a face
   anchor, 0.25 for a fallback). Convert each view to top=0, ground=1, and apply
   the same fused stature. Slices match at the same normalized height. No
   unnecessary resampling is needed. Height-anchor disagreement over 10% and
   unusual front shoulder/height proportions produce warnings.
6. `extract_measurements`: chest/waist/hip/neck levels come from joint-aligned
   interpolation; front widths and side depths use the matching normalized
   slice. Take the median of nearby rows and select the local foreground run
   around the relevant body part, excluding separated arms or the other leg.
   Ellipse circumference is `π[3(a+b) − √((3a+b)(a+3b))]`, where `a=width/2` and
   `b=depth/2`. Arm length sums shoulder–elbow and elbow–wrist distances;
   shoulder width and torso length are Euclidean joint distances in the front
   view. **Inseam is a hip-midpoint–ankle-midpoint proxy**, since Pose has no
   crotch landmark; it is not a tailoring inseam. `torso_length` is included in
   addition to the example response fields.
7. `sanity_checks`: implausible height, waist > hips, excessive depth and broad
   measurement-range violations produce warnings; measurements are not clamped
   to an expected body shape.

**Two uncalibrated images do not determine absolute metric size uniquely.** A
taller person farther away can have the same image dimensions as a shorter one
closer to the camera. The head-size prior supplies the missing metric anchor,
so an incorrect head-size assumption scales every measurement incorrectly.

The code documents an **unvalidated engineering error budget of roughly 10–20%
or worse for lengths and 15–30% or worse for girths**. These are not measured
accuracy figures, guarantees, or statistical confidence intervals. No
ground-truth measurement dataset has been used to validate this app. Errors can
be much larger for children, unusual proportions, head tilt, hair, occlusion,
clothing, perspective or poor segmentation. Selfie Segmentation is particularly
limited on small full-body subjects. Adult priors should not be used for children.

Hidden limb boundaries cannot be inferred reliably from a merged silhouette.
Unreliable wrist/thigh or missing cross-sections return `{ "cm": null, "in": null }`
and a warning; the UI displays **Unavailable**. Horizontal limb slices can also
overestimate circumference when limbs are angled. A successful response means
the pipeline ran, not that every measurement is accurate.

`confidence` is a quality heuristic: **medium** only when both head anchors work
and no capture/anomaly warnings occur; otherwise **low**. It never reports
**high**, because no-calibration absolute sizes have not been validated. Use a
tape measure for tailoring, health decisions, or any precision-dependent use.

## API

- `GET /`: same-origin HTML interface.
- `GET /health`: `{ "status": "ok", "service": "two-photo-body-measurement" }`.
  Reports process availability; models are initialized on measurement requests.
- `POST /measure`: multipart fields `front_image`, `side_image`, `gender`
  (`male` / `female`, case insensitive).

```powershell
curl.exe -F "front_image=@uploads/front.jpg" -F "side_image=@uploads/side.jpg" -F "gender=male" http://127.0.0.1:5000/measure
```

Success has `status`, `height`, `measurements` grouped into `head_neck`, `torso`,
`arms`, `legs`, `confidence`, and `warnings`. Each measurement has `cm` and `in`,
rounded to one decimal; inches are calculated from unrounded centimetres / 2.54.
Errors always use `{ "status": "error", "error": "readable message" }`.
Status codes: **400** invalid input, **413** upload limit, **422** undetectable or
unusable person, **500** unexpected processing failure, **503** busy. Unknown
routes and invalid methods also return JSON (404/405). Unexpected exception
details are logged on the server and not exposed to the browser.

Optional CLI, using the same in-memory validation and measurement pipeline:

```powershell
python -m bodymeasure --front uploads/front.jpg --side uploads/side.jpg --gender male
```

## Verification

```powershell
python -m pip check
python -m unittest discover -s tests -v
node tests/test_frontend.cjs
python tests/verify_http.py
```

The suite includes real MediaPipe inference on the existing `uploads/front.jpg`
and `uploads/side.jpg` samples, plus blank-image rejection, input validation,
JSON errors, CORS, memory-only multipart buffers, unit conversion, ellipse
geometry, separate camera-distance scales and foreground-run selection.
The sample pair wears different/bulky clothes and has no ground-truth dimensions;
it tests execution and warning handling, **not accuracy**. The old calibrated
model-fitting implementation and its obsolete tests have been removed. Existing
sample images, old outputs and unused local model/config assets are left intact
and are never loaded by the app.

Verified in the fresh Python 3.11.9 environment: `pip check` reports no broken
requirements; **19 Python tests** and **10 JavaScript scenarios** pass. The real
localhost HTTP test returns 200 for the page, health check, and measurements with
both gender selections. The sample response correctly reports a missing side
chin fallback, an unavailable wrist due to mismatched arm positions, and an
estimated waist larger than hips. JavaScript checks use a DOM test harness;
an interactive browser was not available for visual testing in this session.
The boundary regression crops the front sample through the shoe bottoms and
verifies real HTTP 200, measurements, frame/feet warnings, low confidence and
no application error logs. Boundary detection, clamped indices and both-view
ankle-ground handling also have regression coverage.

## Open-source components

- [MediaPipe 0.10.21](https://pypi.org/project/mediapipe/0.10.21/): Apache-2.0,
  bundled freely available inference models.
- [MediaPipe Selfie Segmentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md)
  and [Face Mesh](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md).
- [OpenCV](https://opencv.org/license/), [Flask](https://flask.palletsprojects.com/),
  [NumPy](https://numpy.org/) and [Pillow](https://python-pillow.github.io/).
