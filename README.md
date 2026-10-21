# Two-photo adult body measurement estimates

Python **3.11**, Flask, MediaPipe Solutions, OpenCV and NumPy. Supply front and
90-degree side photos of the same adult, gender, and optionally a known height.
The web app, CLI and evaluator use **one** entry point:
`bodymeasure.pipeline.measure(front_bgr, side_bgr, gender, options=None)`.
Images, multipart buffers, masks and inference results stay in memory. HTTP and
CLI inference do not save them. The offline evaluation command explicitly writes
CSV reports, never photos or masks.

## Install and run

```powershell
py -3.11 -m venv .venv-311
.\.venv-311\Scripts\Activate.ps1
python -m pip install ".[dev]"
bodymeasure-server
```

Open `http://127.0.0.1:5000`. `HOST` and `PORT` configure the development server.
`python app.py` remains a source-checkout launcher. `bodymeasure-server` works
from an installed wheel without this checkout. On Linux/macOS use
`python3.11 -m venv .venv-311` and `source .venv-311/bin/activate`.
The prepared Windows environment is `.venv-fresh`.

The page is packaged at `bodymeasure/templates/index.html`; CSS and JavaScript
are inline. No external frontend service, build process or static assets are
needed. Open the server URL rather than the HTML file. `window.API_BASE_URL` can
set a separate backend origin. CORS allows all origins without credentials.

```powershell
bodymeasure --front uploads/front.jpg --side uploads/side.jpg --gender male --height-cm 175
python -m bodymeasure --front uploads/front.jpg --side uploads/side.jpg --gender female
```

Omit `--height-cm` to use the head prior. JSON goes to stdout; CLI errors return
exit code 1 (argument syntax errors use argparse's exit code 2). MediaPipe may
emit diagnostic messages on stderr.

## Shared pipeline

1. `validation.validate_upload`: enforce encoded format and size limits, apply
   EXIF rotation, composite transparency onto white and resize to at most 1280 px.
2. `detection.detect_pose`: 33 landmarks; visible essential shoulders, hips, knees
   and ankles. The front requires both sides; the side requires one side per pair.
   Out-of-frame points are clamped with warnings. Missing essential joints reject
   the capture. The pure `clean_mask` helper processes Selfie Segmentation output
   using morphology and the largest connected component.
3. `prepare_view`: find a head-local silhouette vertex, visible ground and front
   Face Mesh chin (landmark 152). Check that segmentation covers essential joints.
   Boundary contact warns; bottom contact uses visible foot landmarks, falling
   back to ankles, and reduces quality. A body span below 100 px is rejected.
4. `estimate_scale`: when supplied, known height (120-220 cm, inclusive) sets
   each view's independent `cm_per_px = height_cm / body_pixel_height`. A valid
   front head anchor differing by more than 10% warns. Without known height,
   use front vertex-to-chin priors of 23.2 cm male / 21.8 cm female, accepting
   head/body ratios 0.11-0.16. Missing/implausible chin falls back to 175 cm male /
   162 cm female. **No scale comes from the side face.** Reference-object support
   is not implemented; the optional height anchor is the extension point for a
   future validated calibration adapter.
5. `capture_checks`: warn about orientation using shoulder/torso ratios, large
   differences in normalized torso proportions, and wide silhouettes suggesting
   bulky clothing or merged arms. Multiple large segmentation components warn
   about possible extra people or fragments. These are heuristics, not identity
   verification, reliable people counts or clothing classifiers.
6. Measurement functions: use outer shoulder spans, matching normalized torso
   slices, Ramanujan ellipse circumferences and joint-chain lengths. Contaminated
   torso slices use explicit proportion fallbacks; missing wrist geometry uses a
   height prior. Merged front thighs are split at the hip midpoint and labeled
   estimated. Inseam is a hip-to-ankle proxy, not a tailoring crotch measurement.
7. `sanity_checks`: broad gender-specific absolute and height-relative ranges for
   every output. An implausible value is retained, named in a warning and changed
   to `estimated`; overall confidence becomes low. These are engineering screening
   ranges, not validated population norms. A 70.9 cm neck is explicitly flagged.
8. `measurement_confidence`: only `medium` or `low`, never `high`. Assumed/prior
   scale, capture warnings, implausible values or estimated major girths lower
   confidence. This is a quality heuristic, not a probability or error interval.

Geometry, scale, mask processing and quality stages are independently testable.
MediaPipe adapters are isolated, and tests can inject synthetic landmarks/masks.
The optional Python-only `options={"debug": True}` returns overlays as in-memory
data URIs; the HTTP endpoint does not enable this option.

## Sources and accuracy

Every height/measurement value contains `{ "cm": 175.0, "in": 68.9, "source": "..." }`.
Values are rounded to one decimal; inches use unrounded cm / 2.54.
The UI displays source badges with keyboard-accessible tooltip descriptions:

| Source | Meaning |
| --- | --- |
| `measured` | User-supplied height, or image-derived geometry using that scale. Not a validated tape measurement. |
| `estimated` | Head-prior scale, proportion fallback, uncertain geometry or a plausibility warning. |
| `assumed` | Fixed baseline height because a reliable scale anchor was unavailable. |

Uncalibrated photos cannot uniquely determine absolute metric size. Known height
fixes the scale ambiguity but does not resolve perspective, clothing, missing
boundaries, segmentation errors or anatomical landmark bias. All estimates remain
approximate. Adult priors are unsuitable for children. Do not use this for medical
decisions or precise tailoring. A successful response means processing completed,
not that its dimensions are accurate.

The six-subject archived evaluation is reported in [reports/README.md](reports/README.md)
and `reports/uncalibrated/summary.csv`. It is a small convenience sample, not a
validated accuracy guarantee. Reports include signed bias, MAE, RMSE, +/-2 cm and
+/-5 cm rates, exclusions and failed captures. `_tbr` ground-truth entries are
excluded; ambiguous anatomical columns are not substituted. Known-height
evaluation excludes height itself from scoring to avoid input leakage.

## API

- `GET /`: packaged single-page interface.
- `GET /health`: process availability, not model readiness.
- `POST /measure`: multipart `front_image`, `side_image`, `gender` (`male` or
  `female`, case-insensitive), optional `height_cm` (finite number, 120-220).

```powershell
curl.exe -F "front_image=@uploads/front.jpg" -F "side_image=@uploads/side.jpg" -F "gender=male" -F "height_cm=175" http://127.0.0.1:5000/measure
```

Success retains `status`, `height`, `measurements`, `confidence`, `warnings` and
adds `scale_anchor` (`user_height`, `head_prior`, `adult_baseline`). Measurement
groups remain `head_neck` (neck), `torso` (shoulder_width, chest, waist, hips,
torso_length), `arms` (arm_length, wrist), `legs` (inseam, thigh). Source fields
are additive; the cm/in keys are unchanged.

Errors use `{ "status": "error", "error": "readable message" }`:
400 invalid input, 413 upload limits, 422 unusable capture, 503 busy, 500 internal
failure. Unknown routes/methods use JSON 404/405. Exception details stay in logs.

Limits: JPEG, PNG or still WebP; 12 MiB per image, 25 MiB per request,
20 megapixels, at least 128 px per side; maximum eight multipart parts and
128 KiB non-file form memory. Responses are `no-store`. Browser requests time out
after 120 seconds. Browser abort does not cancel native server inference.

## Capture guidance

Use one adult, the same person in both views, fitted clothing, even light and a
contrasting plain background. Include margins above the head and below feet.
Stand straight with separated feet and arms away from the torso; side-view arms
should be slightly forward. Keep the camera level near hip height at the same
distance/zoom. Avoid wide-angle lenses, hats, bulky clothes and thick shoes.

## Testing and evaluation

```powershell
python -m pip check
python -m ruff check app.py bodymeasure scripts tests gunicorn.conf.py
python -m pytest -q
node tests/test_frontend.cjs
python tests/verify_http.py
python scripts/evaluate.py --output-dir reports/uncalibrated
python scripts/evaluate.py --known-height --output-dir reports/known_height
python -m build
```

Synthetic pytest tests cover the current shared algorithm without photos.
Optional real-photo integration tests use `uploads/front.jpg` and `side.jpg`;
they skip when those local files are not distributed. `verify_http.py` explicitly
requires them. Tests include format limits, EXIF, in-memory multipart streams,
JSON errors, CLI routing, source badges, scale anchors, contamination, merged
thighs, confidence and sanity checks. The JS suite is a DOM harness, not full
interactive browser coverage. CI runs lint, tests, package and Docker builds.

## Production

```sh
pip install '.[server]'
gunicorn --config gunicorn.conf.py bodymeasure.web:app
docker build -t bodymeasure .
docker run --rm -p 5000:5000 --read-only --tmpfs /tmp:rw,noexec,nosuid,size=128m bodymeasure
```

Gunicorn is for Unix/Linux; the Windows command is a development server. The
default configuration uses **one worker, two gthread threads**, one inference
slot and a 150-second worker/graceful timeout. Model objects are created and
closed per request; no unsafe cross-thread model sharing. A second inference
request gets 503 while a health check can run on the other thread. The semaphore
is per-process: raising worker count permits one inference per worker and raises
memory/CPU demand. There is no distributed request queue or cancellation.

Gunicorn's gthread timeout is a worker-liveness timeout, **not a hard per-request
deadline**. Set a reverse proxy request-body limit of 25 MiB and an upstream read
timeout near 150 seconds. Client/proxy timeout still cannot interrupt native
inference. Container heartbeat/cache files use ephemeral storage, and bytecode
writing is disabled. Production photos and masks never reach disk. Restrict CORS
and add access control at the deployment boundary if exposing a private service.

Deployment choices follow [Gunicorn settings](https://docs.gunicorn.org/en/stable/settings.html)
and its [threading design](https://docs.gunicorn.org/en/stable/design.html).
HTML is included as [setuptools package data](https://setuptools.pypa.io/en/stable/userguide/datafiles.html).

## Dependencies and legacy assets

MediaPipe 0.10.21 is pinned for bundled Pose, Selfie Segmentation and Face Mesh
Solutions APIs on Python 3.11. No PyTorch, training, licensed body model or runtime
model download is needed. MediaPipe requires `opencv-contrib-python`; it coexists
with the requested `opencv-python-headless`, both pinned to 4.11.0.86 and sharing
`cv2`. Keep the environment isolated; do not independently upgrade/uninstall one.
The Linux image includes libraries needed by the contrib wheel, so this is not
a purely GUI-library-free dependency set. MediaPipe also brings JAX/audio/plotting
dependencies that the app does not directly use.

[LEGACY_ASSETS.md](LEGACY_ASSETS.md) lists reconciled unused configurations,
models, samples and old outputs, plus deleted/replaced tests. They are not wheel
or Docker contents. Dataset and binary assets were retained, not deleted.
