# Implementation and verification

## Changes by requested priority

| Item | Delivered |
| --- | --- |
| 1. Shared pipeline | `bodymeasure.pipeline.measure` is the single entry point used by HTTP, CLI and evaluation. Root `app.py` is a launcher. Detection and image validation are isolated package modules; duplicated legacy geometry was removed. |
| 2. Sources | Every metric has `source`; UI badges describe measured, estimated and assumed values. Heights supplied by the user are measured; uncalibrated image-derived values are estimated. |
| 3. Plausibility | Gender-specific absolute and height-relative screening ranges cover all ten measurements. Suspicious numbers remain visible but receive named warnings, estimated labels and low confidence. Confidence never exceeds medium. |
| 4. Scale | Optional `height_cm`, `--height-cm` and UI field. Separate per-view scales use the same known stature. Head-anchor disagreement over 10% warns. Head priors and explicit baseline fallback remain without a known height. |
| 5. Tests | Current synthetic-mask/landmark pytest tests, endpoint and CLI tests, source-badge frontend checks, memory/format checks and optional real-photo regressions. Older geometry and frontend tests replaced. |
| 6. Evaluation | `scripts/evaluate.py` generates predictions, summary and failure CSVs. Both uncalibrated and known-height sample reports are committed under `reports/`. No ground truth is fabricated; uncertain `_tbr` values are excluded. |
| 7. Capture checks | Non-blocking component-count, orientation, normalized-proportion and wide-clothing warnings; retained essential-joint and segmentation rejection checks. Identity is not verified. |
| 8. Installation | Package contains server, CLI, validation, pipeline and HTML. Python 3.11 wheel/sdist build successfully. A fresh isolated environment ran both installed entry points outside the checkout. |
| 9. Documentation | README rewritten; legacy files reconciled in `LEGACY_ASSETS.md`; evaluation assumptions and limitations documented. |
| 10. Production | Single-worker/two-thread Gunicorn configuration, Python 3.11 Dockerfile, request limits/timeouts and GitHub Actions lint/test/build workflow. No publication or deployment performed. |

## Verified locally

- Python 3.11 on Windows, using `.venv-fresh`.
- **67 pytest tests passed, plus four unittest subtests**; optional real sample
  inference and cropped-boundary HTTP regression included.
- **11 frontend scenarios passed**, including source badges and height validation.
- Ruff and `pip check` passed.
- Real HTTP page, health, CORS and both gender selections passed.
- Wheel and source distribution built with `python -m build --no-isolation`.
- Installed the wheel and all dependencies in fresh `.venv-package` (no inherited
  site-packages). `scripts/verify_install.py` used a temporary working directory
  outside the checkout, confirmed imports from installed site-packages, ran real
  CLI inference, launched the installed server, loaded its packaged HTML and
  asserted that a real HTTP measurement exactly matched CLI output. Fresh
  environment `pip check` passed.
- Both evaluation modes processed all six archived pairs; no capture failures.

Reproduce ordinary checks after `python -m pip install ".[dev]"`:

```powershell
python -m pytest -q
node tests/test_frontend.cjs
python -m ruff check app.py bodymeasure scripts tests gunicorn.conf.py
python -m pip check
python tests/verify_http.py
python -m build
```

Standalone installation check (source checkout with optional sample photos):

```powershell
py -3.11 -m venv .venv-package
.\.venv-package\Scripts\python.exe -m pip install dist\two_photo_bodymeasure-0.3.0-py3-none-any.whl
.\.venv-package\Scripts\python.exe scripts/verify_install.py --sample-dir uploads
```

## Sample accuracy

| Measurement | Uncalibrated MAE (cm) | Known-height MAE (cm) |
| --- | ---: | ---: |
| Height | 11.78 | Excluded: supplied input |
| Arm length | 13.85 | 14.73 |
| Chest | 6.66 | 4.72 |
| Waist | 8.22 | 4.60 |
| Hips | 17.68 | 15.73 |
| Thigh | 16.12 | 14.80 |

Chest has five comparisons; other scored measurements have six. RMSE, signed
bias and percentages within 2/5 cm are in each mode's `summary.csv`. The results
show substantial residual error, including after supplying height. They are not
a general accuracy claim or a statistically representative validation study.

## Remaining limitations and unverified work

- Reference-object detection was deliberately deferred as allowed; the optional
  metric scale anchor is documented as its extension point.
- Docker is not installed on this machine, and the available WSL Python is 3.14,
  not the required 3.11. Docker runtime and Gunicorn execution were **not** verified
  locally. Configuration syntax was checked; Linux CI includes Gunicorn config
  validation and a Docker build. The workflow has not been executed on GitHub.
- Frontend checks use a DOM harness, not an interactive browser visual review.
- Capture heuristics cannot reliably count people, verify identity or classify
  clothing. Matching proportions do not prove matching subjects.
- Anatomical proxy definitions, body shape, hair, posture, perspective and merged
  silhouettes remain major errors. The six-subject dataset is too small and its
  tape-measure definitions too uncertain to establish broad validity.
- The lock is process-local. Default one-worker deployment is intentional;
  changing worker count changes concurrency and resource use. No distributed
  queue or hard native-inference cancellation was added. Browser/proxy timeouts
  do not stop work already running.
- MediaPipe still requires the contrib OpenCV distribution alongside headless
  OpenCV. Both are pinned consistently; do not upgrade one independently.

## Deletions and moves

- Deleted `tests/frontend.test.cjs` (obsolete UI contract).
- Deleted `tests/test_geometry.py` (obsolete pipeline); replaced by
  `tests/test_pipeline.py` and current endpoint/CLI/evaluation tests.
- Moved root `index.html` to `bodymeasure/templates/index.html` for installation.
- Removed duplicated legacy measurement implementation; one shared pipeline
  remains. No images, datasets, configuration files or binary model assets were
  deleted. Their retained legacy status is listed in `LEGACY_ASSETS.md`.
- Temporary refactoring scripts created during this task were removed.
