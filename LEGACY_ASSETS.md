# Retained legacy and local assets

These files are retained for provenance and are not loaded by the application:

- `config.json`: obsolete calibrated SMPL-X fitting settings. None of its keys
  configure the current application.
- `model_weights.pth`: unused historical model weights; never unpickle/load them.
- `pose_landmarker.task`: unused Tasks model. This implementation uses the
  pinned MediaPipe Solutions models bundled in its wheel.
- `archive.zip`: original dataset archive; evaluation uses its extracted folder.
- `outputs/detection_check/*`: old masks and landmark JSON, not current outputs.
- `uploads/*`: optional local integration samples, not an upload destination.
- `templates/`: empty legacy root directory; the active page is now packaged at
  `bodymeasure/templates/index.html`.
- `.venv`, `.venv-fresh`, `.tools`: local runtimes, dependencies and historical
  server logs. Not application source or distribution contents.
- `.kilo`: editor-specific agent settings, not runtime configuration.

Deleted obsolete tests: `tests/frontend.test.cjs` and `tests/test_geometry.py`.
The former targeted a removed UI; the latter tested the removed algorithm.
They are replaced by the current frontend harness and `tests/test_pipeline.py`.
The root `index.html` was moved into package data, not discarded. Old duplicated
measurement logic was removed from the root app and package pipeline; one shared
implementation remains. No datasets, images or binary model assets were deleted.
