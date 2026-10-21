# Archived dataset evaluation

## Observed sample results

All six image pairs completed in both modes. Chest has five scored values because
subject 4's `86.0_tbr` entry is excluded. MAE below is in centimetres:

| Measurement | Uncalibrated MAE | Known-height MAE |
| --- | ---: | ---: |
| Height | 11.78 | Not scored (input) |
| Arm length | 13.85 | 14.73 |
| Chest | 6.66 | 4.72 |
| Waist | 8.22 | 4.60 |
| Hips | 17.68 | 15.73 |
| Thigh | 16.12 | 14.80 |

These results show substantial remaining error. Known height improves scale for
some measurements but cannot resolve landmark definitions, clothing or silhouette
geometry. See each mode's `summary.csv` for RMSE, bias and tolerance percentages.

## Reproduction and assumptions

Run from the checkout after installing the package:

```powershell
python scripts/evaluate.py --output-dir reports/uncalibrated
python scripts/evaluate.py --known-height --output-dir reports/known_height
```

The source is `archive/Body Measurements Image Dataset.csv`: six rows, joined by
`set_id` to `archive/<set_id>/front_img.jpg` and `side_img.jpg`. The root CSV is an
identical copy. The much larger tab-separated `archive/for notebook/Unidata__Body_Measure  META.csv`
has unrelated long IDs, so it is not joined to the six local image pairs.

The available tape-measure columns are taken as supplied ground truth, without
independent verification. Height is assumed to be centimetres, as indicated by
the plausible adult values and adjacent measurement columns. No values are
inferred from annotation images. Numeric values suffixed `_tbr` are excluded
because the suffix is not documented and may indicate review is needed.

Only height, arm length, chest, waist, hips and thigh have reasonably matching
outputs. `front_build_cm` is not assumed equivalent to outer shoulder width;
pelvis is not silently substituted for hips. Neck, wrist, torso and inseam have
no matched ground truth in this six-row file and are not scored. Anatomical
definitions may still differ between dataset tape measures and image proxies.

`predictions.csv` includes subject-level predictions, sources, raw truth,
warnings and exclusion reasons. `summary.csv` includes sample count, MAE, RMSE,
signed bias (prediction minus truth) and percentages with absolute error <=2/5 cm.
`failures.csv` records all rejected captures. Failed captures do not enter metric
denominators; their count is repeated in the summary. Missing comparisons stay
blank, never zero. Scores use the API's one-decimal predictions.

The known-height mode supplies the CSV height as input and explicitly excludes
height from scoring to avoid reporting input reuse as prediction accuracy. The
uncalibrated mode does not supply height. These are separate reports, not pooled.

This is a small convenience sample, not a held-out validation study. Rejections,
clothing, pose, uncertain annotation definitions and only six subjects limit
interpretation. No tuning against these six rows was performed. The evaluator
is an offline command explicitly allowed to write report CSVs; HTTP/CLI inference
never saves images, masks or results.
