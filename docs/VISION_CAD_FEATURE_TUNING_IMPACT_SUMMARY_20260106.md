# Vision CAD Feature Tuning Impact Summary (2026-01-06)

## Dataset
- Source: `data/dedup_report_train_local_version_profile_spatial_full_package/assets/images`
- Samples: 20

## Threshold Overrides
- `min_area=24`
- `line_aspect=6`
- `line_elongation=8`
- `circle_fill_min=0.4`
- `arc_fill_min=0.08`

## Baseline vs Tuned Summary
| Metric | Baseline | Tuned | Delta |
| --- | --- | --- | --- |
| total_lines | 165 | 83 | -82 |
| total_circles | 36 | 15 | -21 |
| total_arcs | 6 | 9 | +3 |
| avg_ink_ratio | 0.0195 | 0.0195 | 0.0 |
| avg_components | 10.35 | 5.35 | -5.0 |

## Notable Sample Deltas (components)
| Sample | lines | circles | arcs | components |
| --- | --- | --- | --- | --- |
| BTJ01231501522-00短轴承座(盖)v1__d80be3da.png | -8 | -1 | 0 | -9 |
| BTJ01231501522-00短轴承座(盖)v2__923f3197.png | -8 | -1 | 0 | -9 |
| BTJ01231501522-00短轴承座(盖)v3__42db91c9.png | -8 | -1 | 0 | -9 |
| BTJ01230901522-00汽水分离器v1__f36cd05a.png | -5 | -3 | 0 | -8 |
| BTJ01230901522-00汽水分离器v2__9f41a97f.png | -5 | -3 | 0 | -8 |

## Interpretation
- Stricter line and circle thresholds reduce detections significantly.
- Arc counts increased slightly with the higher `arc_fill_min` threshold.
- Overall component density drops ~48% on this dataset.

## Source Reports
- Baseline: `reports/vision_cad_feature_baseline_spatial_20260106.json`
- Tuned compare: `reports/vision_cad_feature_tuning_compare_20260106.json`
