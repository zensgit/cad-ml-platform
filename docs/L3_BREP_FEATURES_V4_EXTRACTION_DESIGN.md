# L3 B-Rep Features for V4 Extraction

## Summary
This change wires L3 B-Rep features into the v4 feature extraction path so STEP/IGES inputs report
non-zero `surface_count` and a meaningful `shape_entropy`. The analysis pipeline now runs the
GeometryEngine (L3) before 2D feature extraction and passes its B-Rep output into
`FeatureExtractor.extract`.

## Motivation
Some STEP/IGES inputs have sparse 2D entities, causing v4 features like `surface_count` and
`shape_entropy` to be zero or uninformative. The L3 geometry engine already computes B-Rep data
(e.g., `faces`, `surface_types`), which is a more accurate source for those metrics.

## Data Flow
1. `src/api/v1/analyze.py`
   - Run L3 geometry extraction early for STEP/IGES.
   - Collect B-Rep features in `features_3d`.
   - Pass `features_3d` to `FeatureExtractor.extract`.
2. `src/core/feature_extractor.py`
   - `extract(..., brep_features=...)` uses B-Rep `faces`/`surface_types` when present.
   - `shape_entropy` is computed from B-Rep `surface_types` when available; otherwise it falls back
     to entity counts.

## API/Code Changes
- `FeatureExtractor.extract` now accepts an optional `brep_features` dict.
- `compute_surface_count` prioritizes `faces`/`surface_count`/`surface_types` from B-Rep features.
- `shape_entropy` uses normalized B-Rep `surface_types` when available.

## Compatibility
- The new parameter is optional; callers that do not pass B-Rep features retain existing behavior.
- Cached feature vectors are still honored; the B-Rep integration only affects fresh extraction.

## Metrics and Observability
- Existing histograms (`v4_surface_count`, `v4_shape_entropy`) continue to track v4 values.
- `features_3d` remains included in the analysis payload when available.

## Risks and Mitigations
- **Missing/invalid B-Rep**: Guarded by `valid_3d` and type checks; fallback to legacy counts.
- **Cache mismatch**: Cached vectors stay as-is; only new extractions use B-Rep data.

## Test Plan
- `pytest tests/unit/test_feature_extractor_v4_real.py -v`
  - New test confirms `surface_count` and `shape_entropy` reflect B-Rep surface types.

## Local Validation (macOS arm64)
- `pythonocc-core` wheels are not available for macOS arm64; use linux/amd64 via Docker.
- Run `bash scripts/validate_brep_features_linux_amd64.sh` to provision pythonocc-core,
  start the API, generate STEP fixtures, and write a validation report.
