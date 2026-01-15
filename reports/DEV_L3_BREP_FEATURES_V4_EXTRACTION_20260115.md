# DEV_L3_BREP_FEATURES_V4_EXTRACTION_20260115

## Summary
Validated that v4 feature extraction consumes L3 B-Rep surface data to produce non-zero
surface counts and entropy for STEP inputs. Confirmed extraction behavior with new unit test.

## Scope
- L3 3D extraction now runs before 2D feature extraction in the analyze pipeline.
- FeatureExtractor v4 accepts optional B-Rep features for `surface_count` and `shape_entropy`.

## Validation
- `pytest tests/unit/test_feature_extractor_v4_real.py -v`
  - Result: 8 passed, 1 skipped (VECTOR_V4_LENGTH not defined in this environment).
- `pytest tests/unit/test_analyzer_rules.py -v`
  - Result: 2 passed.

## Notes
- B-Rep surface types are preferred for entropy when available.
- Legacy behavior is preserved when B-Rep data is missing or invalid.
- `pythonocc-core` is not installed in this environment, so live `/api/v1/analyze` STEP validation was skipped.
