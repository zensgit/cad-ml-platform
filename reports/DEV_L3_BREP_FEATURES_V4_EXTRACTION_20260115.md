# DEV_L3_BREP_FEATURES_V4_EXTRACTION_20260115

## Summary
Validated that v4 feature extraction consumes L3 B-Rep surface data to produce non-zero
surface counts and entropy for STEP inputs. Confirmed extraction behavior with unit tests
and live API calls in a linux/amd64 pythonocc-core environment.

## Scope
- L3 3D extraction now runs before 2D feature extraction in the analyze pipeline.
- FeatureExtractor v4 accepts optional B-Rep features for `surface_count` and `shape_entropy`.

## Validation
- `pytest tests/unit/test_feature_extractor_v4_real.py -v`
  - Result: 8 passed, 1 skipped (VECTOR_V4_LENGTH not defined in this environment).
- `pytest tests/unit/test_analyzer_rules.py -v`
  - Result: 2 passed.
- Live STEP validation (linux/amd64 micromamba container with python=3.10, pythonocc-core=7.9.0):
  - Started API: `micromamba run -n cadml uvicorn src.main:app --host 0.0.0.0 --port 8000`
  - `examples/sample_part.step` failed OCC parse (`ERR StepFile : Incorrect Syntax`), so it cannot be used
    for B-Rep validation with pythonocc.
  - Generated STEP fixtures via pythonocc and posted to `/api/v1/analyze`:
    - `tmp/box.step`: `surface_count=6`, `shape_entropy=0.0` (single surface type expected).
    - `tmp/cylinder.step`: `surface_count=3`, `shape_entropy=0.9709505944546685` (non-zero entropy).

## Notes
- B-Rep surface types are preferred for entropy when available.
- Legacy behavior is preserved when B-Rep data is missing or invalid.
- `pythonocc-core` is not installed on the host arm64/Python 3.13; `python3 -m pip install pythonocc-core`
  had no matching distribution.
- Attempted `docker build --platform linux/amd64 -f deployments/docker/Dockerfile -t cad-ml-platform:l3 .`;
  build failed because `pythonocc-core>=7.7.0` had no matching distribution on pip.
- Temporary STEP fixtures were generated under `tmp/` and removed after validation.
