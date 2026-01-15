# BREP_PYTHONOCC_SWIG_WARNING_FILTER_DESIGN

## Goal
Reduce pytest noise by filtering known SWIG-related DeprecationWarnings emitted by pythonocc-core.

## Approach
- Add targeted `filterwarnings` entries in `pytest.ini` to ignore the three SWIG bootstrap warnings:
  - SwigPyPacked
  - SwigPyObject
  - swigvarlink

## Scope
- Only suppresses these specific warnings; other DeprecationWarnings remain visible.

## Testing
- `pytest tests/integration/test_brep_features_v4.py -v` (linux/amd64 micromamba, pythonocc-core 7.9.0).
