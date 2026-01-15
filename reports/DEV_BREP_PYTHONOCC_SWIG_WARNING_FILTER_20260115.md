# DEV_BREP_PYTHONOCC_SWIG_WARNING_FILTER_20260115

## Summary
Filtered pythonocc-core SWIG DeprecationWarnings in pytest output to keep B-Rep integration runs clean.
Verified that the integration test passes without warning noise.

## Environment
- Docker image: mambaorg/micromamba:1.5.8 (linux/amd64)
- Python: 3.10.19
- pytest: 7.4.3
- pythonocc-core: 7.9.0

## Steps
- Added SWIG warning filters to `pytest.ini`.
- Ran: `pytest tests/integration/test_brep_features_v4.py -v`

## Results
- `tests/integration/test_brep_features_v4.py::test_brep_surface_metrics_from_generated_steps`: passed
- Warnings: none emitted for SWIG bootstrap deprecations.
