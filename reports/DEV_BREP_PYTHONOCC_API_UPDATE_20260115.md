# DEV_BREP_PYTHONOCC_API_UPDATE_20260115

## Summary
Updated pythonocc-core API usage to remove deprecation warnings in B-Rep feature extraction and STEP/IGES adapter paths.
Validated with the existing B-Rep integration test in a linux/amd64 micromamba environment; only SWIG warnings remain.

## Environment
- Docker image: mambaorg/micromamba:1.5.8 (linux/amd64)
- Python: 3.10.19
- pytest: 7.4.3
- pythonocc-core: 7.9.0

## Steps
- Created a micromamba environment with pythonocc-core.
- Installed repo requirements.
- Ran: `pytest tests/integration/test_brep_features_v4.py -v`

## Results
- `tests/integration/test_brep_features_v4.py::test_brep_surface_metrics_from_generated_steps`: passed

## Notes
- The pythonocc-core deprecation warnings for BRepGProp/BRepBndLib no longer appear; remaining warnings are SWIG bootstrap deprecations.
