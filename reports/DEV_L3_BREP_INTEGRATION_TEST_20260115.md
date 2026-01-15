# DEV_L3_BREP_INTEGRATION_TEST_20260115

## Summary
Ran the L3 B-Rep integration test in a linux/amd64 micromamba environment with pythonocc-core.
The STEP-generated box and cylinder metrics test passed; warnings were limited to pythonocc-core deprecations.

## Environment
- Docker image: mambaorg/micromamba:1.5.8 (linux/amd64)
- Python: 3.10.19
- pytest: 7.4.3
- pythonocc-core: installed via conda-forge

## Steps
- Created micromamba env with pythonocc-core and installed repo requirements.
- Ran: `pytest tests/integration/test_brep_features_v4.py -v`

## Results
- `tests/integration/test_brep_features_v4.py::test_brep_surface_metrics_from_generated_steps`: passed

## Notes
- Warnings observed: pythonocc-core deprecations in `src/core/geometry/engine.py` and SWIG bootstrap warnings.
