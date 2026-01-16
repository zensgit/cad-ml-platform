# DEV_BREP_V4_INTEGRATION_EXTENDED_SHAPES_20260115

## Summary
Extended the B-Rep v4 integration test with sphere and torus fixtures plus bounding-box and surface-type
assertions. Local macOS run skips without pythonocc-core; linux/amd64 validation remains pending due to
Docker CLI timeouts while provisioning micromamba.

## Environment
- Local: macOS, Python 3.13.9, pytest 9.0.1
- Intended: linux/amd64 micromamba with pythonocc-core 7.9.0

## Steps
- Added new STEP fixtures (sphere, torus) and extra assertions.
- Ran: `pytest tests/integration/test_brep_features_v4.py -v`

## Results
- Local run: skipped (pythonocc-core not installed).

## Notes
- Attempted linux/amd64 micromamba setup in Docker, but `docker ps`/`docker rm` operations timed out;
  rerun in a healthy Docker environment to complete validation.
