# DEV_BREP_V4_INTEGRATION_EXTENDED_SHAPES_20260115

## Summary
Extended the B-Rep v4 integration test with sphere and torus fixtures plus bounding-box and surface-type
assertions. Local macOS run skips without pythonocc-core; linux/amd64 validation remains pending due to
micromamba provisioning stalls inside Docker.

## Environment
- Local: macOS, Python 3.13.9, pytest 9.0.1
- Intended: linux/amd64 micromamba with pythonocc-core 7.9.0

## Steps
- Added new STEP fixtures (sphere, torus) and extra assertions.
- Ran: `pytest tests/integration/test_brep_features_v4.py -v`

## Results
- Local run: skipped (pythonocc-core not installed).

## Notes
- Restarted Docker Desktop to recover the CLI and reran the linux/amd64 setup.
- micromamba encountered conda-forge download issues (SSL error on repodata) and later hung while
  resolving packages; validation was aborted to avoid an indefinite wait.
- Added `scripts/validate_brep_features_linux_amd64_cached.sh` with a docker volume cache; the
  micromamba solver still stalled without creating the cadml env.
- Retried with `-vvv` logging; downloads/SSL traces appeared but `/opt/conda/pkgs` remained empty
  and the run was interrupted. The script now captures stderr logs for the next retry.
- Rerun in a stable linux/amd64 environment to complete validation.
