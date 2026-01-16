# DEV_L3_BREP_LINUX_AMD64_VALIDATION_20260116

## Summary
Attempted linux/amd64 B-Rep validation with a persistent micromamba package cache; conda env
provisioning did not finish, so analysis could not run.

## Environment
- Host: macOS (arm64)
- Docker image: mambaorg/micromamba:1.5.8 (linux/amd64)
- Cache volume: cadml-micromamba-cache
- FEATURE_VERSION: v4

## Steps
- Ran `bash scripts/validate_brep_features_linux_amd64_cached.sh`.
- Container started with `/opt/conda/pkgs` volume cache and `MAMBA_NO_REPODATA_ZST=1`.
- Began `micromamba create -n cadml -c conda-forge python=3.10 pythonocc-core`.

## Results
- micromamba solver ran for an extended period with no packages written to `/opt/conda/pkgs`.
- Env `cadml` was not created; `micromamba run -n cadml python -m pip install -r requirements.txt`
  failed with: `The given prefix does not exist: "/opt/conda/envs/cadml"`.

## Next Steps
- Retry on a native linux/amd64 host or allow more time for the solver to finish.
- Consider adding verbose micromamba logging to capture solver progress if the stall persists.
