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
- Began `micromamba create -vvv -n cadml -c conda-forge python=3.10 pythonocc-core`.

## Results
- micromamba solver ran for an extended period and produced download/SSL traces, but no packages
  were written to `/opt/conda/pkgs` before the run was interrupted.
- Env `cadml` was not created; the follow-on pip install step could not start.
## Notes
- The verbose output streamed to the console during the run; the script is now updated to capture
  stderr via `2>&1 | tee` for future retries.
- Log path: `reports/DEV_L3_BREP_LINUX_AMD64_VALIDATION_20260116_micromamba.log` (empty for this run).

## Next Steps
- Retry on a native linux/amd64 host or allow more time for the solver to finish.
- Consider adding verbose micromamba logging to capture solver progress if the stall persists.
