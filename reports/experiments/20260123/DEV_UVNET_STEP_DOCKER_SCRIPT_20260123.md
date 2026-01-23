# DEV_UVNET_STEP_DOCKER_SCRIPT_20260123

## Summary
- Added a reusable Docker micromamba wrapper to run the UV-Net STEP dry-run on linux/amd64.
- Script supports configurable Docker image, conda env name, Python version, and torch index.

## Deliverables
- Script: `scripts/validate_uvnet_step_docker.sh`
- Defaults: micromamba image `mambaorg/micromamba:1.5.8`, env `cadml`, Python `3.10`, CPU torch wheels.

## Notes
- Use this script once STEP data is available locally.
- The script mounts the repo at `/workspace` and runs `scripts/train_uvnet_graph_dryrun.py`.
