# DEV_GHCR_PUBLISH_VALIDATION_20260113

## Commands
- gh workflow run ghcr-publish.yml
- gh run watch 20955972127 --exit-status
- docker pull --platform linux/amd64 ghcr.io/zensgit/cad-ml-platform:main

## Results
- GHCR Publish run 20955972127 succeeded after skipping L3 deps via INSTALL_L3_DEPS=0.
- Image pulled successfully for linux/amd64.

## Notes
- Prior run 20955870231 failed when installing pythonocc-core from requirements-l3.txt.
