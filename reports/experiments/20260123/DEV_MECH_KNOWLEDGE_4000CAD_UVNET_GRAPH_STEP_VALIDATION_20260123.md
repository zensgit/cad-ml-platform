# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_STEP_VALIDATION_20260123

## Summary
- Attempted STEP-based UV-Net graph dry-run validation locally; pip install failed on macOS.
- Completed STEP validation via linux/amd64 micromamba Docker container with pythonocc-core.

## Result
- Local pip install failed with "No matching distribution found".
- Docker run succeeded: non-zero node/edge counts and valid logits/embedding shapes.

## Notes
- Docker-based validation is the recommended path for macOS arm64.
- Use a persistent micromamba cache volume to speed up repeat validations.
