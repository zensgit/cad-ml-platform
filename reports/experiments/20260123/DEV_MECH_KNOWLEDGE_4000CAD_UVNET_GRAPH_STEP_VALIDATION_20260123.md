# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_STEP_VALIDATION_20260123

## Summary
- Attempted STEP-based UV-Net graph dry-run validation in the current environment.
- Tried installing `pythonocc-core` via pip in `.venv-graph` but no distribution was available.

## Result
- `pythonocc-core` is not available locally, so STEP-based graph extraction cannot execute.
- The pip installation failed with "No matching distribution found".

## Notes
- Re-run on a machine with `pythonocc-core` installed to validate real B-Rep graph extraction.
- Consider using a Linux environment with conda-forge packages or a container that bundles OpenCascade.
