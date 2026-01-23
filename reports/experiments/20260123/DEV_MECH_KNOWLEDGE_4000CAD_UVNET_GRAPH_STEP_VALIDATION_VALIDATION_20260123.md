# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_STEP_VALIDATION_VALIDATION_20260123

## Checks
- Confirmed the local environment does not have `pythonocc-core` installed.
- Attempted to install `pythonocc-core` in `.venv-graph` via pip.

## Runtime Output
- Command:
  - `python3 - <<'PY'
from src.core.geometry.engine import HAS_OCC
print("HAS_OCC", HAS_OCC)
PY`
- Result:
  - `HAS_OCC False`
- Command:
  - `.venv-graph/bin/pip install pythonocc-core`
- Result:
  - `ERROR: No matching distribution found for pythonocc-core`

## Notes
- STEP validation is blocked until `pythonocc-core` is installed.
