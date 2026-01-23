# DEV_MECH_KNOWLEDGE_4000CAD_UVNET_GRAPH_STEP_VALIDATION_VALIDATION_20260123

## Checks
- Confirmed the local environment does not have `pythonocc-core` installed.
- Attempted to install `pythonocc-core` in `.venv-graph` via pip.
- Executed STEP-based dry-run in a linux/amd64 micromamba container.

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
- Command:
  - `micromamba create -y -n cadml -c conda-forge python=3.10 pythonocc-core`
- Result:
  - `pythonocc-core 7.9.0` installed in container
- Command:
  - `micromamba run -n cadml python -m pip install --index-url https://download.pytorch.org/whl/cpu torch`
- Result:
  - `torch-2.10.0+cpu` installed in container
- Command:
  - `micromamba run -n cadml env LD_LIBRARY_PATH=/opt/conda/envs/cadml/lib python scripts/train_uvnet_graph_dryrun.py --data-dir tests/fixtures --batch-size 2 --limit 2`
- Result:
  - `UV-Net Graph Dry-Run`
  - `Batch nodes: 25`
  - `Batch edges: 32`
  - `Logits shape: (2, 10)`
  - `Embedding shape: (2, 1024)`

## Notes
- Local macOS pip installs remain blocked; Docker validation succeeded.
