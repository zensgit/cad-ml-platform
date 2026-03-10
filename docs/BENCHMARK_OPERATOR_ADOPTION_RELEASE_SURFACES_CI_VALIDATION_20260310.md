# Benchmark Operator Adoption Release Surfaces CI Validation

## Goal

Propagate operator-adoption `release_surface_alignment` from downstream benchmark
surfaces into workflow-visible outputs.

Covered layers:

- `benchmark artifact bundle`
- `benchmark companion summary`
- `evaluation-report.yml` job summary
- `evaluation-report.yml` PR comment

## Changes

- Added bundle/companion step outputs:
  - `operator_adoption_release_surface_alignment_status`
  - `operator_adoption_release_surface_alignment_summary`
  - `operator_adoption_release_surface_alignment_mismatches`
- Added job summary rows for bundle and companion alignment
- Added PR comment rows and JS bindings for bundle and companion alignment
- Extended workflow regression coverage in
  `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path(".github/workflows/evaluation-report.yml").read_text())
print("yaml-ok")
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Result

- workflow YAML parse: pass
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `3 passed`

## Outcome

`release_surface_alignment` is no longer trapped inside exporter JSON. It now
appears in benchmark bundle/companion workflow outputs, GitHub job summary, and
PR comment surfaces.
