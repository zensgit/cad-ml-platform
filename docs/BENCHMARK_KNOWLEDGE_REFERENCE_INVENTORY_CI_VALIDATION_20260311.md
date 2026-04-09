# Benchmark Knowledge Reference Inventory CI Validation

## Scope

- Wire `benchmark_knowledge_reference_inventory` into `evaluation-report.yml`.
- Expose workflow inputs, env vars, build step, artifact upload, and job summary.
- Pass the generated JSON into:
  - `benchmark artifact bundle`
  - `benchmark companion summary`

## Workflow Coverage

- `workflow_dispatch` inputs:
  - `benchmark_knowledge_reference_inventory_enable`
  - `benchmark_knowledge_reference_inventory_snapshot_json`
  - `benchmark_artifact_bundle_knowledge_reference_inventory_json`
  - `benchmark_companion_summary_knowledge_reference_inventory_json`
- workflow env:
  - `BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_ENABLE`
  - `BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_TITLE`
  - `BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_SNAPSHOT_JSON`
  - `BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_OUTPUT_JSON`
  - `BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_OUTPUT_MD`
  - bundle / companion JSON passthrough vars
- evaluate job:
  - `Build benchmark knowledge reference inventory (optional)`
  - `Upload benchmark knowledge reference inventory`
  - job summary lines for standalone inventory status
  - job summary lines for bundle / companion inventory status

## Validation Commands

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml-ok')
PY

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

## Results

- `py_compile` passed
- `flake8` passed
- workflow YAML parse passed
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
  - `4 passed, 1 warning`
