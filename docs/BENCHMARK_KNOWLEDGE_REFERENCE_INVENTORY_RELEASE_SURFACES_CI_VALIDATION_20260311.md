# Benchmark Knowledge Reference Inventory Release Surfaces CI Validation

## Goal

Wire `knowledge_reference_inventory` release-surface outputs into the benchmark
CI control plane so `evaluation-report.yml` can consume and publish:

- release decision inventory status
- release decision inventory summary
- release decision inventory priority domains
- release decision inventory total reference items
- release runbook inventory status
- release runbook inventory summary
- release runbook inventory priority domains
- release runbook inventory total reference items

## Scope

Updated:

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

Added:

- `docs/BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_RELEASE_SURFACES_CI_VALIDATION_20260311.md`

## Changes

### Workflow inputs

Added new optional `workflow_dispatch` inputs:

- `benchmark_release_decision_knowledge_reference_inventory_json`
- `benchmark_release_runbook_knowledge_reference_inventory_json`

### Release decision wiring

`Build benchmark release decision (optional)` now accepts
`--benchmark-knowledge-reference-inventory` from:

- workflow input
- `steps.benchmark_knowledge_reference_inventory.outputs.output_json`
- `BENCHMARK_RELEASE_DECISION_KNOWLEDGE_REFERENCE_INVENTORY_JSON`

It now exports:

- `knowledge_reference_inventory_status`
- `knowledge_reference_inventory_summary`
- `knowledge_reference_inventory_priority_domains`
- `knowledge_reference_inventory_total_reference_items`
- `knowledge_reference_inventory_recommendations`

### Release runbook wiring

`Build benchmark release runbook (optional)` now accepts
`--benchmark-knowledge-reference-inventory` from:

- workflow input
- `steps.benchmark_knowledge_reference_inventory.outputs.output_json`
- `BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_REFERENCE_INVENTORY_JSON`

It now exports:

- `knowledge_reference_inventory_status`
- `knowledge_reference_inventory_summary`
- `knowledge_reference_inventory_priority_domains`
- `knowledge_reference_inventory_total_reference_items`
- `knowledge_reference_inventory_recommendations`

### Job summary

`Create job summary` now prints:

- `Benchmark release knowledge reference inventory`
- `Benchmark release runbook knowledge reference inventory`

including summary, priority domains, total references, and recommendations.

## Validation

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
python3 - <<'PY'
import yaml, pathlib
yaml.safe_load(pathlib.Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml-ok')
PY
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
git diff --check
```

Results:

- `py_compile`: pass
- `flake8`: pass
- workflow YAML parse: `yaml-ok`
- `pytest`: `5 passed, 1 warning`
- `git diff --check`: pending final pre-push check

## Outcome

The benchmark control-plane now carries `knowledge_reference_inventory` from the
release-surface exporters into workflow-level CI outputs and job summary, ready
for the next stacked PR to extend PR comment and signal-light surfaces.
