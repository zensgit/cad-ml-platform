# Benchmark Knowledge Readiness CI Validation

## Goal

Wire `knowledge_readiness` into the benchmark workflow so the signal is visible
in CI artifacts, job summary output, PR comments, and downstream benchmark
surfaces.

## Delivered

### 1. Standalone workflow artifact

- Added `Build benchmark knowledge readiness (optional)` to
  [evaluation-report.yml](../.github/workflows/evaluation-report.yml)
- Added `Upload benchmark knowledge readiness`
- Exposed workflow outputs:
  - `status`
  - `total_reference_items`
  - `ready_component_count`
  - `partial_component_count`
  - `missing_component_count`
  - `recommendations`

### 2. Scorecard integration

- `Generate benchmark scorecard (optional)` now accepts:
  - workflow-dispatch input `benchmark_scorecard_knowledge_readiness_summary`
  - env `BENCHMARK_SCORECARD_KNOWLEDGE_READINESS_JSON`
  - upstream artifact from `steps.benchmark_knowledge_readiness.outputs.output_json`
- Scorecard workflow outputs now expose:
  - `knowledge_status`
  - `knowledge_total_reference_items`

### 3. Downstream surface integration

These workflow-driven benchmark surfaces now accept
`benchmark_knowledge_readiness`:

- benchmark artifact bundle
- benchmark companion summary
- benchmark release decision
- benchmark release runbook

Each surface now exports `knowledge_status` in workflow outputs where
applicable.

### 4. CI / PR visibility

The workflow summary now shows:

- benchmark knowledge readiness status
- total reference items
- ready / partial / missing component counts
- recommendations

The PR comment now shows:

- scorecard-level `knowledge=...`
- standalone knowledge readiness row
- bundle / companion / release / runbook `knowledge=...` status details

## Validation

Commands:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result:

- `yaml_ok`
- `flake8` passed
- `3 passed`

## Notes

- This branch is stacked on top of
  `feat/benchmark-knowledge-signals`, which already delivered the exporter and
  reusable benchmark-surface contracts.
- The workflow keeps knowledge readiness optional by default. It runs when
  explicitly enabled or when a snapshot override is provided.
