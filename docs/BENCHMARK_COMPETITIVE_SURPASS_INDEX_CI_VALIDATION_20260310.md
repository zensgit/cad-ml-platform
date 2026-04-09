# Benchmark Competitive Surpass Index CI Validation

## Scope

This change wires `benchmark_competitive_surpass_index` into the evaluation
workflow so the new benchmark component is available to CI consumers instead of
remaining a standalone exporter only.

Integrated areas:

- `workflow_dispatch` inputs
- workflow `env`
- standalone build step
- artifact upload
- job summary
- downstream passthrough into:
  - `benchmark artifact bundle`
  - `benchmark companion summary`

## Design

The CI layer now treats `competitive_surpass_index` like the other benchmark
components:

1. Build a standalone JSON/Markdown artifact from existing benchmark signals.
2. Expose the component status through `GITHUB_OUTPUT`.
3. Feed the generated artifact into bundle and companion exporters.
4. Surface the result in the GitHub job summary.

The standalone component emits:

- `status`
- `score`
- `ready_pillars`
- `partial_pillars`
- `blocked_pillars`
- `primary_gaps`
- `recommendations`

The bundle and companion summary steps now expose:

- `competitive_surpass_index_status`
- `competitive_surpass_primary_gaps`
- `competitive_surpass_recommendations`

## Files

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Commands run:

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed, 1 warning`

## Outcome

`competitive_surpass_index` is now CI-visible and can be consumed by downstream
benchmark surfaces. The next stacked step is PR comment / signal-light wiring.
