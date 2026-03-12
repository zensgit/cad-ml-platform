# Benchmark Knowledge API Surface Matrix CI Validation

## Goal

Wire the `knowledge_domain_api_surface_matrix` benchmark component into
`evaluation-report.yml` so CI can build it, upload it, summarize it, and pass it
through to downstream benchmark surfaces.

## Scope

- workflow dispatch inputs
- workflow env defaults
- optional exporter build step
- artifact upload
- job summary lines
- passthrough into bundle / companion / release decision / release runbook

## Verification

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Additional parse check:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml-ok')
PY
```

## Expected Outputs

The exporter step emits:

- `status`
- `ready_domain_count`
- `partial_domain_count`
- `blocked_domain_count`
- `total_domain_count`
- `total_api_route_count`
- `focus_areas`
- `priority_domains`
- `public_api_gap_domains`
- `reference_gap_domains`
- `domain_statuses`
- `recommendations`

## Outcome

The benchmark control-plane now exposes public API coverage gaps for
`tolerance`, `standards`, and `gdt` through the same CI path already used by
other benchmark governance components.
