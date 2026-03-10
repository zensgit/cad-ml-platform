# Benchmark Knowledge Domain Control Plane CI Validation

## Scope

- wire `knowledge_domain_control_plane` into `evaluation-report.yml`
- add workflow-dispatch inputs and env defaults
- add optional build/upload steps
- propagate control-plane JSON into:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- expose control-plane status in job summary

## Workflow Coverage

The CI layer now includes:

- `benchmark_knowledge_domain_control_plane_enable`
- source inputs for:
  - capability matrix
  - capability drift
  - realdata correlation
  - outcome correlation
  - domain action plan
- downstream env passthrough for:
  - `BENCHMARK_ARTIFACT_BUNDLE_KNOWLEDGE_DOMAIN_CONTROL_PLANE_JSON`
  - `BENCHMARK_COMPANION_SUMMARY_KNOWLEDGE_DOMAIN_CONTROL_PLANE_JSON`
  - `BENCHMARK_RELEASE_DECISION_KNOWLEDGE_DOMAIN_CONTROL_PLANE_JSON`
  - `BENCHMARK_RELEASE_RUNBOOK_KNOWLEDGE_DOMAIN_CONTROL_PLANE_JSON`

## Verification

Commands run in isolated worktree:
`/private/tmp/cad-ml-platform-knowledge-domain-control-plane-ci-20260310`

```bash
python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

python3 - <<'PY'
import yaml
from pathlib import Path

yaml.safe_load(
    Path(".github/workflows/evaluation-report.yml").read_text(encoding="utf-8")
)
print("yaml-ok")
PY

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `yaml.safe_load`: pass
- `pytest`: `3 passed`

## Expected Result

- workflow can build and upload `benchmark_knowledge_domain_control_plane`
- downstream benchmark surfaces receive control-plane JSON through workflow wiring
- job summary includes:
  - status
  - ready / partial / blocked / missing domains
  - total actions
  - high-priority actions
  - release blockers
  - priority domains
  - focus areas
  - recommendations

## Limitations

- this layer stops at CI summary and downstream outputs
- PR comment / signal light exposure stays in stacked follow-up branch
