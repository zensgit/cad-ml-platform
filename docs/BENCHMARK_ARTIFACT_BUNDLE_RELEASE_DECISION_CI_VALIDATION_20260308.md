# Benchmark Artifact Bundle Release Decision CI Validation

## Scope

Wire `benchmark_release_decision` into the benchmark artifact bundle workflow
path so `evaluation-report.yml` can pass the release decision JSON to the
bundle exporter from either workflow inputs or prior step outputs.

## Validation

```bash
python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY
```

## Result

- workflow input and env contract extended
- artifact bundle step accepts release decision input from dispatch, steps, and env
- regression test now checks release decision wiring in the bundle path
