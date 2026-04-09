# Benchmark Release Decision PR Comment Validation

## Scope

Keep the benchmark PR comment compact and non-duplicative while exposing the
release decision state alongside bundle and companion summary signals.

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

- release decision remains visible in PR comments
- duplicate artifact-bundle and companion-summary rows are removed
- PR comment signal-light table is compact again
