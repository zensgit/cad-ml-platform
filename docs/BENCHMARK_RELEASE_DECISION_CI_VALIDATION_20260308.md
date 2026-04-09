# Benchmark Release Decision CI Validation

## Scope

Wire the standalone benchmark release decision export into
`.github/workflows/evaluation-report.yml` so CI can:

- build the release decision summary
- upload it as an artifact
- expose it in the job summary
- expose it in the PR comment payload

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

- workflow input/env contract extended
- optional export/upload path added
- job summary receives release decision lines
- PR comment script receives release decision state and signal-light fields
