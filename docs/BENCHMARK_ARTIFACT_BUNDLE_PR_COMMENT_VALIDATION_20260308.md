# Benchmark Artifact Bundle PR Comment Validation

## Goal

Expose benchmark artifact bundle status in the PR comment and signal-light
section of `evaluation-report.yml`.

## Changes

- Added PR comment variables for benchmark artifact bundle outputs:
  - `benchmarkArtifactBundleEnabled`
  - `benchmarkArtifactBundleOverall`
  - `benchmarkArtifactBundleAvailableArtifacts`
  - `benchmarkArtifactBundleFeedbackStatus`
  - `benchmarkArtifactBundleAssistantStatus`
  - `benchmarkArtifactBundleReviewQueueStatus`
  - `benchmarkArtifactBundleOcrStatus`
  - `benchmarkArtifactBundleBlockers`
  - `benchmarkArtifactBundleRecommendations`
  - `benchmarkArtifactBundleArtifact`
- Added derived status line:
  - `benchmarkArtifactBundleStatus`
- Added signal-light state:
  - `benchmarkArtifactBundleLight`
- Added PR comment table row:
  - `Benchmark Artifact Bundle`
- Added Graph2D signal-light table row:
  - `Benchmark Artifact Bundle`
- Extended workflow regression tests to assert the new PR comment fields.

## Validation

Commands run:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
from pathlib import Path
import yaml
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('yaml_ok')
PY
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: passed
- YAML parse: `yaml_ok`

## Notes

- This branch only changes PR comment and signal-light rendering.
- Bundle build/upload behavior remains in the preceding CI wiring branch.
