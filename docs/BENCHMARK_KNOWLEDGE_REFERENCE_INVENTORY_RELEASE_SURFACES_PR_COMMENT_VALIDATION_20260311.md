# Benchmark Knowledge Reference Inventory Release Surfaces PR Comment Validation

## Goal

Extend PR comment and signal-light surfaces so release-layer benchmark views expose
`knowledge_reference_inventory` for:

- `Benchmark Release Decision`
- `Benchmark Release Runbook`

## Scope

Updated:

- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

Added:

- `docs/BENCHMARK_KNOWLEDGE_REFERENCE_INVENTORY_RELEASE_SURFACES_PR_COMMENT_VALIDATION_20260311.md`

## Changes

### PR comment variables

Added release decision variables:

- `benchmarkReleaseKnowledgeReferenceInventoryStatus`
- `benchmarkReleaseKnowledgeReferenceInventorySummary`
- `benchmarkReleaseKnowledgeReferenceInventoryPriorityDomains`
- `benchmarkReleaseKnowledgeReferenceInventoryTotalReferenceItems`
- `benchmarkReleaseKnowledgeReferenceInventoryRecommendations`
- `benchmarkReleaseKnowledgeReferenceInventoryStatusLine`
- `benchmarkReleaseDecisionKnowledgeReferenceInventoryLight`

Added release runbook variables:

- `benchmarkReleaseRunbookKnowledgeReferenceInventoryStatus`
- `benchmarkReleaseRunbookKnowledgeReferenceInventorySummary`
- `benchmarkReleaseRunbookKnowledgeReferenceInventoryPriorityDomains`
- `benchmarkReleaseRunbookKnowledgeReferenceInventoryTotalReferenceItems`
- `benchmarkReleaseRunbookKnowledgeReferenceInventoryRecommendations`
- `benchmarkReleaseRunbookKnowledgeReferenceInventoryStatusLine`
- `benchmarkReleaseRunbookKnowledgeReferenceInventoryLight`

### PR comment tables

Added rows:

- `Benchmark Release Decision Knowledge Reference Inventory`
- `Benchmark Release Runbook Knowledge Reference Inventory`

to both the detailed results table and the signal-lights table.

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

`knowledge_reference_inventory` is now visible not only in standalone benchmark
surfaces, but also in release decision and release runbook PR comment/status-light
views, closing the review surface gap for this benchmark component.
