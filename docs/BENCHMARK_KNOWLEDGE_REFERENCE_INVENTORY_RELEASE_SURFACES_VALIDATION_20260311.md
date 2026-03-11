# Benchmark Knowledge Reference Inventory Release Surfaces Validation

## Scope

- Extend `benchmark_knowledge_reference_inventory` into:
  - `scripts/export_benchmark_release_decision.py`
  - `scripts/export_benchmark_release_runbook.py`
- Keep field semantics aligned with existing bundle / companion surfaces.

## Added Outputs

- `knowledge_reference_inventory_status`
- `knowledge_reference_inventory_summary`
- `knowledge_reference_inventory_priority_domains`
- `knowledge_reference_inventory_total_reference_items`
- `knowledge_reference_inventory_focus_tables_detail`
- `knowledge_reference_inventory_recommendations`

## Markdown Sections

- `## Knowledge Reference Inventory` in release decision output
- `## Knowledge Reference Inventory` in release runbook output

## Artifact Wiring

- Added `benchmark_knowledge_reference_inventory` to:
  - decision artifact rows
  - runbook artifact rows
  - CLI arguments / artifact path passthrough

## Tests Updated

- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Validation Commands

```bash
python3 -m py_compile \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

git diff --check
```

## Results

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `16 passed, 1 warning`
- `git diff --check`: passed

## Notes

- Warning source is existing third-party `python_multipart` deprecation noise from test imports.
- No production behavior was changed outside the release decision / runbook surfaces for this component.
