# Benchmark Knowledge Reference Inventory Validation

## Scope

- Wire `benchmark_knowledge_reference_inventory` through `artifact bundle`.
- Surface the inventory status, summary, recommendations, and artifact path in
  `companion summary`.
- Add focused unit coverage for the new exporter and the bundle/companion
  passthrough behavior.

## Expected Surfaces

- `scripts/export_benchmark_artifact_bundle.py`
  - `component_statuses.knowledge_reference_inventory`
  - `artifacts.benchmark_knowledge_reference_inventory`
  - `knowledge_reference_inventory_status`
  - `knowledge_reference_inventory_summary`
  - `knowledge_reference_inventory_priority_domains`
  - `knowledge_reference_inventory_total_reference_items`
  - `knowledge_reference_inventory_recommendations`
  - `## Knowledge Reference Inventory` markdown section
- `scripts/export_benchmark_companion_summary.py`
  - `component_statuses.knowledge_reference_inventory`
  - `artifacts.benchmark_knowledge_reference_inventory`
  - `knowledge_reference_inventory_status`
  - `knowledge_reference_inventory_summary`
  - `knowledge_reference_inventory_priority_domains`
  - `knowledge_reference_inventory_total_reference_items`
  - `knowledge_reference_inventory_recommendations`
  - `recommended_actions` fallback when higher-priority recommendation sources
    are absent
  - `## Knowledge Reference Inventory` markdown section

## Validation Commands

```bash
python3 -m py_compile \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_knowledge_reference_inventory.py

flake8 \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_knowledge_reference_inventory.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_knowledge_reference_inventory.py
```

## Results

- `python3 -m py_compile scripts/export_benchmark_artifact_bundle.py scripts/export_benchmark_companion_summary.py tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_benchmark_companion_summary.py tests/unit/test_benchmark_knowledge_reference_inventory.py`
  - Passed.
- `flake8 scripts/export_benchmark_artifact_bundle.py scripts/export_benchmark_companion_summary.py tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_benchmark_companion_summary.py tests/unit/test_benchmark_knowledge_reference_inventory.py --max-line-length=100`
  - Passed.
- `pytest -q tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_benchmark_companion_summary.py tests/unit/test_benchmark_knowledge_reference_inventory.py`
  - Passed: `26 passed, 1 warning in 2.36s`
  - Warning: existing environment warning from `starlette.formparsers`
    (`PendingDeprecationWarning` for `python_multipart` import path).

## Delivered

- Exposed inventory status, summary, recommendations, and markdown sections
  through:
  - `scripts/export_benchmark_companion_summary.py`
  - `scripts/export_benchmark_artifact_bundle.py`
- Added focused unit coverage for:
  - exporter behavior
  - companion summary passthrough
  - artifact bundle passthrough
