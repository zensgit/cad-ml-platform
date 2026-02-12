# DEV_PROVIDER_PLUGIN_METRICS_EXPOSURE_20260212

## Summary
- Added dedicated regression coverage to verify provider plugin bootstrap metrics are exported by `/metrics`.

## Changes
- Added `tests/unit/test_provider_plugin_metrics_exposed.py`:
  - verifies `core_provider_plugin_bootstrap_total{result="reload_ok"}` and `{result="cache_hit"}` are emitted.
  - verifies `core_provider_plugin_bootstrap_total{result="strict_error"}` is emitted on strict plugin import failure.
  - verifies corresponding histogram bucket lines exist for each result label.
- Updated `Makefile`:
  - included `tests/unit/test_provider_plugin_metrics_exposed.py` in `make test-provider-core`.

## Validation
- `pytest -q tests/unit/test_provider_plugin_metrics_exposed.py`
  - `2 passed`
- `make test-provider-core`
  - `59 passed`
- `make validate-core-fast`
  - `ISO286 validators OK`, `48 passed`, `103 passed`, `59 passed`

## Notes
- Assertions are label-based and resilient to Prometheus label ordering differences.
