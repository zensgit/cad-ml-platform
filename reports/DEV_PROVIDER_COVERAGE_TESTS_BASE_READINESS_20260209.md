# DEV_PROVIDER_COVERAGE_TESTS_BASE_READINESS_20260209

## Goal
Increase unit-test coverage for the core provider framework by explicitly
covering edge paths in:
- `src/core/providers/base.py`
- `src/core/providers/readiness.py`

This improves confidence in provider health/readiness behavior, and reduces the
risk of regressions when adding new providers (e.g., `knowledge/*`, `classifier/*`).

## Changes
### 1) BaseProvider coverage suite
File: `tests/unit/test_base_provider_coverage.py`

Covers:
- `BaseProvider.name` fallback behavior (missing/None/empty `config.name`)
- `BaseProvider.provider_type` behavior (missing/None/empty `config.provider_type`)
- `BaseProvider.health_check()` edge cases:
  - `timeout_seconds <= 0` minimum handling
  - timeout -> `DOWN` + `last_error="timeout"`
  - exception -> `DOWN` + `last_error` captured
  - unhealthy `False` -> `DOWN` + `last_error="unhealthy"`
- `mark_degraded()` / `mark_healthy()` status transitions
- `status_snapshot()` shape and updates after a health check
- lifecycle no-ops (`warmup`, `shutdown`) and `process()` delegation

### 2) Readiness coverage suite
File: `tests/unit/test_readiness_coverage.py`

Covers:
- `parse_provider_id_list()` parsing branches (commas/spaces, `/` vs `:`,
  whitespace trimming, invalid token handling)
- `ProviderReadinessSummary.to_dict()` shape
- `check_provider_readiness()` edge cases:
  - `timeout_seconds <= 0` defaulting and capping
  - provider init exception handling (`init_error:*`)
  - duplicate provider IDs deduplication
  - required vs optional provider semantics (`ok` vs `degraded`)
- `load_provider_readiness_config_from_env()` parsing

## Validation
Commands run:
```bash
pytest tests/unit/test_base_provider_coverage.py -q
pytest tests/unit/test_readiness_coverage.py -q

# Coverage spot checks (module-level)
pytest tests/unit/test_base_provider_coverage.py --cov=src.core.providers.base --cov-report=term-missing -q
pytest tests/unit/test_readiness_coverage.py --cov=src.core.providers.readiness --cov-report=term-missing -q

# Lint
flake8 tests/unit/test_base_provider_coverage.py tests/unit/test_readiness_coverage.py
```

Results (spot checks):
- `src/core/providers/base.py`: 100% (0 missing)
- `src/core/providers/readiness.py`: 100% (0 missing)

## Notes
- These tests are intentionally isolated from external services and do not
  require Redis, Docker, or a running API server.

