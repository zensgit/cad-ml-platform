# DEV_PROVIDER_COVERAGE_TESTS_KNOWLEDGE_BOOTSTRAP_20260209

## Goal
Complete unit-test coverage for provider-framework glue that is critical for
ops/runtime observability:
- Core registry bootstrap snapshot (`src/core/providers/bootstrap.py`)
- Knowledge provider bridge bootstrap and health probes (`src/core/providers/knowledge.py`)

## Changes
### 1) Core provider bootstrap snapshot coverage
File: `tests/unit/test_bootstrap_coverage.py`
- Covers bootstrap globals (`_BOOTSTRAPPED`, `_BOOTSTRAP_TS`) behavior and idempotency.
- Covers `get_core_provider_registry_snapshot(lazy_bootstrap=...)` behavior.
- Covers `_build_snapshot` resilience when `get_provider_class` fails (records `"unknown"`).
- Validates snapshot structure and core domain presence (`vision`, `ocr`, `classifier`, `knowledge`).

Coverage spot check:
- `src/core/providers/bootstrap.py`: 100%

### 2) Knowledge provider bridge coverage
File: `tests/unit/test_knowledge_provider_coverage.py`
- Covers `ToleranceKnowledgeProviderAdapter` and `StandardsKnowledgeProviderAdapter`:
  - health-check failure paths (None returns / exceptions)
  - healthy probe path
  - lightweight `process()` status payload shape
- Covers `bootstrap_core_knowledge_providers()`:
  - registrations + idempotency
  - instantiation-time default config wiring for `knowledge/tolerance` and `knowledge/standards`

Coverage spot check:
- `src/core/providers/knowledge.py`: 100%

## Validation
Commands run:
```bash
pytest tests/unit/test_bootstrap_coverage.py -q
pytest tests/unit/test_bootstrap_coverage.py --cov=src.core.providers.bootstrap --cov-report=term-missing -q

pytest tests/unit/test_knowledge_provider_coverage.py -q
pytest tests/unit/test_knowledge_provider_coverage.py --cov=src.core.providers.knowledge --cov-report=term-missing -q

flake8 tests/unit/test_bootstrap_coverage.py tests/unit/test_knowledge_provider_coverage.py
```

Results:
- All tests passed locally.
- Coverage spot checks reached 100% for both target modules.

