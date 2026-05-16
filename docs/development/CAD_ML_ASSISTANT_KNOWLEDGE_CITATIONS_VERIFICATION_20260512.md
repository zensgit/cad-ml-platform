# CAD ML Assistant Knowledge Citations Verification

Date: 2026-05-12

## Scope

Validated assistant citation propagation from DecisionService knowledge evidence to
assistant explainability, answer citation notes, and OpenAPI schema.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/api/v1/assistant.py \
  tests/unit/assistant/test_llm_api.py
```

```bash
.venv311/bin/flake8 \
  src/api/v1/assistant.py \
  tests/unit/assistant/test_llm_api.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/assistant/test_llm_api.py \
  tests/unit/test_decision_service.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  tests/unit/test_export_assistant_evidence_report.py
```

```bash
.venv311/bin/python scripts/ci/generate_openapi_schema_snapshot.py \
  --output config/openapi_schema_snapshot.json
```

```bash
.venv311/bin/pytest -q tests/contract/test_openapi_schema_snapshot.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched assistant code and tests.
- Flake8 passed for touched assistant code and tests.
- Assistant, DecisionService, analyze knowledge, and assistant evidence tests passed:
  `31 passed, 7 warnings in 2.98s`.
- OpenAPI snapshot was regenerated after adding `QueryKnowledgeCitation`.
- OpenAPI contract test passed: `1 passed, 7 warnings in 2.37s`.
- `git diff --check` passed.

## Notes

- The OpenAPI generator printed existing environment messages:
  `router_import_failed` and `PyTorch not found. PointNet module running in stub mode.`
  It still exited successfully and wrote the snapshot.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.
