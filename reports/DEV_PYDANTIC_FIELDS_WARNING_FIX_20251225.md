# Pydantic __fields__ warning fix

- Date: 2025-12-25
- Change: Replaced MagicMock spec on `CadDocument` with a local stub to avoid Pydantic `__fields__` deprecation warnings.
- Command: `.venv/bin/python -m pytest tests/unit/test_adapter_factory_coverage.py::TestBaseAdapter::test_base_adapter_convert_calls_parse -q`
- Result: PASS (1 passed in 1.93s)
