# make test Report

- Date: 2026-01-04
- Scope: `make test`

## Command
- make test

## Result
- PASS

## Summary
- 3993 passed, 21 skipped, 3 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 153.86s

## Update (warnings enabled)
### Command
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=html -W default

### Result
- 3993 passed, 21 skipped, 170 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 105.37s

### Notes
- Warnings summary dominated by ResourceWarning: unclosed event loop.

## Update (ResourceWarning as error)
### Command
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=html -W error::ResourceWarning

### Result
- 3993 passed, 21 skipped, 158 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 104.37s

### Notes
- PytestUnraisableExceptionWarning surfaced unclosed socket ResourceWarnings.

## Update (asyncio debug)
### Command
- PYTHONASYNCIODEBUG=1 PYTHONWARNINGS=default .venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=html -W default

### Result
- 3993 passed, 21 skipped, 180 warnings
- Coverage: 72% (htmlcov generated)
- Duration: 110.02s

### Notes
- ResourceWarning for unclosed event loop remains the dominant warning.

## Update (PytestUnraisableExceptionWarning filter attempt)
### Command
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=html -W error::PytestUnraisableExceptionWarning

### Result
- FAILED: pytest could not resolve `PytestUnraisableExceptionWarning` without module qualification.

### Notes
- Error: `AttributeError: module 'builtins' has no attribute 'PytestUnraisableExceptionWarning'`.

## Update (PytestUnraisableExceptionWarning filter qualified)
### Command
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests -v --cov=src --cov-report=term-missing --cov-report=html -W error::pytest.PytestUnraisableExceptionWarning

### Result
- 3993 passed, 21 skipped, 170 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 106.79s

### Notes
- Warnings still surfaced; filter did not convert them into errors.

## Update (targeted warning isolation)
### Commands
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/test_metrics_contract.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/test_provider_timeout_simulation.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/test_render.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/ocr/test_dimension_matching.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/ocr/test_fallback.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/ocr/test_ocr_endpoint.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/resilience/test_adaptive_rate_limiter.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/unit/test_adapter_factory_coverage.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/unit/test_cache_coverage.py -v -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONWARNINGS=default .venv/bin/python -m pytest tests/unit/test_dedup2d_file_storage_s3.py -v -W error::pytest.PytestUnraisableExceptionWarning

### Results
- tests/test_metrics_contract.py: 19 passed, 3 skipped, 1 warning (ResourceWarning: unclosed event loop in test_rejection_reasons_valid).
- tests/test_provider_timeout_simulation.py: 13 passed, 1 warning (ResourceWarning: unclosed event loop in test_high_load_timeout_behavior).
- tests/test_render.py: 3 passed, 0 warnings.
- tests/ocr/test_dimension_matching.py: 30 passed, 0 warnings.
- tests/ocr/test_fallback.py: 18 passed, 0 warnings.
- tests/ocr/test_ocr_endpoint.py: 1 passed, 0 warnings.
- tests/resilience/test_adaptive_rate_limiter.py: 21 passed, 0 warnings.
- tests/unit/test_adapter_factory_coverage.py: 38 passed, 3 warnings (ResourceWarning: unclosed event loop).
- tests/unit/test_cache_coverage.py: 23 passed, 1 warning (ResourceWarning: unclosed event loop; extra warning emitted after summary).
- tests/unit/test_dedup2d_file_storage_s3.py: 16 passed, 1 warning (ResourceWarning: unclosed event loop).

### Notes
- Initial attempt used `tests/test_ocr_endpoint.py` (not found); reran with `tests/ocr/test_ocr_endpoint.py`.

## Update (targeted warning isolation with asyncio debug)
### Commands
- PYTHONASYNCIODEBUG=1 PYTHONWARNINGS=default .venv/bin/python -m pytest tests/test_metrics_contract.py -v -s -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONASYNCIODEBUG=1 PYTHONWARNINGS=default .venv/bin/python -m pytest tests/test_provider_timeout_simulation.py -v -s -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONASYNCIODEBUG=1 PYTHONWARNINGS=default .venv/bin/python -m pytest tests/unit/test_adapter_factory_coverage.py -v -s -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONASYNCIODEBUG=1 PYTHONWARNINGS=default .venv/bin/python -m pytest tests/unit/test_cache_coverage.py -v -s -W error::pytest.PytestUnraisableExceptionWarning
- PYTHONASYNCIODEBUG=1 PYTHONWARNINGS=default .venv/bin/python -m pytest tests/unit/test_dedup2d_file_storage_s3.py -v -s -W error::pytest.PytestUnraisableExceptionWarning

### Results
- tests/test_metrics_contract.py: 19 passed, 3 skipped, 1 warning (ResourceWarning: unclosed event loop in test_rejection_reasons_valid; debug=True).
- tests/test_provider_timeout_simulation.py: 13 passed, 1 warning (ResourceWarning: unclosed event loop in test_high_load_timeout_behavior; debug=True).
- tests/unit/test_adapter_factory_coverage.py: 38 passed, 3 warnings (ResourceWarning in test_dxf_adapter_parse_with_ezdxf, test_stl_adapter_parse_without_trimesh, test_step_adapter_parse_without_occ; debug=True; extra warning printed after summary).
- tests/unit/test_cache_coverage.py: 23 passed, 2 warnings (ResourceWarning in test_redis_healthy_when_connected, test_redis_module_available_check; debug=True).
- tests/unit/test_dedup2d_file_storage_s3.py: 16 passed, 1 warning (ResourceWarning in test_create_s3_storage; debug=True).

### Notes
- Warnings remain unclosed event loop warnings even with asyncio debug enabled.
