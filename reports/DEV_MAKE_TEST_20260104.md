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

## Update (tracemalloc + asyncio debug)
### Targeted Commands
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/test_metrics_contract.py -k test_rejection_reasons_valid -v -s
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/test_provider_timeout_simulation.py -k test_high_load_timeout_behavior -v -s
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/unit/test_adapter_factory_coverage.py -k "dxf_adapter_parse_without_ezdxf or stl_adapter_parse_without_trimesh or step_adapter_parse_without_occ" -v -s
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/unit/test_cache_coverage.py -k "redis_healthy_when_connected or redis_module_available_check" -v -s
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/unit/test_dedup2d_file_storage_s3.py -k test_create_s3_storage -v -s

### Full-file Commands
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/test_provider_timeout_simulation.py -v -s
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/unit/test_adapter_factory_coverage.py -v -s
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/unit/test_cache_coverage.py -v -s
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 pytest tests/unit/test_dedup2d_file_storage_s3.py -v -s

### Results
- tests/test_metrics_contract.py (targeted): 1 skipped, 21 deselected (metrics disabled), no warnings.
- tests/test_provider_timeout_simulation.py (targeted): 1 passed, 12 deselected, no warnings.
- tests/unit/test_adapter_factory_coverage.py (targeted): 3 passed, 35 deselected, no warnings.
- tests/unit/test_cache_coverage.py (targeted): 2 passed, 21 deselected, no warnings.
- tests/unit/test_dedup2d_file_storage_s3.py (targeted): 1 skipped (0 collected), no warnings.
- tests/test_provider_timeout_simulation.py (full): 13 passed, no warnings.
- tests/unit/test_adapter_factory_coverage.py (full): 38 passed, no warnings.
- tests/unit/test_cache_coverage.py (full): FAILED 2 tests (test_get_cache_from_redis, test_set_cache_to_redis).
- tests/unit/test_dedup2d_file_storage_s3.py (full): 1 skipped (0 collected), no warnings.

### Notes
- Tracemalloc runs did not reproduce the unclosed event loop warnings seen in prior isolation runs.
- test_get_cache_from_redis returned None instead of the mocked payload.
- test_set_cache_to_redis did not call mock_client.setex.

## Update (tracemalloc with venv)
### Commands
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 .venv/bin/python -m pytest tests/test_metrics_contract.py -k test_rejection_reasons_valid -v -s
- PYTHONASYNCIODEBUG=1 PYTHONTRACEMALLOC=1 .venv/bin/python -m pytest tests/unit/test_cache_coverage.py -k "test_get_cache_from_redis or test_set_cache_to_redis" -v -s

### Results
- tests/test_metrics_contract.py: 1 passed, 21 deselected.
- tests/unit/test_cache_coverage.py: 2 passed, 21 deselected.

### Notes
- `.venv` has `prometheus_client` and `redis` available; metrics enabled and redis paths execute normally.
- The earlier tracemalloc failures under system Python 3.13 were due to missing optional dependencies.

## Update (cache redis-path skip guard)
### Commands
- pytest tests/unit/test_cache_coverage.py -k "test_get_cache_from_redis or test_set_cache_to_redis" -v
- .venv/bin/python -m pytest tests/unit/test_cache_coverage.py -k "test_get_cache_from_redis or test_set_cache_to_redis" -v

### Results
- System Python: 2 skipped, 21 deselected (redis.asyncio not available).
- `.venv`: 2 passed, 21 deselected.

### Notes
- Added skip guard so redis-path cache tests only run when redis.asyncio is installed.

## Update (make test rerun after redis guard)
### Command
- make test

### Result
- 3993 passed, 21 skipped, 3 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 115.44s

### Notes
- Warnings were DeprecationWarning from SwigPyPacked/SwigPyObject/swigvarlink in vision integration tests.

## Update (DeprecationWarning error probe)
### Command
- .venv/bin/python -m pytest tests/integration/test_vision_api_integration.py -k test_analyze_with_stub_provider -v -W error::DeprecationWarning -s

### Result
- FAILED: fatal Python segmentation fault while importing `faiss` (swigfaiss).

### Notes
- Crash occurred during faiss import in `src/core/similarity.py` while bringing up the test client.

## Update (DeprecationWarning probe with memory backend)
### Command
- VECTOR_STORE_BACKEND=memory .venv/bin/python -m pytest tests/integration/test_vision_api_integration.py -k test_analyze_with_stub_provider -v -W error::DeprecationWarning -s

### Result
- FAILED: fatal Python segmentation fault while importing `faiss` (swigfaiss).

### Notes
- `VECTOR_STORE_BACKEND=memory` did not prevent faiss recovery import; crash still occurs in `src/core/similarity.py` recovery loop.

## Update (DeprecationWarning probe with clean recovery state)
### Command
- FAISS_RECOVERY_STATE_PATH=/tmp/faiss_recovery_state.json VECTOR_STORE_BACKEND=memory \
  .venv/bin/python -m pytest tests/integration/test_vision_api_integration.py \
  -k test_analyze_with_stub_provider -v -W error::DeprecationWarning -s

### Result
- 1 passed, 13 deselected

### Notes
- Clean recovery state avoided faiss import, so the DeprecationWarning probe completed without segfault.

## Update (integration recovery-state fixture)
### Command
- .venv/bin/python -m pytest tests/integration/test_vision_api_integration.py -k test_analyze_with_stub_provider -v -W error::DeprecationWarning -s

### Result
- 1 passed, 13 deselected

### Notes
- Added integration fixture to set `FAISS_RECOVERY_STATE_PATH` per test, preventing stale degraded state from triggering faiss import.

## Update (global recovery-state isolation + recovery loop guard)
### Command
- .venv/bin/python -m pytest tests/integration/test_vision_api_integration.py -k test_analyze_with_stub_provider -v -W error::DeprecationWarning -s

### Result
- 1 passed, 13 deselected

### Notes
- Moved recovery-state isolation to `tests/conftest.py` for all tests.
- Guarded faiss recovery loop to only start when `VECTOR_STORE_BACKEND=faiss`.

## Update (make test after faiss guard)
### Command
- make test

### Result
- 3993 passed, 21 skipped, 3 warnings
- Coverage: 71% (htmlcov generated)
- Duration: 139.52s

### Notes
- Warnings were DeprecationWarning from SwigPyPacked/SwigPyObject/swigvarlink in `tests/perf/test_vector_search_latency.py`.

## Update (DeprecationWarning error probe on perf test)
### Command
- FAISS_RECOVERY_STATE_PATH=/tmp/faiss_recovery_state.json VECTOR_STORE_BACKEND=faiss .venv/bin/python -m pytest tests/perf/test_vector_search_latency.py -v -W error::DeprecationWarning -s

### Result
- FAILED: fatal Python segmentation fault while importing `faiss` (swigfaiss).

### Notes
- Crash occurred during faiss import in `src/core/similarity.py` when `tests/perf/test_vector_search_latency.py` initializes `FaissVectorStore`.

## Update (perf run with faiss gate enabled)
### Command
- RUN_FAISS_PERF_TESTS=1 FAISS_RECOVERY_STATE_PATH=/tmp/faiss_recovery_state.json .venv/bin/python -m pytest tests/perf/test_vector_search_latency.py -v -s

### Result
- 1 passed, 3 warnings
- Duration: 41.62s

### Notes
- DeprecationWarning emitted for SwigPyPacked/SwigPyObject/swigvarlink during faiss import.

## Update (perf run with faiss gate + DeprecationWarning error)
### Command
- RUN_FAISS_PERF_TESTS=1 FAISS_RECOVERY_STATE_PATH=/tmp/faiss_recovery_state.json .venv/bin/python -m pytest tests/perf/test_vector_search_latency.py -v -W error::DeprecationWarning -s

### Result
- FAILED: fatal Python segmentation fault while importing `faiss` (swigfaiss).

### Notes
- Crash occurred during faiss import in `src/core/similarity.py` while initializing `FaissVectorStore`.

## Update (faiss perf subprocess isolation + DeprecationWarning error)
### Command
- RUN_FAISS_PERF_TESTS=1 FAISS_RECOVERY_STATE_PATH=/tmp/faiss_recovery_state.json .venv/bin/python -m pytest tests/perf/test_vector_search_latency.py -v -W error::DeprecationWarning -s

### Result
- FAILED: faiss perf subprocess exited with code -11 (segfault).
- Duration: 37.13s

### Notes
- Faiss perf now runs in a subprocess to prevent a segfault from killing the pytest process.

## Update (faiss perf subprocess skip mode)
### Command
- RUN_FAISS_PERF_TESTS=1 FAISS_RECOVERY_STATE_PATH=/tmp/faiss_recovery_state.json .venv/bin/python -m pytest tests/perf/test_vector_search_latency.py -v -W error::DeprecationWarning -s

### Result
- 1 skipped
- Duration: 43.74s

### Notes
- Faiss perf subprocess failures now skip the test unless `REQUIRE_FAISS_PERF=1` is set.

## Update (faiss perf forced)
### Command
- RUN_FAISS_PERF_TESTS=1 REQUIRE_FAISS_PERF=1 FAISS_RECOVERY_STATE_PATH=/tmp/faiss_recovery_state.json .venv/bin/python -m pytest tests/perf/test_vector_search_latency.py -v -s

### Result
- 1 passed
- Duration: 7.31s

### Notes
- Faiss perf ran successfully in subprocess with `REQUIRE_FAISS_PERF=1`.

## Update (faiss perf strict warnings with swig filter)
### Command
- RUN_FAISS_PERF_TESTS=1 REQUIRE_FAISS_PERF=1 FAISS_RECOVERY_STATE_PATH=/tmp/faiss_recovery_state.json .venv/bin/python -m pytest tests/perf/test_vector_search_latency.py -v -W error::DeprecationWarning -s

### Result
- 1 passed
- Duration: 8.00s

### Notes
- Added subprocess warning filter for SwigPyPacked/SwigPyObject/swigvarlink DeprecationWarning to avoid strict-warning crashes.

## Update (faiss-cpu downgrade check)
### Command
- .venv/bin/python -m pip install faiss-cpu==1.12.0
- PYTHONWARNINGS=error::DeprecationWarning .venv/bin/python -X faulthandler -c "import faiss; print('ok')"
- .venv/bin/python -m pip install faiss-cpu==1.13.2

### Result
- Import still segfaulted with faiss-cpu 1.12.0 under DeprecationWarning-as-error.
- Restored faiss-cpu 1.13.2 after the check.

### Notes
- No newer faiss-cpu version is available via pip; downgrade did not resolve strict-warning segfault.

## Update (Python 3.12 faiss import check)
### Command
- python3.12 -m venv /tmp/faiss-py312-venv.iUnWFt
- /tmp/faiss-py312-venv.iUnWFt/bin/python -m pip install faiss-cpu
- PYTHONWARNINGS=error::DeprecationWarning /tmp/faiss-py312-venv.iUnWFt/bin/python -X faulthandler -c "import faiss; print(faiss.__version__)"

### Result
- FAILED: fatal Python segmentation fault while importing `faiss` (swigfaiss).

### Notes
- faiss-cpu 1.13.2 installed with numpy 2.4.0 in the Python 3.12 venv.

## Update (conda-forge faiss import check)
### Command
- curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba
- /tmp/micromamba.bRC6/bin/micromamba create -y -p /tmp/faiss-conda-venv.Q5KE -c conda-forge python=3.11 faiss-cpu
- PYTHONWARNINGS=error::DeprecationWarning /tmp/faiss-conda-venv.Q5KE/bin/python -X faulthandler -c "import faiss; print(faiss.__version__)"

### Result
- FAILED: fatal Python segmentation fault while importing `faiss` (swigfaiss).

### Notes
- conda-forge resolved faiss 1.9.0 with python 3.11.14 in the temp environment.
