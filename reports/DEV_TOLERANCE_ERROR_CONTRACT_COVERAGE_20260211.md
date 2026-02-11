# DEV_TOLERANCE_ERROR_CONTRACT_COVERAGE_20260211

## Summary
- Added explicit tolerance API error-path coverage for unsupported or invalid inputs.
- Covered both integration-level behavior and contract-level response shape.

## Changes
- Added `tests/integration/test_tolerance_api_errors.py`
  - `/api/v1/tolerance/it` unsupported grade -> `400`
  - `/api/v1/tolerance/it` invalid diameter -> `422`
  - `/api/v1/tolerance/limit-deviations` unknown symbol -> `404`
  - `/api/v1/tolerance/fit` unsupported fit code -> `404`
- Updated `tests/contract/test_api_contract.py`
  - Added 3 tolerance error-path checks in `TestKnowledgeApiContracts`.
- Updated `Makefile`
  - Included `tests/integration/test_tolerance_api_errors.py` in `make test-tolerance`.

## Validation
- `pytest -q tests/integration/test_tolerance_api_errors.py`
  - Result: `4 passed`
- `pytest -q tests/contract/test_api_contract.py::TestKnowledgeApiContracts::test_tolerance_it_endpoint_rejects_unsupported_grade tests/contract/test_api_contract.py::TestKnowledgeApiContracts::test_tolerance_limit_deviations_not_found_response_shape tests/contract/test_api_contract.py::TestKnowledgeApiContracts::test_tolerance_fit_endpoint_not_found_response_shape`
  - Result: `3 passed`
- `make test-tolerance`
  - Result: `48 passed`

