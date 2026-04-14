# PR398 Remote Failure Triage Batch 4 Development Plan

Date: 2026-04-14
Branch: `submit/local-main-20260414`
Scope: Remote `CI` regression on Graph2D DXF prediction contract tests

## Background

PR #398 remote `CI` run `24401743395` failed in both Python jobs:

- `tests (3.10)` job `71275155556`
- `tests (3.11)` job `71275155564`

The same three integration tests failed in both jobs:

- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py::test_analyze_dxf_graph2d_prediction_defaults_to_hybrid_config`
- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py::test_analyze_dxf_graph2d_soft_override_defaults_to_graph2d_min_conf`
- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py::test_analyze_dxf_graph2d_model_unavailable_still_attaches_prediction`

Failure symptom:

- test expected `graph2d_prediction.min_confidence == 0.5`
- runtime returned `0.35`

## Root Cause

`src/ml/hybrid_config.py` already changed the Graph2D default threshold from `0.5` to `0.35` in the B4.4 configuration update. The API implementation in `src/api/v1/analyze.py` correctly reads the Graph2D default from Hybrid config when `GRAPH2D_MIN_CONF` is unset.

The integration tests were still hardcoding the old default and also used stub confidences (`0.49`) that were only below the legacy threshold, not below the current threshold.

## Remediation Plan

1. Keep the contract aligned to config-driven defaults instead of hardcoded legacy numbers.
2. Preserve the original behavioral intent:
   - for threshold-blocking cases, use a stub confidence just below the config default
   - for model-unavailable case, still assert the attached threshold metadata is config-derived
3. Avoid tautological assertions that merely recompute runtime logic from the response payload.
4. Re-run the targeted integration contract file locally.
5. Run a read-only `Claude Code CLI` sidecar review for the patch.

## Intended Code Changes

File:

- `tests/integration/test_analyze_dxf_graph2d_prediction_contract.py`

Changes:

- import `get_config()` and derive `GRAPH2D_DEFAULT_MIN_CONF`
- replace hardcoded `0.5` threshold assertions with `GRAPH2D_DEFAULT_MIN_CONF`
- replace hardcoded stub confidence `0.49` with `GRAPH2D_DEFAULT_MIN_CONF - 0.01` in tests that are supposed to stay below threshold
- keep explicit boolean assertions (`passed_threshold is False`, `eligible is False`) so the file remains a real contract test

