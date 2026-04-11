# Hybrid Superpass Gate Development & Validation (2026-03-13)

## Scope
- Land executable "superpass" benchmark gate for Hybrid outputs.
- Wire gate into `Makefile` and `.github/workflows/evaluation-report.yml`.
- Add regression tests for script logic and workflow wiring.

## Implemented

### 1) Superpass gate script + config
- Added [check_hybrid_superpass_targets.py](/Users/huazhou/Downloads/Github/cad-ml-platform/scripts/ci/check_hybrid_superpass_targets.py)
- Added [hybrid_superpass_targets.yaml](/Users/huazhou/Downloads/Github/cad-ml-platform/config/hybrid_superpass_targets.yaml)

Checks covered:
- `hybrid_accuracy >= min_hybrid_accuracy`
- `hybrid_gain_vs_graph2d >= min_hybrid_gain_vs_graph2d`
- `calibration_ece <= max_calibration_ece`

Missing input behavior:
- `missing_mode=skip`: warning + skipped checks
- `missing_mode=fail`: gate fails

Output report fields:
- `status`
- `failures`
- `warnings`
- `checks`
- `thresholds`

### 2) Local Make integration
- Updated [Makefile](/Users/huazhou/Downloads/Github/cad-ml-platform/Makefile)
- Added target: `hybrid-superpass-gate`
- Added/updated vars:
  - `HYBRID_SUPERPASS_GATE_REPORT_JSON`
  - `HYBRID_SUPERPASS_CALIBRATION_JSON`
  - `HYBRID_SUPERPASS_CONFIG`
  - `HYBRID_SUPERPASS_OUTPUT_JSON`
  - `HYBRID_SUPERPASS_MISSING_MODE`

### 3) CI workflow integration
- Updated [evaluation-report.yml](/Users/huazhou/Downloads/Github/cad-ml-platform/.github/workflows/evaluation-report.yml)

Added dispatch inputs:
- `hybrid_superpass_enable`
- `hybrid_superpass_missing_mode`
- `hybrid_superpass_fail_on_failed`

Added env vars:
- `HYBRID_SUPERPASS_ENABLE`
- `HYBRID_SUPERPASS_CONFIG`
- `HYBRID_SUPERPASS_OUTPUT_JSON`
- `HYBRID_SUPERPASS_MISSING_MODE`
- `HYBRID_SUPERPASS_GATE_REPORT_JSON`
- `HYBRID_SUPERPASS_CALIBRATION_JSON`
- `HYBRID_SUPERPASS_FAIL_ON_FAILED`

Added CI steps:
- `Check Hybrid superpass gate (optional)`
- `Evaluate Hybrid superpass strict mode (optional)`
- `Fail workflow when Hybrid superpass strict check requires blocking`
- `Upload Hybrid superpass gate artifact`

Job summary now includes superpass strict fields.

### 4) Tests
- Added [test_check_hybrid_superpass_targets.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_check_hybrid_superpass_targets.py)
- Added [test_evaluation_report_workflow_hybrid_superpass_step.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py)
- Added [test_hybrid_superpass_workflow_integration.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_hybrid_superpass_workflow_integration.py)
- Updated [test_hybrid_calibration_make_targets.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_hybrid_calibration_make_targets.py)
- Updated [test_evaluation_report_workflow_graph2d_extensions.py](/Users/huazhou/Downloads/Github/cad-ml-platform/tests/unit/test_evaluation_report_workflow_graph2d_extensions.py)

## Validation

Executed:

```bash
pytest -q \
  tests/unit/test_check_hybrid_superpass_targets.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_calibration_make_targets.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Result:
- `37 passed`

Script smoke (missing inputs):

```bash
python3 scripts/ci/check_hybrid_superpass_targets.py \
  --hybrid-blind-gate-report /tmp/not_found_gate.json \
  --hybrid-calibration-json /tmp/not_found_calibration.json \
  --missing-mode skip \
  --output reports/experiments/20260313/hybrid_superpass_gate_smoke_skip.json
```
- `status=passed` with warnings (as expected).

```bash
python3 scripts/ci/check_hybrid_superpass_targets.py \
  --hybrid-blind-gate-report /tmp/not_found_gate.json \
  --hybrid-calibration-json /tmp/not_found_calibration.json \
  --missing-mode fail \
  --output reports/experiments/20260313/hybrid_superpass_gate_smoke_fail.json
```
- exit code `1` with failed status (as expected).

## Notes
- Superpass flow is optional by default (`HYBRID_SUPERPASS_ENABLE=false`).
- To enforce blocking policy in CI: enable superpass + set `hybrid_superpass_fail_on_failed=true`.
