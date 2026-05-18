# CAD ML Reviewed Manifest Hybrid Gate Verification

Date: 2026-05-13

## Scope

Validated the optional hybrid blind gate requirement that release evaluation must
consume the merged reviewed benchmark manifest. The checks cover gate behavior,
workflow variable wiring, CLI wiring, and regression coverage around the existing
hybrid superpass workflow.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/ci/check_hybrid_blind_gate.py \
  tests/unit/test_hybrid_blind_gate_check.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py
```

```bash
.venv311/bin/flake8 \
  scripts/ci/check_hybrid_blind_gate.py \
  tests/unit/test_hybrid_blind_gate_check.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_hybrid_blind_gate_check.py \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py
```

```bash
git diff --check
```

## Results

- Python compile passed for the hybrid blind gate and touched tests.
- Flake8 passed for the hybrid blind gate and touched tests.
- Targeted pytest passed: `11 passed, 7 warnings in 3.02s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings imported
  through existing DXF evaluation helpers.

## Verified Behavior

- The hybrid blind gate report includes `input_summary.manifest_source`.
- The hybrid blind gate report includes `input_summary.required_manifest_source`.
- `--require-manifest-source reviewed_benchmark_manifest` fails when the actual
  manifest source is `configured`.
- `--require-manifest-source reviewed_benchmark_manifest` passes when the actual
  manifest source is `reviewed_benchmark_manifest`.
- The workflow exposes `HYBRID_BLIND_REQUIRE_REVIEWED_MANIFEST`.
- The workflow passes `steps.hybrid_blind_eval.outputs.manifest_source` to the hybrid
  blind gate.
- The workflow adds
  `--require-manifest-source reviewed_benchmark_manifest` when
  `HYBRID_BLIND_REQUIRE_REVIEWED_MANIFEST=true`.
