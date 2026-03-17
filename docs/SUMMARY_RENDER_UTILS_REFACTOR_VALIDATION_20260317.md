# Summary Render Utils Refactor Validation

Date: 2026-03-17

## Goal

Extract shared JSON/boolean and markdown section helper logic from CI renderers so:

- direct renderer behavior stays unchanged
- repeated `read json / boolish / zeroish / top list / section rendering` logic stops drifting
- renderer validation targets cover the shared helper explicitly

## Added

- `scripts/ci/summary_render_utils.py`
- `tests/unit/test_summary_render_utils.py`

Shared markdown helpers added:

- `append_markdown_section(...)`
- `append_failure_diagnostics_section(...)`
- `render_inline_items(...)`

Refactored renderers:

- `scripts/ci/render_hybrid_blind_strict_real_dispatch_summary.py`
- `scripts/ci/render_hybrid_superpass_dispatch_summary.py`
- `scripts/ci/render_hybrid_superpass_validation_summary.py`
- `scripts/ci/render_soft_mode_smoke_summary.py`

Validation targets updated:

- `validate-render-soft-mode-smoke-summary`
- `validate-render-hybrid-blind-strict-real-dispatch-summary`
- `validate-render-hybrid-superpass-dispatch-summary`
- `validate-render-hybrid-superpass-validation-summary`

## Validation

```bash
pytest -q \
  tests/unit/test_summary_render_utils.py \
  tests/unit/test_render_hybrid_blind_strict_real_dispatch_summary.py \
  tests/unit/test_render_hybrid_superpass_dispatch_summary.py \
  tests/unit/test_render_hybrid_superpass_validation_summary.py \
  tests/unit/test_render_soft_mode_smoke_summary.py \
  tests/unit/test_hybrid_calibration_make_targets.py
```

```bash
make validate-render-hybrid-blind-strict-real-dispatch-summary
make validate-render-hybrid-superpass-dispatch-summary
make validate-render-hybrid-superpass-validation-summary
make validate-render-soft-mode-smoke-summary
```

Results:

- `pytest -q tests/unit/test_summary_render_utils.py tests/unit/test_render_hybrid_blind_strict_real_dispatch_summary.py tests/unit/test_render_hybrid_superpass_dispatch_summary.py tests/unit/test_render_hybrid_superpass_validation_summary.py tests/unit/test_render_soft_mode_smoke_summary.py tests/unit/test_hybrid_calibration_make_targets.py` -> `62 passed`
- `make validate-render-hybrid-blind-strict-real-dispatch-summary` -> `50 passed`
- `make validate-render-hybrid-superpass-dispatch-summary` -> `50 passed`
- `make validate-render-hybrid-superpass-validation-summary` -> `50 passed`
- `make validate-render-soft-mode-smoke-summary` -> `50 passed`

Follow-up validation after extracting the shared `Failure Diagnostics` block from both dispatch renderers:

```bash
pytest -q \
  tests/unit/test_summary_render_utils.py \
  tests/unit/test_render_hybrid_blind_strict_real_dispatch_summary.py \
  tests/unit/test_render_hybrid_superpass_dispatch_summary.py
```

```bash
pytest -q \
  tests/unit/test_summary_render_utils.py \
  tests/unit/test_render_hybrid_superpass_validation_summary.py \
  tests/unit/test_render_soft_mode_smoke_summary.py
```

```bash
make validate-render-hybrid-blind-strict-real-dispatch-summary
make validate-render-hybrid-superpass-dispatch-summary
make validate-render-hybrid-superpass-validation-summary
make validate-render-soft-mode-smoke-summary
```

Follow-up results:

- `pytest -q tests/unit/test_summary_render_utils.py tests/unit/test_render_hybrid_blind_strict_real_dispatch_summary.py tests/unit/test_render_hybrid_superpass_dispatch_summary.py` -> `17 passed`
- `pytest -q tests/unit/test_summary_render_utils.py tests/unit/test_render_hybrid_superpass_validation_summary.py tests/unit/test_render_soft_mode_smoke_summary.py` -> `17 passed`
- `make validate-render-hybrid-blind-strict-real-dispatch-summary` -> `52 passed`
- `make validate-render-hybrid-superpass-dispatch-summary` -> `52 passed`
- `make validate-render-hybrid-superpass-validation-summary` -> `52 passed`
- `make validate-render-soft-mode-smoke-summary` -> `52 passed`

## Notes

- This is an internal refactor only. No workflow YAML changes were required.
- The `Verdict / Snapshot` headers and row ordering remained stable across all 4 renderers.
- The compact `### Failure Diagnostics` layout also remained stable after being moved behind the shared helper.
- Direct script execution is preserved via a small import fallback for `scripts.ci.summary_render_utils`.
