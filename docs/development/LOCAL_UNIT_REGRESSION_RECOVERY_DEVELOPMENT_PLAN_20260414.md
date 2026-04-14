# Local Unit Regression Recovery Development Plan

Date: 2026-04-14
Scope: local unit-regression recovery on branch `submit/local-main-20260414`

## Objective

Recover the local regression surface exposed after the earlier CI / workflow fixes, then drive the repository back to a clean `tests/unit` baseline under `.venv311`.

## Problem Clusters

1. `HybridClassifier` behavior drift
   - `_dxf_text` could be referenced before initialization.
   - Guardrailed `graph2d_non_matching_ignored -> filename_only` paths were still nulled by final reject-threshold logic.

2. Helper / contract drift
   - `src/api/v1/analyze.py` no longer exported the Graph2D helper functions expected by unit tests.
   - `scripts/eval_trend.py::load_history()` had already converged on a 3-tuple contract while one helper test still unpacked 2 values.

3. Make / workflow compatibility regressions
   - `code-quality.yml` comment steps still used raw PR comment creation instead of the shared upsert helper.
   - `Makefile` had lost multiple legacy wrapper targets asserted by regression tests:
     - workflow-guardrail wrappers
     - hybrid calibration wrappers
     - hybrid blind wrappers
     - soft-mode smoke wrappers
     - new-modules wrapper targets

4. Assistant factory drift
   - `create_semantic_retriever(use_transformers=False)` started selecting `DomainEmbeddingProvider` by default, breaking the explicit lightweight-baseline contract expected by tests.

## Implemented Changes

### 1. Lint / workflow / local runner stabilization

- Added narrow `.flake8` per-file ignores for known legacy files:
  - `src/core/materials/cost.py`
  - `src/core/materials/data_models.py`
  - `src/ml/embeddings/model.py`
  - `src/ml/monitoring/auto_remediation.py`
- Updated `.github/workflows/code-quality.yml` to use `scripts/ci/comment_pr_utils.js` marker-based PR comment upsert flow for both quality and mypy report comments.
- Updated `scripts/test_with_local_api.sh` to prefer `.venv311` over the older `.venv` / Python 3.13 environment.

### 2. Graph2D helper recovery

- Re-extracted and reused:
  - `_enrich_graph2d_prediction()`
  - `_build_graph2d_soft_override_suggestion()`
- The helpers are now defined in `src/api/v1/analyze.py` and reused by the route logic, restoring the unit-test import contract without reintroducing duplicated inline branches.

### 3. Hybrid classifier regression fixes

- Initialized `_dxf_text = ""` before the optional text-content path in `src/ml/hybrid_classifier.py`.
- Narrowed final rejection so it does not wipe the explicit guardrail-protected path where:
  - Graph2D introduced a non-matching label
  - Graph2D was ignored for fusion
  - filename remained the only source
- Kept general rejection-policy behavior intact for normal low-confidence results.

### 4. Eval-trend test contract alignment

- Updated `tests/unit/test_graph2d_eval_helpers.py` to unpack the current 3-value `load_history()` return contract and assert the empty history-sequence slot explicitly.

### 5. Makefile compatibility recovery

- Reintroduced workflow guardrail / health wrapper targets.
- Added hybrid calibration targets:
  - `hybrid-calibrate-confidence`
  - `hybrid-calibration-gate`
  - `update-hybrid-calibration-baseline`
  - `refresh-hybrid-calibration-baseline`
  - `validate-hybrid-calibration-workflow`
- Added hybrid blind targets and wrappers:
  - `hybrid-blind-build-synth`
  - `hybrid-blind-eval`
  - `hybrid-blind-gate`
  - `hybrid-blind-history-bootstrap`
  - `hybrid-blind-drift-alert`
  - `hybrid-blind-drift-suggest-thresholds`
  - `hybrid-blind-drift-apply-suggestion-gh`
  - `hybrid-blind-drift-activate`
  - `hybrid-blind-strict-real`
  - `hybrid-blind-strict-real-e2e-gh`
  - `hybrid-blind-strict-real-template-gh`
  - `hybrid-blind-strict-real-apply-gh-vars`
  - `validate-hybrid-blind-strict-real-e2e-gh`
  - `validate-hybrid-blind-workflow`
- Added soft-mode smoke targets and wrappers:
  - `validate-soft-mode-smoke`
  - `validate-soft-mode-smoke-auto-pr`
  - `render-soft-mode-smoke-summary`
  - `validate-render-soft-mode-smoke-summary`
  - `validate-soft-mode-smoke-workflow`
  - `validate-soft-mode-smoke-comment`
  - `soft-mode-smoke-comment-pr`
  - `validate-soft-mode-smoke-comment-pr`
- Added missing render wrappers:
  - `render-hybrid-blind-strict-real-dispatch-summary`
  - `render-hybrid-superpass-dispatch-summary`
  - `render-hybrid-superpass-validation-summary`
  - matching `validate-render-*` targets
- Added missing new-module compatibility targets:
  - `test-new-modules`
  - `smoke-new-modules`
  - `test-cost`
  - `test-ai-intelligence`
  - `test-diff`
  - `test-pointcloud`
  - `test-embeddings`
  - `test-copilot`
  - `test-training-scripts`
- Expanded `validate-hybrid-superpass-workflow` to include the soft-mode smoke / calibration-related tests expected by regression coverage.

### 6. Missing strict-real template utility

- Added `scripts/ci/print_hybrid_blind_strict_real_gh_template.py`
- Added `tests/unit/test_print_hybrid_blind_strict_real_gh_template.py`

### 7. Semantic retriever factory compatibility

- Changed `create_semantic_retriever()` signature to:
  - `use_transformers=None` => auto mode, keep DomainEmbeddingProvider preference
  - `use_transformers=True` => allow sentence-transformers fallback
  - `use_transformers=False` => explicit lightweight SimpleEmbeddingProvider path
- This preserves current default behavior while restoring backward compatibility for explicit lightweight callers.

## Files Touched

- `.flake8`
- `.github/workflows/code-quality.yml`
- `Makefile`
- `scripts/test_with_local_api.sh`
- `scripts/ci/print_hybrid_blind_strict_real_gh_template.py`
- `src/api/v1/analyze.py`
- `src/core/assistant/semantic_retrieval.py`
- `src/ml/hybrid_classifier.py`
- `src/ml/text_extractor.py`
- `tests/unit/test_graph2d_eval_helpers.py`
- `tests/unit/test_print_hybrid_blind_strict_real_gh_template.py`

## Notes

- `Claude Code CLI` was available and used only as a read-only sidecar reviewer.
- Main debugging, edits, and verification stayed on the local toolchain and did not depend on Claude CLI being available.
