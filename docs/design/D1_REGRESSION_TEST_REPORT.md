# D1 Regression Test Report

**Date:** 2026-04-13
**Branch:** main (post 9-merge integration)
**Environment:** Python 3.9.6, macOS Darwin 25.4.0, pytest 7.4.3

---

## 1. Test Execution Summary

| Metric | Count |
|--------|-------|
| **Total collected** | 10,429 (excluding 3 collection errors) |
| **Passed** | 10,237 |
| **Failed** | 68 |
| **Skipped** | 124 |
| **Collection Errors** | 3 |
| **Pass Rate** | 98.1% (of collected) |
| **Wall Time** | 4m 01s |

---

## 2. Collection Errors (CRITICAL)

Three test files failed to collect (import-time errors):

| Test File | Error | Root Cause |
|-----------|-------|------------|
| `tests/unit/test_graph_augmentations.py` | `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed in test environment |
| `tests/unit/test_contrastive_pretrain.py` | `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed in test environment |
| `tests/unit/test_analyze_graph2d_gate_helpers.py` | `ImportError: cannot import name '_build_graph2d_soft_override_suggestion' from 'src.api.v1.analyze'` | Missing function in analyze module |

**Recommendation:** Graph2D tests should use `pytest.importorskip("torch")` instead of bare `import torch` at module level. The `_build_graph2d_soft_override_suggestion` function appears to be referenced in tests but not yet defined in `src/api/v1/analyze.py`.

---

## 3. Per-Module Test Results

| Module | Tests | Passed | Failed | Skipped | Status |
|--------|-------|--------|--------|---------|--------|
| Materials (`test_materials_*.py`) | 183 | 183 | 0 | 0 | PASS |
| vLLM (`test_vllm_*.py` + integration) | 75 | 75 | 0 | 0 | PASS |
| Fusion/Calibration (`test_hybrid_fusion_strategies.py`, `test_hybrid_calibration.py`) | 64 | 64 | 0 | 0 | PASS |
| LLM Providers (`test_llm_providers.py`) | 32 | 32 | 0 | 0 | PASS |
| Graph2D (`test_graph_augmentations.py`, `test_contrastive_pretrain.py`) | 0 | 0 | 0 | 0 | COLLECTION ERROR (torch) |
| Integration tests | 185 | 185 | 0 | 10 | PASS |
| Makefile targets (`test_makefile_targets.py`) | 3 | 0 | 3 | 0 | FAIL |
| Hybrid calibration make targets | 35 | 0 | 35 | 0 | FAIL |
| Workflow file health make targets | 14 | 0 | 14 | 0 | FAIL |
| Auth middleware integration | 5 | 0 | 5 | 0 | FAIL |
| Compare endpoint | 5 | 0 | 5 | 0 | FAIL |
| Semantic retrieval | 1 | 0 | 1 | 0 | FAIL |
| Other unit/integration | ~9,827 | ~9,698 | 5 | ~114 | PASS (mostly) |

---

## 4. Failure Analysis (68 Failures)

### 4.1 Makefile / CI target tests (52 failures)

The bulk of failures (52/68) are in Makefile target validation tests:

- `test_hybrid_calibration_make_targets.py` — 35 failures
- `test_workflow_file_health_make_target.py` — 14 failures
- `test_makefile_targets.py` — 3 failures

These tests validate that Makefile targets contain expected flags and invoke expected test files. Likely caused by Makefile changes not yet reflected in tests, or vice versa.

### 4.2 Auth middleware tests (5 failures)

- `test_integration_auth_middleware.py` — 5 failures
- Tests for token validation, claims checking, tenant matching

### 4.3 Compare endpoint tests (5 failures)

- `test_compare_endpoint.py` — 5 failures
- Tests for `/compare` API endpoint functionality

### 4.4 Other failures (6 failures)

- `test_additional_workflow_comment_helper_adoption.py` — 2 failures (code quality workflow comment helpers)
- `test_filename_classifier.py::test_hybrid_prefers_graph2d_when_filename_low` — 1 failure
- `test_graph2d_eval_helpers.py::test_eval_trend_load_history_recognizes_explicit_ocr_type` — 1 failure
- `test_semantic_retrieval.py::TestCreateSemanticRetriever::test_create_with_simple_provider` — 1 failure

---

## 5. Import Verification Results

| Module | Import Statement | Status |
|--------|-----------------|--------|
| Materials (classify) | `from src.core.materials.classify import classify_material_detailed` | OK |
| Materials (cost) | `from src.core.materials.cost import get_material_cost` | OK |
| Materials (compatibility) | `from src.core.materials.compatibility import check_full_compatibility` | OK |
| Materials (equivalence) | `from src.core.materials.equivalence import get_material_equivalence` | OK |
| Materials (export) | `from src.core.materials.export import export_materials_csv` | OK |
| Materials (backward compat) | `from src.core.materials.classifier import classify_material_detailed` | OK |
| Vision (`from src.core.vision import *`) | Wildcard import | FAIL — `AttributeError: module 'src.core.vision' has no attribute 'InMemoryAuditStore'` |
| LLM Providers | `from src.core.assistant.llm_providers import VLLMProvider, get_best_available_provider` | OK |
| Graph augmentations | `from src.ml.graph_augmentations import node_feature_masking, ...` | FAIL — `No module named 'torch'` |
| Graph2D model | `from src.ml.train.model_2d import GraphEncoder, ...` | FAIL — `No module named 'torch'` (transitive) |
| Circular imports | `import src.core; import src.ml; import src.api` | OK (no circular imports) |
| Core dependencies | `import yaml; import ezdxf; import numpy; import sklearn` | OK |

**Notes:**
- Vision `import *` fails due to `__all__` exporting `InMemoryAuditStore` which is not defined in the module. Specific named imports work fine.
- Graph2D modules require PyTorch, which is expected (optional dependency), but the modules lack graceful fallback.

---

## 6. Code Quality Results (flake8)

### Batch 1: `src/core/materials/`, `src/ml/graph_augmentations.py`, `src/core/assistant/llm_providers.py`

- **79 warnings** — all are `F601: dictionary key repeated with different values` in `src/core/materials/data_models.py`
- Duplicate dictionary keys: `60Si2Mn`, `50CrVA`, `AgCdO`, `AgSnO2`, `GCr15`, `GCr15SiMn`, `20CrMnTi`, `YG8`, `YT15`, `QAl9-4`, `QBe2`
- **Risk:** Later dictionary entries silently overwrite earlier ones, potentially losing material data.

### Batch 2: `src/core/assistant/prompts/`, `src/core/ocr/providers/vllm_ocr_enhancer.py`, `src/core/vision/providers/vllm_vision.py`

- **0 warnings** — clean

---

## 7. Script Dry-Run Results

| Script | Status | Notes |
|--------|--------|-------|
| `scripts/pretrain_graph2d_contrastive.py --dry-run` | FAIL | `ModuleNotFoundError: No module named 'torch'` |
| `scripts/finetune_graph2d_from_pretrained.py --dry-run` | FAIL | `ModuleNotFoundError: No module named 'torch'` |
| `scripts/benchmark_vllm_quantization.py --dry-run` | PASS | Benchmark completed successfully, report generated |

---

## 8. Issues Summary

### Critical (3)
1. **`_build_graph2d_soft_override_suggestion` missing** — referenced in test but not defined in `src.api.v1.analyze`
2. **Vision `__all__` export error** — `InMemoryAuditStore` listed in `__all__` but not available
3. **79 duplicate dictionary keys** in `src/core/materials/data_models.py` (data silently lost)

### High (2)
4. **68 test failures** across Makefile target validation, auth middleware, compare endpoint, and other modules
5. **Graph2D modules lack torch import guards** — bare `import torch` at module level prevents graceful degradation

### Medium (1)
6. **Graph2D scripts and tests require PyTorch** — no `pytest.importorskip` or try/except guards

---

## 9. Recommendations

1. **Fix `__all__` in `src/core/vision/__init__.py`** — remove or properly import `InMemoryAuditStore`
2. **Add `_build_graph2d_soft_override_suggestion`** to `src/api/v1/analyze.py` or update the test
3. **Deduplicate material keys** in `data_models.py` — consolidate entries to prevent data loss
4. **Add `pytest.importorskip("torch")`** to Graph2D test files instead of bare `import torch`
5. **Investigate Makefile target test failures** — likely Makefile and tests are out of sync after the 9-merge integration
6. **Investigate auth middleware failures** — may indicate a regression in JWT/token handling
7. **Add `--dry-run` guard** for PyTorch in Graph2D training scripts (try/except with helpful error message)
