# A3: Test Coverage for Critical Modules

## Overview

Phase A3 adds comprehensive unit tests for modules split/refactored in A1-A2,
plus key ML and integration modules that lacked coverage.

## Coverage Targets

| Module | File | Target | Tests Added |
|--------|------|--------|-------------|
| Materials - Classify | `src/core/materials/classify.py` | 90%+ | 28 |
| Materials - Cost | `src/core/materials/cost.py` | 85%+ | 18 |
| Materials - Compatibility | `src/core/materials/compatibility.py` | 85%+ | 18 |
| Materials - Equivalence | `src/core/materials/equivalence.py` | 90%+ | 14 |
| Materials - Export | `src/core/materials/export.py` | 85%+ | 10 |
| Hybrid Fusion | `src/ml/hybrid/fusion.py` | 90%+ | 37 |
| Hybrid Calibration | `src/ml/hybrid/calibration.py` | 85%+ | 27 |
| LLM Providers | `src/core/assistant/llm_providers.py` | 85%+ | 32 |

## Test Matrix

### Materials Module Tests

- **test_materials_classify.py**: classify_material_detailed (known materials, case-insensitive, empty/None/unknown, alias, keyword), classify_material_simple, search_materials (exact, fuzzy, pinyin, filters, limits), _calculate_similarity
- **test_materials_cost.py**: get_material_cost (known/unknown, structure, group defaults), compare_material_costs (sorting, relative_to_cheapest, include_missing), search_by_cost (tier/index filters, limits), get_cost_tier_info
- **test_materials_compatibility.py**: check_weld_compatibility (same group, cross-group, symmetric, unknown), check_galvanic_corrosion (same/distant materials, anode/cathode ID), check_heat_treatment_compatibility, check_full_compatibility
- **test_materials_equivalence.py**: get_material_equivalence (forward/reverse lookup), find_equivalent_material (CN/US/JP/DE), list_material_standards
- **test_materials_export.py**: export_materials_csv (string output, header, data rows, parseability, file write), export_equivalence_csv

### ML Module Tests

- **test_hybrid_fusion_strategies.py**: WeightedAverageFusion, VotingFusion (hard/soft), DempsterShaferFusion, AttentionFusion (temperature, learned weights), MultiSourceFusion (auto-selection, engine caching), SourcePrediction/FusionResult dataclasses
- **test_hybrid_calibration.py**: TemperatureScaling, PlattScaling, IsotonicCalibration, HistogramBinning, BetaCalibration, ConfidenceCalibrator (global/per-source), CalibrationMetrics (ECE/MCE/Brier)

### Integration Tests

- **test_llm_providers.py**: LLMConfig defaults, OfflineProvider (always available, knowledge extraction), ClaudeProvider (mocked), OpenAIProvider (model override, mocked API), QwenProvider (mocked), OllamaProvider (mocked requests), get_provider (aliases, unknown fallback), get_best_available_provider

## Test Design Principles

- All external dependencies (HTTP, LLM APIs) are mocked with unittest.mock
- No real network calls, no GPU, no large I/O
- Synthetic data for calibration tests (deterministic via fixed random seed)
- pytest fixtures for shared test data (fusion predictions)
- Fast execution: full suite completes in <3 seconds

## Summary

| Metric | Value |
|--------|-------|
| Total new test files | 8 |
| Total new tests | 184 |
| All passing | Yes |
| Execution time | ~2.5s |

## Verification

```
python3 -m pytest tests/unit/test_materials_classify.py tests/unit/test_materials_cost.py \
  tests/unit/test_materials_compatibility.py tests/unit/test_materials_equivalence.py \
  tests/unit/test_materials_export.py tests/unit/test_hybrid_fusion_strategies.py \
  tests/unit/test_hybrid_calibration.py tests/unit/test_llm_providers.py -q --timeout=30
# Result: 184 passed in 2.48s
```
