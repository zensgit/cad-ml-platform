# Commit Scope Proposal

- Date: 2025-12-30
- Status: working tree has tracked modifications + untracked reports/tests.

## Proposed commit buckets

### 1) Hardening + hashing updates
**Suggested message**: `fix(hardening): tighten hashing and eval paths`
**Key paths**:
- src/utils/safe_eval.py
- src/core/vision/observability.py
- src/core/vision/workflow_engine.py
- src/core/vision/embedding.py
- src/core/vision/deduplication.py
- src/core/vision/* (SHA256 migration)
- src/core/dedupcad_precision/cad_pipeline.py
- src/core/active_learning.py
- src/core/config.py
- src/core/assembly/confidence_calibrator.py
- src/core/vectors/stores/faiss_store.py
- src/ml/classifier.py
- tests/unit/test_safe_eval.py
- tests/unit/test_vision_phase4.py
- tests/unit/test_vision_phase6.py
- tests/integration/test_e2e_api_smoke.py

### 2) Integration auth + dependency updates
**Suggested message**: `fix(auth): switch integration JWT to PyJWT`
**Key paths**:
- src/api/middleware/integration_auth.py
- tests/unit/test_integration_auth_middleware.py
- requirements.txt
- requirements-dev.txt
- requirements-dev-lite.txt
- reports/DEV_ECDSA_REMOVAL_20251230.md

### 3) DeepSeek HF pinning + config/docs
**Suggested message**: `feat(ocr): require DeepSeek HF pinned revision`
**Key paths**:
- src/core/ocr/providers/deepseek_hf.py
- .env.example
- README.md
- reports/DEV_DEEPSEEK_HF_REVISION_20251230.md

### 4) Prometheus recording rules cleanup
**Suggested message**: `chore(prometheus): dedupe recording rules`
**Key paths**:
- config/prometheus/recording_rules.yml
- reports/DEV_PROMTOOL_RULES_VALIDATE_20251230.md

### 5) Verification logs + baselines
**Suggested message**: `docs(reports): refresh validation logs`
**Key paths**:
- FINAL_VERIFICATION_LOG.md
- IMPLEMENTATION_RESULTS.md
- reports/DEV_COMMIT_SCOPE_20251230.md
- reports/DEV_HASH_COMPAT_20251230.md
- reports/DEV_MAKE_TEST_20251230_FULL.md
- reports/DEV_REGRESSION_VALIDATION_20251230.md
- reports/regression_validation.md
- reports/DEV_METRICS_CONSISTENCY_20251230.md
- reports/DEV_PERFORMANCE_BASELINE_20251230.md
- reports/DEV_PERFORMANCE_BASELINE_COMPARE_20251230.md
- reports/DEV_PROMTOOL_RULES_VALIDATE_20251230.md
- reports/performance_baseline_day0_20251230.json
- reports/security/*

## Notes
- Untracked count: 45 files (mostly reports). Decide whether to commit all reports or keep them local.
