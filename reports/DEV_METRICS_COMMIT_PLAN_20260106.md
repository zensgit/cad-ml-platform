# DEV_METRICS_COMMIT_PLAN_20260106

## Goal
Provide a clean, review-friendly commit plan for the 2026-01-06 metrics updates.

## Suggested Commit Breakdown
1. feat(metrics): cache tuning endpoint + metrics
   - Files: `src/api/v1/features.py`, `src/api/v1/health.py`, `src/utils/analysis_metrics.py`, `tests/unit/test_cache_tuning.py`, `tests/test_metrics_contract.py`
   - Design: `docs/FEATURE_CACHE_TUNING_POST_ENDPOINT_DESIGN.md`, `docs/FEATURE_CACHE_TUNING_RECOMMENDATION_GAUGES_DESIGN.md`, `docs/FEATURE_CACHE_TUNING_METRICS_CONTRACT_DESIGN.md`
   - Validation: `reports/DEV_FEATURE_CACHE_TUNING_POST_ENDPOINT_VALIDATION_20260106.md`, `reports/DEV_FEATURE_CACHE_TUNING_RECOMMENDATION_GAUGES_VALIDATION_20260106.md`, `reports/DEV_FEATURE_CACHE_TUNING_METRICS_CONTRACT_VALIDATION_20260106.md`

2. feat(metrics): model security + rollback + opcode metrics
   - Files: `src/ml/classifier.py`, `src/api/v1/health.py`, `src/utils/analysis_metrics.py`, `tests/unit/test_model_security_validation.py`, `tests/unit/test_model_rollback_health.py`, `tests/unit/test_model_rollback_level3.py`, `tests/unit/test_model_opcode_modes.py`
   - Design: `docs/MODEL_INTERFACE_VALIDATION_METRICS_DESIGN.md`, `docs/MODEL_OPCODE_MODE_GAUGE_DESIGN.md`, `docs/DASHBOARD_METRICS_ALIGNMENT_SECURITY_ROLLBACK_V4_DESIGN.md`
   - Validation: `reports/DEV_MODEL_INTERFACE_VALIDATION_METRICS_VALIDATION_20260106.md`, `reports/DEV_MODEL_OPCODE_MODE_GAUGE_VALIDATION_20260106.md`, `reports/DEV_DASHBOARD_METRICS_ALIGNMENT_SECURITY_ROLLBACK_V4_VALIDATION_20260106.md`

3. feat(metrics): v4 feature histograms + vector migrate metrics
   - Files: `src/core/feature_extractor.py`, `src/utils/analysis_metrics.py`, `tests/unit/test_v4_feature_performance.py`, `tests/unit/test_vector_migrate_metrics.py`, `tests/unit/test_vector_migrate_dimension_histogram.py`
   - Design: `docs/V4_FEATURE_METRICS_HISTOGRAM_COUNT_DESIGN.md`, `docs/VECTOR_MIGRATE_DOWNGRADE_METRICS_TESTS_DESIGN.md`, `docs/VECTOR_MIGRATE_DIMENSION_HISTOGRAM_COUNT_DESIGN.md`
   - Validation: `reports/DEV_V4_FEATURE_METRICS_HISTOGRAM_COUNT_VALIDATION_20260106.md`, `reports/DEV_VECTOR_MIGRATE_DOWNGRADE_METRICS_TESTS_VALIDATION_20260106.md`, `reports/DEV_VECTOR_MIGRATE_DIMENSION_HISTOGRAM_COUNT_VALIDATION_20260106.md`

4. chore(metrics): dashboard alignment and validation tooling
   - Files: `config/grafana/dashboard_main.json`, `scripts/validate_dashboard_metrics.py`
   - Validation: `reports/DEV_DASHBOARD_METRICS_ALIGNMENT_SECURITY_ROLLBACK_V4_VALIDATION_20260106.md`, `reports/DEV_DASHBOARD_METRICS_REVALIDATION_20260106.md`

5. test(metrics): contract + metrics-enabled unit sweeps
   - Files: test updates only as needed
   - Validation: `reports/DEV_METRICS_CONTRACT_FULL_VALIDATION_20260106.md`, `reports/DEV_METRICS_UNIT_SUBSET_VENV_VALIDATION_20260106.md`, `reports/DEV_METRICS_UNIT_SECURITY_ROLLBACK_VECTOR_VENV_VALIDATION_20260106.md`, `reports/DEV_METRICS_UNIT_FILTER_VENV_VALIDATION_20260106.md`

6. docs(metrics): handoff summaries and addenda
   - Files: `FINAL_VALIDATION_REPORT.md`, `PROJECT_HANDOVER.md`, `FINAL_HANDOVER_PACKAGE_V3.md`, `FINAL_SUMMARY.md`, `PROJECT_COMPLETION_REPORT.md`, `PROJECT_HANDOVER_PHASE8.md`, `PROJECT_HANDOVER_PHASE5.md`, `DELIVERABLES_SUMMARY.md`, `DESIGN_SUMMARY.md`, `PRODUCTION_VERIFICATION_PLAN.md`, `DEVELOPMENT_SUMMARY.md`, `PHASE5_V2_COMPLETION_REPORT.md`, `PHASE3_V2_COMPLETION_REPORT.md`, `PHASE2_ENHANCEMENT_SUMMARY.md`, `PHASE7_IMPLEMENTATION_LOG.md`, `IMPLEMENTATION_RESULTS.md`, `FINAL_VERIFICATION_LOG.md`, `docs/DEVELOPMENT_SUMMARY_FINAL.md`, `docs/DEVELOPMENT_REPORT_FINAL.md`, `docs/IMPLEMENTATION_SUMMARY.md`, `docs/FINAL_IMPLEMENTATION_CHECKLIST.md`
   - Validation: `reports/DEV_METRICS_HANDOFF_SUMMARY_20260106.md`, `reports/DEV_METRICS_DELIVERY_INDEX_20260106.md`, `reports/DEV_METRICS_PR_SUMMARY_20260106.md`, `reports/DEV_METRICS_DELIVERY_CHECKLIST_20260106.md`, `reports/DEV_METRICS_FINAL_DELIVERY_SUMMARY_20260106.md`, `reports/DEV_METRICS_COMMIT_PLAN_20260106.md`, `reports/DEV_METRICS_CHANGED_FILES_20260106.md`, `reports/DEV_METRICS_DEVELOPMENT_REPORT_20260106.md`, `reports/DEV_METRICS_VERIFICATION_REPORT_20260106.md`

7. ci(metrics): docker staging smoke workflow
   - Files: `.github/workflows/docker-staging-smoke.yml`, `scripts/ci/docker_staging_smoke.sh`
   - Design: `docs/GITHUB_DOCKER_STAGING_WORKFLOW_DESIGN.md`
   - Validation: `reports/DEV_GITHUB_DOCKER_STAGING_WORKFLOW_VALIDATION_20260110.md`

## Notes
- Use `.venv` test results for metrics-enabled validation.
- Keep commit messages conventional: `feat:`, `test:`, `chore:`, `docs:`.
