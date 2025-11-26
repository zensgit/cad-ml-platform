"""Prometheus metrics for CAD analysis pipeline."""

from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, Gauge  # type: ignore
except Exception:  # pragma: no cover - fallback dummy
    class _Dummy:
        def labels(self, *a, **kw):
            return self
        def inc(self, *a, **kw):
            pass
        def observe(self, *a, **kw):
            pass
        def set(self, *a, **kw):
            pass
    def Counter(*a, **kw):  # type: ignore
        return _Dummy()
    def Histogram(*a, **kw):  # type: ignore
        return _Dummy()
    def Gauge(*a, **kw):  # type: ignore
        return _Dummy()


analysis_requests_total = Counter(
    "analysis_requests_total", "CAD analysis requests", ["status"]
)
analysis_errors_total = Counter(
    "analysis_errors_total", "CAD analysis errors", ["stage", "code"]
)
analysis_stage_duration_seconds = Histogram(
    "analysis_stage_duration_seconds",
    "Per-stage duration for CAD analysis",
    ["stage"],
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)
parse_stage_latency_seconds = Histogram(
    "parse_stage_latency_seconds",
    "Latency of parse stage (seconds)",
    ["format"],
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

analysis_feature_vector_dimension = Histogram(
    "analysis_feature_vector_dimension",
    "Distribution of feature vector dimensions",
    buckets=[1, 5, 10, 20, 50, 100],
)

# Per-version feature extraction latency
feature_extraction_latency_seconds = Histogram(
    "feature_extraction_latency_seconds",
    "Latency of feature extraction by version",
    ["version"],
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
analysis_material_usage_total = Counter(
    "analysis_material_usage_total",
    "Material field usage counts",
    ["material"],
)

analysis_rejections_total = Counter(
    "analysis_rejections_total", "Rejected analyses due to limits", ["reason"]
)

analysis_error_code_total = Counter(
    "analysis_error_code_total", "Extended error code occurrences", ["code"]
)

analysis_parse_latency_budget_ratio = Histogram(
    "analysis_parse_latency_budget_ratio",
    "Ratio of parse stage latency to target budget (p95 target).",
    buckets=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
)

analysis_vector_count = Counter(
    "analysis_vector_count",
    "Total vectors registered (labeled by backend)",
    ["backend"],
)

analysis_cache_hits_total = Counter(
    "analysis_cache_hits_total",
    "CAD analysis cache hits",
)
analysis_cache_miss_total = Counter(
    "analysis_cache_miss_total",
    "CAD analysis cache misses",
)

vector_dimension_rejections_total = Counter(
    "vector_dimension_rejections_total",
    "Vector registration/update dimension rejection occurrences",
    ["reason"],
)

vector_stats_requests_total = Counter(
    "vector_stats_requests_total",
    "Vector stats endpoint requests",
    ["status"],
)

process_rule_version_total = Counter(
    "process_rule_version_total",
    "Process rule version occurrences",
    ["version"],
)

classification_latency_seconds = Histogram(
    "classification_latency_seconds",
    "Latency of classification stage (wall clock seconds)",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

process_recommend_latency_seconds = Histogram(
    "process_recommend_latency_seconds",
    "Latency of process recommendation stage (wall clock seconds)",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

vector_store_material_total = Counter(
    "vector_store_material_total",
    "Material distribution of stored vectors",
    ["material"],
)

analysis_parallel_enabled = Gauge(
    "analysis_parallel_enabled",
    "Indicates if parallel execution of post-feature stages was used (1) or not (0)",
)

analysis_parallel_savings_seconds = Histogram(
    "analysis_parallel_savings_seconds",
    "Estimated time saved by parallel execution (sum(serial stage durations) - wall time)",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

material_drift_ratio = Histogram(
    "material_drift_ratio",
    "Ratio of dominant material count to total vectors (used for drift detection)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

signature_validation_fail_total = Counter(
    "signature_validation_fail_total",
    "CAD file signature validation failures (format mismatch heuristic)",
    ["format"],
)

faiss_index_size = Gauge(
    "faiss_index_size",
    "Current number of vectors stored in Faiss index",
)
faiss_init_errors_total = Counter(
    "faiss_init_errors_total",
    "Faiss initialization errors",
)
vector_query_backend_total = Counter(
    "vector_query_backend_total",
    "Vector similarity queries by backend",
    ["backend"],
)

vector_query_latency_seconds = Histogram(
    "vector_query_latency_seconds",
    "Latency of vector similarity top-k queries",
    ["backend"],
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

vector_query_batch_latency_seconds = Histogram(
    "vector_query_batch_latency_seconds",
    "Latency of batch vector similarity queries",
    ["batch_size_range"],  # small: 1-5, medium: 6-20, large: 21+
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

feature_migration_total = Counter(
    "feature_migration_total",
    "Feature version migration outcomes",
    ["status"],  # success|skipped|error
)

# Vector migrate API outcomes (API-level rather than script)
vector_migrate_total = Counter(
    "vector_migrate_total",
    "Vector migrate endpoint item outcomes",
    ["status"],  # migrated|skipped|dry_run|error|not_found|downgraded
)

# Vector dimension change during migration
vector_migrate_dimension_delta = Histogram(
    "vector_migrate_dimension_delta",
    "Dimension delta (new_dim - old_dim) during vector migration",
    buckets=(-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100),  # Dimension differences
)

# Process rules audit endpoint requests
process_rules_audit_requests_total = Counter(
    "process_rules_audit_requests_total",
    "Process rules audit endpoint requests",
    ["status"],  # ok|error
)

faiss_rebuild_total = Counter(
    "faiss_rebuild_total",
    "Faiss index rebuild outcomes",
    ["status"],  # success|error|skipped
)
faiss_rebuild_duration_seconds = Histogram(
    "faiss_rebuild_duration_seconds",
    "Duration of Faiss index rebuild operations",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
faiss_rebuild_backoff_seconds = Gauge(
    "faiss_rebuild_backoff_seconds",
    "Current backoff interval (seconds) before next auto rebuild attempt",
)

format_validation_fail_total = Counter(
    "format_validation_fail_total",
    "Deep format validation failures",
    ["format", "reason"],
)
strict_mode_enabled = Gauge(
    "strict_mode_enabled",
    "Indicates if strict format validation mode is enabled (1/0)",
)

faiss_export_total = Counter(
    "faiss_export_total",
    "Faiss index export attempts",
    ["status"],  # success|error|skipped
)
faiss_export_duration_seconds = Histogram(
    "faiss_export_duration_seconds",
    "Duration of Faiss index export operations",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
faiss_import_total = Counter(
    "faiss_import_total",
    "Faiss index import attempts",
    ["status"],  # success|error|skipped
)
faiss_import_duration_seconds = Histogram(
    "faiss_import_duration_seconds",
    "Duration of Faiss index import operations",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
classification_model_load_total = Counter(
    "classification_model_load_total",
    "Classification model load outcomes",
    ["status", "version"],  # success|error|absent
)
classification_model_inference_seconds = Histogram(
    "classification_model_inference_seconds",
    "ML classification inference latency",
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
)
classification_prediction_distribution = Counter(
    "classification_prediction_distribution",
    "Distribution of ML predicted types",
    ["label", "version"],
)
classification_model_version_total = Counter(
    "classification_model_version_total",
    "Observed classification model version usage",
    ["version"],
)
faiss_auto_rebuild_total = Counter(
    "faiss_auto_rebuild_total",
    "Faiss auto rebuild trigger outcomes",
    ["status"],  # success|error|skipped
)
faiss_index_dim_mismatch_total = Counter(
    "faiss_index_dim_mismatch_total",
    "Faiss index dimension mismatch occurrences triggering fallback",
)
vector_orphan_total = Counter(
    "vector_orphan_total",
    "Vectors without corresponding cached analysis result detected",
)

# === New metrics (drift / reload / pruning / diff) ===
classification_prediction_drift_score = Histogram(
    "classification_prediction_drift_score",
    "Drift score for classification predictions (PSI-like, 0-1)",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
)
material_distribution_drift_score = Histogram(
    "material_distribution_drift_score",
    "Drift score for material distribution (PSI-like, 0-1)",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
)
model_reload_total = Counter(
    "model_reload_total",
    "Model hot reload attempts",
    ["status", "version"],  # success|error|not_found|mismatch
)
vector_cold_pruned_total = Counter(
    "vector_cold_pruned_total",
    "Cold (idle) vectors pruned",
    ["reason"],  # idle
)
parse_timeout_total = Counter(
    "parse_timeout_total",
    "CAD parse stage timeouts",
)
features_diff_requests_total = Counter(
    "features_diff_requests_total",
    "Feature diff endpoint requests",
    ["status"],  # ok|not_found|dimension_mismatch|error
)
feature_slot_delta_magnitude = Histogram(
    "feature_slot_delta_magnitude",
    "Magnitude of normalized per-slot feature delta",
    buckets=[0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
)
feature_cache_hits_total = Counter(
    "feature_cache_hits_total",
    "Feature cache vector hits",
)
feature_cache_miss_total = Counter(
    "feature_cache_miss_total",
    "Feature cache vector misses",
)
feature_cache_hits_last_hour = Gauge(
    "feature_cache_hits_last_hour",
    "Feature cache hits in the last sliding hour (approx, updated per request)",
)
feature_cache_miss_last_hour = Gauge(
    "feature_cache_miss_last_hour",
    "Feature cache misses in the last sliding hour (approx, updated per request)",
)
feature_cache_evictions_total = Counter(
    "feature_cache_evictions_total",
    "Feature cache evictions",
)
feature_cache_size = Gauge(
    "feature_cache_size",
    "Current number of cached feature vectors",
)
feature_cache_lookup_seconds = Histogram(
    "feature_cache_lookup_seconds",
    "Latency of feature cache lookup",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01],
)
feature_cache_prewarm_total = Counter(
    "feature_cache_prewarm_total",
    "Feature cache prewarm attempts",
    ["result"],  # ok|error
)
faiss_index_age_seconds = Gauge(
    "faiss_index_age_seconds",
    "Seconds since last Faiss index export/import",
)
vector_orphan_scan_last_seconds = Gauge(
    "vector_orphan_scan_last_seconds",
    "Seconds since last orphan vector scan",
)
baseline_material_age_seconds = Gauge(
    "baseline_material_age_seconds",
    "Seconds since material distribution baseline established",
)
baseline_prediction_age_seconds = Gauge(
    "baseline_prediction_age_seconds",
    "Seconds since prediction distribution baseline established",
)
drift_baseline_created_total = Counter(
    "drift_baseline_created_total",
    "Drift baseline creation events",
    ["type"],  # material|prediction
)
drift_baseline_refresh_total = Counter(
    "drift_baseline_refresh_total",
    "Total times drift baseline refreshed (manual or auto)",
    ["type", "trigger"],  # type: material|prediction, trigger: manual|auto|stale
)
model_security_fail_total = Counter(
    "model_security_fail_total",
    "Model security validation failures",
    ["reason"],  # hash_mismatch|magic_number_invalid|forged_file|opcode_blocked|opcode_scan_error
)
vector_store_reload_total = Counter(
    "vector_store_reload_total",
    "Vector store backend reload requests",
    ["status"],  # success|error
)

# Model health check requests
model_health_checks_total = Counter(
    "model_health_checks_total",
    "Model health endpoint requests",
    ["status"],  # ok|absent|rollback|error
)

# Similarity degraded / restored events (Faiss fallback lifecycle)
similarity_degraded_total = Counter(
    "similarity_degraded_total",
    "Vector similarity degraded mode events",
    ["event"],  # degraded|restored
)

# Faiss auto-recovery metrics
faiss_recovery_attempts_total = Counter(
    "faiss_recovery_attempts_total",
    "Faiss recovery attempts",
    ["result"],  # success|skipped|error|suppressed
)
faiss_degraded_duration_seconds = Gauge(
    "faiss_degraded_duration_seconds",
    "Current degraded duration in seconds (0 when healthy)",
)

# Next scheduled recovery attempt epoch seconds (0 = none scheduled / healthy)
faiss_next_recovery_eta_seconds = Gauge(
    "faiss_next_recovery_eta_seconds",
    "Epoch timestamp for next automatic Faiss recovery attempt (0 if not scheduled)",
)

# Compatibility alias (singular) to avoid confusion in dashboards/rules

# Recovery suppression events (e.g. flapping protection)
faiss_recovery_suppressed_total = Counter(
    "faiss_recovery_suppressed_total",
    "Faiss recovery attempts suppressed due to flapping or manual override",
    ["reason"],  # flapping
)

# Seconds remaining in current suppression window (0 = none active)
faiss_recovery_suppression_remaining_seconds = Gauge(
    "faiss_recovery_suppression_remaining_seconds",
    "Seconds remaining in current recovery suppression window (0 if none)",
)

# Process start time (used for alert quiet periods)
process_start_time_seconds = Gauge(
    "process_start_time_seconds",
    "Process start time epoch seconds",
)
try:
    import time as _t
    process_start_time_seconds.set(_t.time())
except Exception:
    pass

# Opcode security / audit metrics
model_opcode_audit_total = Counter(
    "model_opcode_audit_total",
    "Observed pickle opcodes during model reload scans",
    ["opcode"],
)
model_opcode_whitelist_violations_total = Counter(
    "model_opcode_whitelist_violations_total",
    "Whitelist violations (disallowed opcodes) during model reload",
    ["opcode"],
)

# Recovery state backend label for observability (file|redis)
faiss_recovery_state_backend = Gauge(
    "faiss_recovery_state_backend",
    "Active recovery state backend",
    ["backend"],
)

__all__ = [
    "analysis_requests_total",
    "analysis_errors_total",
    "analysis_stage_duration_seconds",
    "analysis_feature_vector_dimension",
    "feature_extraction_latency_seconds",
    "analysis_material_usage_total",
    "analysis_rejections_total",
    "analysis_error_code_total",
    "analysis_parse_latency_budget_ratio",
    "analysis_vector_count",
    "vector_stats_requests_total",
    "process_rule_version_total",
    "classification_latency_seconds",
    "process_recommend_latency_seconds",
    "vector_store_material_total",
    "vector_dimension_rejections_total",
    "analysis_parallel_enabled",
    "analysis_parallel_savings_seconds",
    "analysis_cache_hits_total",
    "analysis_cache_miss_total",
    "material_drift_ratio",
    "signature_validation_fail_total",
    "faiss_index_size",
    "faiss_init_errors_total",
    "vector_query_backend_total",
    "vector_query_latency_seconds",
    "vector_query_batch_latency_seconds",
    "feature_migration_total",
    "vector_migrate_total",
    "vector_migrate_dimension_delta",
    "process_rules_audit_requests_total",
    "faiss_rebuild_total",
    "faiss_rebuild_duration_seconds",
    "faiss_rebuild_backoff_seconds",
    "format_validation_fail_total",
    "strict_mode_enabled",
    "faiss_export_total",
    "faiss_export_duration_seconds",
    "classification_model_load_total",
    "classification_model_inference_seconds",
    "classification_prediction_distribution",
    "classification_model_version_total",
    "faiss_import_total",
    "faiss_import_duration_seconds",
    "faiss_auto_rebuild_total",
    "faiss_index_dim_mismatch_total",
    "vector_orphan_total",
    "parse_stage_latency_seconds",
    "classification_prediction_drift_score",
    "material_distribution_drift_score",
    "model_reload_total",
    "vector_cold_pruned_total",
    "parse_timeout_total",
    "features_diff_requests_total",
    "feature_slot_delta_magnitude",
    "vector_orphan_scan_last_seconds",
    "feature_cache_hits_total",
    "feature_cache_miss_total",
    "feature_cache_evictions_total",
    "feature_cache_size",
    "feature_cache_lookup_seconds",
    "feature_cache_prewarm_total",
    "feature_cache_hits_last_hour",
    "feature_cache_miss_last_hour",
    "faiss_index_age_seconds",
    "baseline_material_age_seconds",
    "baseline_prediction_age_seconds",
    "drift_baseline_created_total",
    "drift_baseline_refresh_total",
    "model_security_fail_total",
    "model_health_checks_total",
    "vector_store_reload_total",
    "similarity_degraded_total",
    "faiss_recovery_attempts_total",
    "faiss_degraded_duration_seconds",
    "faiss_next_recovery_eta_seconds",
    "faiss_recovery_suppressed_total",
    "faiss_recovery_suppression_remaining_seconds",
    "process_start_time_seconds",
    "model_opcode_audit_total",
    "model_opcode_whitelist_violations_total",
    "faiss_recovery_state_backend",
]
