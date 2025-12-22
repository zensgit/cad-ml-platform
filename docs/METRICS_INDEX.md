# Metrics Index

This document lists all Prometheus metrics exposed by the CAD ML Platform at the `/metrics` endpoint.

## Analysis & Feature Extraction

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `analysis_requests_total` | Counter | Total CAD analysis requests | `status` |
| `analysis_errors_total` | Counter | Total analysis errors | `stage`, `code` |
| `analysis_stage_duration_seconds` | Histogram | Duration of each analysis stage | `stage` |
| `analysis_feature_vector_dimension` | Histogram | Distribution of feature vector dimensions | - |
| `feature_extraction_latency_seconds` | Histogram | Latency of feature extraction by version | `version` |
| `classification_latency_seconds` | Histogram | Classification stage latency | - |
| `dfm_analysis_latency_seconds` | Histogram | DFM analysis latency | - |
| `process_recommend_latency_seconds` | Histogram | Process recommendation latency | - |
| `cost_estimation_latency_seconds` | Histogram | Cost estimation latency | - |
| `feature_version_usage_total` | Counter | Feature extraction operations per version | `version` |
| `feature_cache_hits_total` | Counter | Feature cache hits | - |
| `feature_cache_miss_total` | Counter | Feature cache misses | - |
| `feature_cache_size` | Gauge | Number of cached feature vectors | - |
| `feature_cache_tuning_requests_total` | Counter | Cache tuning endpoint requests | `status` |
| `feature_embedding_generation_seconds` | Histogram | Latency of metric embedding generation | - |
| `feature_embedding_errors_total` | Counter | Errors during metric embedding generation | `type` |

## Vector Store (Faiss & Redis)

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `vector_query_backend_total` | Counter | Vector similarity queries by backend | `backend` |
| `vector_query_latency_seconds` | Histogram | Latency of vector similarity queries | `backend` |
| `faiss_index_size` | Gauge | Number of vectors in Faiss index | - |
| `faiss_rebuild_total` | Counter | Faiss index rebuild outcomes | `status` |
| `faiss_recovery_attempts_total` | Counter | Faiss recovery attempts | `result` |
| `similarity_degraded_total` | Counter | Vector similarity degraded mode events | `event` |
| `vector_migrate_total` | Counter | Vector migrate endpoint outcomes | `status` |
| `vector_migrate_dimension_delta` | Histogram | Dimension delta during migration | - |
| `vector_orphan_total` | Counter | Vectors without cached analysis result | - |

## Model & Classification

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `classification_model_load_total` | Counter | Model load outcomes | `status`, `version` |
| `classification_model_inference_seconds` | Histogram | Classification inference latency | - |
| `classification_prediction_distribution` | Counter | Distribution of predicted types | `label`, `version` |
| `model_reload_total` | Counter | Model hot reload attempts | `status`, `version` |
| `model_security_fail_total` | Counter | Model security validation failures | `reason` |
| `model_opcode_mode` | Gauge | Current opcode validation mode (0=audit, 1=blocklist, 2=whitelist) | - |
| `model_opcode_audit_total` | Counter | Observed pickle opcodes during reload | `opcode` |
| `model_opcode_scan_total` | Counter | Model opcode scan results | `mode`, `safe` |
| `model_health_checks_total` | Counter | Model health endpoint requests | `status` |

## System & Process

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `process_start_time_seconds` | Gauge | Process start time epoch seconds | - |
| `process_rules_audit_requests_total` | Counter | Process rules audit requests | `status` |
| `drift_baseline_created_total` | Counter | Drift baseline creation events | `type` |
| `drift_baseline_refresh_total` | Counter | Drift baseline refresh events | `type`, `trigger` |

For a complete list of metrics, query the `/metrics` endpoint or check `src/utils/analysis_metrics.py`.
