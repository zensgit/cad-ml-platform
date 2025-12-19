"""Prometheus metrics for dedup2d subsystem (Phase 4 Observability).

All metric objects are defined at import time. If prometheus_client is not
available, dummy objects are provided preserving the .labels() chain.
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore
except Exception:  # provide no-op dummies if prometheus not installed

    class _Dummy:
        def labels(self, **kwargs):  # type: ignore
            return self

        def inc(self, *a, **kw):  # type: ignore
            pass

        def dec(self, *a, **kw):  # type: ignore
            pass

        def observe(self, *a, **kw):  # type: ignore
            pass

        def set(self, *a, **kw):  # type: ignore
            pass

    def Counter(*a, **kw):  # type: ignore
        return _Dummy()

    def Histogram(*a, **kw):  # type: ignore
        return _Dummy()

    def Gauge(*a, **kw):  # type: ignore
        return _Dummy()


# =============================================================================
# Job Metrics
# =============================================================================

dedup2d_jobs_submitted_total = Counter(
    "dedup2d_jobs_submitted_total",
    "Total number of dedup2d jobs submitted",
    ["backend"],  # inprocess | redis
)

dedup2d_jobs_completed_total = Counter(
    "dedup2d_jobs_completed_total",
    "Total number of dedup2d jobs completed",
    ["backend", "status"],  # status: success | error
)

dedup2d_job_duration_seconds = Histogram(
    "dedup2d_job_duration_seconds",
    "Dedup2d job processing duration in seconds",
    ["backend"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

dedup2d_jobs_queued = Gauge(
    "dedup2d_jobs_queued",
    "Number of dedup2d jobs currently queued",
    ["backend"],
)

dedup2d_jobs_active = Gauge(
    "dedup2d_jobs_active",
    "Number of dedup2d jobs currently being processed",
    ["backend"],
)

# =============================================================================
# File Storage Metrics
# =============================================================================

dedup2d_file_uploads_total = Counter(
    "dedup2d_file_uploads_total",
    "Total number of file uploads to storage",
    ["backend", "status"],  # backend: local | s3, status: success | error
)

dedup2d_file_downloads_total = Counter(
    "dedup2d_file_downloads_total",
    "Total number of file downloads from storage",
    ["backend", "status"],
)

dedup2d_file_deletes_total = Counter(
    "dedup2d_file_deletes_total",
    "Total number of file deletes from storage",
    ["backend", "status"],
)

dedup2d_file_upload_bytes = Histogram(
    "dedup2d_file_upload_bytes",
    "Size of uploaded files in bytes",
    ["backend"],
    buckets=[
        1_000,       # 1KB
        10_000,      # 10KB
        100_000,     # 100KB
        1_000_000,   # 1MB
        5_000_000,   # 5MB
        10_000_000,  # 10MB
        50_000_000,  # 50MB
        100_000_000, # 100MB
    ],
)

dedup2d_file_operation_duration_seconds = Histogram(
    "dedup2d_file_operation_duration_seconds",
    "File storage operation duration in seconds",
    ["backend", "operation"],  # operation: upload | download | delete
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# =============================================================================
# Callback Metrics
# =============================================================================

dedup2d_callbacks_total = Counter(
    "dedup2d_callbacks_total",
    "Total number of webhook callbacks attempted",
    ["status"],  # status: success | error | blocked
)

dedup2d_callback_duration_seconds = Histogram(
    "dedup2d_callback_duration_seconds",
    "Webhook callback duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

dedup2d_callback_retries_total = Counter(
    "dedup2d_callback_retries_total",
    "Total number of callback retry attempts",
)

dedup2d_callback_blocked_total = Counter(
    "dedup2d_callback_blocked_total",
    "Callbacks blocked by security policy",
    ["reason"],  # reason: private_network | http_not_allowed | dns_mismatch | not_in_allowlist
)

# =============================================================================
# GC Metrics
# =============================================================================

dedup2d_gc_runs_total = Counter(
    "dedup2d_gc_runs_total",
    "Total number of GC script runs",
    ["backend", "status"],  # status: success | error
)

dedup2d_gc_files_deleted_total = Counter(
    "dedup2d_gc_files_deleted_total",
    "Total number of files deleted by GC",
    ["backend"],
)

dedup2d_gc_bytes_freed_total = Counter(
    "dedup2d_gc_bytes_freed_total",
    "Total bytes freed by GC",
    ["backend"],
)

dedup2d_gc_duration_seconds = Histogram(
    "dedup2d_gc_duration_seconds",
    "GC run duration in seconds",
    ["backend"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# =============================================================================
# Rolling Upgrade Metrics
# =============================================================================

dedup2d_payload_format = Counter(
    "dedup2d_payload_format_total",
    "Payload format used for job submission",
    ["format"],  # format: file_ref_only | file_ref_and_b64 | b64_only (legacy)
)

dedup2d_legacy_b64_fallback_total = Counter(
    "dedup2d_legacy_b64_fallback_total",
    "Number of times worker fell back to file_bytes_b64 format",
)

# =============================================================================
# Error Rate EMA (for health checks)
# =============================================================================

dedup2d_error_rate_ema = Gauge(
    "dedup2d_error_rate_ema",
    "Exponential moving average of dedup2d job error rate (0..1)",
)

_EMA_ALPHA = 0.2
_error_rate_value = 0.0


def update_dedup2d_error_ema(is_error: bool) -> None:
    """Update the EMA error rate tracker."""
    global _error_rate_value
    target = 1.0 if is_error else 0.0
    _error_rate_value = _EMA_ALPHA * target + (1 - _EMA_ALPHA) * _error_rate_value
    try:
        dedup2d_error_rate_ema.set(_error_rate_value)
    except Exception:
        pass


def get_dedup2d_error_rate_ema() -> float:
    """Return the current in-process EMA value for dedup2d error rate."""
    return float(_error_rate_value)


__all__ = [
    # Job metrics
    "dedup2d_jobs_submitted_total",
    "dedup2d_jobs_completed_total",
    "dedup2d_job_duration_seconds",
    "dedup2d_jobs_queued",
    "dedup2d_jobs_active",
    # File storage metrics
    "dedup2d_file_uploads_total",
    "dedup2d_file_downloads_total",
    "dedup2d_file_deletes_total",
    "dedup2d_file_upload_bytes",
    "dedup2d_file_operation_duration_seconds",
    # Callback metrics
    "dedup2d_callbacks_total",
    "dedup2d_callback_duration_seconds",
    "dedup2d_callback_retries_total",
    "dedup2d_callback_blocked_total",
    # GC metrics
    "dedup2d_gc_runs_total",
    "dedup2d_gc_files_deleted_total",
    "dedup2d_gc_bytes_freed_total",
    "dedup2d_gc_duration_seconds",
    # Rolling upgrade metrics
    "dedup2d_payload_format",
    "dedup2d_legacy_b64_fallback_total",
    # Error rate
    "dedup2d_error_rate_ema",
    "update_dedup2d_error_ema",
    "get_dedup2d_error_rate_ema",
]
