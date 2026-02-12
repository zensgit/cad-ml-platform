"""Prometheus metrics registration for OCR subsystem.

All metric objects are defined at import time. If prometheus_client is not
available, dummy objects are provided preserving the .labels() chain so that
calling code does not need conditional branches.
"""

from __future__ import annotations

from src.core.config import get_settings

try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore
except Exception:  # provide no-op dummies if prometheus not installed

    class _Dummy:
        def labels(self, **kwargs):  # type: ignore
            return self

        def inc(self, *a, **kw):  # type: ignore
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


ocr_requests_total = Counter(
    "ocr_requests_total",
    "Number of OCR requests",
    ["provider", "status"],
)
ocr_processing_duration_seconds = Histogram(
    "ocr_processing_duration_seconds",
    "OCR end-to-end processing duration",
    ["provider"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
ocr_fallback_triggered = Counter(
    "ocr_fallback_triggered",
    "Fallback triggered count",
    ["reason"],
)
ocr_model_loaded = Gauge(
    "ocr_model_loaded",
    "Provider model loaded flag",
    ["provider"],
)

ocr_errors_total = Counter(
    "ocr_errors_total",
    "OCR errors",
    ["provider", "code", "stage"],
)

# Input validation rejections (parity with vision_input_rejected_total)
ocr_input_rejected_total = Counter(
    "ocr_input_rejected_total",
    "OCR input validation rejections",
    ["reason"],
)

# New metrics for confidence & completeness distributions
try:
    from prometheus_client import Summary  # type: ignore
except Exception:  # reuse dummy if not installed

    def Summary(*a, **kw):  # type: ignore
        return _Dummy()


ocr_confidence_distribution = Histogram(
    "ocr_confidence_distribution",
    "Raw provider confidence distribution",
    ["provider"],
    buckets=[0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0],
)
ocr_completeness_ratio = Histogram(
    "ocr_completeness_ratio",
    "Parsing completeness ratio",
    ["provider"],
    buckets=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
)
ocr_cold_start_seconds = Gauge(
    "ocr_cold_start_seconds",
    "Provider cold start duration seconds",
    ["provider"],
)

ocr_stage_duration_seconds = Histogram(
    "ocr_stage_duration_seconds",
    "Per-stage OCR processing duration seconds",
    ["provider", "stage"],
    buckets=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
)

# Per-item confidence distribution (dimensions/symbols)
ocr_item_confidence_distribution = Histogram(
    "ocr_item_confidence_distribution",
    "Per-item confidence distribution (dimensions & symbols)",
    ["provider", "item_type"],
    buckets=[0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0],
)

# Dynamic threshold metrics
ocr_confidence_fallback_threshold = Gauge(
    "ocr_confidence_fallback_threshold",
    "Current dynamic confidence fallback threshold (EMA-adjusted)",
)
ocr_confidence_ema = Gauge(
    "ocr_confidence_ema",
    "EMA of (calibrated) confidence used for threshold adaptation",
)

# Distributed control metrics
ocr_rate_limited_total = Counter(
    "ocr_rate_limited_total",
    "Requests rejected by rate limiter",
)
ocr_circuit_state = Gauge(
    "ocr_circuit_state",
    "Circuit breaker state per key/provider (0=closed,1=half_open,2=open)",
    ["key"],
)

# ========== Vision Metrics ==========
vision_requests_total = Counter(
    "vision_requests_total",
    "Number of vision analysis requests",
    ["provider", "status"],
)
vision_processing_duration_seconds = Histogram(
    "vision_processing_duration_seconds",
    "Vision analysis processing duration seconds",
    ["provider"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)
vision_errors_total = Counter(
    "vision_errors_total",
    "Vision analysis errors",
    ["provider", "code"],
)
vision_input_rejected_total = Counter(
    "vision_input_rejected_total",
    "Vision input validation rejections",
    ["reason"],
)

# Input image size distribution (bytes)
vision_image_size_bytes = Histogram(
    "vision_image_size_bytes",
    "Size of input images in bytes for vision analysis",
    buckets=[
        1_000,
        5_000,
        10_000,
        25_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        25_000_000,
        50_000_000,
    ],
)

# OCR input size distribution (bytes)
ocr_image_size_bytes = Histogram(
    "ocr_image_size_bytes",
    "Size of input images in bytes for OCR extraction",
    buckets=[
        1_000,
        5_000,
        10_000,
        25_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        25_000_000,
        50_000_000,
    ],
)

# Rolling error rates (EMA) exposed for /health
ocr_error_rate_ema = Gauge(
    "ocr_error_rate_ema",
    "Exponential moving average of OCR error rate (0..1)",
)
vision_error_rate_ema = Gauge(
    "vision_error_rate_ema",
    "Exponential moving average of Vision error rate (0..1)",
)

# Health endpoints ( /health, /health/extended, /ready )
health_requests_total = Counter(
    "health_requests_total",
    "Health endpoint requests",
    ["endpoint", "status"],
)
health_request_duration_seconds = Histogram(
    "health_request_duration_seconds",
    "Health endpoint request duration seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# Core provider framework health/readiness checks
core_provider_checks_total = Counter(
    "core_provider_checks_total",
    "Core provider readiness/health checks",
    ["source", "domain", "provider", "result"],
)
core_provider_check_duration_seconds = Histogram(
    "core_provider_check_duration_seconds",
    "Core provider readiness/health check duration seconds",
    ["source", "domain", "provider"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
core_provider_plugin_bootstrap_total = Counter(
    "core_provider_plugin_bootstrap_total",
    "Core provider plugin bootstrap outcomes",
    ["result"],
)
core_provider_plugin_bootstrap_duration_seconds = Histogram(
    "core_provider_plugin_bootstrap_duration_seconds",
    "Core provider plugin bootstrap duration seconds",
    ["result"],
    buckets=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)
core_provider_plugin_configured = Gauge(
    "core_provider_plugin_configured",
    "Number of configured core provider plugins",
)
core_provider_plugin_loaded = Gauge(
    "core_provider_plugin_loaded",
    "Number of loaded core provider plugins",
)
core_provider_plugin_errors = Gauge(
    "core_provider_plugin_errors",
    "Number of core provider plugin bootstrap errors",
)

# Simple in-process EMA trackers; callers should update on success/error events.
_EMA_ALPHA = float(get_settings().ERROR_EMA_ALPHA)
_ocr_error_rate_value = 0.0
_vision_error_rate_value = 0.0


def update_ocr_error_ema(is_error: bool) -> None:
    global _ocr_error_rate_value
    target = 1.0 if is_error else 0.0
    _ocr_error_rate_value = _EMA_ALPHA * target + (1 - _EMA_ALPHA) * _ocr_error_rate_value
    try:
        ocr_error_rate_ema.set(_ocr_error_rate_value)
    except Exception:
        pass


def update_vision_error_ema(is_error: bool) -> None:
    global _vision_error_rate_value
    target = 1.0 if is_error else 0.0
    _vision_error_rate_value = _EMA_ALPHA * target + (1 - _EMA_ALPHA) * _vision_error_rate_value
    try:
        vision_error_rate_ema.set(_vision_error_rate_value)
    except Exception:
        pass


def get_ocr_error_rate_ema() -> float:
    """Return the current in-process EMA value for OCR error rate."""
    return float(_ocr_error_rate_value)


def get_vision_error_rate_ema() -> float:
    """Return the current in-process EMA value for Vision error rate."""
    return float(_vision_error_rate_value)
