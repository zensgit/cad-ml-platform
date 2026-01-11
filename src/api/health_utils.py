"""Shared health payload builder to avoid circular imports."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.api.health_models import HealthResponse
from src.api.health_resilience import get_resilience_health
from src.core.config import get_settings
from src.utils.metrics import health_request_duration_seconds, health_requests_total
from src.utils.metrics import get_ocr_error_rate_ema, get_vision_error_rate_ema

try:
    from prometheus_client import make_asgi_app as _make_asgi_app  # noqa: F401

    _METRICS_ENABLED = True
except Exception:  # pragma: no cover - optional dependency
    _METRICS_ENABLED = False


def metrics_enabled() -> bool:
    return _METRICS_ENABLED


def record_health_request(endpoint: str, status: str, duration_seconds: float) -> None:
    try:
        health_requests_total.labels(endpoint=endpoint, status=status).inc()
        health_request_duration_seconds.labels(endpoint=endpoint).observe(duration_seconds)
    except Exception:
        # Metrics collection should never block health responses.
        pass


def build_health_payload(metrics_enabled_override: Optional[bool] = None) -> Dict[str, Any]:
    """Build health payload shared by /health and /api/v1/health."""
    metrics_enabled = (
        _METRICS_ENABLED if metrics_enabled_override is None else metrics_enabled_override
    )
    current_settings = get_settings()

    base: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "up",
            "ml": "up",
            "redis": "up" if current_settings.REDIS_ENABLED else "disabled",
        },
        "runtime": {
            "python_version": sys.version.split(" ")[0],
            "metrics_enabled": metrics_enabled,
            "vision_max_base64_bytes": current_settings.VISION_MAX_BASE64_BYTES,
            "error_rate_ema": {
                "ocr": get_ocr_error_rate_ema(),
                "vision": get_vision_error_rate_ema(),
            },
        },
        "config": {
            "limits": {
                "vision_max_base64_bytes": current_settings.VISION_MAX_BASE64_BYTES,
                "vision_max_base64_mb": round(
                    current_settings.VISION_MAX_BASE64_BYTES / 1024 / 1024, 2
                ),
                "ocr_timeout_ms": current_settings.OCR_TIMEOUT_MS,
                "ocr_timeout_seconds": current_settings.OCR_TIMEOUT_MS / 1000,
            },
            "providers": {
                "ocr_default": current_settings.OCR_PROVIDER_DEFAULT,
                "confidence_fallback": current_settings.CONFIDENCE_FALLBACK,
            },
            "monitoring": {
                "error_ema_alpha": current_settings.ERROR_EMA_ALPHA,
                "metrics_enabled": metrics_enabled,
                "redis_enabled": current_settings.REDIS_ENABLED,
            },
            "network": {
                "cors_origins": current_settings.CORS_ORIGINS,
                "allowed_hosts": current_settings.ALLOWED_HOSTS,
            },
            "debug": {
                "debug_mode": current_settings.DEBUG,
                "log_level": current_settings.LOG_LEVEL,
            },
        },
    }

    resilience_payload = None
    try:
        resilience_payload = get_resilience_health().get("resilience")
    except Exception:
        resilience_payload = None

    if resilience_payload is not None:
        base["resilience"] = resilience_payload

    return HealthResponse(**base).model_dump()
