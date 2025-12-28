"""Shared health payload builder to avoid circular imports."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.api.health_resilience import get_resilience_health
from src.core.config import get_settings
from src.utils.metrics import get_ocr_error_rate_ema, get_vision_error_rate_ema

try:
    from prometheus_client import make_asgi_app as _make_asgi_app  # noqa: F401

    _METRICS_ENABLED = True
except Exception:  # pragma: no cover - optional dependency
    _METRICS_ENABLED = False


def metrics_enabled() -> bool:
    return _METRICS_ENABLED


def build_health_payload(metrics_enabled_override: Optional[bool] = None) -> Dict[str, Any]:
    """Build health payload shared by /health and /api/v1/health."""
    metrics_enabled = (
        _METRICS_ENABLED if metrics_enabled_override is None else metrics_enabled_override
    )
    current_settings = get_settings()

    base = {
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

    try:
        base.update(get_resilience_health())
    except Exception:
        pass

    return base
