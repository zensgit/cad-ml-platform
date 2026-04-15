"""Helpers for Hybrid override orchestration in analyze flows."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from src.core.classification.override_policy import apply_hybrid_override

logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "true" if default else "false")
    return str(raw or "").strip().lower() == "true"


def _safe_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%s; using default %.2f", name, raw, default)
        return float(default)


def build_hybrid_override_context(
    payload: Optional[Dict[str, Any]],
    *,
    hybrid_result: Optional[Dict[str, Any]],
    is_drawing_type: Callable[[Optional[str]], bool],
) -> Dict[str, Any]:
    """Apply env-driven Hybrid override orchestration without changing caller order."""
    cls_payload = dict(payload or {})
    override_enabled = _env_enabled("HYBRID_CLASSIFIER_OVERRIDE")
    auto_override_enabled = _env_enabled(
        "HYBRID_CLASSIFIER_AUTO_OVERRIDE", default=True
    )
    min_confidence = _safe_float_env("HYBRID_OVERRIDE_MIN_CONF", 0.8)
    base_max_confidence = _safe_float_env("HYBRID_OVERRIDE_BASE_MAX_CONF", 0.7)

    if hybrid_result:
        cls_payload = apply_hybrid_override(
            cls_payload,
            hybrid_result=hybrid_result,
            override_enabled=override_enabled,
            auto_override_enabled=auto_override_enabled,
            min_confidence=min_confidence,
            base_max_confidence=base_max_confidence,
            is_drawing_type=is_drawing_type,
        )

    return {
        "payload": cls_payload,
        "override_enabled": override_enabled,
        "auto_override_enabled": auto_override_enabled,
        "min_confidence": min_confidence,
        "base_max_confidence": base_max_confidence,
    }


__all__ = ["build_hybrid_override_context"]
