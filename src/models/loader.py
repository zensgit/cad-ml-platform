"""Model readiness loader facade.

This module keeps the legacy ``load_models`` / ``models_loaded`` API while
delegating truth to the model readiness registry.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.models.readiness_registry import (
    ModelReadinessSnapshot,
    build_model_readiness_snapshot,
)

_last_snapshot: Optional[ModelReadinessSnapshot] = None


async def load_models() -> None:
    """Refresh model readiness evidence during application startup."""
    global _last_snapshot
    _last_snapshot = build_model_readiness_snapshot()


def get_model_readiness_snapshot(*, refresh: bool = False) -> ModelReadinessSnapshot:
    global _last_snapshot
    if refresh or _last_snapshot is None:
        _last_snapshot = build_model_readiness_snapshot()
    return _last_snapshot


def models_loaded() -> bool:
    """Return whether required model readiness gates pass."""
    return bool(get_model_readiness_snapshot(refresh=True).ok)


def models_readiness_check() -> Dict[str, Any]:
    """Return a readiness-check payload consumed by `/ready`."""
    if not models_loaded():
        snapshot = get_model_readiness_snapshot()
        return {
            "ok": False,
            "degraded": snapshot.degraded,
            "detail": ",".join(snapshot.blocking_reasons) or "model readiness failed",
        }

    snapshot = get_model_readiness_snapshot(refresh=True)
    detail = None
    if snapshot.degraded_reasons:
        detail = "degraded=" + ",".join(snapshot.degraded_reasons)
    return {
        "ok": True,
        "degraded": snapshot.degraded,
        "detail": detail,
    }


__all__ = [
    "get_model_readiness_snapshot",
    "load_models",
    "models_loaded",
    "models_readiness_check",
]
