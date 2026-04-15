"""Helpers for FusionAnalyzer orchestration in analyze flows."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional

from src.core.classification.override_policy import apply_fusion_override

logger = logging.getLogger(__name__)

FusionAnalyzerFactory = Callable[[], Any]


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


def _default_get_fusion_analyzer() -> Any:
    from src.core.knowledge.fusion_analyzer import get_fusion_analyzer

    return get_fusion_analyzer()


def _build_l4_prediction(
    *,
    graph2d_fusion_enabled: bool,
    graph2d_fusable: Optional[Mapping[str, Any]],
    ml_result: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if (
        graph2d_fusion_enabled
        and graph2d_fusable
        and graph2d_fusable.get("label")
    ):
        return {
            "label": graph2d_fusable["label"],
            "confidence": float(graph2d_fusable.get("confidence", 0.0) or 0.0),
            "source": "graph2d",
        }

    if ml_result and ml_result.get("predicted_type"):
        return {
            "label": ml_result["predicted_type"],
            "confidence": float(ml_result.get("confidence", 0.0) or 0.0),
            "source": "ml",
        }

    return None


def build_fusion_classification_context(
    payload: Optional[Dict[str, Any]],
    *,
    doc_metadata: Mapping[str, Any],
    l2_features: Mapping[str, Any],
    l3_features: Mapping[str, Any],
    ml_result: Optional[Mapping[str, Any]],
    graph2d_fusable: Optional[Mapping[str, Any]],
    fusion_analyzer_factory: Optional[FusionAnalyzerFactory] = None,
) -> Dict[str, Any]:
    """Build FusionAnalyzer payload updates without changing caller order."""
    cls_payload = dict(payload or {})
    fusion_enabled = _env_enabled("FUSION_ANALYZER_ENABLED")
    fusion_override_enabled = _env_enabled("FUSION_ANALYZER_OVERRIDE")
    fusion_override_min_confidence = _safe_float_env(
        "FUSION_ANALYZER_OVERRIDE_MIN_CONF", 0.5
    )
    graph2d_fusion_enabled = _env_enabled("GRAPH2D_FUSION_ENABLED")

    if not fusion_enabled:
        return {
            "payload": cls_payload,
            "enabled": False,
            "graph2d_fusion_enabled": graph2d_fusion_enabled,
            "l4_prediction": None,
        }

    analyzer = (fusion_analyzer_factory or _default_get_fusion_analyzer)()
    l4_prediction = _build_l4_prediction(
        graph2d_fusion_enabled=graph2d_fusion_enabled,
        graph2d_fusable=graph2d_fusable,
        ml_result=ml_result,
    )
    fusion_decision = analyzer.analyze(
        doc_metadata=dict(doc_metadata),
        l2_features=dict(l2_features),
        l3_features=dict(l3_features),
        l4_prediction=l4_prediction,
    )

    cls_payload["fusion_decision"] = fusion_decision.model_dump()
    cls_payload["fusion_inputs"] = {
        "l1": dict(doc_metadata),
        "l2": dict(l2_features),
        "l3": dict(l3_features),
        "l4": l4_prediction,
    }
    cls_payload = apply_fusion_override(
        cls_payload,
        fusion_decision=fusion_decision,
        override_enabled=fusion_override_enabled,
        min_confidence=fusion_override_min_confidence,
    )
    return {
        "payload": cls_payload,
        "enabled": True,
        "graph2d_fusion_enabled": graph2d_fusion_enabled,
        "l4_prediction": l4_prediction,
    }


__all__ = ["build_fusion_classification_context"]
