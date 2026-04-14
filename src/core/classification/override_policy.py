"""Helpers for Fusion/Hybrid override policy decisions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional


def _source_text(value: Any) -> str:
    source_value = getattr(value, "value", value)
    return str(source_value or "").strip().lower()


def apply_fusion_override(
    payload: Optional[Dict[str, Any]],
    *,
    fusion_decision: Any,
    override_enabled: bool,
    min_confidence: float,
) -> Dict[str, Any]:
    """Apply FusionAnalyzer override policy without changing caller order."""
    cls_payload = dict(payload or {})
    if fusion_decision is None or not override_enabled:
        return cls_payload

    confidence = float(getattr(fusion_decision, "confidence", 0.0) or 0.0)
    if confidence < float(min_confidence):
        cls_payload["fusion_override_skipped"] = {
            "min_confidence": float(min_confidence),
            "decision_confidence": confidence,
        }
        return cls_payload

    is_default_rule = (
        _source_text(getattr(fusion_decision, "source", None)) == "rule_based"
        and list(getattr(fusion_decision, "rule_hits", []) or []) == ["RULE_DEFAULT"]
    )
    if is_default_rule:
        cls_payload["fusion_override_skipped"] = {
            "min_confidence": float(min_confidence),
            "decision_confidence": confidence,
            "reason": "default_rule_only",
        }
        return cls_payload

    cls_payload["part_type"] = getattr(fusion_decision, "primary_label", None)
    cls_payload["confidence"] = confidence
    cls_payload["rule_version"] = (
        f"FusionAnalyzer-{getattr(fusion_decision, 'schema_version', 'unknown')}"
    )
    cls_payload["confidence_source"] = "fusion"
    return cls_payload


def apply_hybrid_override(
    payload: Optional[Dict[str, Any]],
    *,
    hybrid_result: Optional[Dict[str, Any]],
    override_enabled: bool,
    auto_override_enabled: bool,
    min_confidence: float,
    base_max_confidence: float,
    is_drawing_type: Callable[[Optional[str]], bool],
    placeholder_types: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Apply Hybrid override policy without changing caller order."""
    cls_payload = dict(payload or {})
    if not hybrid_result:
        return cls_payload

    placeholder_type_set = set(
        placeholder_types
        or {"", "simple_plate", "moderate_component", "complex_assembly", "unknown", "other"}
    )
    hybrid_label = hybrid_result.get("label")
    hybrid_conf = float(hybrid_result.get("confidence", 0.0) or 0.0)

    current_part_type = str(cls_payload.get("part_type") or "").strip()
    current_is_drawing_type = bool(is_drawing_type(current_part_type))
    is_placeholder_rule = (
        str(cls_payload.get("confidence_source") or "") == "rules"
        and str(cls_payload.get("rule_version") or "") == "v1"
        and current_part_type in placeholder_type_set
    )
    base_conf = float(cls_payload.get("confidence", 0.0) or 0.0)
    is_low_conf_base = (
        str(cls_payload.get("confidence_source") or "") == "rules"
        and base_conf < float(base_max_confidence)
    )

    mode: Optional[str] = None
    should_override = False
    if override_enabled:
        mode = "env"
        should_override = bool(hybrid_label) and hybrid_conf >= float(min_confidence)
    elif auto_override_enabled and is_placeholder_rule:
        mode = "auto"
        should_override = bool(hybrid_label) and hybrid_conf >= float(min_confidence)
    elif auto_override_enabled and is_low_conf_base:
        mode = "auto_low_conf"
        should_override = bool(hybrid_label) and hybrid_conf >= float(min_confidence)
    elif auto_override_enabled and current_is_drawing_type:
        mode = "auto_drawing_type"
        should_override = bool(hybrid_label) and hybrid_conf >= float(min_confidence)

    if should_override:
        cls_payload["hybrid_override_applied"] = {
            "mode": mode,
            "min_confidence": float(min_confidence),
            "base_max_confidence": float(base_max_confidence),
            "previous_part_type": cls_payload.get("part_type"),
            "previous_confidence": cls_payload.get("confidence"),
            "previous_rule_version": cls_payload.get("rule_version"),
            "previous_confidence_source": cls_payload.get("confidence_source"),
        }
        cls_payload["part_type"] = hybrid_label
        cls_payload["confidence"] = hybrid_conf
        cls_payload["rule_version"] = "HybridClassifier-v1"
        cls_payload["confidence_source"] = "hybrid"
    elif override_enabled:
        cls_payload["hybrid_override_skipped"] = {
            "min_confidence": float(min_confidence),
            "decision_confidence": hybrid_conf,
            "label": hybrid_label,
        }
    return cls_payload


__all__ = ["apply_fusion_override", "apply_hybrid_override"]
