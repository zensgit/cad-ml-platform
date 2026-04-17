"""Manufacturing decision summary helpers for analyze flows."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def _resolve_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _build_primary_process_payload(process_payload: Mapping[str, Any]) -> Dict[str, Any]:
    primary = process_payload.get("primary_recommendation")
    if isinstance(primary, Mapping) and primary:
        return dict(primary)
    if "process" in process_payload:
        return {
            "process": process_payload.get("process"),
            "method": process_payload.get("method"),
        }
    return {}


def _build_cost_range(total_cost: Any) -> Optional[Dict[str, float]]:
    if isinstance(total_cost, (int, float)):
        normalized_cost = float(total_cost)
        return {
            "low": round(normalized_cost * 0.9, 2),
            "high": round(normalized_cost * 1.1, 2),
        }
    return None


def build_manufacturing_decision_summary(
    *,
    quality_payload: Optional[Mapping[str, Any]] = None,
    process_payload: Optional[Mapping[str, Any]] = None,
    cost_payload: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the combined manufacturing decision summary for analyze results."""
    quality = _resolve_payload(quality_payload)
    process = _resolve_payload(process_payload)
    cost = _resolve_payload(cost_payload)

    if not (quality or process or cost):
        return None

    primary_process = _build_primary_process_payload(process)
    cost_range = _build_cost_range(cost.get("total_unit_cost"))

    return {
        "feasibility": quality.get("manufacturability"),
        "risks": quality.get("issues", []),
        "process": primary_process or None,
        "cost_estimate": cost,
        "cost_range": cost_range,
        "currency": cost.get("currency"),
    }


__all__ = ["build_manufacturing_decision_summary"]
