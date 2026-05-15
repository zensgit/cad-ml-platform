"""Manufacturing decision summary helpers for analyze flows."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def _resolve_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _build_primary_process_payload(
    process_payload: Mapping[str, Any]
) -> Dict[str, Any]:
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


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_manufacturing_evidence(
    *,
    quality_payload: Optional[Mapping[str, Any]] = None,
    process_payload: Optional[Mapping[str, Any]] = None,
    cost_payload: Optional[Mapping[str, Any]] = None,
    manufacturing_decision: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build DecisionService-compatible evidence rows for manufacturing outputs."""
    quality = _resolve_payload(quality_payload)
    process = _resolve_payload(process_payload)
    cost = _resolve_payload(cost_payload)
    decision = _resolve_payload(manufacturing_decision)

    evidence: List[Dict[str, Any]] = []

    if quality:
        score = _safe_float(quality.get("score") or quality.get("dfm_score"))
        manufacturability = _clean_text(quality.get("manufacturability"))
        quality_details = {
            key: value
            for key, value in {
                "mode": quality.get("mode"),
                "score": score,
                "issues": quality.get("issues", []),
                "suggestions": quality.get("suggestions", []),
            }.items()
            if value not in (None, "", [], {})
        }
        evidence.append(
            {
                "source": "dfm",
                "kind": "manufacturability_check",
                "label": manufacturability,
                "confidence": round(score / 100.0, 6) if score is not None else None,
                "status": manufacturability or "checked",
                "details": quality_details,
            }
        )

    if process:
        primary_process = _build_primary_process_payload(process)
        alternatives = process.get("alternatives")
        alternatives_count = len(alternatives) if isinstance(alternatives, list) else 0
        confidence = _safe_float(primary_process.get("confidence"))
        process_details = {
            key: value
            for key, value in {
                "primary_recommendation": primary_process or None,
                "alternatives_count": alternatives_count,
                "analysis_mode": process.get("analysis_mode"),
                "rule_version": process.get("rule_version"),
            }.items()
            if value not in (None, "", [], {})
        }
        evidence.append(
            {
                "source": "manufacturing_process",
                "kind": "process_recommendation",
                "label": _clean_text(primary_process.get("process")),
                "confidence": confidence,
                "status": "recommended" if primary_process else "available",
                "details": process_details,
            }
        )

    if cost:
        total_cost = cost.get("total_unit_cost")
        cost_details = dict(cost)
        cost_details.setdefault("cost_range", _build_cost_range(total_cost))
        evidence.append(
            {
                "source": "manufacturing_cost",
                "kind": "cost_estimate",
                "label": _clean_text(cost.get("currency")),
                "confidence": None,
                "status": "estimated",
                "details": {
                    key: value
                    for key, value in cost_details.items()
                    if value not in (None, "", [], {})
                },
            }
        )

    if decision:
        decision_process = _resolve_payload(decision.get("process"))
        risks = decision.get("risks", [])
        risk_count = len(risks) if isinstance(risks, list) else 0
        evidence.append(
            {
                "source": "manufacturing_decision",
                "kind": "manufacturing_summary",
                "label": _clean_text(decision_process.get("process")),
                "confidence": None,
                "status": _clean_text(decision.get("feasibility")) or "summarized",
                "details": {
                    key: value
                    for key, value in {
                        "feasibility": decision.get("feasibility"),
                        "risks_count": risk_count,
                        "cost_range": decision.get("cost_range"),
                        "currency": decision.get("currency"),
                        "has_cost_estimate": bool(decision.get("cost_estimate")),
                    }.items()
                    if value not in (None, "", [], {})
                },
            }
        )

    return evidence


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


__all__ = [
    "build_manufacturing_decision_summary",
    "build_manufacturing_evidence",
]
