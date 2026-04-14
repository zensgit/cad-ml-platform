"""Stable helpers for final classification decision contracts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from src.core.classification.coarse_labels import normalize_coarse_label


def _clean_contract_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_contract_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def extract_label_decision_contract(
    payload: Optional[Dict[str, Any]],
    *,
    part_type_key: str = "part_type",
    fine_part_type_key: str = "fine_part_type",
    coarse_part_type_key: str = "coarse_part_type",
    decision_source_keys: Iterable[str] = (
        "final_decision_source",
        "decision_source",
        "confidence_source",
    ),
) -> Dict[str, Any]:
    """Extract a stable fine/coarse decision contract from a payload."""
    source_payload = dict(payload or {})

    part_type = _clean_contract_text(source_payload.get(part_type_key))
    fine_part_type = (
        _clean_contract_text(source_payload.get(fine_part_type_key)) or part_type
    )
    coarse_part_type = _clean_contract_text(source_payload.get(coarse_part_type_key))
    if not coarse_part_type:
        coarse_part_type = _clean_contract_text(
            normalize_coarse_label(fine_part_type or part_type)
        )
    if not part_type:
        part_type = fine_part_type or coarse_part_type

    decision_source = None
    for key in decision_source_keys:
        decision_source = _clean_contract_text(source_payload.get(key))
        if decision_source:
            break

    is_coarse_label = _coerce_contract_bool(source_payload.get("is_coarse_label"))
    if is_coarse_label is None and fine_part_type and coarse_part_type:
        is_coarse_label = fine_part_type == coarse_part_type

    return {
        "part_type": part_type,
        "fine_part_type": fine_part_type,
        "coarse_part_type": coarse_part_type,
        "decision_source": decision_source,
        "final_decision_source": decision_source,
        "is_coarse_label": is_coarse_label,
    }


def build_classification_decision_contract(
    payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the stable final-classification contract exposed by analyze."""
    contract = extract_label_decision_contract(payload)
    return {
        **contract,
        "confidence_source": _clean_contract_text(
            (payload or {}).get("confidence_source")
        )
        or contract.get("decision_source"),
        "rule_version": _clean_contract_text((payload or {}).get("rule_version")),
    }


__all__ = [
    "extract_label_decision_contract",
    "build_classification_decision_contract",
]
