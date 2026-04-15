"""Stable helpers for vector metadata derived from classification outputs."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.core.classification.decision_contract import extract_label_decision_contract


def extract_vector_label_contract(meta: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract stable fine/coarse semantic fields from vector metadata."""
    contract = extract_label_decision_contract(dict(meta or {}))
    return {
        "part_type": contract.get("part_type"),
        "fine_part_type": contract.get("fine_part_type"),
        "coarse_part_type": contract.get("coarse_part_type"),
        "decision_source": contract.get("decision_source"),
        "is_coarse_label": contract.get("is_coarse_label"),
    }


def build_vector_registration_metadata(
    *,
    material: Optional[str],
    doc: Any,
    features: Mapping[str, Any],
    feature_vector: list[float],
    feature_version: str,
    vector_layout: str,
    classification_meta: Optional[Mapping[str, Any]] = None,
    l3_dim: Optional[int] = None,
) -> Dict[str, str]:
    """Build the metadata payload stored alongside similarity vectors."""
    meta: Dict[str, str] = {
        "material": material or "unknown",
        "complexity": doc.complexity_bucket(),
        "format": doc.format,
        "feature_version": feature_version,
        "vector_layout": vector_layout,
        "geometric_dim": str(len(features.get("geometric", []))),
        "semantic_dim": str(len(features.get("semantic", []))),
        "total_dim": str(len(feature_vector)),
    }
    if l3_dim is not None:
        meta["l3_3d_dim"] = str(l3_dim)

    if isinstance(classification_meta, Mapping):
        contract = extract_vector_label_contract(classification_meta)
        final_part_type = str(contract.get("part_type") or "").strip()
        fine_part_type = str(contract.get("fine_part_type") or "").strip()
        coarse_part_type = str(contract.get("coarse_part_type") or "").strip()
        final_decision_source = str(contract.get("decision_source") or "").strip()
        if final_part_type:
            meta["part_type"] = final_part_type
        if fine_part_type:
            meta["fine_part_type"] = fine_part_type
        if coarse_part_type:
            meta["coarse_part_type"] = coarse_part_type
        is_coarse_label = contract.get("is_coarse_label")
        if is_coarse_label is not None:
            meta["is_coarse_label"] = str(bool(is_coarse_label)).lower()
        if final_decision_source:
            meta["final_decision_source"] = final_decision_source

    return meta


__all__ = [
    "build_vector_registration_metadata",
    "extract_vector_label_contract",
]
