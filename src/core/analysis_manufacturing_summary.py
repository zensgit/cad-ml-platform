from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from src.core.process.manufacturing_summary import build_manufacturing_evidence


SummaryBuilder = Callable[..., Optional[Dict[str, Any]]]


def _append_manufacturing_evidence_to_classification(
    *,
    results: Dict[str, Any],
    evidence: list[Dict[str, Any]],
) -> None:
    classification = results.get("classification")
    if not isinstance(classification, dict) or not evidence:
        return

    existing = classification.get("evidence")
    merged_evidence = (
        [item for item in existing if isinstance(item, dict)]
        if isinstance(existing, list)
        else []
    )
    merged_evidence.extend(evidence)
    classification["evidence"] = merged_evidence
    classification["manufacturing_evidence"] = evidence

    decision_contract = classification.get("decision_contract")
    if isinstance(decision_contract, dict):
        decision_contract["evidence"] = merged_evidence


def attach_manufacturing_decision_summary(
    *,
    results: Dict[str, Any],
    summary_builder: SummaryBuilder,
    logger_instance: logging.Logger,
) -> Optional[Dict[str, Any]]:
    try:
        manufacturing_decision = summary_builder(
            quality_payload=(
                results.get("quality") if isinstance(results, dict) else None
            ),
            process_payload=(
                results.get("process") if isinstance(results, dict) else None
            ),
            cost_payload=(
                results.get("cost_estimation") if isinstance(results, dict) else None
            ),
        )
        if manufacturing_decision is not None:
            results["manufacturing_decision"] = manufacturing_decision
        evidence = build_manufacturing_evidence(
            quality_payload=(
                results.get("quality") if isinstance(results, dict) else None
            ),
            process_payload=(
                results.get("process") if isinstance(results, dict) else None
            ),
            cost_payload=(
                results.get("cost_estimation") if isinstance(results, dict) else None
            ),
            manufacturing_decision=manufacturing_decision,
        )
        if evidence:
            results["manufacturing_evidence"] = evidence
            _append_manufacturing_evidence_to_classification(
                results=results,
                evidence=evidence,
            )
        return manufacturing_decision
    except Exception as exc:  # pragma: no cover
        logger_instance.warning("Manufacturing decision summary failed: %s", exc)
        return None
