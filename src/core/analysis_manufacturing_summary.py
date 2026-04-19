from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional


SummaryBuilder = Callable[..., Optional[Dict[str, Any]]]


def attach_manufacturing_decision_summary(
    *,
    results: Dict[str, Any],
    summary_builder: SummaryBuilder,
    logger_instance: logging.Logger,
) -> Optional[Dict[str, Any]]:
    try:
        manufacturing_decision = summary_builder(
            quality_payload=results.get("quality") if isinstance(results, dict) else None,
            process_payload=results.get("process") if isinstance(results, dict) else None,
            cost_payload=(
                results.get("cost_estimation") if isinstance(results, dict) else None
            ),
        )
        if manufacturing_decision is not None:
            results["manufacturing_decision"] = manufacturing_decision
        return manufacturing_decision
    except Exception as exc:  # pragma: no cover - behavior is tested via warning side effect
        logger_instance.warning("Manufacturing decision summary failed: %s", exc)
        return None
