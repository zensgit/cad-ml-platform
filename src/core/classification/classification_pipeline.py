"""End-to-end classification orchestration for analyze flows."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from src.core.classification.active_learning_policy import (
    flag_classification_for_review,
)
from src.core.classification.baseline_policy import (
    build_baseline_classification_context,
)
from src.core.classification.decision_service import DecisionService
from src.core.classification.finalization import finalize_classification_payload
from src.core.classification.fusion_pipeline import (
    build_fusion_classification_context,
)
from src.core.classification.hybrid_override_pipeline import (
    build_hybrid_override_context,
)
from src.core.classification.shadow_pipeline import (
    _graph2d_is_drawing_type,
    _safe_float_env,
    build_shadow_classification_context,
)

logger = logging.getLogger(__name__)

BaselineClassifierFn = Callable[[Any, Mapping[str, Any]], Awaitable[Dict[str, Any]]]


async def run_classification_pipeline(
    *,
    analysis_id: str,
    doc: Any,
    features: Mapping[str, Any],
    features_3d: Optional[Mapping[str, Any]],
    file_name: Optional[str],
    file_format: str,
    content: bytes,
    analysis_options: Any,
    classify_part: BaselineClassifierFn,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run the full classification decision chain without API-side orchestration."""
    active_logger = logger_instance or logger

    baseline_context = await build_baseline_classification_context(
        doc=doc,
        features=features,
        features_3d=features_3d,
        classify_part=classify_part,
        logger_instance=active_logger,
    )
    cls_payload = baseline_context["payload"]
    text_signals = baseline_context["text_signals"]
    ent_counts = baseline_context["entity_counts"]
    doc_metadata = baseline_context["doc_metadata"]
    l2_features = baseline_context["l2_features"]
    l3_features = baseline_context["l3_features"]

    shadow_context = await build_shadow_classification_context(
        cls_payload,
        features=features,
        file_name=file_name,
        file_format=file_format,
        content=content,
        analysis_options=analysis_options,
    )
    cls_payload = shadow_context["payload"]
    ml_result = shadow_context["ml_result"]
    graph2d_fusable = shadow_context["graph2d_fusable"]
    hybrid_result = shadow_context["hybrid_result"]

    try:
        fusion_context = build_fusion_classification_context(
            cls_payload,
            doc_metadata=doc_metadata,
            l2_features=l2_features,
            l3_features=l3_features,
            ml_result=ml_result,
            graph2d_fusable=graph2d_fusable,
        )
        cls_payload = fusion_context["payload"]
    except Exception as exc:  # noqa: BLE001
        active_logger.error("FusionAnalyzer failed: %s", exc)

    hybrid_context = build_hybrid_override_context(
        cls_payload,
        hybrid_result=hybrid_result,
        is_drawing_type=_graph2d_is_drawing_type,
    )
    cls_payload = hybrid_context["payload"]

    review_low_conf_threshold = _safe_float_env(
        "ANALYSIS_REVIEW_LOW_CONFIDENCE_THRESHOLD",
        _safe_float_env("ACTIVE_LEARNING_CONFIDENCE_THRESHOLD", 0.6),
    )
    review_high_conf_threshold = _safe_float_env(
        "ANALYSIS_REVIEW_HIGH_CONFIDENCE_THRESHOLD",
        0.85,
    )
    decision_service = DecisionService(finalize_fn=finalize_classification_payload)
    cls_payload = decision_service.decide(
        cls_payload,
        text_signals=text_signals,
        text_items=doc.metadata.get("text_content"),
        geometric_features=l2_features,
        entity_counts=ent_counts,
        features_3d=l3_features,
        low_confidence_threshold=review_low_conf_threshold,
        high_confidence_threshold=review_high_conf_threshold,
    )

    try:
        flag_classification_for_review(
            analysis_id=analysis_id,
            cls_payload=cls_payload,
        )
    except Exception as exc:  # noqa: BLE001
        active_logger.warning("Active learning flag failed: %s", exc)

    return cls_payload


__all__ = ["run_classification_pipeline"]
