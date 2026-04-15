"""Helpers for baseline L1/L2/L3 classification policy."""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

BaselineClassifierFn = Callable[[Any, Mapping[str, Any]], Awaitable[Dict[str, Any]]]
FusionClassifierFactory = Callable[[], Any]
L2FeatureBuilder = Callable[[Any], Dict[str, Any]]
DocMetadataBuilder = Callable[[Any], Dict[str, Any]]


def _build_text_signals(doc: Any) -> str:
    parts: list[str] = []
    stem, _ = os.path.splitext(str(getattr(doc, "file_name", "") or ""))
    if stem:
        parts.append(stem)

    metadata = getattr(doc, "metadata", {}) or {}
    text = metadata.get("text")
    if text:
        parts.append(str(text))

    text_content = metadata.get("text_content")
    if isinstance(text_content, list):
        parts.extend([str(item) for item in text_content if str(item).strip()])

    meta = metadata.get("meta")
    if isinstance(meta, dict):
        for key in (
            "drawing_number",
            "drawing_no",
            "drawingNo",
            "drawingNumber",
            "number",
        ):
            value = meta.get(key)
            if value:
                parts.append(str(value))
    return " ".join(parts).strip()


def _build_entity_counts(doc: Any) -> Dict[str, int]:
    entity_counts: Dict[str, int] = {}
    for entity in getattr(doc, "entities", []) or []:
        kind = str(getattr(entity, "kind", "") or "").strip()
        if not kind:
            continue
        entity_counts[kind] = entity_counts.get(kind, 0) + 1
    return entity_counts


def _build_fused_classification(
    fused_result: Mapping[str, Any], *, rule_version: str
) -> Dict[str, Any]:
    return {
        "type": fused_result["type"],
        "confidence": fused_result["confidence"],
        "sub_type": None,
        "characteristics": [],
        "rule_version": rule_version,
        "alternatives": fused_result.get("alternatives", []),
        "confidence_breakdown": fused_result.get("fusion_breakdown"),
    }


def _build_initial_payload(classification: Mapping[str, Any]) -> Dict[str, Any]:
    rule_version = str(classification.get("rule_version") or "")
    return {
        "part_type": classification["type"],
        "confidence": classification["confidence"],
        "sub_type": classification.get("sub_type"),
        "characteristics": classification.get("characteristics", []),
        "rule_version": classification.get("rule_version"),
        "alternatives": classification.get("alternatives", []),
        "confidence_breakdown": classification.get("confidence_breakdown"),
        "confidence_source": (
            "fusion"
            if rule_version.startswith("L3-Fusion")
            or rule_version.startswith("L2-Fusion")
            else "rules"
        ),
    }


def _default_build_l2_features(doc: Any) -> Dict[str, Any]:
    from src.core.knowledge.fusion_analyzer import build_l2_features

    return build_l2_features(doc)


def _default_build_doc_metadata(doc: Any) -> Dict[str, Any]:
    from src.core.knowledge.fusion_analyzer import build_doc_metadata

    return build_doc_metadata(doc)


def _default_get_fusion_classifier() -> Any:
    from src.core.knowledge.fusion import get_fusion_classifier

    return get_fusion_classifier()


async def build_baseline_classification_context(
    *,
    doc: Any,
    features: Mapping[str, Any],
    features_3d: Optional[Mapping[str, Any]],
    classify_part: BaselineClassifierFn,
    logger_instance: Optional[logging.Logger] = None,
    build_l2_features_fn: Optional[L2FeatureBuilder] = None,
    build_doc_metadata_fn: Optional[DocMetadataBuilder] = None,
    fusion_classifier_factory: Optional[FusionClassifierFactory] = None,
) -> Dict[str, Any]:
    """Build baseline classification payload plus signals reused downstream."""
    active_logger = logger_instance or logger
    l2_feature_builder = build_l2_features_fn or _default_build_l2_features
    doc_metadata_builder = build_doc_metadata_fn or _default_build_doc_metadata
    fusion_factory = fusion_classifier_factory or _default_get_fusion_classifier

    text_signals = _build_text_signals(doc)
    entity_counts = _build_entity_counts(doc)

    try:
        l2_features = l2_feature_builder(doc)
    except Exception:
        l2_features = {}
    try:
        doc_metadata = doc_metadata_builder(doc)
    except Exception:
        doc_metadata = {}
    l3_features = {
        key: value
        for key, value in dict(features_3d or {}).items()
        if key != "embedding_vector"
    }

    if features_3d:
        try:
            fusion = fusion_factory()
            fused_result = fusion.classify(
                text_signals=text_signals,
                features_2d={
                    "geometric_features": l2_features,
                    "entity_counts": entity_counts,
                },
                features_3d=dict(features_3d),
            )
            classification = _build_fused_classification(
                fused_result, rule_version="L3-Fusion-v1"
            )
        except Exception as exc:  # noqa: BLE001
            active_logger.error("Fusion failed, falling back to L1: %s", exc)
            classification = await classify_part(doc, features)
    else:
        classification = await classify_part(doc, features)
        if text_signals or entity_counts:
            try:
                fusion = fusion_factory()
                fused_result = fusion.classify(
                    text_signals=text_signals,
                    features_2d={
                        "geometric_features": l2_features,
                        "entity_counts": entity_counts,
                    },
                    features_3d={},
                )
                if (
                    fused_result.get("type") not in {None, "unknown"}
                    and float(fused_result.get("confidence") or 0.0) > 0.0
                ):
                    classification = _build_fused_classification(
                        fused_result, rule_version="L2-Fusion-v1"
                    )
            except Exception as exc:  # noqa: BLE001
                active_logger.error("Fusion failed, falling back to L1: %s", exc)

    return {
        "payload": _build_initial_payload(classification),
        "text_signals": text_signals,
        "entity_counts": entity_counts,
        "doc_metadata": doc_metadata,
        "l2_features": l2_features,
        "l3_features": l3_features,
    }


async def build_baseline_classification_payload(
    *,
    doc: Any,
    features: Mapping[str, Any],
    features_3d: Optional[Mapping[str, Any]],
    classify_part: BaselineClassifierFn,
    logger_instance: Optional[logging.Logger] = None,
    build_l2_features_fn: Optional[L2FeatureBuilder] = None,
    build_doc_metadata_fn: Optional[DocMetadataBuilder] = None,
    fusion_classifier_factory: Optional[FusionClassifierFactory] = None,
) -> Dict[str, Any]:
    """Build only the initial payload for direct callers and unit tests."""
    baseline_context = await build_baseline_classification_context(
        doc=doc,
        features=features,
        features_3d=features_3d,
        classify_part=classify_part,
        logger_instance=logger_instance,
        build_l2_features_fn=build_l2_features_fn,
        build_doc_metadata_fn=build_doc_metadata_fn,
        fusion_classifier_factory=fusion_classifier_factory,
    )
    return dict(baseline_context["payload"])


__all__ = [
    "build_baseline_classification_context",
    "build_baseline_classification_payload",
]
