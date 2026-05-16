from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Callable, Optional, Sequence

from fastapi import UploadFile

from src.core.classification.coarse_labels import normalize_coarse_label
from src.core.classification.decision_service import DecisionService
from src.core.classification.finalization import finalize_classification_payload

ClassifierGetter = Callable[[], Any]


def _merge_review_reasons(*reason_groups: Any) -> list[str]:
    merged: list[str] = []
    for group in reason_groups:
        if group is None:
            continue
        if isinstance(group, str):
            values = [group]
        elif isinstance(group, (list, tuple, set)):
            values = list(group)
        else:
            values = [group]
        for value in values:
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _finalize_batch_payload(payload: Optional[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    source_payload = dict(payload or {})
    result = finalize_classification_payload(source_payload, **kwargs)
    classifier_reasons = _merge_review_reasons(
        source_payload.get("review_reasons"),
        source_payload.get("review_reason"),
    )
    if source_payload.get("needs_review") and not classifier_reasons:
        classifier_reasons.append("classifier_review")
    if source_payload.get("needs_review") or classifier_reasons:
        result["needs_review"] = True
        result["review_reasons"] = _merge_review_reasons(
            result.get("review_reasons"),
            classifier_reasons,
        )
        result["review_reason_text"] = ";".join(result["review_reasons"])
        if not str(result.get("review_priority") or "").strip() or (
            result.get("review_priority") == "none"
        ):
            result["review_priority"] = "medium"
    return result


def _build_batch_decision_payload(
    *,
    category: Optional[str],
    confidence: Optional[float],
    probabilities: Optional[dict[str, float]],
    classifier: Optional[str],
    needs_review: bool,
    review_reason: Optional[str],
    top2_category: Optional[str],
    top2_confidence: Optional[float],
) -> dict[str, Any]:
    fine_category = str(category or "").strip() or None
    classifier_name = str(classifier or "batch_classifier").strip() or "batch_classifier"
    return {
        "part_type": fine_category,
        "fine_part_type": fine_category,
        "confidence": confidence,
        "confidence_source": classifier_name,
        "rule_version": classifier_name,
        "probabilities": probabilities,
        "needs_review": bool(needs_review),
        "review_reason": review_reason,
        "review_reasons": [review_reason] if review_reason else [],
        "alternatives": (
            [{"label": top2_category, "confidence": top2_confidence}]
            if top2_category
            else []
        ),
        "part_classifier_prediction": {
            "label": fine_category,
            "confidence": confidence,
            "status": "ok" if fine_category else "no_prediction",
            "model_version": classifier_name,
        },
    }


def build_batch_classify_item(
    *,
    file_name: str,
    category: Optional[str],
    confidence: Optional[float],
    probabilities: Optional[dict[str, float]],
    classifier: Optional[str],
    needs_review: bool = False,
    review_reason: Optional[str] = None,
    top2_category: Optional[str] = None,
    top2_confidence: Optional[float] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    fine_category = str(category or "").strip() or None
    coarse_category = normalize_coarse_label(fine_category)
    is_coarse_label = None
    if fine_category:
        is_coarse_label = fine_category == coarse_category
    decision_payload = _build_batch_decision_payload(
        category=fine_category,
        confidence=confidence,
        probabilities=probabilities,
        classifier=classifier,
        needs_review=needs_review,
        review_reason=review_reason,
        top2_category=top2_category,
        top2_confidence=top2_confidence,
    )
    decided = DecisionService(finalize_fn=_finalize_batch_payload).decide(
        decision_payload,
    )
    return {
        "file_name": file_name,
        "category": fine_category,
        "fine_category": fine_category,
        "coarse_category": coarse_category,
        "is_coarse_label": is_coarse_label,
        "part_type": decided.get("part_type"),
        "fine_part_type": decided.get("fine_part_type"),
        "coarse_part_type": decided.get("coarse_part_type"),
        "decision_source": decided.get("decision_source"),
        "branch_conflicts": decided.get("branch_conflicts"),
        "evidence": decided.get("evidence"),
        "review_reasons": decided.get("review_reasons"),
        "fallback_flags": decided.get("fallback_flags"),
        "contract_version": decided.get("contract_version"),
        "decision_contract": decided.get("decision_contract"),
        "confidence": confidence,
        "probabilities": probabilities,
        "needs_review": bool(decided.get("needs_review")),
        "review_reason": review_reason,
        "top2_category": top2_category,
        "top2_confidence": top2_confidence,
        "classifier": classifier,
        "error": error,
    }


async def run_batch_classify_pipeline(
    *,
    files: Sequence[UploadFile],
    max_workers: Optional[int],
    logger,
    get_v16_classifier: Optional[ClassifierGetter] = None,
    get_ml_classifier: Optional[ClassifierGetter] = None,
) -> dict[str, Any]:
    if get_v16_classifier is None:
        from src.core.analyzer import _get_v16_classifier

        get_v16_classifier = _get_v16_classifier

    if get_ml_classifier is None:
        from src.core.analyzer import _get_ml_classifier

        get_ml_classifier = _get_ml_classifier

    start_time = time.time()
    results: list[dict[str, Any]] = []
    temp_files: list[str] = []
    valid_result_indices: list[int] = []

    try:
        for file in files:
            suffix = os.path.splitext(file.filename or "")[1].lower()
            if suffix not in (".dxf", ".dwg"):
                results.append(
                    {
                        "file_name": file.filename or "unknown",
                        "error": (
                            f"Unsupported format: {suffix}, only .dxf and .dwg are supported"
                        ),
                    }
                )
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_files.append(tmp.name)
                valid_result_indices.append(len(results))
                results.append({"file_name": file.filename or tmp.name})

        classifier = get_v16_classifier()

        if classifier is None:
            logger.warning("V16 classifier not available, falling back to sequential")
            ml_classifier = get_ml_classifier()
            for result_idx, temp_path in zip(valid_result_indices, temp_files):
                try:
                    if ml_classifier:
                        result = ml_classifier.predict(temp_path)
                        if result:
                            results[result_idx] = build_batch_classify_item(
                                file_name=results[result_idx]["file_name"],
                                category=result.category,
                                confidence=round(result.confidence, 4),
                                probabilities={
                                    k: round(v, 4)
                                    for k, v in result.probabilities.items()
                                },
                                classifier="ml_v6",
                                needs_review=bool(
                                    getattr(result, "needs_review", False)
                                ),
                                review_reason=getattr(result, "review_reason", None),
                                top2_category=getattr(result, "top2_category", None),
                                top2_confidence=getattr(
                                    result, "top2_confidence", None
                                ),
                            )
                        else:
                            results[result_idx] = {
                                "file_name": results[result_idx]["file_name"],
                                "error": "Classification returned None",
                            }
                    else:
                        results[result_idx] = {
                            "file_name": results[result_idx]["file_name"],
                            "error": "No classifier available",
                        }
                except Exception as exc:
                    results[result_idx] = {
                        "file_name": results[result_idx]["file_name"],
                        "error": str(exc),
                    }
        else:
            batch_results = classifier.predict_batch(temp_files, max_workers=max_workers)
            for result_idx, result in zip(valid_result_indices, batch_results):
                if result:
                    results[result_idx] = build_batch_classify_item(
                        file_name=results[result_idx]["file_name"],
                        category=result.category,
                        confidence=round(result.confidence, 4),
                        probabilities={
                            k: round(v, 4) for k, v in result.probabilities.items()
                        },
                        needs_review=getattr(result, "needs_review", False),
                        review_reason=getattr(result, "review_reason", None),
                        top2_category=getattr(result, "top2_category", None),
                        top2_confidence=getattr(result, "top2_confidence", None),
                        classifier=getattr(result, "model_version", "v16"),
                    )
                else:
                    results[result_idx] = {
                        "file_name": results[result_idx]["file_name"],
                        "error": "Classification returned None",
                    }

        success_count = sum(1 for item in results if item.get("category") is not None)
        failed_count = len(results) - success_count
        processing_time = round(time.time() - start_time, 3)

        try:
            from src.utils.analysis_metrics import (
                v16_batch_classify_files_total,
                v16_batch_classify_requests_total,
                v16_classifier_batch_seconds,
                v16_classifier_batch_size,
            )

            v16_classifier_batch_seconds.observe(processing_time)
            v16_classifier_batch_size.observe(len(files))

            if failed_count == 0:
                v16_batch_classify_requests_total.labels(status="success").inc()
            elif success_count == 0:
                v16_batch_classify_requests_total.labels(status="failed").inc()
            else:
                v16_batch_classify_requests_total.labels(status="partial").inc()

            v16_batch_classify_files_total.labels(result="success").inc(success_count)
            v16_batch_classify_files_total.labels(result="failed").inc(failed_count)
        except Exception:
            pass

        return {
            "total": len(files),
            "success": success_count,
            "failed": failed_count,
            "processing_time": processing_time,
            "results": results,
        }
    finally:
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


__all__ = ["build_batch_classify_item", "run_batch_classify_pipeline"]
