"""
Feedback API.

Collects user corrections to improve L3/L4 models (Data Flywheel).
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.classification.coarse_labels import normalize_coarse_label

router = APIRouter()


class FeedbackRequest(BaseModel):
    analysis_id: str = Field(..., description="The ID of the analysis being corrected")
    corrected_part_type: Optional[str] = Field(None, description="Correct classification")
    corrected_fine_part_type: Optional[str] = Field(
        None, description="Correct fine-grained classification"
    )
    corrected_coarse_part_type: Optional[str] = Field(
        None, description="Correct coarse-grained classification"
    )
    corrected_process: Optional[str] = Field(None, description="Correct manufacturing process")
    original_part_type: Optional[str] = Field(
        None, description="Original classification returned by the system"
    )
    original_fine_part_type: Optional[str] = Field(
        None, description="Original fine-grained classification"
    )
    original_coarse_part_type: Optional[str] = Field(
        None, description="Original coarse-grained classification"
    )
    original_decision_source: Optional[str] = Field(
        None, description="Decision source that produced the original result"
    )
    review_outcome: Optional[str] = Field(
        None, description="Human review outcome, e.g. accepted/rejected/updated"
    )
    review_reasons: List[str] = Field(
        default_factory=list,
        description="Structured review reasons associated with this correction",
    )
    dfm_feedback: Optional[str] = Field(None, description="Comments on DFM accuracy")
    rating: int = Field(..., ge=1, le=5, description="1-5 star rating of the AI result")
    label_source: Optional[str] = Field(None, description="human_feedback | claude_suggestion")
    review_source: Optional[str] = Field(None, description="human | claude_assisted")
    verified_by: Optional[str] = Field(None, description="Identifier of the verifying user")


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    message: str


class FeedbackStatsResponse(BaseModel):
    status: str
    total: int
    correction_count: int
    coarse_correction_count: int
    average_rating: Optional[float] = None
    by_review_outcome: Dict[str, int] = Field(default_factory=dict)
    by_review_reason: Dict[str, int] = Field(default_factory=dict)
    by_corrected_fine_part_type: Dict[str, int] = Field(default_factory=dict)
    by_corrected_coarse_part_type: Dict[str, int] = Field(default_factory=dict)
    by_original_fine_part_type: Dict[str, int] = Field(default_factory=dict)
    by_original_coarse_part_type: Dict[str, int] = Field(default_factory=dict)
    by_original_decision_source: Dict[str, int] = Field(default_factory=dict)


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _clean_reasons(values: List[str]) -> List[str]:
    cleaned: List[str] = []
    for value in values:
        text = _clean_optional_text(value)
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_feedback_entry(payload: FeedbackRequest) -> Dict[str, Any]:
    entry = payload.model_dump()

    corrected_fine = _clean_optional_text(
        payload.corrected_fine_part_type or payload.corrected_part_type
    )
    corrected_coarse = normalize_coarse_label(
        _clean_optional_text(payload.corrected_coarse_part_type) or corrected_fine
    )

    original_fine = _clean_optional_text(
        payload.original_fine_part_type or payload.original_part_type
    )
    original_coarse = normalize_coarse_label(
        _clean_optional_text(payload.original_coarse_part_type) or original_fine
    )

    entry["corrected_part_type"] = corrected_fine
    entry["corrected_fine_part_type"] = corrected_fine
    entry["corrected_coarse_part_type"] = corrected_coarse
    entry["corrected_is_coarse_label"] = bool(
        corrected_fine and corrected_fine == corrected_coarse
    )
    entry["correct_label"] = corrected_fine
    entry["correct_fine_label"] = corrected_fine
    entry["correct_coarse_label"] = corrected_coarse

    entry["original_part_type"] = original_fine
    entry["original_fine_part_type"] = original_fine
    entry["original_coarse_part_type"] = original_coarse
    entry["original_is_coarse_label"] = bool(
        original_fine and original_fine == original_coarse
    )
    entry["original_label"] = original_fine
    entry["original_fine_label"] = original_fine
    entry["original_coarse_label"] = original_coarse
    entry["original_decision_source"] = _clean_optional_text(payload.original_decision_source)
    entry["review_outcome"] = _clean_optional_text(payload.review_outcome)
    entry["review_reasons"] = _clean_reasons(payload.review_reasons)
    return entry


def _feedback_log_path() -> str:
    return os.getenv("FEEDBACK_LOG_PATH", "data/feedback_log.jsonl")


def _read_feedback_entries() -> List[Dict[str, Any]]:
    log_path = _feedback_log_path()
    if not os.path.exists(log_path):
        return []

    entries: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    return entries


def _increment(counter: Dict[str, int], value: Optional[str]) -> None:
    text = _clean_optional_text(value)
    if not text:
        return
    counter[text] = counter.get(text, 0) + 1


def _build_feedback_stats(entries: List[Dict[str, Any]]) -> FeedbackStatsResponse:
    total = len(entries)
    correction_count = 0
    coarse_correction_count = 0
    rating_sum = 0
    rating_count = 0

    by_review_outcome: Dict[str, int] = {}
    by_review_reason: Dict[str, int] = {}
    by_corrected_fine_part_type: Dict[str, int] = {}
    by_corrected_coarse_part_type: Dict[str, int] = {}
    by_original_fine_part_type: Dict[str, int] = {}
    by_original_coarse_part_type: Dict[str, int] = {}
    by_original_decision_source: Dict[str, int] = {}

    for entry in entries:
        corrected_fine = _clean_optional_text(entry.get("corrected_fine_part_type"))
        corrected_coarse = _clean_optional_text(entry.get("corrected_coarse_part_type"))
        original_fine = _clean_optional_text(entry.get("original_fine_part_type"))
        original_coarse = _clean_optional_text(entry.get("original_coarse_part_type"))

        _increment(by_review_outcome, entry.get("review_outcome"))
        _increment(by_corrected_fine_part_type, corrected_fine)
        _increment(by_corrected_coarse_part_type, corrected_coarse)
        _increment(by_original_fine_part_type, original_fine)
        _increment(by_original_coarse_part_type, original_coarse)
        _increment(by_original_decision_source, entry.get("original_decision_source"))

        for reason in entry.get("review_reasons", []):
            _increment(by_review_reason, reason)

        if corrected_fine and original_fine and corrected_fine != original_fine:
            correction_count += 1
        if corrected_coarse and original_coarse and corrected_coarse != original_coarse:
            coarse_correction_count += 1

        rating = entry.get("rating")
        if isinstance(rating, (int, float)):
            rating_sum += int(rating)
            rating_count += 1

    average_rating = None
    if rating_count:
        average_rating = round(rating_sum / rating_count, 4)

    return FeedbackStatsResponse(
        status="success",
        total=total,
        correction_count=correction_count,
        coarse_correction_count=coarse_correction_count,
        average_rating=average_rating,
        by_review_outcome=by_review_outcome,
        by_review_reason=by_review_reason,
        by_corrected_fine_part_type=by_corrected_fine_part_type,
        by_corrected_coarse_part_type=by_corrected_coarse_part_type,
        by_original_fine_part_type=by_original_fine_part_type,
        by_original_coarse_part_type=by_original_coarse_part_type,
        by_original_decision_source=by_original_decision_source,
    )


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest, api_key: str = Depends(get_api_key)):
    """
    Submit feedback for an analysis result.
    This data is logged and used to fine-tune future models.
    """
    feedback_id = str(uuid.uuid4())

    # In a real system, this would write to a database (PostgreSQL/MongoDB).
    # For now, we append to a JSONL file.

    entry = {
        "id": feedback_id,
        "timestamp": datetime.now().isoformat(),
        **_normalize_feedback_entry(payload),
        "label_source": payload.label_source or "human_feedback",
        "review_source": payload.review_source,
        "verified_by": payload.verified_by,
    }

    log_path = _feedback_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

    return FeedbackResponse(
        status="success",
        feedback_id=feedback_id,
        message="Feedback received. Thank you for helping improve the AI.",
    )


@router.get("/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(api_key: str = Depends(get_api_key)):
    """Return aggregated feedback observability counters."""
    try:
        entries = _read_feedback_entries()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read feedback: {str(exc)}")
    return _build_feedback_stats(entries)
