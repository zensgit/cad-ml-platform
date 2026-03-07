"""
Feedback API.

Collects user corrections to improve L3/L4 models (Data Flywheel).
"""

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


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    message: str


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

    entry["original_part_type"] = original_fine
    entry["original_fine_part_type"] = original_fine
    entry["original_coarse_part_type"] = original_coarse
    entry["original_is_coarse_label"] = bool(
        original_fine and original_fine == original_coarse
    )
    entry["original_decision_source"] = _clean_optional_text(payload.original_decision_source)
    entry["review_outcome"] = _clean_optional_text(payload.review_outcome)
    entry["review_reasons"] = _clean_reasons(payload.review_reasons)
    return entry


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackRequest, api_key: str = Depends(get_api_key)):
    """
    Submit feedback for an analysis result.
    This data is logged and used to fine-tune future models.
    """
    import json
    import os
    import uuid

    feedback_id = str(uuid.uuid4())

    # In a real system, this would write to a database (PostgreSQL/MongoDB).
    # For now, we append to a JSONL file.

    entry = {
        "id": feedback_id,
        "timestamp": datetime.now().isoformat(),
        **_normalize_feedback_entry(payload),
    }

    log_path = os.getenv("FEEDBACK_LOG_PATH", "data/feedback_log.jsonl")
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
