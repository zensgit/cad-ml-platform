"""
Feedback API.

Collects user corrections to improve L3/L4 models (Data Flywheel).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from src.api.dependencies import get_api_key

router = APIRouter()

class FeedbackRequest(BaseModel):
    analysis_id: str = Field(..., description="The ID of the analysis being corrected")
    corrected_part_type: Optional[str] = Field(None, description="Correct classification")
    corrected_process: Optional[str] = Field(None, description="Correct manufacturing process")
    dfm_feedback: Optional[str] = Field(None, description="Comments on DFM accuracy")
    rating: int = Field(..., ge=1, le=5, description="1-5 star rating of the AI result")

class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    message: str

@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest, 
    api_key: str = Depends(get_api_key)
):
    """
    Submit feedback for an analysis result.
    This data is logged and used to fine-tune future models.
    """
    import uuid
    import json
    import os
    
    feedback_id = str(uuid.uuid4())
    
    # In a real system, this would write to a database (PostgreSQL/MongoDB).
    # For now, we append to a JSONL file.
    
    entry = {
        "id": feedback_id,
        "timestamp": datetime.now().isoformat(),
        **payload.model_dump()
    }
    
    log_path = os.getenv("FEEDBACK_LOG_PATH", "data/feedback_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
        
    return FeedbackResponse(
        status="success",
        feedback_id=feedback_id,
        message="Feedback received. Thank you for helping improve the AI."
    )
