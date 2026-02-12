"""
Request and response types for model serving.

Provides:
- Inference request/response structures
- Prediction types
- Serialization utilities
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class RequestPriority(int, Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class RequestStatus(str, Enum):
    """Request processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Prediction:
    """A single prediction result."""
    label: Union[int, str]
    confidence: float
    probabilities: Optional[List[float]] = None
    embeddings: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "label": self.label,
            "confidence": round(self.confidence, 4),
        }
        if self.probabilities:
            result["probabilities"] = [round(p, 4) for p in self.probabilities]
        if self.embeddings:
            result["embeddings"] = self.embeddings
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    model_name: str = ""
    inputs: List[Any] = field(default_factory=list)
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: float = 30.0
    batch_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Status
    status: RequestStatus = RequestStatus.PENDING

    @property
    def wait_time(self) -> float:
        """Time spent waiting in queue."""
        if self.started_at:
            return self.started_at - self.created_at
        return time.time() - self.created_at

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent processing."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def total_time(self) -> Optional[float]:
        """Total request time."""
        if self.completed_at:
            return self.completed_at - self.created_at
        return None

    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded timeout."""
        return (time.time() - self.created_at) > self.timeout

    def start(self) -> None:
        """Mark request as started."""
        self.status = RequestStatus.PROCESSING
        self.started_at = time.time()

    def complete(self) -> None:
        """Mark request as completed."""
        self.status = RequestStatus.COMPLETED
        self.completed_at = time.time()

    def fail(self) -> None:
        """Mark request as failed."""
        self.status = RequestStatus.FAILED
        self.completed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "input_count": len(self.inputs),
            "priority": self.priority.value,
            "timeout": self.timeout,
            "batch_id": self.batch_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "wait_time": self.wait_time,
            "processing_time": self.processing_time,
            "total_time": self.total_time,
        }


@dataclass
class InferenceResponse:
    """Response from model inference."""
    request_id: str
    predictions: List[Prediction]
    model_name: str
    model_version: str = ""
    latency_ms: float = 0.0
    status: RequestStatus = RequestStatus.COMPLETED
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if inference succeeded."""
        return self.status == RequestStatus.COMPLETED

    @property
    def prediction_count(self) -> int:
        """Number of predictions."""
        return len(self.predictions)

    def get_prediction(self, index: int = 0) -> Optional[Prediction]:
        """Get prediction by index."""
        if 0 <= index < len(self.predictions):
            return self.predictions[index]
        return None

    def get_labels(self) -> List[Union[int, str]]:
        """Get all prediction labels."""
        return [p.label for p in self.predictions]

    def get_confidences(self) -> List[float]:
        """Get all prediction confidences."""
        return [p.confidence for p in self.predictions]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "request_id": self.request_id,
            "predictions": [p.to_dict() for p in self.predictions],
            "model_name": self.model_name,
            "model_version": self.model_version,
            "latency_ms": round(self.latency_ms, 2),
            "status": self.status.value,
            "prediction_count": self.prediction_count,
        }
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def error_response(
        cls,
        request_id: str,
        model_name: str,
        error: str,
    ) -> "InferenceResponse":
        """Create error response."""
        return cls(
            request_id=request_id,
            predictions=[],
            model_name=model_name,
            status=RequestStatus.FAILED,
            error=error,
        )
