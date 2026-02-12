"""GraphQL Type Definitions.

Defines GraphQL types for the CAD ML Platform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Enums
# ============================================================================

class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ModelStatus(str, Enum):
    """ML model status."""
    TRAINING = "TRAINING"
    READY = "READY"
    DEPLOYED = "DEPLOYED"
    ARCHIVED = "ARCHIVED"


class SortOrder(str, Enum):
    """Sort order direction."""
    ASC = "ASC"
    DESC = "DESC"


# ============================================================================
# Pagination Types
# ============================================================================

@dataclass
class PageInfo:
    """Pagination info for cursor-based pagination."""
    has_next_page: bool = False
    has_previous_page: bool = False
    start_cursor: Optional[str] = None
    end_cursor: Optional[str] = None
    total_count: Optional[int] = None


@dataclass
class Edge(Generic[T]):
    """Edge in a connection."""
    node: T
    cursor: str


@dataclass
class Connection(Generic[T]):
    """Connection for cursor-based pagination."""
    edges: List[Edge[T]]
    page_info: PageInfo
    total_count: int


@dataclass
class PaginationType:
    """Offset-based pagination input."""
    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    sort_order: SortOrder = SortOrder.DESC


# ============================================================================
# Core Domain Types
# ============================================================================

@dataclass
class DocumentType:
    """GraphQL Document type."""
    id: str
    name: str
    file_path: str
    file_type: str
    status: DocumentStatus
    file_size: int = 0
    checksum: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Relationships
    owner_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Features
    feature_vector: Optional[List[float]] = None
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "status": self.status.value,
            "file_size": self.file_size,
            "metadata": self.metadata,
            "tags": self.tags,
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ModelType:
    """GraphQL ML Model type."""
    id: str
    name: str
    model_type: str  # classifier, detector, etc.
    version: str
    status: ModelStatus

    # Model info
    framework: str = "pytorch"
    architecture: Optional[str] = None
    parameters_count: int = 0

    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Training info
    training_dataset_id: Optional[str] = None
    training_config: Dict[str, Any] = field(default_factory=dict)

    # Deployment info
    endpoint_url: Optional[str] = None
    is_default: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Relationships
    owner_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "status": self.status.value,
            "framework": self.framework,
            "metrics": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
            },
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class UserType:
    """GraphQL User type."""
    id: str
    email: str
    username: str
    display_name: Optional[str] = None

    # Role info
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)

    # Profile
    avatar_url: Optional[str] = None
    bio: Optional[str] = None

    # Status
    is_active: bool = True
    is_verified: bool = False

    # Relationships
    tenant_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "display_name": self.display_name,
            "roles": self.roles,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PredictionType:
    """GraphQL Prediction result type."""
    id: str
    model_id: str
    document_id: str

    # Results
    prediction: Any
    confidence: float
    probabilities: Optional[Dict[str, float]] = None

    # Performance
    latency_ms: float = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class JobType:
    """GraphQL Job type."""
    id: str
    job_type: str
    status: str

    # Progress
    progress: float = 0.0
    current_step: Optional[str] = None
    total_steps: int = 0
    completed_steps: int = 0

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    # Relationships
    document_id: Optional[str] = None
    model_id: Optional[str] = None
    owner_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ============================================================================
# Input Types
# ============================================================================

@dataclass
class DocumentInput:
    """Input for creating/updating documents."""
    name: str
    file_path: str
    file_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


@dataclass
class ModelInput:
    """Input for creating/updating models."""
    name: str
    model_type: str
    version: Optional[str] = None
    framework: str = "pytorch"
    architecture: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


@dataclass
class PredictionInput:
    """Input for making predictions."""
    model_id: str
    document_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


@dataclass
class FilterInput:
    """Input for filtering queries."""
    field: str
    operator: str  # eq, ne, gt, gte, lt, lte, in, contains
    value: Any


@dataclass
class SearchInput:
    """Input for search queries."""
    query: str
    filters: Optional[List[FilterInput]] = None
    pagination: Optional[PaginationType] = None


# ============================================================================
# Response Types
# ============================================================================

@dataclass
class MutationResponse:
    """Standard mutation response."""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None


@dataclass
class BatchResponse(Generic[T]):
    """Response for batch operations."""
    success_count: int
    failure_count: int
    results: List[T]
    errors: List[str]


# ============================================================================
# Subscription Types
# ============================================================================

@dataclass
class DocumentEvent:
    """Event for document subscriptions."""
    event_type: str  # created, updated, deleted, processed
    document: DocumentType
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class JobEvent:
    """Event for job subscriptions."""
    event_type: str  # started, progress, completed, failed
    job: JobType
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ModelEvent:
    """Event for model subscriptions."""
    event_type: str  # trained, deployed, archived
    model: ModelType
    timestamp: datetime = field(default_factory=datetime.utcnow)
