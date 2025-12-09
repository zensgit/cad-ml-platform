"""Result persistence for vision analysis.

Provides:
- Database storage for analysis results
- Query and retrieval functions
- Result history and audit trail
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backends."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    REDIS = "redis"


@dataclass
class AnalysisRecord:
    """A persisted analysis record."""

    record_id: str
    image_hash: str
    provider: str
    result: VisionDescription
    created_at: datetime
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    cost_usd: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "image_hash": self.image_hash,
            "provider": self.provider,
            "result": {
                "summary": self.result.summary,
                "details": self.result.details,
                "confidence": self.result.confidence,
            },
            "created_at": self.created_at.isoformat(),
            "request_metadata": self.request_metadata,
            "processing_time_ms": self.processing_time_ms,
            "cost_usd": self.cost_usd,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisRecord":
        """Create from dictionary."""
        return cls(
            record_id=data["record_id"],
            image_hash=data["image_hash"],
            provider=data["provider"],
            result=VisionDescription(
                summary=data["result"]["summary"],
                details=data["result"]["details"],
                confidence=data["result"]["confidence"],
            ),
            created_at=datetime.fromisoformat(data["created_at"]),
            request_metadata=data.get("request_metadata", {}),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            cost_usd=data.get("cost_usd", 0.0),
            tags=data.get("tags", []),
        )


@dataclass
class QueryFilter:
    """Filter for querying records."""

    provider: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    tags: Optional[List[str]] = None
    image_hash: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class QueryResult:
    """Result of a query operation."""

    records: List[AnalysisRecord]
    total_count: int
    has_more: bool


class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    @abstractmethod
    async def save(self, record: AnalysisRecord) -> str:
        """Save a record and return the record ID."""
        pass

    @abstractmethod
    async def get(self, record_id: str) -> Optional[AnalysisRecord]:
        """Get a record by ID."""
        pass

    @abstractmethod
    async def query(self, filter: QueryFilter) -> QueryResult:
        """Query records with filters."""
        pass

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        """Delete a record by ID."""
        pass

    @abstractmethod
    async def update_tags(self, record_id: str, tags: List[str]) -> bool:
        """Update tags for a record."""
        pass


class InMemoryStorage(StorageProvider):
    """In-memory storage implementation for testing."""

    def __init__(self) -> None:
        self._records: Dict[str, AnalysisRecord] = {}

    async def save(self, record: AnalysisRecord) -> str:
        """Save a record."""
        self._records[record.record_id] = record
        return record.record_id

    async def get(self, record_id: str) -> Optional[AnalysisRecord]:
        """Get a record by ID."""
        return self._records.get(record_id)

    async def query(self, filter: QueryFilter) -> QueryResult:
        """Query records with filters."""
        results = list(self._records.values())

        # Apply filters
        if filter.provider:
            results = [r for r in results if r.provider == filter.provider]

        if filter.start_date:
            results = [r for r in results if r.created_at >= filter.start_date]

        if filter.end_date:
            results = [r for r in results if r.created_at <= filter.end_date]

        if filter.min_confidence is not None:
            results = [
                r for r in results
                if r.result.confidence >= filter.min_confidence
            ]

        if filter.max_confidence is not None:
            results = [
                r for r in results
                if r.result.confidence <= filter.max_confidence
            ]

        if filter.tags:
            results = [
                r for r in results
                if any(tag in r.tags for tag in filter.tags)
            ]

        if filter.image_hash:
            results = [r for r in results if r.image_hash == filter.image_hash]

        # Sort by created_at descending
        results.sort(key=lambda r: r.created_at, reverse=True)

        total_count = len(results)
        has_more = (filter.offset + filter.limit) < total_count

        # Apply pagination
        results = results[filter.offset:filter.offset + filter.limit]

        return QueryResult(
            records=results,
            total_count=total_count,
            has_more=has_more,
        )

    async def delete(self, record_id: str) -> bool:
        """Delete a record."""
        if record_id in self._records:
            del self._records[record_id]
            return True
        return False

    async def update_tags(self, record_id: str, tags: List[str]) -> bool:
        """Update tags for a record."""
        if record_id in self._records:
            self._records[record_id].tags = tags
            return True
        return False

    def clear(self) -> int:
        """Clear all records."""
        count = len(self._records)
        self._records.clear()
        return count


class ResultPersistence:
    """
    Manages persistence of vision analysis results.

    Features:
    - Multiple storage backend support
    - Query and filtering
    - Tagging for organization
    - Audit trail
    """

    def __init__(
        self,
        storage: Optional[StorageProvider] = None,
    ):
        """
        Initialize persistence manager.

        Args:
            storage: Storage provider (defaults to in-memory)
        """
        self._storage = storage or InMemoryStorage()

    @staticmethod
    def compute_image_hash(image_data: bytes) -> str:
        """Compute SHA-256 hash of image data."""
        return hashlib.sha256(image_data).hexdigest()

    async def save_result(
        self,
        image_data: bytes,
        provider: str,
        result: VisionDescription,
        processing_time_ms: float = 0.0,
        cost_usd: float = 0.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AnalysisRecord:
        """
        Save an analysis result.

        Args:
            image_data: Original image data
            provider: Provider name
            result: Analysis result
            processing_time_ms: Processing time
            cost_usd: Cost of analysis
            tags: Optional tags for organization
            metadata: Additional metadata

        Returns:
            The saved AnalysisRecord
        """
        record = AnalysisRecord(
            record_id=str(uuid.uuid4()),
            image_hash=self.compute_image_hash(image_data),
            provider=provider,
            result=result,
            created_at=datetime.now(),
            request_metadata=metadata or {},
            processing_time_ms=processing_time_ms,
            cost_usd=cost_usd,
            tags=tags or [],
        )

        await self._storage.save(record)
        logger.debug(f"Saved analysis record: {record.record_id}")
        return record

    async def get_result(self, record_id: str) -> Optional[AnalysisRecord]:
        """Get a result by record ID."""
        return await self._storage.get(record_id)

    async def find_by_image(
        self,
        image_data: bytes,
        provider: Optional[str] = None,
    ) -> List[AnalysisRecord]:
        """
        Find all results for an image.

        Args:
            image_data: Image data to search for
            provider: Optional provider filter

        Returns:
            List of matching records
        """
        image_hash = self.compute_image_hash(image_data)
        filter = QueryFilter(
            image_hash=image_hash,
            provider=provider,
        )
        result = await self._storage.query(filter)
        return result.records

    async def query_results(
        self,
        provider: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """
        Query results with filters.

        Args:
            provider: Filter by provider
            start_date: Filter by start date
            end_date: Filter by end date
            min_confidence: Minimum confidence score
            tags: Filter by tags
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            QueryResult with matching records
        """
        filter = QueryFilter(
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            min_confidence=min_confidence,
            tags=tags,
            limit=limit,
            offset=offset,
        )
        return await self._storage.query(filter)

    async def delete_result(self, record_id: str) -> bool:
        """Delete a result by record ID."""
        return await self._storage.delete(record_id)

    async def add_tags(
        self,
        record_id: str,
        tags: List[str],
    ) -> bool:
        """Add tags to a record."""
        record = await self._storage.get(record_id)
        if record:
            new_tags = list(set(record.tags + tags))
            return await self._storage.update_tags(record_id, new_tags)
        return False

    async def remove_tags(
        self,
        record_id: str,
        tags: List[str],
    ) -> bool:
        """Remove tags from a record."""
        record = await self._storage.get(record_id)
        if record:
            new_tags = [t for t in record.tags if t not in tags]
            return await self._storage.update_tags(record_id, new_tags)
        return False


class PersistentVisionProvider:
    """
    Wrapper that adds persistence to any VisionProvider.

    Automatically saves analysis results to storage.
    """

    def __init__(
        self,
        provider: VisionProvider,
        persistence: ResultPersistence,
        auto_tag: Optional[List[str]] = None,
    ):
        """
        Initialize persistent provider.

        Args:
            provider: The underlying vision provider
            persistence: ResultPersistence instance
            auto_tag: Tags to automatically add to all results
        """
        self._provider = provider
        self._persistence = persistence
        self._auto_tag = auto_tag or []

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[VisionDescription, AnalysisRecord]:
        """
        Analyze image and persist result.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description
            tags: Additional tags for the result
            metadata: Additional metadata

        Returns:
            Tuple of (VisionDescription, AnalysisRecord)
        """
        import time
        start_time = time.time()

        result = await self._provider.analyze_image(
            image_data, include_description
        )
        processing_time_ms = (time.time() - start_time) * 1000

        # Combine auto-tags with provided tags
        all_tags = list(set(self._auto_tag + (tags or [])))

        record = await self._persistence.save_result(
            image_data=image_data,
            provider=self._provider.provider_name,
            result=result,
            processing_time_ms=processing_time_ms,
            tags=all_tags,
            metadata=metadata,
        )

        return result, record

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def persistence(self) -> ResultPersistence:
        """Get the persistence manager."""
        return self._persistence


# Global persistence instance
_global_persistence: Optional[ResultPersistence] = None


def get_persistence() -> ResultPersistence:
    """
    Get the global persistence instance.

    Returns:
        ResultPersistence singleton
    """
    global _global_persistence
    if _global_persistence is None:
        _global_persistence = ResultPersistence()
    return _global_persistence


def create_persistent_provider(
    provider: VisionProvider,
    storage: Optional[StorageProvider] = None,
    auto_tag: Optional[List[str]] = None,
) -> PersistentVisionProvider:
    """
    Factory to create a persistent provider wrapper.

    Args:
        provider: The underlying vision provider
        storage: Optional storage provider
        auto_tag: Tags to automatically add

    Returns:
        PersistentVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> persistent = create_persistent_provider(provider)
        >>> result, record = await persistent.analyze_image(image_bytes)
        >>> print(f"Saved as: {record.record_id}")
    """
    persistence = ResultPersistence(storage=storage)
    return PersistentVisionProvider(
        provider=provider,
        persistence=persistence,
        auto_tag=auto_tag,
    )
