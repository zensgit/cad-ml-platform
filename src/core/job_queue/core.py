"""Job Queue Core.

Provides job queue primitives:
- Job definition
- Queue interface
- Priority handling
"""

from __future__ import annotations

import json
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar


class JobState(Enum):
    """State of a job."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD = "dead"  # Moved to dead letter queue
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 100


@dataclass
class JobOptions:
    """Options for job execution."""
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    retry_backoff_factor: float = 2.0
    timeout_seconds: Optional[float] = None
    delay_seconds: float = 0.0  # Delay before first execution
    unique_key: Optional[str] = None  # For deduplication
    ttl_seconds: Optional[float] = None  # Job expires after this
    tags: List[str] = field(default_factory=list)


@dataclass
class Job:
    """A job to be executed."""
    id: str
    queue_name: str
    handler: str  # Handler name/identifier
    payload: Dict[str, Any]
    options: JobOptions
    state: JobState = JobState.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    last_error: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            JobState.COMPLETED,
            JobState.DEAD,
            JobState.CANCELLED,
        )

    @property
    def can_retry(self) -> bool:
        return self.attempts < self.options.max_retries

    @property
    def next_retry_delay(self) -> float:
        return self.options.retry_delay_seconds * (
            self.options.retry_backoff_factor ** self.attempts
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "queue_name": self.queue_name,
            "handler": self.handler,
            "payload": self.payload,
            "options": {
                "priority": self.options.priority.value,
                "max_retries": self.options.max_retries,
                "retry_delay_seconds": self.options.retry_delay_seconds,
                "retry_backoff_factor": self.options.retry_backoff_factor,
                "timeout_seconds": self.options.timeout_seconds,
                "delay_seconds": self.options.delay_seconds,
                "unique_key": self.options.unique_key,
                "ttl_seconds": self.options.ttl_seconds,
                "tags": self.options.tags,
            },
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "result": self.result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        options_data = data.get("options", {})
        options = JobOptions(
            priority=JobPriority(options_data.get("priority", 5)),
            max_retries=options_data.get("max_retries", 3),
            retry_delay_seconds=options_data.get("retry_delay_seconds", 5.0),
            retry_backoff_factor=options_data.get("retry_backoff_factor", 2.0),
            timeout_seconds=options_data.get("timeout_seconds"),
            delay_seconds=options_data.get("delay_seconds", 0.0),
            unique_key=options_data.get("unique_key"),
            ttl_seconds=options_data.get("ttl_seconds"),
            tags=options_data.get("tags", []),
        )

        return cls(
            id=data["id"],
            queue_name=data["queue_name"],
            handler=data["handler"],
            payload=data["payload"],
            options=options,
            state=JobState(data.get("state", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]),
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]) if data.get("scheduled_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            attempts=data.get("attempts", 0),
            last_error=data.get("last_error"),
            result=data.get("result"),
            metadata=data.get("metadata", {}),
        )


def create_job(
    queue_name: str,
    handler: str,
    payload: Dict[str, Any],
    options: Optional[JobOptions] = None,
    job_id: Optional[str] = None,
) -> Job:
    """Create a new job."""
    job_id = job_id or f"job_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
    options = options or JobOptions()

    scheduled_at = None
    if options.delay_seconds > 0:
        scheduled_at = datetime.utcnow() + timedelta(seconds=options.delay_seconds)

    return Job(
        id=job_id,
        queue_name=queue_name,
        handler=handler,
        payload=payload,
        options=options,
        scheduled_at=scheduled_at,
    )


class JobQueue(ABC):
    """Abstract base class for job queues."""

    @abstractmethod
    async def enqueue(self, job: Job) -> bool:
        """Add job to queue."""
        pass

    @abstractmethod
    async def dequeue(
        self,
        queue_name: str,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[Job]:
        """Get next job from queue."""
        pass

    @abstractmethod
    async def complete(self, job: Job, result: Any = None) -> bool:
        """Mark job as completed."""
        pass

    @abstractmethod
    async def fail(self, job: Job, error: str) -> bool:
        """Mark job as failed (may retry or move to dead letter)."""
        pass

    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        pass

    @abstractmethod
    async def get_queue_size(self, queue_name: str) -> int:
        """Get number of pending jobs in queue."""
        pass

    @abstractmethod
    async def get_dead_letter_jobs(
        self,
        queue_name: str,
        limit: int = 100,
    ) -> List[Job]:
        """Get jobs in dead letter queue."""
        pass

    @abstractmethod
    async def retry_dead_letter(self, job_id: str) -> bool:
        """Retry a dead letter job."""
        pass


@dataclass
class QueueStats:
    """Statistics for a queue."""
    queue_name: str
    pending_count: int
    running_count: int
    completed_count: int
    failed_count: int
    dead_letter_count: int
    avg_processing_time_ms: float
    jobs_per_minute: float


class JobHandler(ABC):
    """Abstract base class for job handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name."""
        pass

    @abstractmethod
    async def handle(self, job: Job) -> Any:
        """Handle a job."""
        pass

    async def on_success(self, job: Job, result: Any) -> None:
        """Called after successful handling."""
        pass

    async def on_failure(self, job: Job, error: Exception) -> None:
        """Called after failed handling."""
        pass

    async def on_retry(self, job: Job, error: Exception, attempt: int) -> None:
        """Called before retry."""
        pass


class FunctionHandler(JobHandler):
    """Job handler from a function."""

    def __init__(
        self,
        handler_name: str,
        func: Callable[[Job], Any],
    ):
        self._name = handler_name
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    async def handle(self, job: Job) -> Any:
        import asyncio
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(job)
        return self._func(job)


class HandlerRegistry:
    """Registry for job handlers."""

    def __init__(self):
        self._handlers: Dict[str, JobHandler] = {}

    def register(self, handler: JobHandler) -> None:
        """Register a handler."""
        self._handlers[handler.name] = handler

    def register_function(
        self,
        name: str,
        func: Callable[[Job], Any],
    ) -> None:
        """Register a function as handler."""
        self._handlers[name] = FunctionHandler(name, func)

    def get(self, name: str) -> Optional[JobHandler]:
        """Get handler by name."""
        return self._handlers.get(name)

    def handler(self, name: str):
        """Decorator to register a function as handler."""
        def decorator(func: Callable[[Job], Any]):
            self.register_function(name, func)
            return func
        return decorator

    @property
    def handlers(self) -> Dict[str, JobHandler]:
        return self._handlers.copy()
