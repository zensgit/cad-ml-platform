"""Arq-based task queue for distributed job processing.

This module provides a production-ready task queue using Arq,
replacing the custom InMemoryMessageStore implementation.

Benefits over custom implementation:
- Redis-backed persistence (survives restarts)
- Distributed workers (horizontal scaling)
- Built-in retry with backoff
- Job result storage
- Dead letter queue support
- ~80% less code to maintain

Example:
    >>> from src.core.tasks import TaskClient, TaskResult
    >>> client = TaskClient()
    >>> await client.connect()
    >>> job_id = await client.submit_analysis("doc.dxf", {"extract_features": True})
    >>> result = await client.get_result(job_id, timeout=60)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Conditional import for arq
try:
    from arq import create_pool
    from arq.connections import RedisSettings
    from arq.jobs import Job, JobStatus

    ARQ_AVAILABLE = True
except ImportError:
    ARQ_AVAILABLE = False
    create_pool = None
    RedisSettings = None
    Job = None
    JobStatus = None


__all__ = [
    "TaskClient",
    "TaskResult",
    "TaskStatus",
    "TaskConfig",
    "ARQ_AVAILABLE",
    "get_task_client",
]


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DEFERRED = "deferred"


@dataclass
class TaskResult:
    """Result of a task execution."""

    job_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for task queue."""

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    max_jobs: int = 10
    job_timeout: int = 300  # 5 minutes
    retry_jobs: bool = True
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "TaskConfig":
        """Create config from environment variables."""
        return cls(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            max_jobs=int(os.getenv("ARQ_MAX_JOBS", "10")),
            job_timeout=int(os.getenv("ARQ_JOB_TIMEOUT", "300")),
            retry_jobs=os.getenv("ARQ_RETRY_JOBS", "true").lower() == "true",
            max_retries=int(os.getenv("ARQ_MAX_RETRIES", "3")),
        )


class TaskClient:
    """Client for submitting and managing tasks.

    This replaces the custom message queue implementation with
    a production-ready Redis-backed task queue.

    Attributes:
        config: Task queue configuration
        _pool: Arq Redis connection pool
    """

    def __init__(self, config: TaskConfig | None = None):
        """Initialize task client.

        Args:
            config: Task configuration. If None, loads from environment.

        Raises:
            ImportError: If arq is not installed.
        """
        if not ARQ_AVAILABLE:
            raise ImportError(
                "arq is not installed. Install with: pip install arq"
            )

        self.config = config or TaskConfig.from_env()
        self._pool = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis and initialize the pool."""
        if self._connected:
            return

        self._pool = await create_pool(
            RedisSettings(
                host=self.config.redis_host,
                port=self.config.redis_port,
                database=self.config.redis_db,
                password=self.config.redis_password,
            )
        )
        self._connected = True
        logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._connected = False

    async def submit_analysis(
        self,
        file_path: str,
        options: dict[str, Any] | None = None,
        priority: int = 0,
    ) -> str:
        """Submit a CAD analysis task.

        Args:
            file_path: Path to the CAD file.
            options: Analysis options.
            priority: Job priority (higher = more urgent).

        Returns:
            Job ID for tracking.
        """
        await self._ensure_connected()

        job = await self._pool.enqueue_job(
            "analyze_cad_file",
            file_path,
            options or {},
            _job_timeout=self.config.job_timeout,
        )
        logger.debug(f"Submitted analysis job {job.job_id} for {file_path}")
        return job.job_id

    async def submit_feature_extraction(
        self,
        document_id: str,
        feature_types: list[str] | None = None,
    ) -> str:
        """Submit a feature extraction task.

        Args:
            document_id: Document identifier.
            feature_types: Types of features to extract.

        Returns:
            Job ID for tracking.
        """
        await self._ensure_connected()

        job = await self._pool.enqueue_job(
            "extract_features",
            document_id,
            feature_types or ["geometric", "topological"],
            _job_timeout=self.config.job_timeout,
        )
        logger.debug(f"Submitted feature extraction job {job.job_id}")
        return job.job_id

    async def submit_similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_conditions: dict[str, Any] | None = None,
    ) -> str:
        """Submit a similarity search task.

        Args:
            query_vector: Query feature vector.
            top_k: Number of results to return.
            filter_conditions: Metadata filters.

        Returns:
            Job ID for tracking.
        """
        await self._ensure_connected()

        job = await self._pool.enqueue_job(
            "search_similar",
            query_vector,
            top_k,
            filter_conditions or {},
            _job_timeout=60,  # Shorter timeout for search
        )
        return job.job_id

    async def get_result(
        self,
        job_id: str,
        timeout: float = 60.0,
    ) -> TaskResult:
        """Get the result of a task.

        Args:
            job_id: Job identifier.
            timeout: Maximum time to wait for result.

        Returns:
            TaskResult with status and data.
        """
        await self._ensure_connected()

        job = Job(job_id, self._pool)

        try:
            result = await job.result(timeout=timeout)
            info = await job.info()

            return TaskResult(
                job_id=job_id,
                status=TaskStatus.COMPLETED,
                result=result,
                started_at=info.start_time if info else None,
                completed_at=info.finish_time if info else None,
            )
        except TimeoutError:
            return TaskResult(
                job_id=job_id,
                status=TaskStatus.TIMEOUT,
                error="Task timed out waiting for result",
            )
        except Exception as e:
            return TaskResult(
                job_id=job_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )

    async def get_status(self, job_id: str) -> TaskStatus:
        """Get the status of a task.

        Args:
            job_id: Job identifier.

        Returns:
            Current task status.
        """
        await self._ensure_connected()

        job = Job(job_id, self._pool)
        status = await job.status()

        status_map = {
            JobStatus.deferred: TaskStatus.DEFERRED,
            JobStatus.queued: TaskStatus.PENDING,
            JobStatus.in_progress: TaskStatus.IN_PROGRESS,
            JobStatus.complete: TaskStatus.COMPLETED,
            JobStatus.not_found: TaskStatus.FAILED,
        }

        return status_map.get(status, TaskStatus.PENDING)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending task.

        Args:
            job_id: Job identifier.

        Returns:
            True if cancelled successfully.
        """
        await self._ensure_connected()

        job = Job(job_id, self._pool)
        await job.abort()
        logger.info(f"Cancelled job {job_id}")
        return True

    async def _ensure_connected(self) -> None:
        """Ensure we're connected to Redis."""
        if not self._connected:
            await self.connect()

    async def __aenter__(self) -> "TaskClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


# Singleton instance
_task_client: TaskClient | None = None


def get_task_client() -> TaskClient:
    """Get or create the global task client instance.

    Returns:
        TaskClient instance.
    """
    global _task_client

    if _task_client is None:
        _task_client = TaskClient()

    return _task_client
