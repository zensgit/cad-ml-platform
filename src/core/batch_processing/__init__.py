"""Batch Processing Module.

Provides batch processing infrastructure:
- Batch job framework
- Progress tracking
- Checkpointing
- Error handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchStatus(Enum):
    """Batch job status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchProgress:
    """Batch job progress."""

    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    last_checkpoint: Optional[datetime] = None

    current_item: Optional[str] = None
    last_error: Optional[str] = None

    @property
    def percent_complete(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def remaining_items(self) -> int:
        """Get remaining items."""
        return self.total_items - self.processed_items

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    @property
    def items_per_second(self) -> float:
        """Calculate processing rate."""
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.processed_items / elapsed

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        """Estimate remaining time."""
        rate = self.items_per_second
        if rate == 0:
            return None
        return self.remaining_items / rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "percent_complete": self.percent_complete,
            "elapsed_seconds": self.elapsed_seconds,
            "items_per_second": self.items_per_second,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
        }


@dataclass
class BatchResult(Generic[R]):
    """Result of batch processing."""

    job_id: str
    status: BatchStatus
    progress: BatchProgress
    results: List[R] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress.to_dict(),
            "error_count": len(self.errors),
            "metadata": self.metadata,
        }


@dataclass
class Checkpoint:
    """Checkpoint for resumable processing."""

    job_id: str
    timestamp: datetime
    processed_count: int
    last_item_id: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "processed_count": self.processed_count,
            "last_item_id": self.last_item_id,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            job_id=data["job_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            processed_count=data["processed_count"],
            last_item_id=data.get("last_item_id"),
            state=data.get("state", {}),
        )


class CheckpointStore(ABC):
    """Abstract checkpoint storage."""

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> bool:
        """Save checkpoint."""
        pass

    @abstractmethod
    async def load(self, job_id: str) -> Optional[Checkpoint]:
        """Load checkpoint."""
        pass

    @abstractmethod
    async def delete(self, job_id: str) -> bool:
        """Delete checkpoint."""
        pass


class InMemoryCheckpointStore(CheckpointStore):
    """In-memory checkpoint storage."""

    def __init__(self):
        self._checkpoints: Dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> bool:
        self._checkpoints[checkpoint.job_id] = checkpoint
        return True

    async def load(self, job_id: str) -> Optional[Checkpoint]:
        return self._checkpoints.get(job_id)

    async def delete(self, job_id: str) -> bool:
        if job_id in self._checkpoints:
            del self._checkpoints[job_id]
            return True
        return False


class FileCheckpointStore(CheckpointStore):
    """File-based checkpoint storage."""

    def __init__(self, directory: Union[str, Path]):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _get_path(self, job_id: str) -> Path:
        return self._directory / f"{job_id}.checkpoint.json"

    async def save(self, checkpoint: Checkpoint) -> bool:
        path = self._get_path(checkpoint.job_id)
        try:
            path.write_text(json.dumps(checkpoint.to_dict(), indent=2))
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    async def load(self, job_id: str) -> Optional[Checkpoint]:
        path = self._get_path(job_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Checkpoint.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def delete(self, job_id: str) -> bool:
        path = self._get_path(job_id)
        if path.exists():
            path.unlink()
            return True
        return False


class ItemProcessor(ABC, Generic[T, R]):
    """Abstract item processor."""

    @abstractmethod
    async def process(self, item: T) -> R:
        """Process a single item."""
        pass

    async def on_error(self, item: T, error: Exception) -> Optional[R]:
        """Handle processing error. Return value to use or None to skip."""
        return None

    def get_item_id(self, item: T) -> str:
        """Get unique identifier for item."""
        return str(hash(str(item)))


class FunctionProcessor(ItemProcessor[T, R]):
    """Processor wrapping a function."""

    def __init__(
        self,
        func: Callable[[T], R],
        error_handler: Optional[Callable[[T, Exception], Optional[R]]] = None,
    ):
        self._func = func
        self._error_handler = error_handler

    async def process(self, item: T) -> R:
        result = self._func(item)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def on_error(self, item: T, error: Exception) -> Optional[R]:
        if self._error_handler:
            result = self._error_handler(item, error)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return None


@dataclass
class BatchConfig:
    """Batch processing configuration."""

    batch_size: int = 100
    max_concurrency: int = 10
    checkpoint_interval: int = 100  # Items between checkpoints
    retry_count: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = True
    timeout_per_item: Optional[float] = None


class BatchJob(Generic[T, R]):
    """A batch processing job."""

    def __init__(
        self,
        job_id: Optional[str] = None,
        processor: Optional[ItemProcessor[T, R]] = None,
        config: Optional[BatchConfig] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        on_progress: Optional[Callable[[BatchProgress], None]] = None,
    ):
        self._job_id = job_id or str(uuid.uuid4())
        self._processor = processor
        self._config = config or BatchConfig()
        self._checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self._on_progress = on_progress

        self._status = BatchStatus.PENDING
        self._progress = BatchProgress()
        self._results: List[R] = []
        self._errors: List[Dict[str, Any]] = []
        self._cancel_requested = False
        self._pause_requested = False

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def status(self) -> BatchStatus:
        return self._status

    @property
    def progress(self) -> BatchProgress:
        return self._progress

    async def run(
        self,
        items: Union[List[T], Iterator[T]],
        processor: Optional[ItemProcessor[T, R]] = None,
        resume: bool = False,
    ) -> BatchResult[R]:
        """Run the batch job."""
        processor = processor or self._processor
        if not processor:
            raise ValueError("No processor provided")

        # Convert to list for counting
        items_list = list(items)
        self._progress.total_items = len(items_list)
        self._progress.start_time = datetime.utcnow()
        self._status = BatchStatus.RUNNING

        # Check for checkpoint
        start_index = 0
        if resume:
            checkpoint = await self._checkpoint_store.load(self._job_id)
            if checkpoint:
                start_index = checkpoint.processed_count
                self._progress.processed_items = checkpoint.processed_count
                logger.info(f"Resuming job {self._job_id} from item {start_index}")

        try:
            # Process in batches
            for batch_start in range(start_index, len(items_list), self._config.batch_size):
                if self._cancel_requested:
                    self._status = BatchStatus.CANCELLED
                    break

                while self._pause_requested:
                    await asyncio.sleep(0.1)
                    if self._cancel_requested:
                        break

                batch_end = min(batch_start + self._config.batch_size, len(items_list))
                batch = items_list[batch_start:batch_end]

                await self._process_batch(batch, processor)

                # Checkpoint
                if self._progress.processed_items % self._config.checkpoint_interval == 0:
                    await self._save_checkpoint(items_list)

                # Progress callback
                if self._on_progress:
                    self._on_progress(self._progress)

            if self._status == BatchStatus.RUNNING:
                self._status = BatchStatus.COMPLETED

        except Exception as e:
            self._status = BatchStatus.FAILED
            self._progress.last_error = str(e)
            logger.error(f"Batch job {self._job_id} failed: {e}")

        finally:
            self._progress.end_time = datetime.utcnow()

            # Clean up checkpoint on completion
            if self._status == BatchStatus.COMPLETED:
                await self._checkpoint_store.delete(self._job_id)

        return BatchResult(
            job_id=self._job_id,
            status=self._status,
            progress=self._progress,
            results=self._results,
            errors=self._errors,
        )

    async def _process_batch(
        self,
        batch: List[T],
        processor: ItemProcessor[T, R],
    ) -> None:
        """Process a batch of items."""
        semaphore = asyncio.Semaphore(self._config.max_concurrency)

        async def process_with_semaphore(item: T) -> None:
            async with semaphore:
                await self._process_item(item, processor)

        await asyncio.gather(*[process_with_semaphore(item) for item in batch])

    async def _process_item(
        self,
        item: T,
        processor: ItemProcessor[T, R],
    ) -> None:
        """Process a single item with retries."""
        item_id = processor.get_item_id(item)
        self._progress.current_item = item_id

        for attempt in range(self._config.retry_count):
            try:
                # Apply timeout if configured
                if self._config.timeout_per_item:
                    result = await asyncio.wait_for(
                        processor.process(item),
                        timeout=self._config.timeout_per_item,
                    )
                else:
                    result = await processor.process(item)

                self._results.append(result)
                self._progress.successful_items += 1
                self._progress.processed_items += 1
                return

            except Exception as e:
                if attempt < self._config.retry_count - 1:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))
                    continue

                # Final failure
                error_result = await processor.on_error(item, e)

                if error_result is not None:
                    self._results.append(error_result)
                    self._progress.successful_items += 1
                else:
                    self._errors.append({
                        "item_id": item_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    })
                    self._progress.failed_items += 1

                    if not self._config.continue_on_error:
                        raise

                self._progress.processed_items += 1
                self._progress.last_error = str(e)

    async def _save_checkpoint(self, items: List[T]) -> None:
        """Save checkpoint."""
        checkpoint = Checkpoint(
            job_id=self._job_id,
            timestamp=datetime.utcnow(),
            processed_count=self._progress.processed_items,
        )
        await self._checkpoint_store.save(checkpoint)
        self._progress.last_checkpoint = checkpoint.timestamp

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancel_requested = True

    def pause(self) -> None:
        """Request pause."""
        self._pause_requested = True
        self._status = BatchStatus.PAUSED

    def resume(self) -> None:
        """Resume from pause."""
        self._pause_requested = False
        self._status = BatchStatus.RUNNING


class BatchJobManager:
    """Manages multiple batch jobs."""

    def __init__(
        self,
        checkpoint_store: Optional[CheckpointStore] = None,
    ):
        self._jobs: Dict[str, BatchJob] = {}
        self._checkpoint_store = checkpoint_store or InMemoryCheckpointStore()

    def create_job(
        self,
        processor: Optional[ItemProcessor] = None,
        config: Optional[BatchConfig] = None,
        job_id: Optional[str] = None,
    ) -> BatchJob:
        """Create a new batch job."""
        job = BatchJob(
            job_id=job_id,
            processor=processor,
            config=config,
            checkpoint_store=self._checkpoint_store,
        )
        self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, status: Optional[BatchStatus] = None) -> List[BatchJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self._jobs.get(job_id)
        if job:
            job.cancel()
            return True
        return False

    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get progress for all jobs."""
        return {
            job_id: {
                "status": job.status.value,
                "progress": job.progress.to_dict(),
            }
            for job_id, job in self._jobs.items()
        }


def batch_process(
    items: Union[List[T], Iterator[T]],
    processor: Callable[[T], R],
    config: Optional[BatchConfig] = None,
    on_progress: Optional[Callable[[BatchProgress], None]] = None,
) -> BatchResult[R]:
    """Convenience function for batch processing."""
    job = BatchJob(
        processor=FunctionProcessor(processor),
        config=config,
        on_progress=on_progress,
    )
    return asyncio.run(job.run(items))


async def async_batch_process(
    items: Union[List[T], Iterator[T]],
    processor: Callable[[T], R],
    config: Optional[BatchConfig] = None,
    on_progress: Optional[Callable[[BatchProgress], None]] = None,
) -> BatchResult[R]:
    """Async convenience function for batch processing."""
    job = BatchJob(
        processor=FunctionProcessor(processor),
        config=config,
        on_progress=on_progress,
    )
    return await job.run(items)


__all__ = [
    "BatchStatus",
    "BatchProgress",
    "BatchResult",
    "Checkpoint",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "FileCheckpointStore",
    "ItemProcessor",
    "FunctionProcessor",
    "BatchConfig",
    "BatchJob",
    "BatchJobManager",
    "batch_process",
    "async_batch_process",
]
