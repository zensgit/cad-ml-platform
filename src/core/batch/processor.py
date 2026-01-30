"""Batch Processing API for bulk operations.

Features:
- Async batch job submission and tracking
- Priority queue processing
- Progress tracking and callbacks
- Configurable concurrency limits
- Result aggregation
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BatchJobStatus(Enum):
    """Status of a batch job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some items succeeded, some failed


class BatchItemStatus(Enum):
    """Status of a batch item."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchItem:
    """Represents a single item in a batch."""

    id: str
    data: Dict[str, Any]
    status: BatchItemStatus = BatchItemStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 0


@dataclass
class BatchJob:
    """Represents a batch processing job."""

    id: str
    operation: str
    items: List[BatchItem]
    status: BatchJobStatus = BatchJobStatus.PENDING
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None
    max_retries: int = 3
    concurrency: int = 5

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def completed_items(self) -> int:
        return sum(
            1
            for item in self.items
            if item.status in (BatchItemStatus.COMPLETED, BatchItemStatus.FAILED, BatchItemStatus.SKIPPED)
        )

    @property
    def successful_items(self) -> int:
        return sum(1 for item in self.items if item.status == BatchItemStatus.COMPLETED)

    @property
    def failed_items(self) -> int:
        return sum(1 for item in self.items if item.status == BatchItemStatus.FAILED)

    @property
    def progress(self) -> float:
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100

    def to_summary(self) -> Dict[str, Any]:
        """Return a summary of the job."""
        return {
            "job_id": self.id,
            "operation": self.operation,
            "status": self.status.value,
            "priority": self.priority,
            "progress": round(self.progress, 2),
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
        }


class BatchProcessor:
    """Processes batch jobs with configurable concurrency."""

    def __init__(
        self,
        max_concurrent_jobs: int = 10,
        max_items_per_job: int = 1000,
        default_concurrency: int = 5,
    ):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_items_per_job = max_items_per_job
        self.default_concurrency = default_concurrency

        # Job storage
        self._jobs: Dict[str, BatchJob] = {}
        self._queue: Optional[asyncio.PriorityQueue] = None
        self._active_jobs: Dict[str, asyncio.Task] = {}

        # Operation handlers
        self._handlers: Dict[str, Callable] = {}

        # State
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._lock: Optional[asyncio.Lock] = None

        # Metrics
        self._metrics = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "items_processed": 0,
        }

    def _get_queue(self) -> asyncio.PriorityQueue:
        """Get or create priority queue (lazy init for Python 3.9)."""
        if self._queue is None:
            self._queue = asyncio.PriorityQueue()
        return self._queue

    def _get_lock(self) -> asyncio.Lock:
        """Get or create lock (lazy init for Python 3.9)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def register_handler(self, operation: str, handler: Callable) -> None:
        """Register a handler for an operation type.

        Handler signature: async def handler(item_data: Dict) -> Dict
        """
        self._handlers[operation] = handler
        logger.info(f"Registered batch handler for operation: {operation}")

    async def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Batch processor started")

    async def stop(self) -> None:
        """Stop the batch processor."""
        self._running = False

        # Cancel active jobs
        for job_id, task in self._active_jobs.items():
            task.cancel()

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("Batch processor stopped")

    async def submit_job(
        self,
        operation: str,
        items: List[Dict[str, Any]],
        priority: int = 5,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        callback_url: Optional[str] = None,
        max_retries: int = 3,
        concurrency: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Submit a new batch job."""
        if operation not in self._handlers:
            raise ValueError(f"Unknown operation: {operation}")

        if len(items) > self.max_items_per_job:
            raise ValueError(f"Too many items: {len(items)} > {self.max_items_per_job}")

        # Create batch items
        batch_items = [
            BatchItem(id=str(uuid.uuid4()), data=item_data) for item_data in items
        ]

        # Create job
        job = BatchJob(
            id=str(uuid.uuid4()),
            operation=operation,
            items=batch_items,
            priority=priority,
            user_id=user_id,
            tenant_id=tenant_id,
            callback_url=callback_url,
            max_retries=max_retries,
            concurrency=concurrency or self.default_concurrency,
            metadata=metadata or {},
        )

        # Store and queue
        self._jobs[job.id] = job
        await self._get_queue().put((10 - priority, time.time(), job.id))  # Lower priority value = higher priority

        self._metrics["jobs_submitted"] += 1
        logger.info(f"Batch job submitted: {job.id} ({len(items)} items)")

        return job

    async def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status summary."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return job.to_summary()

    async def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed job results."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        return {
            **job.to_summary(),
            "items": [
                {
                    "id": item.id,
                    "status": item.status.value,
                    "result": item.result,
                    "error": item.error,
                    "retries": item.retries,
                }
                for item in job.items
            ],
        }

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in (BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED):
            return False

        job.status = BatchJobStatus.CANCELLED

        # Cancel running task if exists
        if job_id in self._active_jobs:
            self._active_jobs[job_id].cancel()

        logger.info(f"Batch job cancelled: {job_id}")
        return True

    async def list_jobs(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[BatchJobStatus] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filtering."""
        jobs = []
        for job in self._jobs.values():
            if user_id and job.user_id != user_id:
                continue
            if tenant_id and job.tenant_id != tenant_id:
                continue
            if status and job.status != status:
                continue
            jobs.append(job.to_summary())

        # Sort by created_at descending
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        return jobs[:limit]

    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        queue_size = self._queue.qsize() if self._queue else 0
        return {
            **self._metrics,
            "active_jobs": len(self._active_jobs),
            "queued_jobs": queue_size,
            "total_jobs": len(self._jobs),
        }

    async def _worker_loop(self) -> None:
        """Main worker loop that processes jobs from the queue."""
        while self._running:
            try:
                # Wait for job with timeout
                try:
                    _, _, job_id = await asyncio.wait_for(
                        self._get_queue().get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                job = self._jobs.get(job_id)
                if not job or job.status == BatchJobStatus.CANCELLED:
                    continue

                # Check concurrency limit
                if len(self._active_jobs) >= self.max_concurrent_jobs:
                    # Re-queue with slight delay
                    await self._get_queue().put((10 - job.priority, time.time(), job_id))
                    await asyncio.sleep(0.1)
                    continue

                # Start processing
                task = asyncio.create_task(self._process_job(job))
                self._active_jobs[job_id] = task

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1.0)

    async def _process_job(self, job: BatchJob) -> None:
        """Process a single batch job."""
        try:
            job.status = BatchJobStatus.RUNNING
            job.started_at = time.time()
            logger.info(f"Processing batch job: {job.id}")

            handler = self._handlers[job.operation]

            # Process items with concurrency limit
            semaphore = asyncio.Semaphore(job.concurrency)

            async def process_item(item: BatchItem) -> None:
                async with semaphore:
                    if job.status == BatchJobStatus.CANCELLED:
                        item.status = BatchItemStatus.SKIPPED
                        return

                    item.status = BatchItemStatus.RUNNING
                    item.started_at = time.time()

                    for attempt in range(job.max_retries + 1):
                        try:
                            result = await handler(item.data)
                            item.result = result
                            item.status = BatchItemStatus.COMPLETED
                            item.completed_at = time.time()
                            self._metrics["items_processed"] += 1
                            break
                        except Exception as e:
                            item.retries = attempt + 1
                            if attempt >= job.max_retries:
                                item.error = str(e)
                                item.status = BatchItemStatus.FAILED
                                item.completed_at = time.time()
                                logger.warning(
                                    f"Batch item failed after {item.retries} retries: {e}"
                                )
                            else:
                                await asyncio.sleep(0.5 * (attempt + 1))  # Backoff

            # Process all items
            tasks = [process_item(item) for item in job.items]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Determine final status
            job.completed_at = time.time()
            if job.status == BatchJobStatus.CANCELLED:
                pass  # Keep cancelled status
            elif job.failed_items == 0:
                job.status = BatchJobStatus.COMPLETED
                self._metrics["jobs_completed"] += 1
            elif job.successful_items == 0:
                job.status = BatchJobStatus.FAILED
                self._metrics["jobs_failed"] += 1
            else:
                job.status = BatchJobStatus.PARTIAL
                self._metrics["jobs_completed"] += 1

            logger.info(
                f"Batch job completed: {job.id} "
                f"({job.successful_items}/{job.total_items} successful)"
            )

            # Trigger callback if configured
            if job.callback_url:
                asyncio.create_task(self._send_callback(job))

        except asyncio.CancelledError:
            job.status = BatchJobStatus.CANCELLED
            raise
        except Exception as e:
            job.status = BatchJobStatus.FAILED
            self._metrics["jobs_failed"] += 1
            logger.error(f"Batch job error: {job.id} - {e}")
        finally:
            self._active_jobs.pop(job.id, None)

    async def _send_callback(self, job: BatchJob) -> None:
        """Send callback notification for completed job."""
        if not job.callback_url:
            return

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                payload = job.to_summary()
                async with session.post(
                    job.callback_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status >= 400:
                        logger.warning(
                            f"Callback failed for job {job.id}: {response.status}"
                        )
        except Exception as e:
            logger.error(f"Callback error for job {job.id}: {e}")


# Global processor instance
_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get the global batch processor instance."""
    global _processor
    if _processor is None:
        _processor = BatchProcessor()
    return _processor
