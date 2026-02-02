"""Batch Job Processing.

Provides batch processing capabilities:
- Job definition and scheduling
- Parallel execution
- Error handling and retries
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from src.core.data_pipeline.transform import Transformer

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class JobStatus(Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class JobConfig:
    """Job configuration."""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0
    timeout_seconds: Optional[float] = None
    priority: JobPriority = JobPriority.NORMAL
    tags: List[str] = field(default_factory=list)


@dataclass
class JobProgress:
    """Job progress information."""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    current_stage: str = ""
    message: str = ""

    @property
    def percentage(self) -> float:
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100


@dataclass
class JobResult:
    """Result of a batch job."""
    job_id: str
    status: JobStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    progress: Optional[JobProgress] = None
    output: Any = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    retries: int = 0


class BatchJob(ABC, Generic[T, R]):
    """Abstract batch job."""

    def __init__(
        self,
        job_id: Optional[str] = None,
        config: Optional[JobConfig] = None,
    ):
        self.job_id = job_id or str(uuid.uuid4())
        self.config = config or JobConfig()
        self._status = JobStatus.PENDING
        self._progress = JobProgress()
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._error: Optional[str] = None
        self._retries = 0

    @property
    def status(self) -> JobStatus:
        return self._status

    @property
    def progress(self) -> JobProgress:
        return self._progress

    @abstractmethod
    async def execute(self, input_data: T) -> R:
        """Execute the job."""
        pass

    def update_progress(
        self,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        failed: Optional[int] = None,
        stage: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update job progress."""
        if processed is not None:
            self._progress.processed_items = processed
        if total is not None:
            self._progress.total_items = total
        if failed is not None:
            self._progress.failed_items = failed
        if stage is not None:
            self._progress.current_stage = stage
        if message is not None:
            self._progress.message = message

    async def run(self, input_data: T) -> JobResult:
        """Run the job with retry logic."""
        self._started_at = datetime.utcnow()
        self._status = JobStatus.RUNNING

        while self._retries <= self.config.max_retries:
            try:
                # Set timeout if configured
                if self.config.timeout_seconds:
                    output = await asyncio.wait_for(
                        self.execute(input_data),
                        timeout=self.config.timeout_seconds,
                    )
                else:
                    output = await self.execute(input_data)

                self._status = JobStatus.COMPLETED
                self._completed_at = datetime.utcnow()

                return self._create_result(output=output)

            except asyncio.TimeoutError:
                self._error = "Job timed out"
                logger.error(f"Job {self.job_id} timed out")
                self._retries += 1

            except asyncio.CancelledError:
                self._status = JobStatus.CANCELLED
                self._error = "Job was cancelled"
                return self._create_result()

            except Exception as e:
                self._error = str(e)
                logger.error(f"Job {self.job_id} failed: {e}")
                self._retries += 1

            if self._retries <= self.config.max_retries:
                self._status = JobStatus.RETRYING
                delay = self.config.retry_delay_seconds * (
                    self.config.retry_backoff_factor ** (self._retries - 1)
                )
                logger.info(f"Retrying job {self.job_id} in {delay}s")
                await asyncio.sleep(delay)

        self._status = JobStatus.FAILED
        self._completed_at = datetime.utcnow()
        return self._create_result()

    def _create_result(self, output: Any = None) -> JobResult:
        """Create job result."""
        duration = None
        if self._started_at and self._completed_at:
            duration = (self._completed_at - self._started_at).total_seconds()

        return JobResult(
            job_id=self.job_id,
            status=self._status,
            started_at=self._started_at,
            completed_at=self._completed_at,
            duration_seconds=duration,
            progress=self._progress,
            output=output,
            error=self._error,
            error_traceback=traceback.format_exc() if self._error else None,
            retries=self._retries,
        )


class FunctionJob(BatchJob[T, R]):
    """Job from a function."""

    def __init__(
        self,
        func: Callable[[T], R],
        job_id: Optional[str] = None,
        config: Optional[JobConfig] = None,
    ):
        super().__init__(job_id, config)
        self.func = func

    async def execute(self, input_data: T) -> R:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(input_data)
        return self.func(input_data)


class TransformJob(BatchJob[List[Dict[str, Any]], List[Dict[str, Any]]]):
    """Job that applies transformations to data."""

    def __init__(
        self,
        transformer: Transformer[Dict[str, Any], Dict[str, Any]],
        job_id: Optional[str] = None,
        config: Optional[JobConfig] = None,
        batch_size: int = 1000,
    ):
        super().__init__(job_id, config)
        self.transformer = transformer
        self.batch_size = batch_size

    async def execute(
        self,
        input_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        self.update_progress(total=len(input_data), processed=0, stage="transforming")

        results = []
        for i in range(0, len(input_data), self.batch_size):
            batch = input_data[i:i + self.batch_size]

            for item in batch:
                try:
                    transformed = self.transformer.transform(item)
                    if transformed is not None:
                        results.append(transformed)
                except Exception as e:
                    self._progress.failed_items += 1
                    logger.warning(f"Transform error: {e}")

            self.update_progress(processed=min(i + self.batch_size, len(input_data)))

            # Yield to event loop
            await asyncio.sleep(0)

        return results


class ChainedJob(BatchJob[T, R]):
    """Job that chains multiple jobs."""

    def __init__(
        self,
        jobs: List[BatchJob],
        job_id: Optional[str] = None,
        config: Optional[JobConfig] = None,
    ):
        super().__init__(job_id, config)
        self.jobs = jobs

    async def execute(self, input_data: T) -> R:
        self.update_progress(total=len(self.jobs), processed=0)

        current: Any = input_data
        for i, job in enumerate(self.jobs):
            self.update_progress(
                stage=f"Running job {i + 1}/{len(self.jobs)}: {job.job_id}"
            )

            result = await job.run(current)
            if result.status != JobStatus.COMPLETED:
                raise RuntimeError(f"Chained job {job.job_id} failed: {result.error}")

            current = result.output
            self.update_progress(processed=i + 1)

        return current


# Job Scheduler

class JobScheduler:
    """Scheduler for batch jobs."""

    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self._jobs: Dict[str, BatchJob] = {}
        self._results: Dict[str, JobResult] = {}
        self._pending: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running: Dict[str, asyncio.Task] = {}
        self._lock: Optional[asyncio.Lock] = None
        self._stopped = False

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def submit(
        self,
        job: BatchJob,
        input_data: Any,
    ) -> str:
        """Submit a job for execution."""
        async with self._get_lock():
            self._jobs[job.job_id] = job
            priority = -job.config.priority.value  # Negative for max heap behavior
            await self._pending.put((priority, time.time(), job.job_id, input_data))
            logger.info(f"Submitted job {job.job_id}")
            return job.job_id

    async def start(self) -> None:
        """Start the scheduler."""
        self._stopped = False
        logger.info("Starting job scheduler")

        while not self._stopped:
            # Clean completed tasks
            completed = [
                jid for jid, task in self._running.items()
                if task.done()
            ]
            for jid in completed:
                task = self._running.pop(jid)
                try:
                    result = task.result()
                    self._results[jid] = result
                except Exception as e:
                    logger.error(f"Job {jid} task error: {e}")

            # Start new jobs if capacity available
            while len(self._running) < self.max_concurrent and not self._pending.empty():
                try:
                    priority, submit_time, job_id, input_data = await asyncio.wait_for(
                        self._pending.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    break

                job = self._jobs.get(job_id)
                if job:
                    task = asyncio.create_task(job.run(input_data))
                    self._running[job_id] = task
                    logger.info(f"Started job {job_id}")

            await asyncio.sleep(0.1)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._stopped = True
        logger.info("Stopping job scheduler")

    async def wait_for(self, job_id: str, timeout: Optional[float] = None) -> JobResult:
        """Wait for a job to complete."""
        start = time.time()

        while True:
            if job_id in self._results:
                return self._results[job_id]

            if job_id in self._running:
                task = self._running[job_id]
                if task.done():
                    try:
                        result = task.result()
                        self._results[job_id] = result
                        return result
                    except Exception as e:
                        return JobResult(
                            job_id=job_id,
                            status=JobStatus.FAILED,
                            error=str(e),
                        )

            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            await asyncio.sleep(0.1)

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        if job_id in self._results:
            return self._results[job_id].status

        job = self._jobs.get(job_id)
        return job.status if job else None

    def get_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get job progress."""
        job = self._jobs.get(job_id)
        return job.progress if job else None


# Parallel Batch Processor

class ParallelBatchProcessor(Generic[T, R]):
    """Process items in parallel batches."""

    def __init__(
        self,
        processor: Callable[[T], R],
        batch_size: int = 100,
        max_workers: int = 4,
        on_error: str = "continue",  # continue, raise, collect
    ):
        self.processor = processor
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.on_error = on_error
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def process(
        self,
        items: List[T],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[R]:
        """Process all items."""
        self._semaphore = asyncio.Semaphore(self.max_workers)

        results: List[R] = []
        errors: List[Exception] = []
        processed = 0
        total = len(items)

        # Process in batches
        tasks = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            task = asyncio.create_task(self._process_batch(batch))
            tasks.append((i, task))

        for start_idx, task in tasks:
            try:
                batch_results = await task
                results.extend(batch_results)
                processed += len(batch_results)

                if progress_callback:
                    progress_callback(processed, total)

            except Exception as e:
                if self.on_error == "raise":
                    raise
                elif self.on_error == "collect":
                    errors.append(e)
                # continue: just log and move on
                logger.error(f"Batch processing error: {e}")

        if errors and self.on_error == "collect":
            logger.warning(f"Collected {len(errors)} errors during processing")

        return results

    async def _process_batch(self, batch: List[T]) -> List[R]:
        """Process a single batch."""
        async with self._semaphore:
            results = []
            for item in batch:
                try:
                    if asyncio.iscoroutinefunction(self.processor):
                        result = await self.processor(item)
                    else:
                        result = self.processor(item)
                    results.append(result)
                except Exception as e:
                    if self.on_error == "raise":
                        raise
                    logger.warning(f"Item processing error: {e}")
            return results


# ETL Pipeline

class ETLPipeline(Generic[T, R]):
    """Extract-Transform-Load pipeline."""

    def __init__(self):
        self._extractor: Optional[Callable[[], List[T]]] = None
        self._transformers: List[Transformer] = []
        self._loader: Optional[Callable[[List[R]], None]] = None
        self._batch_size = 1000

    def extract(self, extractor: Callable[[], List[T]]) -> "ETLPipeline[T, R]":
        """Set extractor."""
        self._extractor = extractor
        return self

    def transform(self, transformer: Transformer) -> "ETLPipeline[T, R]":
        """Add transformer."""
        self._transformers.append(transformer)
        return self

    def load(self, loader: Callable[[List[R]], None]) -> "ETLPipeline[T, R]":
        """Set loader."""
        self._loader = loader
        return self

    def batch_size(self, size: int) -> "ETLPipeline[T, R]":
        """Set batch size."""
        self._batch_size = size
        return self

    async def run(self) -> Dict[str, Any]:
        """Run the ETL pipeline."""
        if not self._extractor:
            raise ValueError("Extractor not set")
        if not self._loader:
            raise ValueError("Loader not set")

        stats = {
            "extracted": 0,
            "transformed": 0,
            "loaded": 0,
            "failed": 0,
            "duration_seconds": 0.0,
        }

        start_time = time.time()

        # Extract
        logger.info("Extracting data...")
        if asyncio.iscoroutinefunction(self._extractor):
            data = await self._extractor()
        else:
            data = self._extractor()

        stats["extracted"] = len(data)
        logger.info(f"Extracted {len(data)} records")

        # Transform in batches
        logger.info("Transforming data...")
        transformed_data = []

        for i in range(0, len(data), self._batch_size):
            batch = data[i:i + self._batch_size]

            for item in batch:
                try:
                    current: Any = item
                    for transformer in self._transformers:
                        if current is None:
                            break
                        current = transformer.transform(current)

                    if current is not None:
                        transformed_data.append(current)
                        stats["transformed"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    logger.warning(f"Transform error: {e}")

            # Yield to event loop
            await asyncio.sleep(0)

        logger.info(f"Transformed {len(transformed_data)} records")

        # Load
        logger.info("Loading data...")
        for i in range(0, len(transformed_data), self._batch_size):
            batch = transformed_data[i:i + self._batch_size]

            try:
                if asyncio.iscoroutinefunction(self._loader):
                    await self._loader(batch)
                else:
                    self._loader(batch)

                stats["loaded"] += len(batch)
            except Exception as e:
                logger.error(f"Load error: {e}")

            await asyncio.sleep(0)

        stats["duration_seconds"] = time.time() - start_time
        logger.info(f"ETL complete: {stats}")

        return stats
