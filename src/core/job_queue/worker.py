"""Job Queue Worker.

Provides worker implementation:
- Job processing
- Graceful shutdown
- Concurrent execution
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.job_queue.core import (
    HandlerRegistry,
    Job,
    JobQueue,
    JobState,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Worker configuration."""
    queue_names: List[str]
    concurrency: int = 4
    poll_interval_seconds: float = 1.0
    shutdown_timeout_seconds: float = 30.0
    job_timeout_seconds: Optional[float] = None


@dataclass
class WorkerStats:
    """Worker statistics."""
    started_at: datetime
    jobs_processed: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0
    jobs_retried: int = 0
    current_jobs: int = 0

    @property
    def uptime_seconds(self) -> float:
        return (datetime.utcnow() - self.started_at).total_seconds()


class Worker:
    """Job queue worker."""

    def __init__(
        self,
        queue: JobQueue,
        registry: HandlerRegistry,
        config: WorkerConfig,
    ):
        self._queue = queue
        self._registry = registry
        self._config = config

        self._running = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._tasks: Set[asyncio.Task] = set()
        self._stats = WorkerStats(started_at=datetime.utcnow())

        # Middleware
        self._before_job: List[Callable[[Job], None]] = []
        self._after_job: List[Callable[[Job, Any], None]] = []
        self._on_error: List[Callable[[Job, Exception], None]] = []

    @property
    def stats(self) -> WorkerStats:
        return self._stats

    @property
    def is_running(self) -> bool:
        return self._running

    def before_job(self, callback: Callable[[Job], None]) -> "Worker":
        """Add before-job middleware."""
        self._before_job.append(callback)
        return self

    def after_job(self, callback: Callable[[Job, Any], None]) -> "Worker":
        """Add after-job middleware."""
        self._after_job.append(callback)
        return self

    def on_error(self, callback: Callable[[Job, Exception], None]) -> "Worker":
        """Add error middleware."""
        self._on_error.append(callback)
        return self

    async def start(self) -> None:
        """Start the worker."""
        if self._running:
            return

        self._running = True
        self._shutdown_event = asyncio.Event()
        self._stats = WorkerStats(started_at=datetime.utcnow())

        logger.info(
            f"Starting worker for queues: {self._config.queue_names} "
            f"(concurrency={self._config.concurrency})"
        )

        # Start processor tasks
        for _ in range(self._config.concurrency):
            task = asyncio.create_task(self._processor_loop())
            self._tasks.add(task)

        # Wait for shutdown
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return

        logger.info("Stopping worker...")
        self._running = False

        # Wait for current jobs with timeout
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=self._config.shutdown_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning("Shutdown timeout, cancelling tasks")
                for task in self._tasks:
                    task.cancel()

        self._tasks.clear()

        if self._shutdown_event:
            self._shutdown_event.set()

        logger.info("Worker stopped")

    async def _processor_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Try each queue
                job = None
                for queue_name in self._config.queue_names:
                    job = await self._queue.dequeue(
                        queue_name,
                        timeout_seconds=0.1,
                    )
                    if job:
                        break

                if job:
                    await self._process_job(job)
                else:
                    await asyncio.sleep(self._config.poll_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processor loop error: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, job: Job) -> None:
        """Process a single job."""
        self._stats.current_jobs += 1
        self._stats.jobs_processed += 1

        try:
            # Run before-job middleware
            for middleware in self._before_job:
                try:
                    if asyncio.iscoroutinefunction(middleware):
                        await middleware(job)
                    else:
                        middleware(job)
                except Exception as e:
                    logger.warning(f"Before-job middleware error: {e}")

            # Get handler
            handler = self._registry.get(job.handler)
            if not handler:
                raise ValueError(f"Unknown handler: {job.handler}")

            # Execute with timeout
            timeout = (
                job.options.timeout_seconds
                or self._config.job_timeout_seconds
            )

            if timeout:
                result = await asyncio.wait_for(
                    handler.handle(job),
                    timeout=timeout,
                )
            else:
                result = await handler.handle(job)

            # Success
            await self._queue.complete(job, result)
            await handler.on_success(job, result)
            self._stats.jobs_succeeded += 1

            # Run after-job middleware
            for middleware in self._after_job:
                try:
                    if asyncio.iscoroutinefunction(middleware):
                        await middleware(job, result)
                    else:
                        middleware(job, result)
                except Exception as e:
                    logger.warning(f"After-job middleware error: {e}")

        except asyncio.TimeoutError:
            error = f"Job timed out after {timeout}s"
            await self._handle_failure(job, TimeoutError(error))

        except Exception as e:
            await self._handle_failure(job, e)

        finally:
            self._stats.current_jobs -= 1

    async def _handle_failure(self, job: Job, error: Exception) -> None:
        """Handle job failure."""
        error_msg = str(error)
        logger.error(f"Job {job.id} failed: {error_msg}")

        # Run error middleware
        for middleware in self._on_error:
            try:
                if asyncio.iscoroutinefunction(middleware):
                    await middleware(job, error)
                else:
                    middleware(job, error)
            except Exception as e:
                logger.warning(f"Error middleware error: {e}")

        # Get handler for callbacks
        handler = self._registry.get(job.handler)

        if job.can_retry:
            self._stats.jobs_retried += 1
            if handler:
                await handler.on_retry(job, error, job.attempts)
        else:
            self._stats.jobs_failed += 1
            if handler:
                await handler.on_failure(job, error)

        await self._queue.fail(job, error_msg)


class WorkerPool:
    """Pool of workers for multiple queues."""

    def __init__(
        self,
        queue: JobQueue,
        registry: HandlerRegistry,
    ):
        self._queue = queue
        self._registry = registry
        self._workers: List[Worker] = []
        self._running = False

    def add_worker(self, config: WorkerConfig) -> Worker:
        """Add a worker to the pool."""
        worker = Worker(self._queue, self._registry, config)
        self._workers.append(worker)
        return worker

    async def start(self) -> None:
        """Start all workers."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting worker pool with {len(self._workers)} workers")

        tasks = [worker.start() for worker in self._workers]
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Stop all workers."""
        if not self._running:
            return

        logger.info("Stopping worker pool")
        self._running = False

        await asyncio.gather(*[
            worker.stop() for worker in self._workers
        ])

    def get_stats(self) -> Dict[str, WorkerStats]:
        """Get stats for all workers."""
        return {
            f"worker_{i}": worker.stats
            for i, worker in enumerate(self._workers)
        }


def setup_signal_handlers(worker: Worker) -> None:
    """Setup signal handlers for graceful shutdown."""
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(worker.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
