"""Async job manager for 2D dedup requests.

The 2D dedup endpoint (`/api/v1/dedup/2d/search`) can be expensive when:
- dedupcad-vision diff rendering is enabled
- local L4 precision verification is enabled (v2 JSON scoring + optional diffs)

This module provides a lightweight in-process job manager so cad-ml-platform can:
- accept the request,
- return a `job_id` immediately,
- let clients poll for the final result.

Implementation notes:
- Dependency-free (no Redis required).
- Uses a dedicated background thread running an asyncio event loop, so jobs keep
  running beyond the lifecycle of a single request coroutine (important for
  uvicorn + for tests using FastAPI TestClient).
- Jobs are stored in-memory with TTL + max size caps.

Phase 1 Enhancements:
- Tenant isolation: jobs are scoped by tenant_id
- Queue capacity protection: JOB_QUEUE_FULL error when max_jobs exceeded
- Multi-worker detection: startup warning for uvicorn --workers > 1
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class JobMetricsCallback(Protocol):
    """Protocol for job metrics callback (dependency injection for observability)."""

    def on_job_completed(self, job: "Dedup2DJob", duration_seconds: float) -> None:
        """Called when a job completes successfully."""
        ...

    def on_job_failed(self, job: "Dedup2DJob", duration_seconds: float) -> None:
        """Called when a job fails."""
        ...

    def on_job_canceled(self, job: "Dedup2DJob") -> None:
        """Called when a job is canceled."""
        ...

    def on_queue_depth_changed(self, depth: int) -> None:
        """Called when the queue depth changes."""
        ...


class JobQueueFullError(Exception):
    """Raised when the job queue is at capacity and cannot accept new jobs."""

    def __init__(self, max_jobs: int, current_jobs: int):
        self.max_jobs = max_jobs
        self.current_jobs = current_jobs
        super().__init__(f"Job queue full: {current_jobs}/{max_jobs} jobs")


class JobNotFoundError(Exception):
    """Raised when a job is not found."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Job not found: {job_id}")


class JobForbiddenError(Exception):
    """Raised when a tenant tries to access another tenant's job."""

    def __init__(self, job_id: str, tenant_id: str):
        self.job_id = job_id
        self.tenant_id = tenant_id
        super().__init__(f"Access denied to job {job_id} for tenant {tenant_id}")


class Dedup2DJobStatus(str, Enum):
    """Lifecycle status for an async dedup job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class Dedup2DJob:
    job_id: str
    tenant_id: str  # Phase 1: tenant isolation
    status: Dedup2DJobStatus = Dedup2DJobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def is_finished(self) -> bool:
        return self.status in {
            Dedup2DJobStatus.COMPLETED,
            Dedup2DJobStatus.FAILED,
            Dedup2DJobStatus.CANCELED,
        }


JobRunner = Callable[[], Awaitable[Dict[str, Any]]]


class Dedup2DJobStore:
    """In-memory async job store backed by a background asyncio loop thread."""

    def __init__(
        self,
        *,
        max_concurrency: int = 2,
        max_jobs: int = 200,
        ttl_seconds: int = 3600,
        metrics_callback: Optional[JobMetricsCallback] = None,
    ) -> None:
        self._max_concurrency = max(1, int(max_concurrency))
        self._max_jobs = max(1, int(max_jobs))
        self._ttl_seconds = max(60, int(ttl_seconds))
        self._metrics_callback = metrics_callback

        self._lock = threading.Lock()
        self._jobs: Dict[str, Dedup2DJob] = {}
        self._futures: Dict[str, concurrent.futures.Future] = {}

        self._loop_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="dedup2d-job-worker",
            daemon=True,
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=5.0):
            raise RuntimeError("Dedup2D job worker loop failed to start")

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._loop_ready.set()
        loop.run_forever()

    def _now(self) -> float:
        return time.time()

    def _is_expired(self, job: Dedup2DJob, now: float) -> bool:
        if not job.is_finished():
            return False
        if job.finished_at is None:
            return False
        return (now - job.finished_at) > float(self._ttl_seconds)

    def _cleanup_locked(self) -> None:
        """Cleanup expired jobs and cap retention (lock must be held)."""
        now = self._now()

        expired_ids = [job_id for job_id, job in self._jobs.items() if self._is_expired(job, now)]
        for job_id in expired_ids:
            self._jobs.pop(job_id, None)
            fut = self._futures.pop(job_id, None)
            if fut is not None and not fut.done():
                fut.cancel()

        if len(self._jobs) <= self._max_jobs:
            return

        finished = [j for j in self._jobs.values() if j.is_finished()]
        finished.sort(key=lambda j: j.finished_at or j.created_at)
        to_drop = len(self._jobs) - self._max_jobs
        for job in finished[:to_drop]:
            self._jobs.pop(job.job_id, None)
            self._futures.pop(job.job_id, None)

    async def submit(
        self,
        runner: JobRunner,
        *,
        tenant_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dedup2DJob:
        """Submit a new async job.

        Args:
            runner: Async function to execute
            tenant_id: Tenant identifier for job isolation
            meta: Optional metadata dict

        Returns:
            The created Dedup2DJob

        Raises:
            JobQueueFullError: When queue is at capacity
            RuntimeError: When worker loop is not initialized
        """
        job_id = str(uuid.uuid4())
        job = Dedup2DJob(job_id=job_id, tenant_id=tenant_id, meta=dict(meta or {}))

        loop = self._loop
        if loop is None:
            raise RuntimeError("Dedup2D job worker loop not initialized")

        with self._lock:
            self._cleanup_locked()

            # Phase 1: Check queue capacity before accepting new job
            pending_or_running = sum(1 for j in self._jobs.values() if not j.is_finished())
            if pending_or_running >= self._max_jobs:
                raise JobQueueFullError(self._max_jobs, pending_or_running)

            self._jobs[job_id] = job

            fut = asyncio.run_coroutine_threadsafe(self._execute(job_id, runner), loop)
            self._futures[job_id] = fut

        return job

    async def get(self, job_id: str) -> Optional[Dedup2DJob]:
        """Get a job by ID (no tenant check - use get_for_tenant for isolation)."""
        with self._lock:
            self._cleanup_locked()
            return self._jobs.get(job_id)

    async def get_for_tenant(self, job_id: str, tenant_id: str) -> Dedup2DJob:
        """Get a job by ID with tenant isolation.

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier for access control

        Returns:
            The Dedup2DJob

        Raises:
            JobNotFoundError: When job does not exist
            JobForbiddenError: When tenant does not own the job
        """
        with self._lock:
            self._cleanup_locked()
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.tenant_id != tenant_id:
                raise JobForbiddenError(job_id, tenant_id)
            return job

    async def cancel(self, job_id: str) -> bool:
        """Cancel a job by ID (no tenant check - use cancel_for_tenant for isolation)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.is_finished():
                return True

            job.status = Dedup2DJobStatus.CANCELED
            job.finished_at = self._now()

            fut = self._futures.get(job_id)
            if fut is not None and not fut.done():
                fut.cancel()
            return True

    async def cancel_for_tenant(self, job_id: str, tenant_id: str) -> bool:
        """Cancel a job by ID with tenant isolation.

        Args:
            job_id: Job identifier
            tenant_id: Tenant identifier for access control

        Returns:
            True if job was canceled or already finished

        Raises:
            JobNotFoundError: When job does not exist
            JobForbiddenError: When tenant does not own the job
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.tenant_id != tenant_id:
                raise JobForbiddenError(job_id, tenant_id)
            if job.is_finished():
                return True

            job.status = Dedup2DJobStatus.CANCELED
            job.finished_at = self._now()

            fut = self._futures.get(job_id)
            if fut is not None and not fut.done():
                fut.cancel()
            return True

    async def list_for_tenant(
        self, tenant_id: str, status: Optional[Dedup2DJobStatus] = None, limit: int = 100
    ) -> List[Dedup2DJob]:
        """List jobs for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            status: Optional status filter
            limit: Maximum number of jobs to return

        Returns:
            List of jobs belonging to the tenant
        """
        with self._lock:
            self._cleanup_locked()
            jobs = [j for j in self._jobs.values() if j.tenant_id == tenant_id]
            if status is not None:
                jobs = [j for j in jobs if j.status == status]
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

    def get_queue_depth(self) -> int:
        """Get the current number of pending/running jobs."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if not j.is_finished())

    def reset(self) -> None:
        """Best-effort reset for tests: cancel all running jobs and clear state."""
        with self._lock:
            for fut in self._futures.values():
                try:
                    if not fut.done():
                        fut.cancel()
                except Exception:
                    pass
            self._jobs.clear()
            self._futures.clear()

    async def _execute(self, job_id: str, runner: JobRunner) -> None:
        semaphore = self._semaphore
        if semaphore is None:
            raise RuntimeError("Dedup2D job worker semaphore not initialized")

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.status == Dedup2DJobStatus.CANCELED:
                self._futures.pop(job_id, None)
                return
            job.status = Dedup2DJobStatus.IN_PROGRESS
            job.started_at = self._now()

        try:
            async with semaphore:
                result = await runner()

            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job.status == Dedup2DJobStatus.CANCELED:
                    # Cancel wins: do not overwrite with success.
                    return
                job.status = Dedup2DJobStatus.COMPLETED
                job.result = result
                job.finished_at = self._now()
                duration = job.finished_at - (job.started_at or job.created_at)
                # Avoid re-acquiring the same lock (threading.Lock is not re-entrant).
                queue_depth = sum(1 for j in self._jobs.values() if not j.is_finished())

            # Call metrics callback outside the lock
            if self._metrics_callback is not None:
                try:
                    self._metrics_callback.on_job_completed(job, duration)
                    self._metrics_callback.on_queue_depth_changed(queue_depth)
                except Exception:
                    logger.warning("dedup2d_metrics_callback_error", exc_info=True)

        except asyncio.CancelledError:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = Dedup2DJobStatus.CANCELED
                    job.finished_at = self._now()
                    queue_depth = sum(1 for j in self._jobs.values() if not j.is_finished())

            # Call metrics callback outside the lock
            if self._metrics_callback is not None and job is not None:
                try:
                    self._metrics_callback.on_job_canceled(job)
                    self._metrics_callback.on_queue_depth_changed(queue_depth)
                except Exception:
                    logger.warning("dedup2d_metrics_callback_error", exc_info=True)
            raise
        except Exception as e:
            logger.exception("dedup_2d_async_job_failed", extra={"job_id": job_id, "error": str(e)})
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job.status == Dedup2DJobStatus.CANCELED:
                    return
                job.status = Dedup2DJobStatus.FAILED
                job.error = str(e)
                job.finished_at = self._now()
                duration = job.finished_at - (job.started_at or job.created_at)
                queue_depth = sum(1 for j in self._jobs.values() if not j.is_finished())

            # Call metrics callback outside the lock
            if self._metrics_callback is not None:
                try:
                    self._metrics_callback.on_job_failed(job, duration)
                    self._metrics_callback.on_queue_depth_changed(queue_depth)
                except Exception:
                    logger.warning("dedup2d_metrics_callback_error", exc_info=True)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
                self._cleanup_locked()


_JOB_STORE: Dedup2DJobStore | None = None
_JOB_STORE_LOCK = threading.Lock()
_MULTI_WORKER_WARNING_ISSUED = False
_GLOBAL_METRICS_CALLBACK: JobMetricsCallback | None = None


def _check_multi_worker_warning() -> None:
    """Issue a warning if running with multiple workers (in-process store limitation)."""
    global _MULTI_WORKER_WARNING_ISSUED
    if _MULTI_WORKER_WARNING_ISSUED:
        return
    _MULTI_WORKER_WARNING_ISSUED = True

    # Check common multi-worker environment variables
    workers = os.getenv("WEB_CONCURRENCY", "1")
    uvicorn_workers = os.getenv("UVICORN_WORKERS", "1")

    try:
        if int(workers) > 1 or int(uvicorn_workers) > 1:
            msg = (
                "⚠️ Multi-worker mode detected (WEB_CONCURRENCY=%s, UVICORN_WORKERS=%s). "
                "In-process job store cannot share state across workers. "
                "Jobs submitted to one worker will not be visible to others. "
                "For production, set DEDUP2D_ASYNC_BACKEND=redis to enable distributed job storage."
            ) % (workers, uvicorn_workers)
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    except (ValueError, TypeError):
        pass


def get_dedup2d_job_store() -> Dedup2DJobStore:
    """Get the global job store instance.

    Env:
      - DEDUP2D_ASYNC_MAX_CONCURRENCY (default 2)
      - DEDUP2D_ASYNC_MAX_JOBS (default 200)
      - DEDUP2D_ASYNC_TTL_SECONDS (default 3600)

    Note:
      In-process store does not share state across workers. For multi-worker
      deployments, set DEDUP2D_ASYNC_BACKEND=redis.
    """
    global _JOB_STORE
    with _JOB_STORE_LOCK:
        if _JOB_STORE is None:
            _check_multi_worker_warning()
            _JOB_STORE = Dedup2DJobStore(
                max_concurrency=int(os.getenv("DEDUP2D_ASYNC_MAX_CONCURRENCY", "2")),
                max_jobs=int(os.getenv("DEDUP2D_ASYNC_MAX_JOBS", "200")),
                ttl_seconds=int(os.getenv("DEDUP2D_ASYNC_TTL_SECONDS", "3600")),
                metrics_callback=_GLOBAL_METRICS_CALLBACK,
            )
        return _JOB_STORE


def reset_dedup2d_job_store() -> None:
    """Reset the job store (testing helper)."""
    store = get_dedup2d_job_store()
    store.reset()


def set_dedup2d_job_metrics_callback(callback: JobMetricsCallback) -> None:
    """Set the metrics callback on the global job store.

    Call this at application startup to enable metrics reporting for job
    completion/failure events.

    Example:
        from src.core.dedupcad_2d_jobs import set_dedup2d_job_metrics_callback

        class Dedup2DJobMetrics:
            def on_job_completed(self, job, duration_seconds):
                dedup2d_jobs_total.labels(status="completed").inc()
                dedup2d_job_duration_seconds.observe(duration_seconds)

            def on_job_failed(self, job, duration_seconds):
                dedup2d_jobs_total.labels(status="failed").inc()
                dedup2d_job_duration_seconds.observe(duration_seconds)

            def on_job_canceled(self, job):
                dedup2d_jobs_total.labels(status="canceled").inc()

            def on_queue_depth_changed(self, depth):
                dedup2d_job_queue_depth.set(depth)

        set_dedup2d_job_metrics_callback(Dedup2DJobMetrics())
    """
    global _GLOBAL_METRICS_CALLBACK
    _GLOBAL_METRICS_CALLBACK = callback
    if _JOB_STORE is None:
        return
    with _JOB_STORE_LOCK:
        if _JOB_STORE is not None:
            _JOB_STORE._metrics_callback = callback


__all__ = [
    "Dedup2DJob",
    "Dedup2DJobStatus",
    "Dedup2DJobStore",
    "JobForbiddenError",
    "JobMetricsCallback",
    "JobNotFoundError",
    "JobQueueFullError",
    "get_dedup2d_job_store",
    "reset_dedup2d_job_store",
    "set_dedup2d_job_metrics_callback",
]
