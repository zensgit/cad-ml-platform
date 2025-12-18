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
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._max_concurrency = max(1, int(max_concurrency))
        self._max_jobs = max(1, int(max_jobs))
        self._ttl_seconds = max(60, int(ttl_seconds))

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

    async def submit(self, runner: JobRunner, *, meta: Optional[Dict[str, Any]] = None) -> Dedup2DJob:
        job_id = str(uuid.uuid4())
        job = Dedup2DJob(job_id=job_id, meta=dict(meta or {}))

        loop = self._loop
        if loop is None:
            raise RuntimeError("Dedup2D job worker loop not initialized")

        with self._lock:
            self._jobs[job_id] = job
            self._cleanup_locked()

            fut = asyncio.run_coroutine_threadsafe(self._execute(job_id, runner), loop)
            self._futures[job_id] = fut

        return job

    async def get(self, job_id: str) -> Optional[Dedup2DJob]:
        with self._lock:
            self._cleanup_locked()
            return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
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

        except asyncio.CancelledError:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = Dedup2DJobStatus.CANCELED
                    job.finished_at = self._now()
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
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
                self._cleanup_locked()


_JOB_STORE: Dedup2DJobStore | None = None
_JOB_STORE_LOCK = threading.Lock()


def get_dedup2d_job_store() -> Dedup2DJobStore:
    """Get the global job store instance.

    Env:
      - DEDUP2D_ASYNC_MAX_CONCURRENCY (default 2)
      - DEDUP2D_ASYNC_MAX_JOBS (default 200)
      - DEDUP2D_ASYNC_TTL_SECONDS (default 3600)
    """
    global _JOB_STORE
    with _JOB_STORE_LOCK:
        if _JOB_STORE is None:
            _JOB_STORE = Dedup2DJobStore(
                max_concurrency=int(os.getenv("DEDUP2D_ASYNC_MAX_CONCURRENCY", "2")),
                max_jobs=int(os.getenv("DEDUP2D_ASYNC_MAX_JOBS", "200")),
                ttl_seconds=int(os.getenv("DEDUP2D_ASYNC_TTL_SECONDS", "3600")),
            )
        return _JOB_STORE


def reset_dedup2d_job_store() -> None:
    """Reset the job store (testing helper)."""
    store = get_dedup2d_job_store()
    store.reset()

