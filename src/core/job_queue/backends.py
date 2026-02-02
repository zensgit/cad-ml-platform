"""Job Queue Backends.

Provides queue implementations:
- In-memory queue (testing)
- Redis-based queue (production)
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from src.core.job_queue.core import (
    Job,
    JobOptions,
    JobPriority,
    JobQueue,
    JobState,
    QueueStats,
)

logger = logging.getLogger(__name__)


class InMemoryJobQueue(JobQueue):
    """In-memory job queue for testing."""

    def __init__(self):
        # Priority queues per queue name: (negative_priority, timestamp, job)
        self._queues: Dict[str, List[tuple]] = defaultdict(list)
        self._scheduled: Dict[str, List[tuple]] = defaultdict(list)  # Delayed jobs
        self._jobs: Dict[str, Job] = {}
        self._running: Set[str] = set()
        self._dead_letter: Dict[str, List[Job]] = defaultdict(list)
        self._unique_keys: Dict[str, str] = {}  # unique_key -> job_id
        self._lock: Optional[asyncio.Lock] = None

        # Stats
        self._completed_count: Dict[str, int] = defaultdict(int)
        self._failed_count: Dict[str, int] = defaultdict(int)
        self._processing_times: Dict[str, List[float]] = defaultdict(list)

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def enqueue(self, job: Job) -> bool:
        async with self._get_lock():
            # Check unique key
            if job.options.unique_key:
                if job.options.unique_key in self._unique_keys:
                    existing_id = self._unique_keys[job.options.unique_key]
                    existing = self._jobs.get(existing_id)
                    if existing and not existing.is_terminal:
                        logger.debug(f"Job with unique key already exists: {job.options.unique_key}")
                        return False

                self._unique_keys[job.options.unique_key] = job.id

            self._jobs[job.id] = job

            # Check if delayed
            if job.scheduled_at and job.scheduled_at > datetime.utcnow():
                job.state = JobState.SCHEDULED
                heapq.heappush(
                    self._scheduled[job.queue_name],
                    (job.scheduled_at.timestamp(), job.id),
                )
            else:
                job.state = JobState.PENDING
                # Use negative priority for max-heap behavior
                heapq.heappush(
                    self._queues[job.queue_name],
                    (-job.options.priority.value, time.time(), job.id),
                )

            logger.debug(f"Enqueued job {job.id} to {job.queue_name}")
            return True

    async def dequeue(
        self,
        queue_name: str,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[Job]:
        start_time = time.time()

        while True:
            async with self._get_lock():
                # Move scheduled jobs that are ready
                self._promote_scheduled(queue_name)

                # Check TTL and remove expired jobs
                self._cleanup_expired(queue_name)

                # Try to get next job
                while self._queues[queue_name]:
                    _, _, job_id = heapq.heappop(self._queues[queue_name])
                    job = self._jobs.get(job_id)

                    if job and job.state == JobState.PENDING:
                        job.state = JobState.RUNNING
                        job.started_at = datetime.utcnow()
                        job.attempts += 1
                        self._running.add(job_id)

                        logger.debug(f"Dequeued job {job.id} (attempt {job.attempts})")
                        return job

            # No job available
            if timeout_seconds is None:
                return None

            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                return None

            await asyncio.sleep(0.1)

    def _promote_scheduled(self, queue_name: str) -> None:
        """Move ready scheduled jobs to main queue."""
        now = datetime.utcnow().timestamp()

        while self._scheduled[queue_name]:
            scheduled_time, job_id = self._scheduled[queue_name][0]
            if scheduled_time <= now:
                heapq.heappop(self._scheduled[queue_name])
                job = self._jobs.get(job_id)
                if job and job.state == JobState.SCHEDULED:
                    job.state = JobState.PENDING
                    heapq.heappush(
                        self._queues[queue_name],
                        (-job.options.priority.value, time.time(), job_id),
                    )
            else:
                break

    def _cleanup_expired(self, queue_name: str) -> None:
        """Remove jobs that have exceeded TTL."""
        now = datetime.utcnow()
        to_remove = []

        for _, _, job_id in self._queues[queue_name]:
            job = self._jobs.get(job_id)
            if job and job.options.ttl_seconds:
                expires_at = job.created_at + timedelta(seconds=job.options.ttl_seconds)
                if now > expires_at:
                    to_remove.append(job_id)
                    job.state = JobState.CANCELLED
                    logger.debug(f"Job {job_id} expired (TTL)")

    async def complete(self, job: Job, result: Any = None) -> bool:
        async with self._get_lock():
            if job.id not in self._jobs:
                return False

            job.state = JobState.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result
            self._running.discard(job.id)

            # Track stats
            self._completed_count[job.queue_name] += 1
            if job.started_at:
                processing_time = (job.completed_at - job.started_at).total_seconds() * 1000
                self._processing_times[job.queue_name].append(processing_time)
                # Keep last 1000 times
                if len(self._processing_times[job.queue_name]) > 1000:
                    self._processing_times[job.queue_name] = self._processing_times[job.queue_name][-1000:]

            logger.debug(f"Job {job.id} completed")
            return True

    async def fail(self, job: Job, error: str) -> bool:
        async with self._get_lock():
            if job.id not in self._jobs:
                return False

            job.last_error = error
            self._running.discard(job.id)

            if job.can_retry:
                # Schedule retry
                job.state = JobState.RETRYING
                retry_at = datetime.utcnow() + timedelta(seconds=job.next_retry_delay)
                job.scheduled_at = retry_at

                heapq.heappush(
                    self._scheduled[job.queue_name],
                    (retry_at.timestamp(), job.id),
                )

                logger.debug(f"Job {job.id} scheduled for retry at {retry_at}")
            else:
                # Move to dead letter
                job.state = JobState.DEAD
                self._dead_letter[job.queue_name].append(job)
                self._failed_count[job.queue_name] += 1

                logger.debug(f"Job {job.id} moved to dead letter queue")

            return True

    async def get_job(self, job_id: str) -> Optional[Job]:
        async with self._get_lock():
            return self._jobs.get(job_id)

    async def get_queue_size(self, queue_name: str) -> int:
        async with self._get_lock():
            return len([
                1 for _, _, jid in self._queues[queue_name]
                if self._jobs.get(jid) and self._jobs[jid].state == JobState.PENDING
            ])

    async def get_dead_letter_jobs(
        self,
        queue_name: str,
        limit: int = 100,
    ) -> List[Job]:
        async with self._get_lock():
            return self._dead_letter[queue_name][:limit]

    async def retry_dead_letter(self, job_id: str) -> bool:
        async with self._get_lock():
            job = self._jobs.get(job_id)
            if not job or job.state != JobState.DEAD:
                return False

            # Remove from dead letter
            self._dead_letter[job.queue_name] = [
                j for j in self._dead_letter[job.queue_name]
                if j.id != job_id
            ]

            # Reset and re-enqueue
            job.state = JobState.PENDING
            job.attempts = 0
            job.last_error = None

            heapq.heappush(
                self._queues[job.queue_name],
                (-job.options.priority.value, time.time(), job.id),
            )

            logger.debug(f"Dead letter job {job_id} re-enqueued")
            return True

    async def get_stats(self, queue_name: str) -> QueueStats:
        async with self._get_lock():
            pending = await self.get_queue_size(queue_name)
            running = len([
                1 for jid in self._running
                if self._jobs.get(jid) and self._jobs[jid].queue_name == queue_name
            ])

            times = self._processing_times.get(queue_name, [])
            avg_time = sum(times) / len(times) if times else 0

            return QueueStats(
                queue_name=queue_name,
                pending_count=pending,
                running_count=running,
                completed_count=self._completed_count[queue_name],
                failed_count=self._failed_count[queue_name],
                dead_letter_count=len(self._dead_letter[queue_name]),
                avg_processing_time_ms=avg_time,
                jobs_per_minute=0,  # Would need time tracking
            )


class RedisJobQueue(JobQueue):
    """Redis-based job queue."""

    def __init__(
        self,
        redis_client: Any = None,
        key_prefix: str = "jobqueue:",
    ):
        self._redis = redis_client
        self._prefix = key_prefix

    def _queue_key(self, queue_name: str) -> str:
        return f"{self._prefix}queue:{queue_name}"

    def _scheduled_key(self, queue_name: str) -> str:
        return f"{self._prefix}scheduled:{queue_name}"

    def _job_key(self, job_id: str) -> str:
        return f"{self._prefix}job:{job_id}"

    def _dead_letter_key(self, queue_name: str) -> str:
        return f"{self._prefix}dead:{queue_name}"

    def _unique_key(self, key: str) -> str:
        return f"{self._prefix}unique:{key}"

    async def enqueue(self, job: Job) -> bool:
        if self._redis is None:
            return False

        try:
            import json

            # Check unique key
            if job.options.unique_key:
                unique_key = self._unique_key(job.options.unique_key)
                existing = await self._redis.get(unique_key)
                if existing:
                    return False
                # Set unique key with TTL
                ttl = job.options.ttl_seconds or 86400
                await self._redis.setex(unique_key, int(ttl), job.id)

            # Store job data
            job_key = self._job_key(job.id)
            await self._redis.set(job_key, json.dumps(job.to_dict()))

            # Add to appropriate queue
            score = -job.options.priority.value  # Negative for highest first

            if job.scheduled_at and job.scheduled_at > datetime.utcnow():
                job.state = JobState.SCHEDULED
                await self._redis.zadd(
                    self._scheduled_key(job.queue_name),
                    {job.id: job.scheduled_at.timestamp()},
                )
            else:
                job.state = JobState.PENDING
                await self._redis.zadd(
                    self._queue_key(job.queue_name),
                    {job.id: score},
                )

            logger.debug(f"Enqueued job {job.id}")
            return True

        except Exception as e:
            logger.error(f"Redis enqueue error: {e}")
            return False

    async def dequeue(
        self,
        queue_name: str,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[Job]:
        if self._redis is None:
            return None

        start_time = time.time()

        while True:
            try:
                import json

                # Move ready scheduled jobs
                await self._promote_scheduled(queue_name)

                # Get highest priority job atomically
                queue_key = self._queue_key(queue_name)
                job_ids = await self._redis.zrange(queue_key, 0, 0)

                if job_ids:
                    job_id = job_ids[0]
                    if isinstance(job_id, bytes):
                        job_id = job_id.decode()

                    # Remove from queue
                    removed = await self._redis.zrem(queue_key, job_id)
                    if removed:
                        job_key = self._job_key(job_id)
                        job_data = await self._redis.get(job_key)

                        if job_data:
                            if isinstance(job_data, bytes):
                                job_data = job_data.decode()

                            job = Job.from_dict(json.loads(job_data))
                            job.state = JobState.RUNNING
                            job.started_at = datetime.utcnow()
                            job.attempts += 1

                            # Update job data
                            await self._redis.set(job_key, json.dumps(job.to_dict()))

                            logger.debug(f"Dequeued job {job.id}")
                            return job

                # No job available
                if timeout_seconds is None:
                    return None

                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    return None

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Redis dequeue error: {e}")
                return None

    async def _promote_scheduled(self, queue_name: str) -> None:
        """Move ready scheduled jobs to main queue."""
        if self._redis is None:
            return

        try:
            import json

            now = time.time()
            scheduled_key = self._scheduled_key(queue_name)
            queue_key = self._queue_key(queue_name)

            # Get ready jobs
            ready = await self._redis.zrangebyscore(scheduled_key, 0, now)

            for job_id in ready:
                if isinstance(job_id, bytes):
                    job_id = job_id.decode()

                # Remove from scheduled
                await self._redis.zrem(scheduled_key, job_id)

                # Get job to find priority
                job_key = self._job_key(job_id)
                job_data = await self._redis.get(job_key)
                if job_data:
                    if isinstance(job_data, bytes):
                        job_data = job_data.decode()
                    job = Job.from_dict(json.loads(job_data))
                    job.state = JobState.PENDING
                    await self._redis.set(job_key, json.dumps(job.to_dict()))

                    # Add to main queue
                    score = -job.options.priority.value
                    await self._redis.zadd(queue_key, {job_id: score})

        except Exception as e:
            logger.error(f"Redis promote scheduled error: {e}")

    async def complete(self, job: Job, result: Any = None) -> bool:
        if self._redis is None:
            return False

        try:
            import json

            job.state = JobState.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result

            job_key = self._job_key(job.id)
            await self._redis.set(job_key, json.dumps(job.to_dict()))

            # Optionally set TTL on completed job
            await self._redis.expire(job_key, 86400)  # Keep for 24 hours

            logger.debug(f"Job {job.id} completed")
            return True

        except Exception as e:
            logger.error(f"Redis complete error: {e}")
            return False

    async def fail(self, job: Job, error: str) -> bool:
        if self._redis is None:
            return False

        try:
            import json

            job.last_error = error

            if job.can_retry:
                job.state = JobState.RETRYING
                retry_at = datetime.utcnow() + timedelta(seconds=job.next_retry_delay)
                job.scheduled_at = retry_at

                job_key = self._job_key(job.id)
                await self._redis.set(job_key, json.dumps(job.to_dict()))

                await self._redis.zadd(
                    self._scheduled_key(job.queue_name),
                    {job.id: retry_at.timestamp()},
                )

                logger.debug(f"Job {job.id} scheduled for retry")
            else:
                job.state = JobState.DEAD

                job_key = self._job_key(job.id)
                await self._redis.set(job_key, json.dumps(job.to_dict()))

                await self._redis.lpush(
                    self._dead_letter_key(job.queue_name),
                    job.id,
                )

                logger.debug(f"Job {job.id} moved to dead letter")

            return True

        except Exception as e:
            logger.error(f"Redis fail error: {e}")
            return False

    async def get_job(self, job_id: str) -> Optional[Job]:
        if self._redis is None:
            return None

        try:
            import json

            job_key = self._job_key(job_id)
            job_data = await self._redis.get(job_key)

            if job_data:
                if isinstance(job_data, bytes):
                    job_data = job_data.decode()
                return Job.from_dict(json.loads(job_data))

            return None

        except Exception as e:
            logger.error(f"Redis get_job error: {e}")
            return None

    async def get_queue_size(self, queue_name: str) -> int:
        if self._redis is None:
            return 0

        try:
            return await self._redis.zcard(self._queue_key(queue_name))
        except Exception:
            return 0

    async def get_dead_letter_jobs(
        self,
        queue_name: str,
        limit: int = 100,
    ) -> List[Job]:
        if self._redis is None:
            return []

        try:
            import json

            job_ids = await self._redis.lrange(
                self._dead_letter_key(queue_name),
                0,
                limit - 1,
            )

            jobs = []
            for job_id in job_ids:
                if isinstance(job_id, bytes):
                    job_id = job_id.decode()

                job = await self.get_job(job_id)
                if job:
                    jobs.append(job)

            return jobs

        except Exception as e:
            logger.error(f"Redis get_dead_letter error: {e}")
            return []

    async def retry_dead_letter(self, job_id: str) -> bool:
        if self._redis is None:
            return False

        try:
            import json

            job = await self.get_job(job_id)
            if not job or job.state != JobState.DEAD:
                return False

            # Remove from dead letter
            await self._redis.lrem(self._dead_letter_key(job.queue_name), 1, job_id)

            # Reset and re-enqueue
            job.state = JobState.PENDING
            job.attempts = 0
            job.last_error = None

            job_key = self._job_key(job.id)
            await self._redis.set(job_key, json.dumps(job.to_dict()))

            await self._redis.zadd(
                self._queue_key(job.queue_name),
                {job.id: -job.options.priority.value},
            )

            logger.debug(f"Dead letter job {job_id} re-enqueued")
            return True

        except Exception as e:
            logger.error(f"Redis retry_dead_letter error: {e}")
            return False
