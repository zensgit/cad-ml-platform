"""Workflow Scheduler.

Provides scheduling capabilities for workflows:
- Cron-based scheduling
- One-time execution
- Interval-based execution
- Schedule management
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Types of schedules."""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"


@dataclass
class Schedule:
    """Schedule configuration."""
    schedule_id: str
    schedule_type: ScheduleType
    # For ONCE
    run_at: Optional[datetime] = None
    # For INTERVAL
    interval_seconds: Optional[float] = None
    # For CRON
    cron_expression: Optional[str] = None
    # Common
    enabled: bool = True
    max_runs: Optional[int] = None
    end_at: Optional[datetime] = None
    timezone: str = "UTC"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.schedule_type == ScheduleType.ONCE and not self.run_at:
            raise ValueError("ONCE schedule requires run_at")
        if self.schedule_type == ScheduleType.INTERVAL and not self.interval_seconds:
            raise ValueError("INTERVAL schedule requires interval_seconds")
        if self.schedule_type == ScheduleType.CRON and not self.cron_expression:
            raise ValueError("CRON schedule requires cron_expression")


@dataclass
class ScheduledJob:
    """A scheduled job."""
    job_id: str
    name: str
    schedule: Schedule
    handler: Callable[[], Any]
    run_count: int = 0
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class CronParser:
    """Simple cron expression parser.

    Supports: minute hour day_of_month month day_of_week
    Special characters: * (any), */n (every n), n-m (range), n,m (list)
    """

    @staticmethod
    def parse(expression: str) -> Dict[str, List[int]]:
        """Parse cron expression into field values.

        Args:
            expression: Cron expression (5 fields).

        Returns:
            Dict mapping field names to allowed values.
        """
        fields = expression.strip().split()
        if len(fields) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        field_names = ["minute", "hour", "day", "month", "weekday"]
        field_ranges = [
            (0, 59),   # minute
            (0, 23),   # hour
            (1, 31),   # day
            (1, 12),   # month
            (0, 6),    # weekday (0=Sunday)
        ]

        result = {}
        for i, (field_value, (min_val, max_val)) in enumerate(zip(fields, field_ranges)):
            result[field_names[i]] = CronParser._parse_field(
                field_value, min_val, max_val
            )

        return result

    @staticmethod
    def _parse_field(value: str, min_val: int, max_val: int) -> List[int]:
        """Parse a single cron field."""
        if value == "*":
            return list(range(min_val, max_val + 1))

        # Handle */n (step)
        if value.startswith("*/"):
            step = int(value[2:])
            return list(range(min_val, max_val + 1, step))

        # Handle n-m (range)
        if "-" in value and "," not in value:
            start, end = value.split("-")
            return list(range(int(start), int(end) + 1))

        # Handle comma-separated values
        if "," in value:
            result = []
            for part in value.split(","):
                result.extend(CronParser._parse_field(part.strip(), min_val, max_val))
            return sorted(set(result))

        # Single value
        return [int(value)]

    @staticmethod
    def get_next_run(expression: str, after: Optional[datetime] = None) -> datetime:
        """Calculate next run time for cron expression.

        Args:
            expression: Cron expression.
            after: Start time (defaults to now).

        Returns:
            Next run datetime.
        """
        parsed = CronParser.parse(expression)
        now = after or datetime.utcnow()
        candidate = now.replace(second=0, microsecond=0)

        # Brute force search for next valid time (max 1 year ahead)
        max_iterations = 365 * 24 * 60
        for _ in range(max_iterations):
            candidate += timedelta(minutes=1)

            if (
                candidate.minute in parsed["minute"]
                and candidate.hour in parsed["hour"]
                and candidate.day in parsed["day"]
                and candidate.month in parsed["month"]
                and candidate.weekday() in parsed["weekday"]
            ):
                return candidate

        raise ValueError(f"Could not find next run time for: {expression}")


class Scheduler:
    """Workflow scheduler."""

    def __init__(self):
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_job(
        self,
        handler: Callable[[], Any],
        schedule: Schedule,
        name: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Add a scheduled job.

        Args:
            handler: Function to execute.
            schedule: Schedule configuration.
            name: Job name.
            job_id: Optional job ID.

        Returns:
            Job ID.
        """
        job_id = job_id or str(uuid.uuid4())
        name = name or handler.__name__

        job = ScheduledJob(
            job_id=job_id,
            name=name,
            schedule=schedule,
            handler=handler,
            next_run=self._calculate_next_run(schedule),
        )

        self._jobs[job_id] = job
        logger.info(f"Added scheduled job: {name} ({job_id}), next run: {job.next_run}")
        return job_id

    def add_once(
        self,
        handler: Callable[[], Any],
        run_at: datetime,
        name: Optional[str] = None,
    ) -> str:
        """Add a one-time job."""
        schedule = Schedule(
            schedule_id=str(uuid.uuid4()),
            schedule_type=ScheduleType.ONCE,
            run_at=run_at,
        )
        return self.add_job(handler, schedule, name)

    def add_interval(
        self,
        handler: Callable[[], Any],
        seconds: float,
        name: Optional[str] = None,
        max_runs: Optional[int] = None,
    ) -> str:
        """Add an interval-based job."""
        schedule = Schedule(
            schedule_id=str(uuid.uuid4()),
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=seconds,
            max_runs=max_runs,
        )
        return self.add_job(handler, schedule, name)

    def add_cron(
        self,
        handler: Callable[[], Any],
        cron_expression: str,
        name: Optional[str] = None,
        max_runs: Optional[int] = None,
    ) -> str:
        """Add a cron-scheduled job."""
        # Validate expression
        CronParser.parse(cron_expression)

        schedule = Schedule(
            schedule_id=str(uuid.uuid4()),
            schedule_type=ScheduleType.CRON,
            cron_expression=cron_expression,
            max_runs=max_runs,
        )
        return self.add_job(handler, schedule, name)

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info(f"Removed scheduled job: {job_id}")
            return True
        return False

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_jobs(self) -> List[ScheduledJob]:
        """Get all jobs."""
        return list(self._jobs.values())

    def pause_job(self, job_id: str) -> bool:
        """Pause a job."""
        job = self._jobs.get(job_id)
        if job:
            job.schedule.enabled = False
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self._jobs.get(job_id)
        if job:
            job.schedule.enabled = True
            job.next_run = self._calculate_next_run(job.schedule)
            return True
        return False

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.utcnow()

            for job_id, job in list(self._jobs.items()):
                if not job.schedule.enabled:
                    continue

                if job.next_run and job.next_run <= now:
                    await self._execute_job(job)

            await asyncio.sleep(1)  # Check every second

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a job."""
        logger.info(f"Executing job: {job.name} ({job.job_id})")

        try:
            result = job.handler()
            if asyncio.iscoroutine(result):
                await result

            job.last_run = datetime.utcnow()
            job.run_count += 1
            job.last_error = None

        except Exception as e:
            job.last_error = str(e)
            logger.error(f"Job {job.name} failed: {e}")

        # Calculate next run
        schedule = job.schedule

        # Check max runs
        if schedule.max_runs and job.run_count >= schedule.max_runs:
            logger.info(f"Job {job.name} reached max runs ({schedule.max_runs})")
            schedule.enabled = False
            job.next_run = None
            return

        # Check end time
        if schedule.end_at and datetime.utcnow() >= schedule.end_at:
            logger.info(f"Job {job.name} reached end time")
            schedule.enabled = False
            job.next_run = None
            return

        # Schedule next run
        if schedule.schedule_type == ScheduleType.ONCE:
            schedule.enabled = False
            job.next_run = None
        else:
            job.next_run = self._calculate_next_run(schedule, job.last_run)

    def _calculate_next_run(
        self,
        schedule: Schedule,
        after: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """Calculate next run time for a schedule."""
        now = after or datetime.utcnow()

        if schedule.schedule_type == ScheduleType.ONCE:
            return schedule.run_at if schedule.run_at > now else None

        elif schedule.schedule_type == ScheduleType.INTERVAL:
            return now + timedelta(seconds=schedule.interval_seconds)

        elif schedule.schedule_type == ScheduleType.CRON:
            return CronParser.get_next_run(schedule.cron_expression, now)

        return None


# Global scheduler instance
_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler


# Decorators for scheduling
def scheduled(
    cron: Optional[str] = None,
    interval: Optional[float] = None,
    run_at: Optional[datetime] = None,
    name: Optional[str] = None,
) -> Callable:
    """Decorator to schedule a function.

    Example:
        @scheduled(cron="0 * * * *")
        async def hourly_task():
            ...

        @scheduled(interval=60)
        async def every_minute():
            ...
    """
    def decorator(func: Callable) -> Callable:
        scheduler = get_scheduler()

        if cron:
            scheduler.add_cron(func, cron, name)
        elif interval:
            scheduler.add_interval(func, interval, name)
        elif run_at:
            scheduler.add_once(func, run_at, name)
        else:
            raise ValueError("Must specify cron, interval, or run_at")

        return func

    return decorator
