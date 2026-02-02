"""Task Scheduler Implementation.

Provides cron-based and interval-based task scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CronSchedule:
    """Cron-like schedule definition.

    Supports: minute, hour, day_of_month, month, day_of_week
    """

    minute: str = "*"
    hour: str = "*"
    day_of_month: str = "*"
    month: str = "*"
    day_of_week: str = "*"

    def matches(self, dt: datetime) -> bool:
        """Check if datetime matches the schedule."""
        if not self._matches_field(self.minute, dt.minute, 0, 59):
            return False
        if not self._matches_field(self.hour, dt.hour, 0, 23):
            return False
        if not self._matches_field(self.day_of_month, dt.day, 1, 31):
            return False
        if not self._matches_field(self.month, dt.month, 1, 12):
            return False
        if not self._matches_field(self.day_of_week, dt.weekday(), 0, 6):
            return False
        return True

    def _matches_field(self, pattern: str, value: int, min_val: int, max_val: int) -> bool:
        """Check if a value matches a cron pattern."""
        if pattern == "*":
            return True

        # Handle ranges like "1-5"
        if "-" in pattern:
            start, end = pattern.split("-")
            return int(start) <= value <= int(end)

        # Handle steps like "*/5"
        if pattern.startswith("*/"):
            step = int(pattern[2:])
            return value % step == 0

        # Handle lists like "1,3,5"
        if "," in pattern:
            return value in [int(v) for v in pattern.split(",")]

        # Simple value
        return int(pattern) == value

    @classmethod
    def every_minute(cls) -> "CronSchedule":
        return cls()

    @classmethod
    def every_hour(cls) -> "CronSchedule":
        return cls(minute="0")

    @classmethod
    def daily(cls, hour: int = 0, minute: int = 0) -> "CronSchedule":
        return cls(minute=str(minute), hour=str(hour))

    @classmethod
    def weekly(cls, day_of_week: int = 0, hour: int = 0) -> "CronSchedule":
        return cls(minute="0", hour=str(hour), day_of_week=str(day_of_week))


@dataclass
class IntervalSchedule:
    """Interval-based schedule definition."""

    seconds: int = 0
    minutes: int = 0
    hours: int = 0
    days: int = 0

    @property
    def total_seconds(self) -> int:
        return self.seconds + self.minutes * 60 + self.hours * 3600 + self.days * 86400

    @classmethod
    def every(cls, **kwargs: int) -> "IntervalSchedule":
        return cls(**kwargs)


class ScheduleType(str, Enum):
    """Type of schedule."""

    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"


@dataclass
class ScheduledTask:
    """A scheduled task definition."""

    task_id: str
    name: str
    func: Callable[..., Any]
    schedule: Union[CronSchedule, IntervalSchedule, datetime]
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def schedule_type(self) -> ScheduleType:
        if isinstance(self.schedule, CronSchedule):
            return ScheduleType.CRON
        elif isinstance(self.schedule, IntervalSchedule):
            return ScheduleType.INTERVAL
        else:
            return ScheduleType.ONCE


class TaskScheduler:
    """Task scheduler for periodic and scheduled task execution."""

    def __init__(self):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def schedule(
        self,
        func: Callable[..., Any],
        schedule: Union[CronSchedule, IntervalSchedule, datetime],
        name: Optional[str] = None,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        max_runs: Optional[int] = None,
    ) -> str:
        """Schedule a task.

        Args:
            func: Function to execute
            schedule: When to run (cron, interval, or one-time)
            name: Task name
            args: Positional arguments
            kwargs: Keyword arguments
            max_runs: Maximum number of executions

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())

        task = ScheduledTask(
            task_id=task_id,
            name=name or func.__name__,
            func=func,
            schedule=schedule,
            args=args,
            kwargs=kwargs or {},
            max_runs=max_runs,
        )

        # Calculate next run time
        task.next_run = self._calculate_next_run(task)

        self._tasks[task_id] = task
        logger.info(f"Scheduled task '{task.name}' ({task_id}), next run: {task.next_run}")

        return task_id

    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate the next run time for a task."""
        now = datetime.utcnow()

        if task.schedule_type == ScheduleType.ONCE:
            return task.schedule if task.schedule > now else None

        elif task.schedule_type == ScheduleType.INTERVAL:
            if task.last_run:
                return task.last_run + timedelta(seconds=task.schedule.total_seconds)
            return now + timedelta(seconds=task.schedule.total_seconds)

        elif task.schedule_type == ScheduleType.CRON:
            # Find next matching time
            next_time = now.replace(second=0, microsecond=0)
            for _ in range(60 * 24 * 31):  # Max 31 days ahead
                next_time += timedelta(minutes=1)
                if task.schedule.matches(next_time):
                    return next_time

        return None

    def unschedule(self, task_id: str) -> bool:
        """Remove a scheduled task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            logger.info(f"Unscheduled task {task_id}")
            return True
        return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a scheduled task."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[ScheduledTask]:
        """List all scheduled tasks."""
        return list(self._tasks.values())

    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        task = self._tasks.get(task_id)
        if task:
            task.enabled = True
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        task = self._tasks.get(task_id)
        if task:
            task.enabled = False
            return True
        return False

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._run_loop())
        logger.info("Task scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        logger.info("Task scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.utcnow()

                async with self._get_lock():
                    # Find tasks ready to run
                    ready_tasks = [
                        task for task in self._tasks.values()
                        if task.enabled
                        and task.next_run
                        and task.next_run <= now
                        and (task.max_runs is None or task.run_count < task.max_runs)
                    ]

                # Execute ready tasks
                for task in ready_tasks:
                    asyncio.create_task(self._execute_task(task))

                # Sleep until next check (every second)
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)

    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a scheduled task."""
        try:
            logger.debug(f"Executing task '{task.name}' ({task.task_id})")

            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)

            # Update task state
            task.last_run = datetime.utcnow()
            task.run_count += 1
            task.next_run = self._calculate_next_run(task)

            logger.debug(f"Task '{task.name}' completed, next run: {task.next_run}")

            # Remove one-time tasks or tasks that reached max runs
            if task.next_run is None or (task.max_runs and task.run_count >= task.max_runs):
                del self._tasks[task.task_id]
                logger.info(f"Task '{task.name}' completed and removed")

        except Exception as e:
            logger.error(f"Task '{task.name}' failed: {e}")
            # Still update last_run to prevent immediate retry
            task.last_run = datetime.utcnow()
            task.next_run = self._calculate_next_run(task)

    def cron(
        self,
        schedule: CronSchedule,
        name: Optional[str] = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator to schedule a function with cron.

        Example:
            @scheduler.cron(CronSchedule.daily(hour=2))
            async def daily_cleanup():
                ...
        """
        def decorator(func: Callable) -> Callable:
            self.schedule(func, schedule, name=name or func.__name__)
            return func
        return decorator

    def interval(
        self,
        **kwargs: int,
    ) -> Callable[[Callable], Callable]:
        """Decorator to schedule a function with interval.

        Example:
            @scheduler.interval(minutes=5)
            async def periodic_check():
                ...
        """
        schedule = IntervalSchedule.every(**kwargs)

        def decorator(func: Callable) -> Callable:
            self.schedule(func, schedule, name=func.__name__)
            return func
        return decorator


# Global scheduler
_scheduler: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get global task scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
