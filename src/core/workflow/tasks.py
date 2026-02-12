"""Workflow Task Definition.

Provides task primitives for workflow execution:
- Task definition and configuration
- Input/output validation
- Retry policies
- Timeout handling
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 100


@dataclass
class RetryPolicy:
    """Retry configuration for tasks."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    retry_on: Optional[List[type]] = None  # Exception types to retry on

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given retry attempt."""
        delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay)


@dataclass
class TaskConfig:
    """Task execution configuration."""
    timeout: Optional[float] = None  # seconds
    retry_policy: Optional[RetryPolicy] = None
    priority: TaskPriority = TaskPriority.NORMAL
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


T = TypeVar('T')
R = TypeVar('R')


class Task(ABC, Generic[T, R]):
    """Base class for workflow tasks."""

    def __init__(
        self,
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[TaskConfig] = None,
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.config = config or TaskConfig()
        self._status = TaskStatus.PENDING
        self._result: Optional[TaskResult] = None

    @property
    def status(self) -> TaskStatus:
        return self._status

    @property
    def result(self) -> Optional[TaskResult]:
        return self._result

    @abstractmethod
    async def execute(self, input_data: T) -> R:
        """Execute the task with given input.

        Args:
            input_data: Input data for the task.

        Returns:
            Task output.
        """
        pass

    async def run(self, input_data: T) -> TaskResult:
        """Run the task with retry and timeout handling.

        Args:
            input_data: Input data for the task.

        Returns:
            TaskResult with execution details.
        """
        self._status = TaskStatus.RUNNING
        started_at = datetime.utcnow()
        retries = 0

        while True:
            try:
                # Apply timeout if configured
                if self.config.timeout:
                    output = await asyncio.wait_for(
                        self.execute(input_data),
                        timeout=self.config.timeout
                    )
                else:
                    output = await self.execute(input_data)

                completed_at = datetime.utcnow()
                duration_ms = (completed_at - started_at).total_seconds() * 1000

                self._status = TaskStatus.COMPLETED
                self._result = TaskResult(
                    task_id=self.task_id,
                    status=TaskStatus.COMPLETED,
                    output=output,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    retries=retries,
                )
                return self._result

            except asyncio.TimeoutError:
                error_msg = f"Task timed out after {self.config.timeout}s"
                logger.warning(f"Task {self.name} ({self.task_id}): {error_msg}")

                # Timeout doesn't retry by default
                self._status = TaskStatus.FAILED
                self._result = TaskResult(
                    task_id=self.task_id,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                    error_type="TimeoutError",
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    retries=retries,
                )
                return self._result

            except asyncio.CancelledError:
                self._status = TaskStatus.CANCELLED
                self._result = TaskResult(
                    task_id=self.task_id,
                    status=TaskStatus.CANCELLED,
                    error="Task was cancelled",
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    retries=retries,
                )
                return self._result

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                # Check if we should retry
                retry_policy = self.config.retry_policy
                if retry_policy and retries < retry_policy.max_retries:
                    # Check if this exception type should be retried
                    should_retry = True
                    if retry_policy.retry_on:
                        should_retry = any(
                            isinstance(e, exc_type)
                            for exc_type in retry_policy.retry_on
                        )

                    if should_retry:
                        retries += 1
                        delay = retry_policy.get_delay(retries - 1)
                        logger.warning(
                            f"Task {self.name} ({self.task_id}) failed: {error_msg}. "
                            f"Retrying in {delay:.1f}s (attempt {retries}/{retry_policy.max_retries})"
                        )
                        self._status = TaskStatus.RETRYING
                        await asyncio.sleep(delay)
                        continue

                # No more retries
                self._status = TaskStatus.FAILED
                self._result = TaskResult(
                    task_id=self.task_id,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                    error_type=error_type,
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    retries=retries,
                )
                logger.error(
                    f"Task {self.name} ({self.task_id}) failed after {retries} retries: {error_msg}"
                )
                return self._result


class FunctionTask(Task[T, R]):
    """Task wrapper for async functions."""

    def __init__(
        self,
        func: Callable[[T], R],
        task_id: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[TaskConfig] = None,
    ):
        super().__init__(task_id, name or func.__name__, config)
        self._func = func

    async def execute(self, input_data: T) -> R:
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(input_data)
        return self._func(input_data)


class LambdaTask(Task[T, R]):
    """Inline task defined with a lambda or simple function."""

    def __init__(
        self,
        func: Callable[[T], R],
        task_id: Optional[str] = None,
        name: str = "LambdaTask",
        config: Optional[TaskConfig] = None,
    ):
        super().__init__(task_id, name, config)
        self._func = func

    async def execute(self, input_data: T) -> R:
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(input_data)
        return self._func(input_data)


class NoOpTask(Task[Any, Any]):
    """A task that does nothing (for testing/placeholder)."""

    def __init__(
        self,
        task_id: Optional[str] = None,
        name: str = "NoOpTask",
        output: Any = None,
    ):
        super().__init__(task_id, name)
        self._output = output

    async def execute(self, input_data: Any) -> Any:
        return self._output


class DelayTask(Task[T, T]):
    """A task that introduces a delay."""

    def __init__(
        self,
        delay_seconds: float,
        task_id: Optional[str] = None,
        name: str = "DelayTask",
    ):
        super().__init__(task_id, name)
        self._delay = delay_seconds

    async def execute(self, input_data: T) -> T:
        await asyncio.sleep(self._delay)
        return input_data


class ConditionalTask(Task[T, R]):
    """Task that executes based on a condition."""

    def __init__(
        self,
        condition: Callable[[T], bool],
        if_true: Task[T, R],
        if_false: Optional[Task[T, R]] = None,
        task_id: Optional[str] = None,
        name: str = "ConditionalTask",
    ):
        super().__init__(task_id, name)
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false

    async def execute(self, input_data: T) -> R:
        if self._condition(input_data):
            result = await self._if_true.run(input_data)
            return result.output
        elif self._if_false:
            result = await self._if_false.run(input_data)
            return result.output
        else:
            return None  # type: ignore


class ParallelTask(Task[T, List[Any]]):
    """Execute multiple tasks in parallel."""

    def __init__(
        self,
        tasks: List[Task],
        task_id: Optional[str] = None,
        name: str = "ParallelTask",
        fail_fast: bool = True,
    ):
        super().__init__(task_id, name)
        self._tasks = tasks
        self._fail_fast = fail_fast

    async def execute(self, input_data: T) -> List[Any]:
        if self._fail_fast:
            # All tasks must succeed
            results = await asyncio.gather(
                *[task.run(input_data) for task in self._tasks],
                return_exceptions=False
            )
        else:
            # Continue even if some fail
            results = await asyncio.gather(
                *[task.run(input_data) for task in self._tasks],
                return_exceptions=True
            )

        return [
            r.output if isinstance(r, TaskResult) else r
            for r in results
        ]


class SequentialTask(Task[T, Any]):
    """Execute tasks sequentially, passing output to next input."""

    def __init__(
        self,
        tasks: List[Task],
        task_id: Optional[str] = None,
        name: str = "SequentialTask",
    ):
        super().__init__(task_id, name)
        self._tasks = tasks

    async def execute(self, input_data: T) -> Any:
        current_input = input_data
        for task in self._tasks:
            result = await task.run(current_input)
            if result.status != TaskStatus.COMPLETED:
                raise RuntimeError(
                    f"Task {task.name} failed: {result.error}"
                )
            current_input = result.output
        return current_input


# Helper function to create tasks
def task(
    name: Optional[str] = None,
    timeout: Optional[float] = None,
    retry_policy: Optional[RetryPolicy] = None,
    priority: TaskPriority = TaskPriority.NORMAL,
) -> Callable[[Callable], FunctionTask]:
    """Decorator to create a task from a function.

    Example:
        @task(name="process_file", timeout=30)
        async def process_file(file_path: str) -> dict:
            ...
    """
    def decorator(func: Callable) -> FunctionTask:
        config = TaskConfig(
            timeout=timeout,
            retry_policy=retry_policy,
            priority=priority,
        )
        return FunctionTask(func, name=name, config=config)
    return decorator
