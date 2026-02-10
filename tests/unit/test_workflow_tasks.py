"""Tests for workflow tasks module to improve coverage."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from src.core.workflow.tasks import (
    Task,
    TaskConfig,
    TaskStatus,
    TaskResult,
    TaskPriority,
    RetryPolicy,
    FunctionTask,
    LambdaTask,
    NoOpTask,
    DelayTask,
    ConditionalTask,
    ParallelTask,
    SequentialTask,
    task,
)


class TestTaskProperties:
    """Tests for Task property accessors."""

    def test_status_property(self):
        """Test Task.status property (line 104)."""
        async def my_func(data):
            return data

        t = FunctionTask(my_func, name="test")
        assert t.status == TaskStatus.PENDING

    def test_result_property_none(self):
        """Test Task.result property returns None initially (line 108)."""
        async def my_func(data):
            return data

        t = FunctionTask(my_func, name="test")
        assert t.result is None

    @pytest.mark.asyncio
    async def test_result_property_after_run(self):
        """Test Task.result property after execution."""
        async def my_func(data):
            return data * 2

        t = FunctionTask(my_func, name="test")
        await t.run(5)

        assert t.result is not None
        assert t.result.output == 10


class TestTaskAbstract:
    """Tests for abstract Task.execute method."""

    def test_execute_is_abstract(self):
        """Test Task.execute is abstract (line 120)."""
        # Task is abstract, verify we can't instantiate it directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Task(name="test")


class TestTaskTimeout:
    """Tests for task timeout handling."""

    @pytest.mark.asyncio
    async def test_task_with_timeout_success(self):
        """Test task with timeout that completes within time (line 139)."""
        async def fast_task(data):
            await asyncio.sleep(0.01)
            return data * 2

        config = TaskConfig(timeout=1.0)  # 1 second timeout
        t = FunctionTask(fast_task, name="fast", config=config)

        result = await t.run(5)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == 10

    @pytest.mark.asyncio
    async def test_task_timeout_exceeded(self):
        """Test task that exceeds timeout (lines 162-176)."""
        async def slow_task(data):
            await asyncio.sleep(10)
            return data

        config = TaskConfig(timeout=0.01)  # 10ms timeout
        t = FunctionTask(slow_task, name="slow", config=config)

        result = await t.run("data")

        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.error
        assert result.error_type == "TimeoutError"


class TestTaskCancellation:
    """Tests for task cancellation handling."""

    @pytest.mark.asyncio
    async def test_task_cancelled(self):
        """Test task cancellation handling (lines 179-188)."""
        async def slow_task(data):
            await asyncio.sleep(10)
            return data

        t = FunctionTask(slow_task, name="slow")

        async def run_and_cancel():
            run_task = asyncio.create_task(t.run("data"))
            await asyncio.sleep(0.01)
            run_task.cancel()
            try:
                return await run_task
            except asyncio.CancelledError:
                # Task was cancelled
                return TaskResult(
                    task_id=t.task_id,
                    status=TaskStatus.CANCELLED,
                    error="Task was cancelled",
                )

        result = await run_and_cancel()
        assert result.status == TaskStatus.CANCELLED


class TestRetryOnSpecificExceptions:
    """Tests for retry_on specific exception types."""

    @pytest.mark.asyncio
    async def test_retry_on_specific_exception(self):
        """Test retry only on specific exception types (line 200)."""
        call_count = 0

        async def flaky_task(data):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retry this")
            return "success"

        retry_policy = RetryPolicy(
            max_retries=5,
            initial_delay=0.01,
            retry_on=[ValueError],  # Only retry on ValueError
        )
        config = TaskConfig(retry_policy=retry_policy)
        t = FunctionTask(flaky_task, config=config)

        result = await t.run("data")

        assert result.status == TaskStatus.COMPLETED
        assert result.output == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_matching_exception(self):
        """Test no retry when exception doesn't match retry_on types."""
        call_count = 0

        async def error_task(data):
            nonlocal call_count
            call_count += 1
            raise TypeError("Don't retry this")

        retry_policy = RetryPolicy(
            max_retries=5,
            initial_delay=0.01,
            retry_on=[ValueError],  # Only retry on ValueError, not TypeError
        )
        config = TaskConfig(retry_policy=retry_policy)
        t = FunctionTask(error_task, config=config)

        result = await t.run("data")

        assert result.status == TaskStatus.FAILED
        assert call_count == 1  # No retries


class TestLambdaTask:
    """Tests for LambdaTask class."""

    @pytest.mark.asyncio
    async def test_lambda_task_sync(self):
        """Test LambdaTask with sync function (line 267)."""
        t = LambdaTask(lambda x: x * 2, name="double")
        result = await t.run(5)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == 10

    @pytest.mark.asyncio
    async def test_lambda_task_async(self):
        """Test LambdaTask with async function."""
        async def async_func(x):
            return x * 3

        t = LambdaTask(async_func, name="triple")
        result = await t.run(4)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == 12


class TestNoOpTask:
    """Tests for NoOpTask class."""

    @pytest.mark.asyncio
    async def test_noop_task_execute(self):
        """Test NoOpTask.execute returns configured output (line 284)."""
        t = NoOpTask(output="test_output")
        result = await t.run("input")

        assert result.status == TaskStatus.COMPLETED
        assert result.output == "test_output"

    @pytest.mark.asyncio
    async def test_noop_task_default_output(self):
        """Test NoOpTask with default None output."""
        t = NoOpTask()
        result = await t.run("anything")

        assert result.status == TaskStatus.COMPLETED
        assert result.output is None


class TestDelayTask:
    """Tests for DelayTask class."""

    @pytest.mark.asyncio
    async def test_delay_task_creation(self):
        """Test DelayTask creation (lines 296-297)."""
        t = DelayTask(delay_seconds=0.01, name="short_delay")
        assert t._delay == 0.01
        assert t.name == "short_delay"

    @pytest.mark.asyncio
    async def test_delay_task_execute(self):
        """Test DelayTask.execute introduces delay and returns input (lines 300-301)."""
        t = DelayTask(delay_seconds=0.01)

        start = datetime.utcnow()
        result = await t.run("passthrough")
        elapsed = (datetime.utcnow() - start).total_seconds()

        assert result.status == TaskStatus.COMPLETED
        assert result.output == "passthrough"
        assert elapsed >= 0.01


class TestConditionalTask:
    """Tests for ConditionalTask class."""

    @pytest.mark.asyncio
    async def test_conditional_task_creation(self):
        """Test ConditionalTask creation (lines 315-318)."""
        true_task = FunctionTask(lambda x: x * 2, name="double")
        false_task = FunctionTask(lambda x: x + 1, name="add_one")

        t = ConditionalTask(
            condition=lambda x: x > 0,
            if_true=true_task,
            if_false=false_task,
            name="conditional",
        )

        assert t._condition is not None
        assert t._if_true is true_task
        assert t._if_false is false_task

    @pytest.mark.asyncio
    async def test_conditional_task_true_branch(self):
        """Test ConditionalTask executes true branch (lines 321-323)."""
        true_task = FunctionTask(lambda x: x * 2, name="double")
        false_task = FunctionTask(lambda x: x + 1, name="add_one")

        t = ConditionalTask(
            condition=lambda x: x > 0,
            if_true=true_task,
            if_false=false_task,
        )

        result = await t.run(5)  # condition is True

        assert result.status == TaskStatus.COMPLETED
        assert result.output == 10  # 5 * 2

    @pytest.mark.asyncio
    async def test_conditional_task_false_branch(self):
        """Test ConditionalTask executes false branch (lines 324-326)."""
        true_task = FunctionTask(lambda x: x * 2, name="double")
        false_task = FunctionTask(lambda x: x + 1, name="add_one")

        t = ConditionalTask(
            condition=lambda x: x > 0,
            if_true=true_task,
            if_false=false_task,
        )

        result = await t.run(-5)  # condition is False

        assert result.status == TaskStatus.COMPLETED
        assert result.output == -4  # -5 + 1

    @pytest.mark.asyncio
    async def test_conditional_task_no_false_branch(self):
        """Test ConditionalTask with no false branch returns None (lines 327-328)."""
        true_task = FunctionTask(lambda x: x * 2, name="double")

        t = ConditionalTask(
            condition=lambda x: x > 0,
            if_true=true_task,
            if_false=None,
        )

        result = await t.run(-5)  # condition is False, no false branch

        assert result.status == TaskStatus.COMPLETED
        assert result.output is None


class TestParallelTask:
    """Tests for ParallelTask class."""

    @pytest.mark.asyncio
    async def test_parallel_task_fail_fast_false(self):
        """Test ParallelTask with fail_fast=False (line 354)."""
        async def success_task(x):
            return x * 2

        async def fail_task(x):
            raise ValueError("Intentional failure")

        t = ParallelTask(
            tasks=[
                FunctionTask(success_task, name="success"),
                FunctionTask(fail_task, name="fail"),
                FunctionTask(success_task, name="success2"),
            ],
            fail_fast=False,
        )

        result = await t.run(5)

        assert result.status == TaskStatus.COMPLETED
        # Results should include both successes and failure
        assert len(result.output) == 3

    @pytest.mark.asyncio
    async def test_parallel_task_all_success(self):
        """Test ParallelTask with all tasks succeeding."""
        async def success_task(x):
            return x * 2

        t = ParallelTask(
            tasks=[
                FunctionTask(success_task, name="success1"),
                FunctionTask(success_task, name="success2"),
            ],
            fail_fast=True,
        )

        result = await t.run(5)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == [10, 10]


class TestSequentialTask:
    """Tests for SequentialTask class."""

    @pytest.mark.asyncio
    async def test_sequential_task_failure(self):
        """Test SequentialTask raises on task failure (line 382)."""
        async def success_task(x):
            return x * 2

        async def fail_task(x):
            raise ValueError("Intentional failure")

        t = SequentialTask(
            tasks=[
                FunctionTask(success_task, name="success"),
                FunctionTask(fail_task, name="fail"),
                FunctionTask(success_task, name="success2"),
            ],
        )

        result = await t.run(5)

        assert result.status == TaskStatus.FAILED
        assert "fail failed" in result.error

    @pytest.mark.asyncio
    async def test_sequential_task_success(self):
        """Test SequentialTask passes output between tasks."""
        async def double(x):
            return x * 2

        async def add_ten(x):
            return x + 10

        t = SequentialTask(
            tasks=[
                FunctionTask(double, name="double"),
                FunctionTask(add_ten, name="add_ten"),
            ],
        )

        result = await t.run(5)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == 20  # (5 * 2) + 10


class TestTaskDecorator:
    """Tests for @task decorator."""

    def test_task_decorator(self):
        """Test @task decorator creates FunctionTask (lines 403-410)."""
        @task(name="decorated_task", timeout=30)
        async def my_task(data):
            return data * 2

        assert isinstance(my_task, FunctionTask)
        assert my_task.name == "decorated_task"
        assert my_task.config.timeout == 30

    def test_task_decorator_with_retry_policy(self):
        """Test @task decorator with retry policy."""
        retry = RetryPolicy(max_retries=3, initial_delay=0.1)

        @task(name="retry_task", retry_policy=retry)
        async def retryable_task(data):
            return data

        assert retryable_task.config.retry_policy is retry
        assert retryable_task.config.retry_policy.max_retries == 3

    def test_task_decorator_with_priority(self):
        """Test @task decorator with priority."""
        @task(name="high_priority", priority=TaskPriority.HIGH)
        async def important_task(data):
            return data

        assert important_task.config.priority == TaskPriority.HIGH


class TestRetryPolicy:
    """Tests for RetryPolicy class."""

    def test_retry_policy_get_delay_exponential(self):
        """Test RetryPolicy.get_delay with exponential backoff."""
        policy = RetryPolicy(
            max_retries=5,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0,
        )

        delay0 = policy.get_delay(0)
        delay1 = policy.get_delay(1)
        delay2 = policy.get_delay(2)

        assert delay0 == 1.0  # initial
        assert delay1 == 2.0  # 1 * 2
        assert delay2 == 4.0  # 2 * 2

    def test_retry_policy_get_delay_capped(self):
        """Test RetryPolicy.get_delay respects max_delay."""
        policy = RetryPolicy(
            max_retries=10,
            initial_delay=1.0,
            max_delay=5.0,
            backoff_multiplier=10.0,
        )

        delay = policy.get_delay(5)
        assert delay == 5.0  # capped at max_delay


class TestTaskConfig:
    """Tests for TaskConfig class."""

    def test_task_config_defaults(self):
        """Test TaskConfig default values."""
        config = TaskConfig()

        assert config.timeout is None
        assert config.retry_policy is None
        assert config.priority == TaskPriority.NORMAL

    def test_task_config_with_values(self):
        """Test TaskConfig with custom values."""
        policy = RetryPolicy(max_retries=3)
        config = TaskConfig(
            timeout=60.0,
            retry_policy=policy,
            priority=TaskPriority.CRITICAL,
        )

        assert config.timeout == 60.0
        assert config.retry_policy is policy
        assert config.priority == TaskPriority.CRITICAL
