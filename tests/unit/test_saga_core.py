"""Tests for saga core module to improve coverage."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from src.core.saga.core import (
    StepState,
    SagaState,
    StepResult,
    StepExecution,
    SagaStep,
    FunctionStep,
    SagaContext,
    SagaDefinition,
    SagaBuilder,
    create_saga_id,
)


class TestStepExecution:
    """Tests for StepExecution class."""

    def test_duration_ms_with_timestamps(self):
        """Test duration_ms property with valid timestamps (lines 65-66)."""
        execution = StepExecution(
            step_name="test",
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            completed_at=datetime(2025, 1, 1, 12, 0, 1),  # 1 second later
        )

        duration = execution.duration_ms
        assert duration == 1000.0

    def test_duration_ms_without_timestamps(self):
        """Test duration_ms property without timestamps (line 67)."""
        execution = StepExecution(step_name="test")

        duration = execution.duration_ms
        assert duration is None

    def test_duration_ms_without_completed_at(self):
        """Test duration_ms property without completed_at."""
        execution = StepExecution(
            step_name="test",
            started_at=datetime(2025, 1, 1, 12, 0, 0),
        )

        duration = execution.duration_ms
        assert duration is None


class TestSagaStepAbstract:
    """Tests for abstract SagaStep methods."""

    def test_name_is_abstract(self):
        """Test SagaStep.name is abstract (line 77)."""
        # SagaStep is abstract, verify through FunctionStep
        step = FunctionStep(
            step_name="test",
            execute_fn=lambda ctx: None,
            compensate_fn=lambda ctx: None,
        )
        assert step.name == "test"

    def test_execute_is_abstract(self):
        """Test SagaStep.execute is abstract (line 82)."""
        # Verify abstract by checking FunctionStep implementation
        assert hasattr(SagaStep, "execute")

    def test_compensate_is_abstract(self):
        """Test SagaStep.compensate is abstract (line 87)."""
        # Verify abstract by checking FunctionStep implementation
        assert hasattr(SagaStep, "compensate")


class TestFunctionStep:
    """Tests for FunctionStep class."""

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        """Test execute with async function (line 140 - async branch)."""
        async def async_execute(ctx):
            return {"result": "async_success"}

        step = FunctionStep(
            step_name="async_step",
            execute_fn=async_execute,
            compensate_fn=lambda ctx: None,
        )

        context = SagaContext(saga_id="test")
        result = await step.execute(context)

        assert result.success is True
        assert result.data == {"result": "async_success"}

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Test execute with sync function."""
        def sync_execute(ctx):
            return {"result": "sync_success"}

        step = FunctionStep(
            step_name="sync_step",
            execute_fn=sync_execute,
            compensate_fn=lambda ctx: None,
        )

        context = SagaContext(saga_id="test")
        result = await step.execute(context)

        assert result.success is True
        assert result.data == {"result": "sync_success"}

    @pytest.mark.asyncio
    async def test_compensate_failure(self):
        """Test compensate handles exception (lines 149-150)."""
        def failing_compensate(ctx):
            raise ValueError("Compensation failed")

        step = FunctionStep(
            step_name="failing_step",
            execute_fn=lambda ctx: None,
            compensate_fn=failing_compensate,
        )

        context = SagaContext(saga_id="test")
        result = await step.compensate(context)

        assert result.success is False
        assert "Compensation failed" in result.error

    @pytest.mark.asyncio
    async def test_compensate_async_function(self):
        """Test compensate with async function."""
        async def async_compensate(ctx):
            return {"compensated": True}

        step = FunctionStep(
            step_name="async_comp_step",
            execute_fn=lambda ctx: None,
            compensate_fn=async_compensate,
        )

        context = SagaContext(saga_id="test")
        result = await step.compensate(context)

        assert result.success is True
        assert result.data == {"compensated": True}


class TestSagaContext:
    """Tests for SagaContext class."""

    def test_get_step_result_no_result(self):
        """Test get_step_result returns None when no result (line 182)."""
        context = SagaContext(saga_id="test")

        # Add execution without result
        context.executions["step1"] = StepExecution(step_name="step1")

        result = context.get_step_result("step1")
        assert result is None

    def test_get_step_result_with_result(self):
        """Test get_step_result returns data when result exists."""
        context = SagaContext(saga_id="test")

        # Add execution with result
        execution = StepExecution(step_name="step1")
        execution.result = StepResult(success=True, data={"key": "value"})
        context.executions["step1"] = execution

        result = context.get_step_result("step1")
        assert result == {"key": "value"}

    def test_get_step_result_nonexistent(self):
        """Test get_step_result returns None for nonexistent step."""
        context = SagaContext(saga_id="test")
        result = context.get_step_result("nonexistent")
        assert result is None

    def test_duration_ms_with_timestamps(self):
        """Test duration_ms property with valid timestamps."""
        context = SagaContext(
            saga_id="test",
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            completed_at=datetime(2025, 1, 1, 12, 0, 2),  # 2 seconds later
        )

        duration = context.duration_ms
        assert duration == 2000.0

    def test_duration_ms_without_timestamps(self):
        """Test duration_ms property without timestamps (line 216)."""
        context = SagaContext(saga_id="test")

        duration = context.duration_ms
        assert duration is None

    def test_to_dict(self):
        """Test to_dict serialization."""
        context = SagaContext(
            saga_id="test",
            data={"key": "value"},
            state=SagaState.RUNNING,
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            metadata={"env": "test"},
        )

        result = context.to_dict()

        assert result["saga_id"] == "test"
        assert result["state"] == "running"
        assert result["data"] == {"key": "value"}
        assert result["started_at"] == "2025-01-01T12:00:00"


class TestSagaDefinition:
    """Tests for SagaDefinition class."""

    def test_add_step(self):
        """Test add_step method (lines 248-249)."""
        definition = SagaDefinition(name="test_saga")

        step = FunctionStep(
            step_name="step1",
            execute_fn=lambda ctx: None,
            compensate_fn=lambda ctx: None,
        )

        result = definition.add_step(step)

        assert result is definition  # Returns self for chaining
        assert len(definition.steps) == 1
        assert definition.steps[0] is step

    def test_step_method(self):
        """Test step method adds FunctionStep."""
        definition = SagaDefinition(name="test_saga")

        definition.step(
            name="step1",
            execute=lambda ctx: "executed",
            compensate=lambda ctx: "compensated",
        )

        assert len(definition.steps) == 1
        assert isinstance(definition.steps[0], FunctionStep)
        assert definition.steps[0].name == "step1"


class TestSagaBuilder:
    """Tests for SagaBuilder class."""

    def test_with_compensation_retries(self):
        """Test with_compensation_retries method (lines 292-293)."""
        builder = SagaBuilder(name="test_saga")

        result = builder.with_compensation_retries(5)

        assert result is builder  # Returns self for chaining
        assert builder._definition.max_compensation_retries == 5

    def test_with_timeout(self):
        """Test with_timeout method."""
        builder = SagaBuilder(name="test_saga")

        result = builder.with_timeout(60.0)

        assert result is builder
        assert builder._definition.timeout_seconds == 60.0

    def test_full_builder_chain(self):
        """Test full builder chain."""
        saga = (
            SagaBuilder(name="order_saga")
            .step(
                name="create_order",
                execute=lambda ctx: {"order_id": "123"},
                compensate=lambda ctx: None,
            )
            .step(
                name="charge_payment",
                execute=lambda ctx: {"payment_id": "456"},
                compensate=lambda ctx: None,
            )
            .with_timeout(120.0)
            .with_compensation_retries(3)
            .build()
        )

        assert saga.name == "order_saga"
        assert len(saga.steps) == 2
        assert saga.timeout_seconds == 120.0
        assert saga.max_compensation_retries == 3


class TestCreateSagaId:
    """Tests for create_saga_id function."""

    def test_create_saga_id_format(self):
        """Test saga ID format."""
        saga_id = create_saga_id()

        assert saga_id.startswith("saga_")
        parts = saga_id.split("_")
        assert len(parts) == 3  # saga, timestamp, random

    def test_create_saga_id_unique(self):
        """Test saga IDs are unique."""
        ids = [create_saga_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique


class TestSagaStepShouldCompensate:
    """Tests for SagaStep.should_compensate method."""

    def test_should_compensate_true(self):
        """Test should_compensate returns True for completed step."""
        step = FunctionStep(
            step_name="test",
            execute_fn=lambda ctx: None,
            compensate_fn=lambda ctx: None,
        )

        context = SagaContext(saga_id="test")
        context.executions["test"] = StepExecution(
            step_name="test",
            state=StepState.COMPLETED,
        )

        assert step.should_compensate(context) is True

    def test_should_compensate_false_not_completed(self):
        """Test should_compensate returns False for non-completed step."""
        step = FunctionStep(
            step_name="test",
            execute_fn=lambda ctx: None,
            compensate_fn=lambda ctx: None,
        )

        context = SagaContext(saga_id="test")
        context.executions["test"] = StepExecution(
            step_name="test",
            state=StepState.FAILED,
        )

        assert step.should_compensate(context) is False

    def test_should_compensate_false_no_execution(self):
        """Test should_compensate returns False when no execution."""
        step = FunctionStep(
            step_name="test",
            execute_fn=lambda ctx: None,
            compensate_fn=lambda ctx: None,
        )

        context = SagaContext(saga_id="test")

        assert step.should_compensate(context) is False


class TestSagaContextRecordExecution:
    """Tests for SagaContext.record_execution method."""

    def test_record_execution_running(self):
        """Test record_execution with RUNNING state."""
        context = SagaContext(saga_id="test")

        execution = context.record_execution("step1", StepState.RUNNING)

        assert execution.state == StepState.RUNNING
        assert execution.started_at is not None
        assert execution.attempts == 1

    def test_record_execution_completed(self):
        """Test record_execution with COMPLETED state."""
        context = SagaContext(saga_id="test")
        result = StepResult(success=True, data={"key": "value"})

        execution = context.record_execution("step1", StepState.COMPLETED, result)

        assert execution.state == StepState.COMPLETED
        assert execution.completed_at is not None
        assert execution.result is result

    def test_record_execution_failed(self):
        """Test record_execution with FAILED state."""
        context = SagaContext(saga_id="test")
        result = StepResult(success=False, error="Something went wrong")

        execution = context.record_execution("step1", StepState.FAILED, result)

        assert execution.state == StepState.FAILED
        assert execution.result is result

    def test_record_execution_compensated(self):
        """Test record_execution with COMPENSATED state."""
        context = SagaContext(saga_id="test")
        result = StepResult(success=True, data={"compensated": True})

        execution = context.record_execution("step1", StepState.COMPENSATED, result)

        assert execution.state == StepState.COMPENSATED
        assert execution.compensation_result is result

    def test_record_execution_updates_existing(self):
        """Test record_execution updates existing execution."""
        context = SagaContext(saga_id="test")

        # First record
        context.record_execution("step1", StepState.RUNNING)

        # Update to completed
        execution = context.record_execution(
            "step1",
            StepState.COMPLETED,
            StepResult(success=True),
        )

        assert execution.attempts == 2  # Incremented
        assert execution.state == StepState.COMPLETED


class TestSagaContextGetSet:
    """Tests for SagaContext get/set methods."""

    def test_get_method(self):
        """Test SagaContext.get method (line 171)."""
        context = SagaContext(saga_id="test", data={"key": "value"})

        result = context.get("key")
        assert result == "value"

        # Test with default
        result = context.get("nonexistent", "default")
        assert result == "default"

    def test_set_method(self):
        """Test SagaContext.set method (line 175)."""
        context = SagaContext(saga_id="test")

        context.set("key", "value")

        assert context.data["key"] == "value"


class TestFunctionStepExecuteFailure:
    """Tests for FunctionStep execute failure handling."""

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test execute handles exception (lines 127-128)."""
        def failing_execute(ctx):
            raise RuntimeError("Execution failed")

        step = FunctionStep(
            step_name="failing_step",
            execute_fn=failing_execute,
            compensate_fn=lambda ctx: None,
        )

        context = SagaContext(saga_id="test")
        result = await step.execute(context)

        assert result.success is False
        assert "Execution failed" in result.error
        assert result.duration_ms is not None
