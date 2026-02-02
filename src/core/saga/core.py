"""Saga Pattern Core.

Provides distributed transaction management:
- Saga step definition
- Compensation handling
- State tracking
"""

from __future__ import annotations

import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class SagaState(Enum):
    """State of a saga execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


class StepState(Enum):
    """State of a saga step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of a step execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class StepExecution:
    """Execution record for a step."""
    step_name: str
    state: StepState = StepState.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[StepResult] = None
    compensation_result: Optional[StepResult] = None
    attempts: int = 0

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


class SagaStep(ABC):
    """Abstract saga step."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Step name."""
        pass

    @abstractmethod
    async def execute(self, context: "SagaContext") -> StepResult:
        """Execute the step."""
        pass

    @abstractmethod
    async def compensate(self, context: "SagaContext") -> StepResult:
        """Compensate (undo) the step."""
        pass

    def should_compensate(self, context: "SagaContext") -> bool:
        """Check if compensation should run."""
        execution = context.get_execution(self.name)
        return execution is not None and execution.state == StepState.COMPLETED


class FunctionStep(SagaStep):
    """Saga step from functions."""

    def __init__(
        self,
        step_name: str,
        execute_fn: Callable[["SagaContext"], Any],
        compensate_fn: Callable[["SagaContext"], Any],
    ):
        self._name = step_name
        self._execute_fn = execute_fn
        self._compensate_fn = compensate_fn

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, context: "SagaContext") -> StepResult:
        import asyncio

        start = time.time()
        try:
            if asyncio.iscoroutinefunction(self._execute_fn):
                data = await self._execute_fn(context)
            else:
                data = self._execute_fn(context)

            return StepResult(
                success=True,
                data=data,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StepResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def compensate(self, context: "SagaContext") -> StepResult:
        import asyncio

        start = time.time()
        try:
            if asyncio.iscoroutinefunction(self._compensate_fn):
                data = await self._compensate_fn(context)
            else:
                data = self._compensate_fn(context)

            return StepResult(
                success=True,
                data=data,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return StepResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


@dataclass
class SagaContext:
    """Context for saga execution."""
    saga_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    executions: Dict[str, StepExecution] = field(default_factory=dict)
    state: SagaState = SagaState.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get data from context."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set data in context."""
        self.data[key] = value

    def get_step_result(self, step_name: str) -> Optional[Any]:
        """Get result data from a step."""
        execution = self.executions.get(step_name)
        if execution and execution.result:
            return execution.result.data
        return None

    def get_execution(self, step_name: str) -> Optional[StepExecution]:
        """Get step execution record."""
        return self.executions.get(step_name)

    def record_execution(
        self,
        step_name: str,
        state: StepState,
        result: Optional[StepResult] = None,
    ) -> StepExecution:
        """Record step execution."""
        if step_name not in self.executions:
            self.executions[step_name] = StepExecution(step_name=step_name)

        execution = self.executions[step_name]
        execution.state = state
        execution.attempts += 1

        if state == StepState.RUNNING:
            execution.started_at = datetime.utcnow()
        elif state in (StepState.COMPLETED, StepState.FAILED):
            execution.completed_at = datetime.utcnow()
            execution.result = result
        elif state == StepState.COMPENSATED:
            execution.compensation_result = result

        return execution

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "saga_id": self.saga_id,
            "state": self.state.value,
            "data": self.data,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "executions": {
                name: {
                    "state": ex.state.value,
                    "attempts": ex.attempts,
                    "duration_ms": ex.duration_ms,
                }
                for name, ex in self.executions.items()
            },
            "metadata": self.metadata,
        }


@dataclass
class SagaDefinition:
    """Definition of a saga."""
    name: str
    steps: List[SagaStep] = field(default_factory=list)
    timeout_seconds: float = 300.0
    max_compensation_retries: int = 3

    def add_step(self, step: SagaStep) -> "SagaDefinition":
        """Add a step to the saga."""
        self.steps.append(step)
        return self

    def step(
        self,
        name: str,
        execute: Callable[[SagaContext], Any],
        compensate: Callable[[SagaContext], Any],
    ) -> "SagaDefinition":
        """Add a function-based step."""
        self.steps.append(FunctionStep(name, execute, compensate))
        return self


def create_saga_id() -> str:
    """Generate unique saga ID."""
    timestamp = int(time.time() * 1000)
    random_part = secrets.token_hex(4)
    return f"saga_{timestamp}_{random_part}"


class SagaBuilder:
    """Builder for saga definitions."""

    def __init__(self, name: str):
        self._definition = SagaDefinition(name=name)

    def step(
        self,
        name: str,
        execute: Callable[[SagaContext], Any],
        compensate: Callable[[SagaContext], Any],
    ) -> "SagaBuilder":
        """Add a step."""
        self._definition.step(name, execute, compensate)
        return self

    def with_timeout(self, seconds: float) -> "SagaBuilder":
        """Set saga timeout."""
        self._definition.timeout_seconds = seconds
        return self

    def with_compensation_retries(self, retries: int) -> "SagaBuilder":
        """Set compensation retry count."""
        self._definition.max_compensation_retries = retries
        return self

    def build(self) -> SagaDefinition:
        """Build the saga definition."""
        return self._definition
