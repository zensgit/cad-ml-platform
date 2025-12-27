"""
Saga Pattern Module for Vision Provider System.

Implements distributed transaction patterns with compensation operations,
saga orchestration, and transaction management for vision operations.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from .base import VisionDescription, VisionProvider

T = TypeVar("T")


class SagaStatus(Enum):
    """Status of a saga execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


class StepStatus(Enum):
    """Status of a saga step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SagaStep:
    """A single step in a saga."""

    name: str
    action: Callable[..., Any]
    compensation: Optional[Callable[..., Any]] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: float = 30.0
    retries: int = 0
    max_retries: int = 3


@dataclass
class SagaContext:
    """Context passed between saga steps."""

    data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self.data.get(key, default)

    def get_step_result(self, step_name: str) -> Optional[Any]:
        """Get the result of a specific step."""
        return self.step_results.get(step_name)


@dataclass
class SagaExecution:
    """Record of a saga execution."""

    saga_id: str
    name: str
    status: SagaStatus
    steps: List[SagaStep]
    context: SagaContext
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    compensation_triggered: bool = False


class SagaDefinition:
    """Defines the structure of a saga."""

    def __init__(self, name: str):
        self.name = name
        self.steps: List[SagaStep] = []

    def add_step(
        self,
        name: str,
        action: Callable[..., Any],
        compensation: Optional[Callable[..., Any]] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> "SagaDefinition":
        """Add a step to the saga."""
        step = SagaStep(
            name=name,
            action=action,
            compensation=compensation,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.steps.append(step)
        return self

    def get_steps(self) -> List[SagaStep]:
        """Get all steps."""
        return self.steps.copy()


class SagaOrchestrator:
    """Orchestrates saga execution."""

    def __init__(self):
        self._executions: Dict[str, SagaExecution] = {}
        self._saga_definitions: Dict[str, SagaDefinition] = {}

    def register_saga(self, definition: SagaDefinition) -> None:
        """Register a saga definition."""
        self._saga_definitions[definition.name] = definition

    def get_saga(self, name: str) -> Optional[SagaDefinition]:
        """Get a registered saga definition."""
        return self._saga_definitions.get(name)

    async def execute(
        self,
        saga_name: str,
        context: Optional[SagaContext] = None,
    ) -> SagaExecution:
        """Execute a saga by name."""
        definition = self._saga_definitions.get(saga_name)
        if not definition:
            raise ValueError(f"Saga '{saga_name}' not found")

        saga_id = str(uuid.uuid4())
        ctx = context or SagaContext()

        # Create fresh steps from definition
        steps = [
            SagaStep(
                name=s.name,
                action=s.action,
                compensation=s.compensation,
                timeout=s.timeout,
                max_retries=s.max_retries,
            )
            for s in definition.steps
        ]

        execution = SagaExecution(
            saga_id=saga_id,
            name=saga_name,
            status=SagaStatus.RUNNING,
            steps=steps,
            context=ctx,
            started_at=datetime.utcnow(),
        )
        self._executions[saga_id] = execution

        try:
            await self._run_steps(execution)
            execution.status = SagaStatus.COMPLETED
        except Exception as e:
            execution.error = str(e)
            execution.status = SagaStatus.COMPENSATING
            execution.compensation_triggered = True
            await self._compensate(execution)
            if execution.status == SagaStatus.COMPENSATING:
                execution.status = SagaStatus.COMPENSATED

        execution.completed_at = datetime.utcnow()
        return execution

    async def _run_steps(self, execution: SagaExecution) -> None:
        """Run saga steps in sequence."""
        for step in execution.steps:
            step.status = StepStatus.RUNNING
            step.started_at = datetime.utcnow()

            try:
                # Execute with timeout and retries
                result = await self._execute_step(step, execution.context)
                step.result = result
                step.status = StepStatus.COMPLETED
                execution.context.step_results[step.name] = result
            except Exception as e:
                step.error = str(e)
                step.status = StepStatus.FAILED
                raise

            step.completed_at = datetime.utcnow()

    async def _execute_step(self, step: SagaStep, context: SagaContext) -> Any:
        """Execute a single step with retry logic."""
        last_error: Optional[Exception] = None

        for attempt in range(step.max_retries + 1):
            try:
                step.retries = attempt
                if asyncio.iscoroutinefunction(step.action):
                    result = await asyncio.wait_for(step.action(context), timeout=step.timeout)
                else:
                    result = step.action(context)
                return result
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Step '{step.name}' timed out after {step.timeout}s")
            except Exception as e:
                last_error = e

            if attempt < step.max_retries:
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff

        raise last_error or Exception(f"Step '{step.name}' failed")

    async def _compensate(self, execution: SagaExecution) -> None:
        """Run compensation for completed steps in reverse order."""
        completed_steps = [s for s in execution.steps if s.status == StepStatus.COMPLETED]

        for step in reversed(completed_steps):
            if step.compensation:
                step.status = StepStatus.COMPENSATING
                try:
                    if asyncio.iscoroutinefunction(step.compensation):
                        await step.compensation(execution.context)
                    else:
                        step.compensation(execution.context)
                    step.status = StepStatus.COMPENSATED
                except Exception as e:
                    step.error = f"Compensation failed: {e}"
                    step.status = StepStatus.FAILED
                    execution.status = SagaStatus.FAILED

    def get_execution(self, saga_id: str) -> Optional[SagaExecution]:
        """Get a saga execution by ID."""
        return self._executions.get(saga_id)

    def list_executions(self, status: Optional[SagaStatus] = None) -> List[SagaExecution]:
        """List all executions, optionally filtered by status."""
        executions = list(self._executions.values())
        if status:
            executions = [e for e in executions if e.status == status]
        return executions


class CompensationManager:
    """Manages compensation operations for distributed transactions."""

    def __init__(self):
        self._compensations: Dict[str, List[Callable[..., Any]]] = {}
        self._executed: Dict[str, List[Callable[..., Any]]] = {}

    def register(self, transaction_id: str, compensation: Callable[..., Any]) -> None:
        """Register a compensation operation."""
        if transaction_id not in self._compensations:
            self._compensations[transaction_id] = []
        self._compensations[transaction_id].append(compensation)

    async def compensate(self, transaction_id: str) -> List[str]:
        """Execute all compensations for a transaction."""
        errors: List[str] = []
        compensations = self._compensations.get(transaction_id, [])
        executed: List[Callable[..., Any]] = []

        for comp in reversed(compensations):
            try:
                if asyncio.iscoroutinefunction(comp):
                    await comp()
                else:
                    comp()
                executed.append(comp)
            except Exception as e:
                errors.append(str(e))

        self._executed[transaction_id] = executed
        return errors

    def clear(self, transaction_id: str) -> None:
        """Clear compensations for a completed transaction."""
        self._compensations.pop(transaction_id, None)
        self._executed.pop(transaction_id, None)

    def get_pending(self, transaction_id: str) -> int:
        """Get count of pending compensations."""
        return len(self._compensations.get(transaction_id, []))


class TransactionCoordinator:
    """Coordinates distributed transactions using two-phase commit."""

    def __init__(self):
        self._participants: Dict[str, List["TransactionParticipant"]] = {}
        self._transaction_states: Dict[str, str] = {}

    def begin_transaction(self) -> str:
        """Begin a new distributed transaction."""
        tx_id = str(uuid.uuid4())
        self._participants[tx_id] = []
        self._transaction_states[tx_id] = "active"
        return tx_id

    def register_participant(self, tx_id: str, participant: "TransactionParticipant") -> None:
        """Register a participant in the transaction."""
        if tx_id not in self._participants:
            raise ValueError(f"Transaction {tx_id} not found")
        self._participants[tx_id].append(participant)

    async def prepare(self, tx_id: str) -> bool:
        """Phase 1: Prepare all participants."""
        if tx_id not in self._participants:
            return False

        participants = self._participants[tx_id]
        self._transaction_states[tx_id] = "preparing"

        for participant in participants:
            try:
                if not await participant.prepare(tx_id):
                    self._transaction_states[tx_id] = "aborting"
                    return False
            except Exception:
                self._transaction_states[tx_id] = "aborting"
                return False

        self._transaction_states[tx_id] = "prepared"
        return True

    async def commit(self, tx_id: str) -> bool:
        """Phase 2: Commit all participants."""
        if self._transaction_states.get(tx_id) != "prepared":
            return False

        participants = self._participants[tx_id]
        self._transaction_states[tx_id] = "committing"

        for participant in participants:
            try:
                await participant.commit(tx_id)
            except Exception:
                # In real 2PC, this would require recovery
                self._transaction_states[tx_id] = "failed"
                return False

        self._transaction_states[tx_id] = "committed"
        return True

    async def rollback(self, tx_id: str) -> bool:
        """Rollback all participants."""
        participants = self._participants.get(tx_id, [])
        self._transaction_states[tx_id] = "rolling_back"

        for participant in participants:
            try:
                await participant.rollback(tx_id)
            except Exception:
                pass  # Best effort rollback

        self._transaction_states[tx_id] = "rolled_back"
        return True

    def get_state(self, tx_id: str) -> Optional[str]:
        """Get the state of a transaction."""
        return self._transaction_states.get(tx_id)


class TransactionParticipant(ABC):
    """Abstract base class for transaction participants."""

    @abstractmethod
    async def prepare(self, tx_id: str) -> bool:
        """Prepare for commit (Phase 1)."""
        pass

    @abstractmethod
    async def commit(self, tx_id: str) -> None:
        """Commit the transaction (Phase 2)."""
        pass

    @abstractmethod
    async def rollback(self, tx_id: str) -> None:
        """Rollback the transaction."""
        pass


class SimpleParticipant(TransactionParticipant):
    """Simple transaction participant implementation."""

    def __init__(self, name: str):
        self.name = name
        self._prepared: Dict[str, Any] = {}
        self._committed: Dict[str, bool] = {}

    async def prepare(self, tx_id: str) -> bool:
        """Prepare for commit."""
        self._prepared[tx_id] = {"ready": True, "timestamp": datetime.utcnow()}
        return True

    async def commit(self, tx_id: str) -> None:
        """Commit the transaction."""
        self._committed[tx_id] = True
        self._prepared.pop(tx_id, None)

    async def rollback(self, tx_id: str) -> None:
        """Rollback the transaction."""
        self._prepared.pop(tx_id, None)
        self._committed.pop(tx_id, None)

    def is_committed(self, tx_id: str) -> bool:
        """Check if transaction was committed."""
        return self._committed.get(tx_id, False)


class ChoreographySaga:
    """Event-driven saga using choreography pattern."""

    def __init__(self, name: str):
        self.name = name
        self._event_handlers: Dict[str, List[Callable[..., Any]]] = {}
        self._compensation_handlers: Dict[str, Callable[..., Any]] = {}
        self._event_history: List[Dict[str, Any]] = []

    def on_event(
        self,
        event_type: str,
        handler: Callable[..., Any],
        compensation: Optional[Callable[..., Any]] = None,
    ) -> "ChoreographySaga":
        """Register handler for an event type."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

        if compensation:
            self._compensation_handlers[event_type] = compensation

        return self

    async def emit(self, event_type: str, data: Any = None) -> List[Any]:
        """Emit an event and trigger handlers."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow(),
        }
        self._event_history.append(event)

        results: List[Any] = []
        handlers = self._event_handlers.get(event_type, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(data)
                else:
                    result = handler(data)
                results.append(result)
            except Exception as e:
                # Trigger compensation for this event
                await self._compensate_event(event_type, data)
                raise e

        return results

    async def _compensate_event(self, event_type: str, data: Any = None) -> None:
        """Run compensation for an event."""
        compensation = self._compensation_handlers.get(event_type)
        if compensation:
            if asyncio.iscoroutinefunction(compensation):
                await compensation(data)
            else:
                compensation(data)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get event history."""
        return self._event_history.copy()


class SagaBuilder:
    """Builder for creating saga definitions."""

    def __init__(self, name: str):
        self._definition = SagaDefinition(name)

    def step(
        self,
        name: str,
        action: Callable[..., Any],
        compensation: Optional[Callable[..., Any]] = None,
    ) -> "SagaBuilder":
        """Add a step to the saga."""
        self._definition.add_step(name, action, compensation)
        return self

    def with_timeout(self, timeout: float) -> "SagaBuilder":
        """Set timeout for the last step."""
        if self._definition.steps:
            self._definition.steps[-1].timeout = timeout
        return self

    def with_retries(self, max_retries: int) -> "SagaBuilder":
        """Set max retries for the last step."""
        if self._definition.steps:
            self._definition.steps[-1].max_retries = max_retries
        return self

    def build(self) -> SagaDefinition:
        """Build the saga definition."""
        return self._definition


class SagaVisionProvider(VisionProvider):
    """Vision provider with saga transaction support."""

    def __init__(
        self,
        base_provider: VisionProvider,
        orchestrator: SagaOrchestrator,
    ):
        self._base = base_provider
        self._orchestrator = orchestrator
        self._analysis_saga_name = "vision_analysis_saga"
        self._setup_analysis_saga()

    def _setup_analysis_saga(self) -> None:
        """Set up the default analysis saga."""
        saga = SagaDefinition(self._analysis_saga_name)

        async def preprocess(ctx: SagaContext) -> Dict[str, Any]:
            return {"preprocessed": True, "image": ctx.get("image_data")}

        async def analyze(ctx: SagaContext) -> Dict[str, Any]:
            image_data = ctx.get("image_data", b"")
            result = await self._base.analyze_image(image_data)
            return {"description": result}

        async def postprocess(ctx: SagaContext) -> Dict[str, Any]:
            desc = ctx.get_step_result("analyze")
            return {"final": desc, "postprocessed": True}

        saga.add_step("preprocess", preprocess)
        saga.add_step("analyze", analyze)
        saga.add_step("postprocess", postprocess)

        self._orchestrator.register_saga(saga)

    @property
    def provider_name(self) -> str:
        return f"saga_{self._base.provider_name}"

    async def analyze_image(self, image_data: bytes, **kwargs: Any) -> VisionDescription:
        """Analyze image using saga pattern."""
        context = SagaContext()
        context.set("image_data", image_data)
        context.set("kwargs", kwargs)

        execution = await self._orchestrator.execute(self._analysis_saga_name, context)

        if execution.status == SagaStatus.COMPLETED:
            result = execution.context.get_step_result("analyze")
            if result and "description" in result:
                return result["description"]

        # Fallback to base provider
        return await self._base.analyze_image(image_data, **kwargs)

    def get_orchestrator(self) -> SagaOrchestrator:
        """Get the saga orchestrator."""
        return self._orchestrator


# Factory functions
def create_saga_orchestrator() -> SagaOrchestrator:
    """Create a new saga orchestrator."""
    return SagaOrchestrator()


def create_saga_builder(name: str) -> SagaBuilder:
    """Create a new saga builder."""
    return SagaBuilder(name)


def create_compensation_manager() -> CompensationManager:
    """Create a new compensation manager."""
    return CompensationManager()


def create_transaction_coordinator() -> TransactionCoordinator:
    """Create a new transaction coordinator."""
    return TransactionCoordinator()


def create_choreography_saga(name: str) -> ChoreographySaga:
    """Create a new choreography saga."""
    return ChoreographySaga(name)


def create_saga_provider(
    base_provider: VisionProvider,
    orchestrator: Optional[SagaOrchestrator] = None,
) -> SagaVisionProvider:
    """Create a vision provider with saga support."""
    orch = orchestrator or create_saga_orchestrator()
    return SagaVisionProvider(base_provider, orch)


__all__ = [
    # Enums
    "SagaStatus",
    "StepStatus",
    # Data classes
    "SagaStep",
    "SagaContext",
    "SagaExecution",
    # Core classes
    "SagaDefinition",
    "SagaOrchestrator",
    "CompensationManager",
    "TransactionCoordinator",
    "TransactionParticipant",
    "SimpleParticipant",
    "ChoreographySaga",
    "SagaBuilder",
    "SagaVisionProvider",
    # Factory functions
    "create_saga_orchestrator",
    "create_saga_builder",
    "create_compensation_manager",
    "create_transaction_coordinator",
    "create_choreography_saga",
    "create_saga_provider",
]
