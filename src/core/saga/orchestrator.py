"""Saga Orchestrator.

Provides saga execution:
- Sequential execution
- Compensation on failure
- State persistence
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.core.saga.core import (
    SagaContext,
    SagaDefinition,
    SagaState,
    SagaStep,
    StepExecution,
    StepResult,
    StepState,
    create_saga_id,
)

logger = logging.getLogger(__name__)


class SagaStore(ABC):
    """Abstract store for saga state persistence."""

    @abstractmethod
    async def save(self, context: SagaContext) -> None:
        """Save saga context."""
        pass

    @abstractmethod
    async def load(self, saga_id: str) -> Optional[SagaContext]:
        """Load saga context."""
        pass

    @abstractmethod
    async def list_pending(self) -> List[str]:
        """List pending saga IDs."""
        pass


class InMemorySagaStore(SagaStore):
    """In-memory saga store."""

    def __init__(self):
        self._sagas: Dict[str, SagaContext] = {}

    async def save(self, context: SagaContext) -> None:
        self._sagas[context.saga_id] = context

    async def load(self, saga_id: str) -> Optional[SagaContext]:
        return self._sagas.get(saga_id)

    async def list_pending(self) -> List[str]:
        return [
            sid for sid, ctx in self._sagas.items()
            if ctx.state in (SagaState.RUNNING, SagaState.COMPENSATING)
        ]


@dataclass
class ExecutionResult:
    """Result of saga execution."""
    saga_id: str
    success: bool
    state: SagaState
    context: SagaContext
    error: Optional[str] = None
    duration_ms: float = 0.0


class SagaOrchestrator:
    """Orchestrates saga execution."""

    def __init__(
        self,
        store: Optional[SagaStore] = None,
    ):
        self._store = store or InMemorySagaStore()
        self._definitions: Dict[str, SagaDefinition] = {}
        self._listeners: List[Callable[[SagaContext], None]] = []

    def register(self, definition: SagaDefinition) -> None:
        """Register a saga definition."""
        self._definitions[definition.name] = definition
        logger.debug(f"Registered saga: {definition.name}")

    def add_listener(
        self,
        listener: Callable[[SagaContext], None],
    ) -> None:
        """Add state change listener."""
        self._listeners.append(listener)

    async def execute(
        self,
        saga_name: str,
        initial_data: Optional[Dict[str, Any]] = None,
        saga_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute a saga."""
        definition = self._definitions.get(saga_name)
        if not definition:
            raise ValueError(f"Unknown saga: {saga_name}")

        # Create context
        saga_id = saga_id or create_saga_id()
        context = SagaContext(
            saga_id=saga_id,
            data=initial_data or {},
            state=SagaState.PENDING,
        )

        return await self._run_saga(definition, context)

    async def _run_saga(
        self,
        definition: SagaDefinition,
        context: SagaContext,
    ) -> ExecutionResult:
        """Run saga execution."""
        context.state = SagaState.RUNNING
        context.started_at = datetime.utcnow()
        await self._save_and_notify(context)

        completed_steps: List[SagaStep] = []

        try:
            # Execute steps
            for step in definition.steps:
                logger.debug(f"[{context.saga_id}] Executing step: {step.name}")

                context.record_execution(step.name, StepState.RUNNING)
                await self._save_and_notify(context)

                try:
                    result = await asyncio.wait_for(
                        step.execute(context),
                        timeout=definition.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    result = StepResult(
                        success=False,
                        error="Step timeout",
                    )

                if result.success:
                    context.record_execution(step.name, StepState.COMPLETED, result)
                    completed_steps.append(step)
                    logger.debug(f"[{context.saga_id}] Step completed: {step.name}")
                else:
                    context.record_execution(step.name, StepState.FAILED, result)
                    context.error = f"Step '{step.name}' failed: {result.error}"
                    logger.warning(f"[{context.saga_id}] Step failed: {step.name}")

                    # Start compensation
                    await self._compensate(
                        definition,
                        context,
                        completed_steps,
                    )
                    break

                await self._save_and_notify(context)

            # Check final state
            if context.state == SagaState.RUNNING:
                context.state = SagaState.COMPLETED
                context.completed_at = datetime.utcnow()

        except Exception as e:
            logger.exception(f"[{context.saga_id}] Saga error: {e}")
            context.error = str(e)
            await self._compensate(definition, context, completed_steps)

        await self._save_and_notify(context)

        return ExecutionResult(
            saga_id=context.saga_id,
            success=context.state == SagaState.COMPLETED,
            state=context.state,
            context=context,
            error=context.error,
            duration_ms=context.duration_ms or 0,
        )

    async def _compensate(
        self,
        definition: SagaDefinition,
        context: SagaContext,
        completed_steps: List[SagaStep],
    ) -> None:
        """Execute compensation for completed steps."""
        context.state = SagaState.COMPENSATING
        await self._save_and_notify(context)

        # Compensate in reverse order
        for step in reversed(completed_steps):
            if not step.should_compensate(context):
                continue

            logger.debug(f"[{context.saga_id}] Compensating step: {step.name}")

            for attempt in range(definition.max_compensation_retries):
                try:
                    context.record_execution(step.name, StepState.COMPENSATING)
                    result = await step.compensate(context)

                    if result.success:
                        context.record_execution(
                            step.name,
                            StepState.COMPENSATED,
                            result,
                        )
                        logger.debug(
                            f"[{context.saga_id}] Step compensated: {step.name}"
                        )
                        break
                    else:
                        logger.warning(
                            f"[{context.saga_id}] Compensation failed for {step.name}: "
                            f"{result.error} (attempt {attempt + 1})"
                        )
                except Exception as e:
                    logger.error(
                        f"[{context.saga_id}] Compensation error for {step.name}: {e}"
                    )

                if attempt < definition.max_compensation_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Backoff

        context.state = SagaState.COMPENSATED
        context.completed_at = datetime.utcnow()

    async def _save_and_notify(self, context: SagaContext) -> None:
        """Save context and notify listeners."""
        await self._store.save(context)

        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(context)
                else:
                    listener(context)
            except Exception as e:
                logger.error(f"Listener error: {e}")

    async def resume(self, saga_id: str) -> Optional[ExecutionResult]:
        """Resume a pending saga."""
        context = await self._store.load(saga_id)
        if not context:
            return None

        if context.state not in (SagaState.RUNNING, SagaState.COMPENSATING):
            return ExecutionResult(
                saga_id=saga_id,
                success=context.state == SagaState.COMPLETED,
                state=context.state,
                context=context,
            )

        # Find saga definition from context
        # In real implementation, we'd store the saga name in context
        saga_name = context.metadata.get("saga_name")
        if not saga_name:
            return None

        definition = self._definitions.get(saga_name)
        if not definition:
            return None

        return await self._run_saga(definition, context)

    async def get_status(self, saga_id: str) -> Optional[SagaContext]:
        """Get saga status."""
        return await self._store.load(saga_id)


class ParallelSagaOrchestrator(SagaOrchestrator):
    """Orchestrator with parallel step support."""

    async def _run_saga(
        self,
        definition: SagaDefinition,
        context: SagaContext,
    ) -> ExecutionResult:
        """Run saga with parallel steps support."""
        # Get parallel groups from definition metadata
        parallel_groups = definition.steps  # Simplified - would have group info

        # For now, use sequential execution
        return await super()._run_saga(definition, context)


class SagaCoordinator:
    """Coordinates multiple saga orchestrators."""

    def __init__(self):
        self._orchestrators: Dict[str, SagaOrchestrator] = {}

    def register_orchestrator(
        self,
        name: str,
        orchestrator: SagaOrchestrator,
    ) -> None:
        """Register an orchestrator."""
        self._orchestrators[name] = orchestrator

    def get_orchestrator(self, name: str) -> Optional[SagaOrchestrator]:
        """Get orchestrator by name."""
        return self._orchestrators.get(name)

    async def execute(
        self,
        orchestrator_name: str,
        saga_name: str,
        initial_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExecutionResult]:
        """Execute saga on specific orchestrator."""
        orchestrator = self._orchestrators.get(orchestrator_name)
        if not orchestrator:
            return None

        return await orchestrator.execute(saga_name, initial_data)
