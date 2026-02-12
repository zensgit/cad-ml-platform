"""Saga Module.

Provides distributed transaction management:
- Saga pattern implementation
- Compensation handling
- State persistence
- Orchestration
"""

from src.core.saga.core import (
    SagaState,
    StepState,
    StepResult,
    StepExecution,
    SagaStep,
    FunctionStep,
    SagaContext,
    SagaDefinition,
    create_saga_id,
    SagaBuilder,
)
from src.core.saga.orchestrator import (
    SagaStore,
    InMemorySagaStore,
    ExecutionResult,
    SagaOrchestrator,
    ParallelSagaOrchestrator,
    SagaCoordinator,
)

__all__ = [
    # Core
    "SagaState",
    "StepState",
    "StepResult",
    "StepExecution",
    "SagaStep",
    "FunctionStep",
    "SagaContext",
    "SagaDefinition",
    "create_saga_id",
    "SagaBuilder",
    # Orchestrator
    "SagaStore",
    "InMemorySagaStore",
    "ExecutionResult",
    "SagaOrchestrator",
    "ParallelSagaOrchestrator",
    "SagaCoordinator",
]
