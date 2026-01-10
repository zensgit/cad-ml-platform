"""
Workflow Engine for Vision Provider.

This module provides workflow orchestration including:
- Workflow definition and execution
- State machine implementation
- Task scheduling and dependencies
- Parallel and sequential execution
- Compensation and rollback
- Workflow monitoring and visualization

Phase 10 Feature.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union

from .base import VisionDescription, VisionProvider
from src.utils.safe_eval import safe_eval

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ============================================================================
# Workflow Enums
# ============================================================================


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATING = "compensating"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    COMPENSATED = "compensated"


class TriggerType(Enum):
    """Workflow trigger types."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    WEBHOOK = "webhook"
    DEPENDENT = "dependent"


class RetryPolicy(Enum):
    """Retry policy types."""

    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


# ============================================================================
# Workflow Data Classes
# ============================================================================


@dataclass
class TaskDefinition:
    """Definition of a workflow task."""

    task_id: str
    name: str
    handler: str  # Handler name or callable reference
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 300.0
    retry_policy: RetryPolicy = RetryPolicy.NONE
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    compensation_handler: Optional[str] = None
    condition: Optional[str] = None  # Condition expression
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Definition of a workflow."""

    workflow_id: str
    name: str
    version: str = "1.0.0"
    tasks: List[TaskDefinition] = field(default_factory=list)
    trigger_type: TriggerType = TriggerType.MANUAL
    timeout_seconds: float = 3600.0
    on_success: Optional[str] = None
    on_failure: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Runtime task execution state."""

    task_id: str
    definition: TaskDefinition
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def duration_ms(self) -> Optional[float]:
        """Get execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


@dataclass
class WorkflowExecution:
    """Runtime workflow execution state."""

    workflow_id: str
    execution_id: str
    definition: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.PENDING
    tasks: Dict[str, TaskExecution] = field(default_factory=dict)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def progress(self) -> float:
        """Get workflow progress percentage."""
        if not self.tasks:
            return 0.0
        completed = sum(
            1 for t in self.tasks.values() if t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
        )
        return (completed / len(self.tasks)) * 100


@dataclass
class WorkflowEvent:
    """Workflow event for monitoring."""

    event_type: str
    workflow_id: str
    execution_id: str
    task_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Task Handlers
# ============================================================================


class TaskHandler(ABC):
    """Abstract base class for task handlers."""

    @abstractmethod
    async def execute(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> Any:
        """Execute the task."""
        pass

    async def compensate(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> None:
        """Compensate for failed execution (rollback)."""
        pass


class VisionAnalysisTask(TaskHandler):
    """Task handler for vision analysis."""

    def __init__(self, provider: VisionProvider) -> None:
        """Initialize with vision provider."""
        self._provider = provider

    async def execute(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> VisionDescription:
        """Execute vision analysis."""
        image_data = task.input_data.get("image_data", b"")
        context_str = task.input_data.get("context")

        return await self._provider.analyze_image(image_data, context_str)


class ConditionalTask(TaskHandler):
    """Task handler that executes conditionally."""

    def __init__(
        self,
        condition: Callable[[Dict[str, Any]], bool],
        if_true: TaskHandler,
        if_false: Optional[TaskHandler] = None,
    ) -> None:
        """Initialize conditional task."""
        self._condition = condition
        self._if_true = if_true
        self._if_false = if_false

    async def execute(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> Any:
        """Execute based on condition."""
        if self._condition(context):
            return await self._if_true.execute(task, context)
        elif self._if_false:
            return await self._if_false.execute(task, context)
        return None


class ParallelTask(TaskHandler):
    """Task handler that executes subtasks in parallel."""

    def __init__(self, subtasks: List[TaskHandler]) -> None:
        """Initialize parallel task."""
        self._subtasks = subtasks

    async def execute(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> List[Any]:
        """Execute subtasks in parallel."""
        results = await asyncio.gather(
            *[st.execute(task, context) for st in self._subtasks],
            return_exceptions=True,
        )

        # Check for exceptions
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise RuntimeError(f"Parallel execution failed: {errors[0]}")

        return list(results)


class SequentialTask(TaskHandler):
    """Task handler that executes subtasks sequentially."""

    def __init__(self, subtasks: List[TaskHandler]) -> None:
        """Initialize sequential task."""
        self._subtasks = subtasks

    async def execute(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> List[Any]:
        """Execute subtasks sequentially."""
        results: List[Any] = []

        for subtask in self._subtasks:
            result = await subtask.execute(task, context)
            results.append(result)
            context["last_result"] = result

        return results


# ============================================================================
# State Machine
# ============================================================================


@dataclass
class StateTransition:
    """State transition definition."""

    from_state: str
    to_state: str
    event: str
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: Optional[Callable[[Dict[str, Any]], None]] = None


class StateMachine:
    """
    Finite state machine implementation.

    Manages workflow state transitions.
    """

    def __init__(
        self,
        initial_state: str,
        transitions: Optional[List[StateTransition]] = None,
    ) -> None:
        """Initialize state machine."""
        self._current_state = initial_state
        self._transitions: Dict[str, List[StateTransition]] = {}
        self._state_enter_actions: Dict[str, List[Callable[..., None]]] = {}
        self._state_exit_actions: Dict[str, List[Callable[..., None]]] = {}
        self._history: List[Tuple[str, str, datetime]] = []

        if transitions:
            for t in transitions:
                self.add_transition(t)

    @property
    def current_state(self) -> str:
        """Get current state."""
        return self._current_state

    def add_transition(self, transition: StateTransition) -> None:
        """Add a state transition."""
        key = f"{transition.from_state}:{transition.event}"
        if key not in self._transitions:
            self._transitions[key] = []
        self._transitions[key].append(transition)

    def on_enter(self, state: str, action: Callable[..., None]) -> None:
        """Add action to execute when entering a state."""
        if state not in self._state_enter_actions:
            self._state_enter_actions[state] = []
        self._state_enter_actions[state].append(action)

    def on_exit(self, state: str, action: Callable[..., None]) -> None:
        """Add action to execute when exiting a state."""
        if state not in self._state_exit_actions:
            self._state_exit_actions[state] = []
        self._state_exit_actions[state].append(action)

    def trigger(self, event: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger an event and transition if valid."""
        context = context or {}
        key = f"{self._current_state}:{event}"

        if key not in self._transitions:
            logger.debug(f"No transition for event '{event}' from state '{self._current_state}'")
            return False

        for transition in self._transitions[key]:
            # Check guard condition
            if transition.guard and not transition.guard(context):
                continue

            # Execute exit actions
            for action in self._state_exit_actions.get(self._current_state, []):
                action(context)

            # Record history
            self._history.append((self._current_state, transition.to_state, datetime.now()))

            # Execute transition action
            if transition.action:
                transition.action(context)

            # Update state
            old_state = self._current_state
            self._current_state = transition.to_state

            # Execute enter actions
            for action in self._state_enter_actions.get(self._current_state, []):
                action(context)

            logger.debug(f"State transition: {old_state} -> {self._current_state}")
            return True

        return False

    def can_trigger(self, event: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if an event can trigger a transition."""
        context = context or {}
        key = f"{self._current_state}:{event}"

        if key not in self._transitions:
            return False

        for transition in self._transitions[key]:
            if transition.guard is None or transition.guard(context):
                return True

        return False

    def get_history(self) -> List[Tuple[str, str, datetime]]:
        """Get state transition history."""
        return list(self._history)

    def reset(self, initial_state: str) -> None:
        """Reset state machine to initial state."""
        self._current_state = initial_state
        self._history.clear()


# ============================================================================
# Workflow Engine
# ============================================================================


class WorkflowEngine:
    """
    Central workflow engine.

    Manages workflow execution, scheduling, and monitoring.
    """

    def __init__(self) -> None:
        """Initialize workflow engine."""
        self._definitions: Dict[str, WorkflowDefinition] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._handlers: Dict[str, TaskHandler] = {}
        self._event_handlers: List[Callable[[WorkflowEvent], None]] = []
        self._lock = asyncio.Lock()

    def register_workflow(self, definition: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self._definitions[definition.workflow_id] = definition
        logger.info(f"Registered workflow: {definition.workflow_id}")

    def register_handler(self, name: str, handler: TaskHandler) -> None:
        """Register a task handler."""
        self._handlers[name] = handler
        logger.debug(f"Registered handler: {name}")

    def on_event(self, callback: Callable[[WorkflowEvent], None]) -> None:
        """Register event callback."""
        self._event_handlers.append(callback)

    async def start_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """Start a new workflow execution."""
        if workflow_id not in self._definitions:
            raise ValueError(f"Workflow '{workflow_id}' not found")

        definition = self._definitions[workflow_id]
        execution_id = str(uuid.uuid4())

        # Create task executions
        tasks: Dict[str, TaskExecution] = {}
        for task_def in definition.tasks:
            tasks[task_def.task_id] = TaskExecution(
                task_id=task_def.task_id,
                definition=task_def,
            )

        # Create workflow execution
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            definition=definition,
            tasks=tasks,
            input_data=input_data or {},
        )

        async with self._lock:
            self._executions[execution_id] = execution

        # Emit event
        self._emit_event("workflow_started", execution)

        # Start execution
        asyncio.create_task(self._run_workflow(execution))

        return execution

    async def _run_workflow(self, execution: WorkflowExecution) -> None:
        """Run workflow execution."""
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = datetime.now()

        try:
            # Build execution context
            context: Dict[str, Any] = {
                "workflow_id": execution.workflow_id,
                "execution_id": execution.execution_id,
                "input": execution.input_data,
                "results": {},
            }

            # Execute tasks in dependency order
            await self._execute_tasks(execution, context)

            # Check final status
            failed_tasks = [t for t in execution.tasks.values() if t.status == TaskStatus.FAILED]

            if failed_tasks:
                execution.status = WorkflowStatus.FAILED
                execution.error = f"Tasks failed: {[t.task_id for t in failed_tasks]}"
            else:
                execution.status = WorkflowStatus.COMPLETED
                execution.output_data = context.get("results")

        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)

            # Trigger compensation
            await self._compensate(execution, context)

        finally:
            execution.completed_at = datetime.now()
            self._emit_event(
                "workflow_completed"
                if execution.status == WorkflowStatus.COMPLETED
                else "workflow_failed",
                execution,
            )

    async def _execute_tasks(
        self,
        execution: WorkflowExecution,
        context: Dict[str, Any],
    ) -> None:
        """Execute tasks in dependency order."""
        pending = set(execution.tasks.keys())
        completed: Set[str] = set()

        while pending:
            # Find ready tasks (all dependencies completed)
            ready: List[str] = []
            for task_id in pending:
                task = execution.tasks[task_id]
                deps = set(task.definition.dependencies)
                if deps.issubset(completed):
                    ready.append(task_id)

            if not ready:
                # No tasks ready - possible circular dependency or all failed
                break

            # Execute ready tasks in parallel
            await asyncio.gather(
                *[self._execute_task(execution, task_id, context) for task_id in ready]
            )

            # Update sets
            for task_id in ready:
                task = execution.tasks[task_id]
                pending.remove(task_id)
                if task.status == TaskStatus.COMPLETED:
                    completed.add(task_id)

    async def _execute_task(
        self,
        execution: WorkflowExecution,
        task_id: str,
        context: Dict[str, Any],
    ) -> None:
        """Execute a single task."""
        task = execution.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        self._emit_event("task_started", execution, task_id)

        try:
            # Check condition
            if task.definition.condition:
                # Simple condition evaluation (in production, use safe eval)
                condition_result = self._evaluate_condition(task.definition.condition, context)
                if not condition_result:
                    task.status = TaskStatus.SKIPPED
                    task.completed_at = datetime.now()
                    self._emit_event("task_skipped", execution, task_id)
                    return

            # Get handler
            handler = self._handlers.get(task.definition.handler)
            if not handler:
                raise ValueError(f"Handler '{task.definition.handler}' not found")

            # Prepare input data
            task.input_data = self._prepare_task_input(task, context)

            # Execute with retry
            result = await self._execute_with_retry(task, handler, context)

            # Store result
            task.output_data = result
            task.status = TaskStatus.COMPLETED
            context["results"][task_id] = result

        except Exception as e:
            logger.error(f"Task '{task_id}' failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self._emit_event("task_failed", execution, task_id, {"error": str(e)})

        finally:
            task.completed_at = datetime.now()
            if task.status == TaskStatus.COMPLETED:
                self._emit_event("task_completed", execution, task_id)

    async def _execute_with_retry(
        self,
        task: TaskExecution,
        handler: TaskHandler,
        context: Dict[str, Any],
    ) -> Any:
        """Execute task with retry logic."""
        definition = task.definition
        max_retries = definition.max_retries
        retry_delay = definition.retry_delay_seconds

        for attempt in range(max_retries + 1):
            try:
                # Execute with timeout
                return await asyncio.wait_for(
                    handler.execute(task, context),
                    timeout=definition.timeout_seconds,
                )
            except asyncio.TimeoutError:
                task.retry_count = attempt + 1
                if attempt >= max_retries:
                    raise TimeoutError(
                        f"Task '{task.task_id}' timed out after {definition.timeout_seconds}s"
                    )
            except Exception as e:
                task.retry_count = attempt + 1
                if attempt >= max_retries:
                    raise

            # Calculate retry delay
            if definition.retry_policy == RetryPolicy.EXPONENTIAL:
                delay = retry_delay * (2**attempt)
            elif definition.retry_policy == RetryPolicy.LINEAR:
                delay = retry_delay * (attempt + 1)
            else:
                delay = retry_delay

            logger.debug(
                f"Retrying task '{task.task_id}' in {delay}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(delay)

        raise RuntimeError("Retry logic error")

    async def _compensate(
        self,
        execution: WorkflowExecution,
        context: Dict[str, Any],
    ) -> None:
        """Execute compensation for failed workflow."""
        execution.status = WorkflowStatus.COMPENSATING
        self._emit_event("compensation_started", execution)

        # Get completed tasks in reverse order
        completed_tasks = [t for t in execution.tasks.values() if t.status == TaskStatus.COMPLETED]
        completed_tasks.sort(key=lambda t: t.completed_at or datetime.min, reverse=True)

        for task in completed_tasks:
            compensation_handler = task.definition.compensation_handler
            if compensation_handler and compensation_handler in self._handlers:
                try:
                    handler = self._handlers[compensation_handler]
                    await handler.compensate(task, context)
                    task.status = TaskStatus.COMPENSATED
                    logger.debug(f"Compensated task: {task.task_id}")
                except Exception as e:
                    logger.error(f"Compensation failed for task '{task.task_id}': {e}")

        self._emit_event("compensation_completed", execution)

    def _prepare_task_input(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare task input data from context and dependencies."""
        input_data: Dict[str, Any] = {}

        # Add workflow input
        input_data.update(context.get("input", {}))

        # Add results from dependencies
        for dep_id in task.definition.dependencies:
            if dep_id in context.get("results", {}):
                input_data[f"dep_{dep_id}"] = context["results"][dep_id]

        # Add task-specific metadata
        input_data.update(task.definition.metadata)

        return input_data

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        # Simple implementation - in production use safe evaluation
        try:
            # Support basic conditions like "results.task1.success == true"
            return bool(safe_eval(condition, context))
        except Exception:
            return True  # Default to true if condition can't be evaluated

    def _emit_event(
        self,
        event_type: str,
        execution: WorkflowExecution,
        task_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit workflow event."""
        event = WorkflowEvent(
            event_type=event_type,
            workflow_id=execution.workflow_id,
            execution_id=execution.execution_id,
            task_id=task_id,
            data=data or {},
        )

        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow."""
        async with self._lock:
            if execution_id not in self._executions:
                return False

            execution = self._executions[execution_id]
            if execution.status != WorkflowStatus.RUNNING:
                return False

            execution.status = WorkflowStatus.PAUSED
            self._emit_event("workflow_paused", execution)
            return True

    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume a paused workflow."""
        async with self._lock:
            if execution_id not in self._executions:
                return False

            execution = self._executions[execution_id]
            if execution.status != WorkflowStatus.PAUSED:
                return False

            execution.status = WorkflowStatus.RUNNING
            self._emit_event("workflow_resumed", execution)
            return True

    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow."""
        async with self._lock:
            if execution_id not in self._executions:
                return False

            execution = self._executions[execution_id]
            if execution.status not in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
                return False

            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.now()
            self._emit_event("workflow_cancelled", execution)
            return True

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        return self._executions.get(execution_id)

    def get_executions_by_workflow(self, workflow_id: str) -> List[WorkflowExecution]:
        """Get all executions for a workflow."""
        return [e for e in self._executions.values() if e.workflow_id == workflow_id]


# ============================================================================
# Workflow Builder
# ============================================================================


class WorkflowBuilder:
    """
    Fluent builder for workflow definitions.

    Provides a convenient API for constructing workflows.
    """

    def __init__(self, workflow_id: str, name: str) -> None:
        """Initialize workflow builder."""
        self._workflow_id = workflow_id
        self._name = name
        self._version = "1.0.0"
        self._tasks: List[TaskDefinition] = []
        self._trigger_type = TriggerType.MANUAL
        self._timeout_seconds = 3600.0
        self._on_success: Optional[str] = None
        self._on_failure: Optional[str] = None
        self._metadata: Dict[str, Any] = {}

    def version(self, version: str) -> "WorkflowBuilder":
        """Set workflow version."""
        self._version = version
        return self

    def timeout(self, seconds: float) -> "WorkflowBuilder":
        """Set workflow timeout."""
        self._timeout_seconds = seconds
        return self

    def trigger(self, trigger_type: TriggerType) -> "WorkflowBuilder":
        """Set trigger type."""
        self._trigger_type = trigger_type
        return self

    def on_success(self, handler: str) -> "WorkflowBuilder":
        """Set success handler."""
        self._on_success = handler
        return self

    def on_failure(self, handler: str) -> "WorkflowBuilder":
        """Set failure handler."""
        self._on_failure = handler
        return self

    def metadata(self, key: str, value: Any) -> "WorkflowBuilder":
        """Add metadata."""
        self._metadata[key] = value
        return self

    def add_task(
        self,
        task_id: str,
        name: str,
        handler: str,
        dependencies: Optional[List[str]] = None,
        timeout_seconds: float = 300.0,
        retry_policy: RetryPolicy = RetryPolicy.NONE,
        max_retries: int = 3,
        compensation_handler: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> "WorkflowBuilder":
        """Add a task to the workflow."""
        task = TaskDefinition(
            task_id=task_id,
            name=name,
            handler=handler,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
            retry_policy=retry_policy,
            max_retries=max_retries,
            compensation_handler=compensation_handler,
            condition=condition,
        )
        self._tasks.append(task)
        return self

    def build(self) -> WorkflowDefinition:
        """Build the workflow definition."""
        return WorkflowDefinition(
            workflow_id=self._workflow_id,
            name=self._name,
            version=self._version,
            tasks=self._tasks,
            trigger_type=self._trigger_type,
            timeout_seconds=self._timeout_seconds,
            on_success=self._on_success,
            on_failure=self._on_failure,
            metadata=self._metadata,
        )


# ============================================================================
# Vision Workflow Tasks
# ============================================================================


class ImagePreprocessTask(TaskHandler):
    """Preprocess image data for analysis."""

    async def execute(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> bytes:
        """Preprocess image."""
        image_data = task.input_data.get("image_data", b"")

        # Simple preprocessing - in production add actual processing
        return image_data


class ImageAnalysisTask(TaskHandler):
    """Analyze image using vision provider."""

    def __init__(self, provider: VisionProvider) -> None:
        """Initialize with provider."""
        self._provider = provider

    async def execute(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze image."""
        image_data = task.input_data.get("image_data", b"")
        context_str = task.input_data.get("context")

        description = await self._provider.analyze_image(image_data, context_str)

        return {
            "summary": description.summary,
            "details": description.details,
            "confidence": description.confidence,
        }


class ResultAggregationTask(TaskHandler):
    """Aggregate results from multiple analyses."""

    async def execute(
        self,
        task: TaskExecution,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate results."""
        results = context.get("results", {})

        summaries: List[str] = []
        all_details: List[str] = []
        confidences: List[float] = []

        for result in results.values():
            if isinstance(result, dict):
                if "summary" in result:
                    summaries.append(result["summary"])
                if "details" in result:
                    all_details.extend(result["details"])
                if "confidence" in result:
                    confidences.append(result["confidence"])

        return {
            "summaries": summaries,
            "details": all_details,
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
        }


# ============================================================================
# Vision Workflow Provider
# ============================================================================


class WorkflowVisionProvider(VisionProvider):
    """
    Vision provider that uses workflow engine.

    Orchestrates analysis through configurable workflows.
    """

    def __init__(
        self,
        engine: WorkflowEngine,
        workflow_id: str,
    ) -> None:
        """Initialize workflow-based provider."""
        self._engine = engine
        self._workflow_id = workflow_id

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"workflow_{self._workflow_id}"

    async def analyze_image(
        self, image_data: bytes, context: Optional[str] = None
    ) -> VisionDescription:
        """Analyze image through workflow."""
        # Start workflow
        execution = await self._engine.start_workflow(
            self._workflow_id,
            input_data={
                "image_data": image_data,
                "context": context,
            },
        )

        # Wait for completion
        while execution.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
            await asyncio.sleep(0.1)
            execution = self._engine.get_execution(execution.execution_id) or execution

        if execution.status != WorkflowStatus.COMPLETED:
            raise RuntimeError(f"Workflow failed: {execution.error}")

        # Extract result
        output = execution.output_data or {}

        return VisionDescription(
            summary=output.get("summary", "Workflow completed"),
            details=output.get("details", []),
            confidence=output.get("confidence", 0.0),
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_workflow_engine() -> WorkflowEngine:
    """Create a workflow engine."""
    return WorkflowEngine()


def create_workflow_builder(workflow_id: str, name: str) -> WorkflowBuilder:
    """Create a workflow builder."""
    return WorkflowBuilder(workflow_id, name)


def create_vision_workflow(
    provider: VisionProvider,
    with_preprocessing: bool = True,
    with_aggregation: bool = False,
) -> Tuple[WorkflowDefinition, Dict[str, TaskHandler]]:
    """Create a standard vision analysis workflow."""
    builder = WorkflowBuilder("vision_analysis", "Vision Analysis Workflow")

    handlers: Dict[str, TaskHandler] = {}

    if with_preprocessing:
        builder.add_task(
            task_id="preprocess",
            name="Image Preprocessing",
            handler="preprocess",
        )
        handlers["preprocess"] = ImagePreprocessTask()

    builder.add_task(
        task_id="analyze",
        name="Image Analysis",
        handler="analyze",
        dependencies=["preprocess"] if with_preprocessing else [],
        retry_policy=RetryPolicy.EXPONENTIAL,
        max_retries=3,
    )
    handlers["analyze"] = ImageAnalysisTask(provider)

    if with_aggregation:
        builder.add_task(
            task_id="aggregate",
            name="Result Aggregation",
            handler="aggregate",
            dependencies=["analyze"],
        )
        handlers["aggregate"] = ResultAggregationTask()

    return builder.build(), handlers
