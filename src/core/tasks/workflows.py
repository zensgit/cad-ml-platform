"""Workflow Engine Implementation.

Provides task orchestration with chains, groups, and chords.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepType(str, Enum):
    """Type of workflow step."""

    TASK = "task"
    CHAIN = "chain"
    GROUP = "group"
    CHORD = "chord"
    CONDITIONAL = "conditional"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""

    step_id: str
    name: str
    step_type: StepType
    func: Optional[Callable[..., Any]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    children: List["WorkflowStep"] = field(default_factory=list)
    callback: Optional[Callable[..., Any]] = None
    condition: Optional[Callable[[Any], bool]] = None

    # Execution state
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3

    @classmethod
    def task(
        cls,
        func: Callable[..., Any],
        *args: Any,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "WorkflowStep":
        """Create a task step."""
        return cls(
            step_id=str(uuid.uuid4()),
            name=name or func.__name__,
            step_type=StepType.TASK,
            func=func,
            args=args,
            kwargs=kwargs,
        )


@dataclass
class WorkflowContext:
    """Context passed through workflow execution."""

    workflow_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)


@dataclass
class Workflow:
    """A workflow definition and execution state."""

    workflow_id: str
    name: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    context: WorkflowContext = field(default_factory=lambda: WorkflowContext(workflow_id=""))
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.context.workflow_id = self.workflow_id


class WorkflowEngine:
    """Engine for executing workflows."""

    def __init__(self):
        self._workflows: Dict[str, Workflow] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def execute(self, workflow: Workflow) -> Any:
        """Execute a workflow.

        Args:
            workflow: Workflow to execute

        Returns:
            Final result of the workflow
        """
        self._workflows[workflow.workflow_id] = workflow
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()

        try:
            result = None
            for i, step in enumerate(workflow.steps):
                workflow.current_step = i
                result = await self._execute_step(step, workflow.context, result)
                workflow.context.results[step.step_id] = result

            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            return result

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            logger.error(f"Workflow {workflow.workflow_id} failed: {e}")
            raise

    async def _execute_step(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
        prev_result: Any = None,
    ) -> Any:
        """Execute a single workflow step."""
        step.status = WorkflowStatus.RUNNING
        step.started_at = datetime.utcnow()

        try:
            if step.step_type == StepType.TASK:
                result = await self._execute_task(step, context, prev_result)

            elif step.step_type == StepType.CHAIN:
                result = await self._execute_chain(step, context, prev_result)

            elif step.step_type == StepType.GROUP:
                result = await self._execute_group(step, context, prev_result)

            elif step.step_type == StepType.CHORD:
                result = await self._execute_chord(step, context, prev_result)

            elif step.step_type == StepType.CONDITIONAL:
                result = await self._execute_conditional(step, context, prev_result)

            else:
                raise ValueError(f"Unknown step type: {step.step_type}")

            step.status = WorkflowStatus.COMPLETED
            step.completed_at = datetime.utcnow()
            step.result = result
            return result

        except Exception as e:
            step.error = str(e)

            # Retry logic
            if step.retries < step.max_retries:
                step.retries += 1
                logger.warning(f"Retrying step {step.name} (attempt {step.retries})")
                await asyncio.sleep(2 ** step.retries)  # Exponential backoff
                return await self._execute_step(step, context, prev_result)

            step.status = WorkflowStatus.FAILED
            step.completed_at = datetime.utcnow()
            raise

    async def _execute_task(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
        prev_result: Any,
    ) -> Any:
        """Execute a task step."""
        args = step.args
        kwargs = {**step.kwargs}

        # Pass previous result if function accepts it
        if prev_result is not None:
            kwargs["prev_result"] = prev_result

        # Pass context if function accepts it
        kwargs["context"] = context

        if asyncio.iscoroutinefunction(step.func):
            return await step.func(*args, **kwargs)
        return step.func(*args, **kwargs)

    async def _execute_chain(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
        prev_result: Any,
    ) -> Any:
        """Execute a chain of steps sequentially."""
        result = prev_result
        for child in step.children:
            result = await self._execute_step(child, context, result)
        return result

    async def _execute_group(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
        prev_result: Any,
    ) -> List[Any]:
        """Execute a group of steps in parallel."""
        tasks = [
            self._execute_step(child, context, prev_result)
            for child in step.children
        ]
        return await asyncio.gather(*tasks)

    async def _execute_chord(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
        prev_result: Any,
    ) -> Any:
        """Execute a group then pass results to callback."""
        # Execute group
        group_results = await self._execute_group(step, context, prev_result)

        # Execute callback with results
        if step.callback:
            if asyncio.iscoroutinefunction(step.callback):
                return await step.callback(group_results, context=context)
            return step.callback(group_results, context=context)

        return group_results

    async def _execute_conditional(
        self,
        step: WorkflowStep,
        context: WorkflowContext,
        prev_result: Any,
    ) -> Any:
        """Execute steps based on condition."""
        if step.condition and step.condition(prev_result):
            # Execute if condition is true
            if step.children:
                return await self._execute_step(step.children[0], context, prev_result)
        elif len(step.children) > 1:
            # Execute else branch
            return await self._execute_step(step.children[1], context, prev_result)

        return prev_result

    def cancel(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self._running_tasks:
            self._running_tasks[workflow_id].cancel()
            if workflow_id in self._workflows:
                self._workflows[workflow_id].status = WorkflowStatus.CANCELLED
            return True
        return False

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[Workflow]:
        """List all workflows."""
        return list(self._workflows.values())


# Workflow builder functions

def chain(*steps: Union[WorkflowStep, Callable]) -> WorkflowStep:
    """Create a chain of steps executed sequentially.

    Args:
        *steps: Steps to chain

    Returns:
        WorkflowStep representing the chain

    Example:
        workflow = chain(
            task1,
            task2,
            task3,
        )
    """
    children = [
        s if isinstance(s, WorkflowStep) else WorkflowStep.task(s)
        for s in steps
    ]
    return WorkflowStep(
        step_id=str(uuid.uuid4()),
        name="chain",
        step_type=StepType.CHAIN,
        children=children,
    )


def group(*steps: Union[WorkflowStep, Callable]) -> WorkflowStep:
    """Create a group of steps executed in parallel.

    Args:
        *steps: Steps to run in parallel

    Returns:
        WorkflowStep representing the group

    Example:
        workflow = group(
            process_image,
            process_text,
            process_metadata,
        )
    """
    children = [
        s if isinstance(s, WorkflowStep) else WorkflowStep.task(s)
        for s in steps
    ]
    return WorkflowStep(
        step_id=str(uuid.uuid4()),
        name="group",
        step_type=StepType.GROUP,
        children=children,
    )


def chord(
    group_steps: List[Union[WorkflowStep, Callable]],
    callback: Callable,
) -> WorkflowStep:
    """Create a chord: parallel tasks followed by a callback.

    Args:
        group_steps: Steps to run in parallel
        callback: Function to call with group results

    Returns:
        WorkflowStep representing the chord

    Example:
        workflow = chord(
            [extract_a, extract_b, extract_c],
            aggregate_results,
        )
    """
    children = [
        s if isinstance(s, WorkflowStep) else WorkflowStep.task(s)
        for s in group_steps
    ]
    return WorkflowStep(
        step_id=str(uuid.uuid4()),
        name="chord",
        step_type=StepType.CHORD,
        children=children,
        callback=callback,
    )


def conditional(
    condition: Callable[[Any], bool],
    if_true: Union[WorkflowStep, Callable],
    if_false: Optional[Union[WorkflowStep, Callable]] = None,
) -> WorkflowStep:
    """Create a conditional step.

    Args:
        condition: Function to evaluate
        if_true: Step to run if condition is true
        if_false: Step to run if condition is false

    Returns:
        WorkflowStep representing the conditional

    Example:
        workflow = conditional(
            lambda x: x > 0.5,
            process_high_confidence,
            process_low_confidence,
        )
    """
    children = [
        if_true if isinstance(if_true, WorkflowStep) else WorkflowStep.task(if_true)
    ]
    if if_false:
        children.append(
            if_false if isinstance(if_false, WorkflowStep) else WorkflowStep.task(if_false)
        )

    return WorkflowStep(
        step_id=str(uuid.uuid4()),
        name="conditional",
        step_type=StepType.CONDITIONAL,
        children=children,
        condition=condition,
    )


class WorkflowBuilder:
    """Builder for creating workflows fluently."""

    def __init__(self, name: str):
        self.name = name
        self._steps: List[WorkflowStep] = []
        self._metadata: Dict[str, Any] = {}

    def add_step(self, step: Union[WorkflowStep, Callable]) -> "WorkflowBuilder":
        """Add a step to the workflow."""
        if not isinstance(step, WorkflowStep):
            step = WorkflowStep.task(step)
        self._steps.append(step)
        return self

    def chain(self, *steps: Union[WorkflowStep, Callable]) -> "WorkflowBuilder":
        """Add a chain of steps."""
        self._steps.append(chain(*steps))
        return self

    def group(self, *steps: Union[WorkflowStep, Callable]) -> "WorkflowBuilder":
        """Add a parallel group."""
        self._steps.append(group(*steps))
        return self

    def chord(
        self,
        group_steps: List[Union[WorkflowStep, Callable]],
        callback: Callable,
    ) -> "WorkflowBuilder":
        """Add a chord."""
        self._steps.append(chord(group_steps, callback))
        return self

    def conditional(
        self,
        condition: Callable[[Any], bool],
        if_true: Union[WorkflowStep, Callable],
        if_false: Optional[Union[WorkflowStep, Callable]] = None,
    ) -> "WorkflowBuilder":
        """Add a conditional step."""
        self._steps.append(conditional(condition, if_true, if_false))
        return self

    def metadata(self, **kwargs: Any) -> "WorkflowBuilder":
        """Add metadata to the workflow."""
        self._metadata.update(kwargs)
        return self

    def build(self) -> Workflow:
        """Build the workflow."""
        return Workflow(
            workflow_id=str(uuid.uuid4()),
            name=self.name,
            steps=self._steps,
            metadata=self._metadata,
        )


# Global workflow engine
_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get global workflow engine."""
    global _engine
    if _engine is None:
        _engine = WorkflowEngine()
    return _engine
