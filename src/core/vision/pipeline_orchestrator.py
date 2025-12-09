"""Pipeline Orchestrator Module for Vision System.

This module provides ML pipeline orchestration including:
- DAG-based pipeline definition and execution
- Task scheduling and dependency management
- Parallel and sequential execution
- Pipeline versioning and templating
- Retry logic and error handling
- Resource management and optimization

Phase 18: Advanced ML Pipeline & AutoML
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import VisionDescription, VisionProvider


# ========================
# Enums
# ========================


class TaskStatus(str, Enum):
    """Pipeline task status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRY = "retry"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TriggerType(str, Enum):
    """Pipeline trigger types."""

    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    API = "api"
    DEPENDENCY = "dependency"


class RetryPolicy(str, Enum):
    """Task retry policies."""

    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class ExecutionMode(str, Enum):
    """Pipeline execution modes."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class ResourceType(str, Enum):
    """Resource types for tasks."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"


# ========================
# Dataclasses
# ========================


@dataclass
class TaskDefinition:
    """Definition of a pipeline task."""

    task_id: str
    name: str
    task_type: str = "python"
    handler: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_policy: RetryPolicy = RetryPolicy.NONE
    max_retries: int = 3
    timeout_seconds: int = 3600
    resources: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskInstance:
    """An instance of a running task."""

    instance_id: str
    task_id: str
    pipeline_run_id: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempt_number: int = 1
    result: Optional[Any] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineDefinition:
    """Definition of a pipeline."""

    pipeline_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    tasks: List[TaskDefinition] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.HYBRID
    default_resources: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[str] = None  # Cron expression
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineRun:
    """A pipeline execution run."""

    run_id: str
    pipeline_id: str
    status: PipelineStatus = PipelineStatus.CREATED
    trigger_type: TriggerType = TriggerType.MANUAL
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    task_instances: Dict[str, TaskInstance] = field(default_factory=dict)
    triggered_by: str = "system"
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DAGNode:
    """A node in the pipeline DAG."""

    task_id: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    level: int = 0


@dataclass
class ScheduleConfig:
    """Pipeline schedule configuration."""

    schedule_id: str
    pipeline_id: str
    cron_expression: str
    enabled: bool = True
    timezone: str = "UTC"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    catchup: bool = False
    max_concurrent_runs: int = 1


@dataclass
class ResourceRequirements:
    """Resource requirements for a task."""

    cpu_cores: float = 1.0
    memory_mb: int = 1024
    gpu_count: int = 0
    storage_mb: int = 100
    timeout_seconds: int = 3600


# ========================
# Core Classes
# ========================


class DAGBuilder:
    """Build and validate pipeline DAGs."""

    def __init__(self):
        self._nodes: Dict[str, DAGNode] = {}
        self._validated = False

    def add_task(
        self,
        task_id: str,
        dependencies: Optional[List[str]] = None,
    ) -> DAGBuilder:
        """Add a task to the DAG."""
        node = DAGNode(
            task_id=task_id,
            dependencies=set(dependencies or []),
        )
        self._nodes[task_id] = node

        # Update dependents for parent nodes
        for dep in node.dependencies:
            if dep in self._nodes:
                self._nodes[dep].dependents.add(task_id)

        self._validated = False
        return self

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the DAG for cycles and missing dependencies."""
        errors = []

        # Check for missing dependencies
        for task_id, node in self._nodes.items():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    errors.append(f"Task {task_id} has missing dependency: {dep}")

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for dependent in self._nodes.get(node_id, DAGNode(node_id)).dependents:
                if dependent not in visited:
                    if has_cycle(dependent):
                        return True
                elif dependent in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for task_id in self._nodes:
            if task_id not in visited:
                if has_cycle(task_id):
                    errors.append(f"Cycle detected involving task: {task_id}")
                    break

        self._validated = len(errors) == 0
        return self._validated, errors

    def compute_levels(self) -> Dict[str, int]:
        """Compute execution levels for parallel scheduling."""
        levels: Dict[str, int] = {}

        # Find root nodes (no dependencies)
        roots = [
            task_id for task_id, node in self._nodes.items()
            if not node.dependencies
        ]

        # BFS to assign levels
        queue = deque([(root, 0) for root in roots])
        while queue:
            task_id, level = queue.popleft()
            levels[task_id] = max(levels.get(task_id, 0), level)
            self._nodes[task_id].level = levels[task_id]

            for dependent in self._nodes[task_id].dependents:
                queue.append((dependent, level + 1))

        return levels

    def get_execution_order(self) -> List[List[str]]:
        """Get tasks grouped by execution level."""
        self.compute_levels()

        level_groups: Dict[int, List[str]] = defaultdict(list)
        for task_id, node in self._nodes.items():
            level_groups[node.level].append(task_id)

        return [level_groups[i] for i in sorted(level_groups.keys())]

    def get_ready_tasks(self, completed: Set[str]) -> List[str]:
        """Get tasks ready for execution."""
        ready = []
        for task_id, node in self._nodes.items():
            if task_id in completed:
                continue
            if node.dependencies.issubset(completed):
                ready.append(task_id)
        return ready


class TaskExecutor:
    """Execute pipeline tasks."""

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running_tasks: Dict[str, TaskInstance] = {}
        self._lock = threading.RLock()

    def start(self) -> None:
        """Start the executor."""
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

    def stop(self) -> None:
        """Stop the executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def execute_task(
        self,
        task: TaskDefinition,
        instance: TaskInstance,
        context: Dict[str, Any],
    ) -> TaskInstance:
        """Execute a single task."""
        instance.status = TaskStatus.RUNNING
        instance.start_time = datetime.now()

        try:
            if task.handler is not None:
                # Execute the handler
                result = task.handler(context, task.parameters)
                instance.result = result
                instance.status = TaskStatus.COMPLETED
            else:
                # Simulate task execution
                time.sleep(0.01)
                instance.result = {"success": True}
                instance.status = TaskStatus.COMPLETED

        except Exception as e:
            instance.error = str(e)
            instance.status = TaskStatus.FAILED

        instance.end_time = datetime.now()
        return instance

    def execute_tasks_parallel(
        self,
        tasks: List[Tuple[TaskDefinition, TaskInstance]],
        context: Dict[str, Any],
    ) -> List[TaskInstance]:
        """Execute multiple tasks in parallel."""
        if not self._executor:
            self.start()

        futures = []
        for task, instance in tasks:
            future = self._executor.submit(
                self.execute_task, task, instance, context
            )
            futures.append(future)

        results = []
        for future in as_completed(futures):
            results.append(future.result())

        return results


class PipelineScheduler:
    """Schedule pipeline executions."""

    def __init__(self):
        self._schedules: Dict[str, ScheduleConfig] = {}
        self._next_runs: Dict[str, datetime] = {}
        self._lock = threading.RLock()

    def add_schedule(self, config: ScheduleConfig) -> None:
        """Add a pipeline schedule."""
        with self._lock:
            self._schedules[config.schedule_id] = config
            if config.enabled:
                self._compute_next_run(config.schedule_id)

    def remove_schedule(self, schedule_id: str) -> None:
        """Remove a schedule."""
        with self._lock:
            self._schedules.pop(schedule_id, None)
            self._next_runs.pop(schedule_id, None)

    def _compute_next_run(self, schedule_id: str) -> Optional[datetime]:
        """Compute the next run time for a schedule."""
        config = self._schedules.get(schedule_id)
        if not config or not config.enabled:
            return None

        # Simplified: just add 1 hour from now
        # In real implementation, would parse cron expression
        next_run = datetime.now() + timedelta(hours=1)

        if config.end_date and next_run > config.end_date:
            return None

        self._next_runs[schedule_id] = next_run
        return next_run

    def get_due_schedules(self) -> List[ScheduleConfig]:
        """Get schedules that are due for execution."""
        now = datetime.now()
        due = []

        with self._lock:
            for schedule_id, next_run in list(self._next_runs.items()):
                if next_run <= now:
                    config = self._schedules.get(schedule_id)
                    if config and config.enabled:
                        due.append(config)
                        # Compute next run
                        self._compute_next_run(schedule_id)

        return due

    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule."""
        with self._lock:
            config = self._schedules.get(schedule_id)
            if config:
                config.enabled = True
                self._schedules[schedule_id] = config
                self._compute_next_run(schedule_id)
                return True
        return False

    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule."""
        with self._lock:
            config = self._schedules.get(schedule_id)
            if config:
                config.enabled = False
                self._schedules[schedule_id] = config
                self._next_runs.pop(schedule_id, None)
                return True
        return False


class PipelineOrchestrator:
    """Main orchestrator for pipeline execution."""

    def __init__(self, max_workers: int = 4):
        self._pipelines: Dict[str, PipelineDefinition] = {}
        self._runs: Dict[str, PipelineRun] = {}
        self._executor = TaskExecutor(max_workers=max_workers)
        self._scheduler = PipelineScheduler()
        self._lock = threading.RLock()

    def register_pipeline(self, pipeline: PipelineDefinition) -> None:
        """Register a pipeline definition."""
        with self._lock:
            self._pipelines[pipeline.pipeline_id] = pipeline

            # Add schedule if configured
            if pipeline.schedule:
                schedule = ScheduleConfig(
                    schedule_id=f"schedule-{pipeline.pipeline_id}",
                    pipeline_id=pipeline.pipeline_id,
                    cron_expression=pipeline.schedule,
                )
                self._scheduler.add_schedule(schedule)

    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineDefinition]:
        """Get a pipeline definition."""
        return self._pipelines.get(pipeline_id)

    def list_pipelines(self) -> List[PipelineDefinition]:
        """List all pipelines."""
        return list(self._pipelines.values())

    def create_run(
        self,
        pipeline_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        trigger_type: TriggerType = TriggerType.MANUAL,
        triggered_by: str = "system",
    ) -> PipelineRun:
        """Create a new pipeline run."""
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        run_id = hashlib.md5(
            f"{pipeline_id}:{time.time()}".encode()
        ).hexdigest()[:12]

        run = PipelineRun(
            run_id=run_id,
            pipeline_id=pipeline_id,
            parameters=parameters or {},
            trigger_type=trigger_type,
            triggered_by=triggered_by,
        )

        # Create task instances
        for task in pipeline.tasks:
            instance_id = f"{run_id}-{task.task_id}"
            run.task_instances[task.task_id] = TaskInstance(
                instance_id=instance_id,
                task_id=task.task_id,
                pipeline_run_id=run_id,
            )

        with self._lock:
            self._runs[run_id] = run

        return run

    def execute_run(self, run_id: str) -> PipelineRun:
        """Execute a pipeline run."""
        run = self._runs.get(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        pipeline = self._pipelines.get(run.pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {run.pipeline_id}")

        run.status = PipelineStatus.RUNNING
        run.start_time = datetime.now()

        # Build DAG
        dag = DAGBuilder()
        task_map: Dict[str, TaskDefinition] = {}
        for task in pipeline.tasks:
            dag.add_task(task.task_id, task.dependencies)
            task_map[task.task_id] = task

        valid, errors = dag.validate()
        if not valid:
            run.status = PipelineStatus.FAILED
            run.end_time = datetime.now()
            return run

        # Execute tasks by level
        completed: Set[str] = set()
        context: Dict[str, Any] = {
            "run_id": run_id,
            "pipeline_id": run.pipeline_id,
            "parameters": run.parameters,
            "results": {},
        }

        try:
            execution_order = dag.get_execution_order()

            for level_tasks in execution_order:
                # Prepare tasks for this level
                tasks_to_run = []
                for task_id in level_tasks:
                    task = task_map[task_id]
                    instance = run.task_instances[task_id]
                    instance.status = TaskStatus.QUEUED
                    tasks_to_run.append((task, instance))

                # Execute level
                if pipeline.execution_mode == ExecutionMode.SEQUENTIAL:
                    for task, instance in tasks_to_run:
                        result = self._executor.execute_task(task, instance, context)
                        if result.status == TaskStatus.COMPLETED:
                            completed.add(task.task_id)
                            context["results"][task.task_id] = result.result
                        else:
                            # Handle failure
                            if not self._handle_failure(task, instance):
                                run.status = PipelineStatus.FAILED
                                break
                else:
                    results = self._executor.execute_tasks_parallel(tasks_to_run, context)
                    for result in results:
                        if result.status == TaskStatus.COMPLETED:
                            completed.add(result.task_id)
                            context["results"][result.task_id] = result.result
                        else:
                            run.status = PipelineStatus.FAILED
                            break

                if run.status == PipelineStatus.FAILED:
                    break

            if run.status != PipelineStatus.FAILED:
                run.status = PipelineStatus.COMPLETED

        except Exception as e:
            run.status = PipelineStatus.FAILED

        run.end_time = datetime.now()

        with self._lock:
            self._runs[run_id] = run

        return run

    def _handle_failure(
        self,
        task: TaskDefinition,
        instance: TaskInstance,
    ) -> bool:
        """Handle task failure with retry logic."""
        if task.retry_policy == RetryPolicy.NONE:
            return False

        if instance.attempt_number >= task.max_retries:
            return False

        instance.attempt_number += 1
        instance.status = TaskStatus.RETRY

        # Apply retry delay
        if task.retry_policy == RetryPolicy.EXPONENTIAL:
            delay = 2 ** instance.attempt_number
        elif task.retry_policy == RetryPolicy.LINEAR:
            delay = instance.attempt_number * 5
        else:
            delay = 5

        time.sleep(min(delay, 60) * 0.001)  # Scale down for testing
        return True

    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get a pipeline run."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        pipeline_id: Optional[str] = None,
        status: Optional[PipelineStatus] = None,
    ) -> List[PipelineRun]:
        """List pipeline runs."""
        runs = list(self._runs.values())
        if pipeline_id:
            runs = [r for r in runs if r.pipeline_id == pipeline_id]
        if status:
            runs = [r for r in runs if r.status == status]
        return sorted(runs, key=lambda r: r.start_time or datetime.min, reverse=True)

    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running pipeline."""
        run = self._runs.get(run_id)
        if run is None:
            return False

        if run.status != PipelineStatus.RUNNING:
            return False

        run.status = PipelineStatus.CANCELLED
        run.end_time = datetime.now()

        # Cancel running tasks
        for instance in run.task_instances.values():
            if instance.status == TaskStatus.RUNNING:
                instance.status = TaskStatus.CANCELLED

        with self._lock:
            self._runs[run_id] = run

        return True

    def pause_run(self, run_id: str) -> bool:
        """Pause a running pipeline."""
        run = self._runs.get(run_id)
        if run and run.status == PipelineStatus.RUNNING:
            run.status = PipelineStatus.PAUSED
            self._runs[run_id] = run
            return True
        return False

    def resume_run(self, run_id: str) -> bool:
        """Resume a paused pipeline."""
        run = self._runs.get(run_id)
        if run and run.status == PipelineStatus.PAUSED:
            run.status = PipelineStatus.RUNNING
            self._runs[run_id] = run
            # Resume execution would continue here
            return True
        return False


class PipelineTemplate:
    """Reusable pipeline templates."""

    def __init__(self, template_id: str, name: str):
        self._template_id = template_id
        self._name = name
        self._task_templates: List[TaskDefinition] = []
        self._parameters: Dict[str, Any] = {}

    def add_task_template(self, task: TaskDefinition) -> PipelineTemplate:
        """Add a task template."""
        self._task_templates.append(task)
        return self

    def set_parameters(self, parameters: Dict[str, Any]) -> PipelineTemplate:
        """Set template parameters."""
        self._parameters = parameters
        return self

    def instantiate(
        self,
        pipeline_id: str,
        name: str,
        parameter_values: Optional[Dict[str, Any]] = None,
    ) -> PipelineDefinition:
        """Instantiate a pipeline from the template."""
        tasks = []
        params = {**self._parameters, **(parameter_values or {})}

        for template in self._task_templates:
            # Apply parameter substitution
            task_params = {}
            for key, value in template.parameters.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    param_name = value[2:-1]
                    task_params[key] = params.get(param_name, value)
                else:
                    task_params[key] = value

            tasks.append(TaskDefinition(
                task_id=template.task_id,
                name=template.name,
                task_type=template.task_type,
                handler=template.handler,
                parameters=task_params,
                dependencies=template.dependencies,
                retry_policy=template.retry_policy,
                max_retries=template.max_retries,
            ))

        return PipelineDefinition(
            pipeline_id=pipeline_id,
            name=name,
            tasks=tasks,
            parameters=params,
        )


# ========================
# Vision Provider
# ========================


class PipelineOrchestratorVisionProvider(VisionProvider):
    """Vision provider for pipeline orchestration capabilities."""

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers
        self._orchestrator: Optional[PipelineOrchestrator] = None

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Pipeline Orchestrator Vision Provider",
            version="1.0.0",
            description="ML pipeline orchestration with DAG execution",
            capabilities=[
                "dag_execution",
                "task_scheduling",
                "parallel_execution",
                "retry_handling",
                "pipeline_templating",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._orchestrator = PipelineOrchestrator(max_workers=self._max_workers)
        self._orchestrator._executor.start()

    def shutdown(self) -> None:
        """Shutdown the provider."""
        if self._orchestrator:
            self._orchestrator._executor.stop()
        self._orchestrator = None

    def get_orchestrator(self) -> PipelineOrchestrator:
        """Get the pipeline orchestrator."""
        if self._orchestrator is None:
            self.initialize()
        return self._orchestrator


# ========================
# Factory Functions
# ========================


def create_pipeline_orchestrator(max_workers: int = 4) -> PipelineOrchestrator:
    """Create a pipeline orchestrator."""
    return PipelineOrchestrator(max_workers=max_workers)


def create_pipeline_definition(
    pipeline_id: str,
    name: str,
    description: str = "",
    tasks: Optional[List[TaskDefinition]] = None,
    execution_mode: ExecutionMode = ExecutionMode.HYBRID,
) -> PipelineDefinition:
    """Create a pipeline definition."""
    return PipelineDefinition(
        pipeline_id=pipeline_id,
        name=name,
        description=description,
        tasks=tasks or [],
        execution_mode=execution_mode,
    )


def create_task_definition(
    task_id: str,
    name: str,
    handler: Optional[Callable] = None,
    dependencies: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    retry_policy: RetryPolicy = RetryPolicy.NONE,
) -> TaskDefinition:
    """Create a task definition."""
    return TaskDefinition(
        task_id=task_id,
        name=name,
        handler=handler,
        dependencies=dependencies or [],
        parameters=parameters or {},
        retry_policy=retry_policy,
    )


def create_dag_builder() -> DAGBuilder:
    """Create a DAG builder."""
    return DAGBuilder()


def create_task_executor(max_workers: int = 4) -> TaskExecutor:
    """Create a task executor."""
    return TaskExecutor(max_workers=max_workers)


def create_pipeline_scheduler() -> PipelineScheduler:
    """Create a pipeline scheduler."""
    return PipelineScheduler()


def create_schedule_config(
    schedule_id: str,
    pipeline_id: str,
    cron_expression: str,
    enabled: bool = True,
) -> ScheduleConfig:
    """Create a schedule configuration."""
    return ScheduleConfig(
        schedule_id=schedule_id,
        pipeline_id=pipeline_id,
        cron_expression=cron_expression,
        enabled=enabled,
    )


def create_pipeline_template(template_id: str, name: str) -> PipelineTemplate:
    """Create a pipeline template."""
    return PipelineTemplate(template_id=template_id, name=name)


def create_pipeline_orchestrator_provider(
    max_workers: int = 4,
) -> PipelineOrchestratorVisionProvider:
    """Create a pipeline orchestrator vision provider."""
    return PipelineOrchestratorVisionProvider(max_workers=max_workers)
