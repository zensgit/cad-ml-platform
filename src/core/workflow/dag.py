"""Workflow DAG Execution Engine.

Provides directed acyclic graph (DAG) based workflow execution:
- Node and edge definitions
- Dependency resolution
- Parallel execution
- State persistence
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.core.workflow.tasks import (
    Task,
    TaskResult,
    TaskStatus,
    FunctionTask,
    TaskConfig,
)

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class NodeType(str, Enum):
    """DAG node types."""
    TASK = "task"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    SUBWORKFLOW = "subworkflow"


@dataclass
class DAGNode:
    """A node in the workflow DAG."""
    node_id: str
    task: Task
    node_type: NodeType = NodeType.TASK
    dependencies: Set[str] = field(default_factory=set)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.node_id)


@dataclass
class DAGEdge:
    """An edge connecting two nodes."""
    source_id: str
    target_id: str
    condition: Optional[Callable[[Any], bool]] = None
    transform: Optional[Callable[[Any], Any]] = None


@dataclass
class WorkflowContext:
    """Execution context shared across workflow."""
    workflow_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, TaskResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    context: WorkflowContext
    error: Optional[str] = None
    duration_ms: Optional[float] = None


class DAG:
    """Directed Acyclic Graph for workflow definition."""

    def __init__(self, dag_id: Optional[str] = None, name: str = "Workflow"):
        self.dag_id = dag_id or str(uuid.uuid4())
        self.name = name
        self._nodes: Dict[str, DAGNode] = {}
        self._edges: List[DAGEdge] = []
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_node(
        self,
        task: Task,
        node_id: Optional[str] = None,
        node_type: NodeType = NodeType.TASK,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        **metadata: Any,
    ) -> str:
        """Add a task node to the DAG.

        Args:
            task: The task to execute.
            node_id: Optional node identifier.
            node_type: Type of node.
            condition: Optional condition for execution.
            **metadata: Additional metadata.

        Returns:
            Node ID.
        """
        node_id = node_id or task.task_id
        node = DAGNode(
            node_id=node_id,
            task=task,
            node_type=node_type,
            condition=condition,
            metadata=metadata,
        )
        self._nodes[node_id] = node
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        condition: Optional[Callable[[Any], bool]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            condition: Optional condition for edge traversal.
            transform: Optional data transformation function.
        """
        if source_id not in self._nodes:
            raise ValueError(f"Source node {source_id} not found")
        if target_id not in self._nodes:
            raise ValueError(f"Target node {target_id} not found")

        edge = DAGEdge(
            source_id=source_id,
            target_id=target_id,
            condition=condition,
            transform=transform,
        )
        self._edges.append(edge)
        self._adjacency[source_id].add(target_id)
        self._reverse_adjacency[target_id].add(source_id)
        self._nodes[target_id].dependencies.add(source_id)

    def add_dependency(self, dependent_id: str, dependency_id: str) -> None:
        """Add a dependency between nodes.

        Args:
            dependent_id: Node that depends on another.
            dependency_id: Node that must complete first.
        """
        self.add_edge(dependency_id, dependent_id)

    def get_node(self, node_id: str) -> Optional[DAGNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes(self) -> List[DAGNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def get_edges(self) -> List[DAGEdge]:
        """Get all edges."""
        return self._edges.copy()

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get direct dependencies of a node."""
        return self._reverse_adjacency.get(node_id, set())

    def get_dependents(self, node_id: str) -> Set[str]:
        """Get nodes that depend on this node."""
        return self._adjacency.get(node_id, set())

    def get_root_nodes(self) -> List[str]:
        """Get nodes with no dependencies."""
        return [
            node_id for node_id, node in self._nodes.items()
            if not node.dependencies
        ]

    def get_leaf_nodes(self) -> List[str]:
        """Get nodes with no dependents."""
        return [
            node_id for node_id in self._nodes
            if not self._adjacency.get(node_id)
        ]

    def topological_sort(self) -> List[str]:
        """Get topologically sorted node IDs.

        Returns:
            List of node IDs in execution order.

        Raises:
            ValueError: If graph contains cycles.
        """
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                in_degree[node.node_id] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for dependent in self._adjacency.get(node_id, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            raise ValueError("DAG contains cycles")

        return result

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the DAG structure.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors = []

        # Check for cycles
        try:
            self.topological_sort()
        except ValueError:
            errors.append("DAG contains cycles")

        # Check for orphan nodes (no path to them)
        roots = set(self.get_root_nodes())
        reachable = set()

        def dfs(node_id: str):
            if node_id in reachable:
                return
            reachable.add(node_id)
            for dep in self._adjacency.get(node_id, set()):
                dfs(dep)

        for root in roots:
            dfs(root)

        orphans = set(self._nodes.keys()) - reachable
        if orphans:
            errors.append(f"Unreachable nodes: {orphans}")

        return len(errors) == 0, errors


class DAGExecutor:
    """Executes a DAG workflow."""

    def __init__(
        self,
        dag: DAG,
        max_parallel: int = 10,
    ):
        self.dag = dag
        self.max_parallel = max_parallel
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def execute(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> WorkflowResult:
        """Execute the workflow DAG.

        Args:
            inputs: Initial input data.
            workflow_id: Optional workflow instance ID.

        Returns:
            WorkflowResult with execution details.
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        self._semaphore = asyncio.Semaphore(self.max_parallel)

        context = WorkflowContext(
            workflow_id=workflow_id,
            inputs=inputs or {},
            started_at=datetime.utcnow(),
        )

        # Validate DAG
        is_valid, errors = self.dag.validate()
        if not is_valid:
            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                context=context,
                error=f"Invalid DAG: {', '.join(errors)}",
            )

        try:
            await self._execute_dag(context)
            context.completed_at = datetime.utcnow()

            # Check if any tasks failed
            failed_tasks = [
                node_id for node_id, result in context.results.items()
                if result.status == TaskStatus.FAILED
            ]

            if failed_tasks:
                duration_ms = (context.completed_at - context.started_at).total_seconds() * 1000
                return WorkflowResult(
                    workflow_id=workflow_id,
                    status=WorkflowStatus.FAILED,
                    context=context,
                    error=f"Tasks failed: {failed_tasks}",
                    duration_ms=duration_ms,
                )

            duration_ms = (context.completed_at - context.started_at).total_seconds() * 1000
            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.COMPLETED,
                context=context,
                duration_ms=duration_ms,
            )

        except asyncio.CancelledError:
            context.completed_at = datetime.utcnow()
            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.CANCELLED,
                context=context,
                error="Workflow was cancelled",
            )

        except Exception as e:
            context.completed_at = datetime.utcnow()
            logger.exception(f"Workflow {workflow_id} failed: {e}")
            return WorkflowResult(
                workflow_id=workflow_id,
                status=WorkflowStatus.FAILED,
                context=context,
                error=str(e),
            )

    async def _execute_dag(self, context: WorkflowContext) -> None:
        """Execute DAG nodes in dependency order."""
        completed: Set[str] = set()
        running: Set[str] = set()
        pending_tasks: Dict[str, asyncio.Task] = {}

        # Get initial nodes (no dependencies)
        ready = set(self.dag.get_root_nodes())

        while ready or running:
            # Start ready nodes
            for node_id in list(ready):
                node = self.dag.get_node(node_id)
                if not node:
                    continue

                # Check condition
                if node.condition and not node.condition(context.outputs):
                    logger.info(f"Skipping node {node_id}: condition not met")
                    context.results[node_id] = TaskResult(
                        task_id=node.task.task_id,
                        status=TaskStatus.SKIPPED,
                    )
                    completed.add(node_id)
                    ready.remove(node_id)
                    continue

                # Get input data
                input_data = self._get_node_input(node_id, context)

                # Start task
                async_task = asyncio.create_task(
                    self._execute_node(node, input_data, context)
                )
                pending_tasks[node_id] = async_task
                running.add(node_id)
                ready.remove(node_id)

            if not running:
                break

            # Wait for any task to complete
            done, _ = await asyncio.wait(
                pending_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process completed tasks
            for done_task in done:
                # Find node ID for this task
                done_node_id = None
                for node_id, task in pending_tasks.items():
                    if task == done_task:
                        done_node_id = node_id
                        break

                if done_node_id:
                    running.remove(done_node_id)
                    completed.add(done_node_id)
                    del pending_tasks[done_node_id]

                    # Check for newly ready nodes
                    for dependent_id in self.dag.get_dependents(done_node_id):
                        deps = self.dag.get_dependencies(dependent_id)
                        if deps.issubset(completed):
                            ready.add(dependent_id)

    async def _execute_node(
        self,
        node: DAGNode,
        input_data: Any,
        context: WorkflowContext,
    ) -> None:
        """Execute a single node."""
        async with self._semaphore:
            logger.info(f"Executing node {node.node_id}")
            result = await node.task.run(input_data)
            context.results[node.node_id] = result

            if result.status == TaskStatus.COMPLETED:
                context.outputs[node.node_id] = result.output
                logger.info(f"Node {node.node_id} completed")
            else:
                logger.warning(
                    f"Node {node.node_id} failed: {result.error}"
                )

    def _get_node_input(
        self,
        node_id: str,
        context: WorkflowContext,
    ) -> Any:
        """Get input data for a node from context and dependencies."""
        # Find edges that target this node
        incoming_edges = [
            e for e in self.dag.get_edges()
            if e.target_id == node_id
        ]

        if not incoming_edges:
            # Root node - use workflow inputs
            return context.inputs

        # Collect outputs from dependencies
        inputs = {}
        for edge in incoming_edges:
            output = context.outputs.get(edge.source_id)

            # Apply transform if specified
            if edge.transform and output is not None:
                output = edge.transform(output)

            inputs[edge.source_id] = output

        # If single input, return it directly
        if len(inputs) == 1:
            return list(inputs.values())[0]

        return inputs


class DAGBuilder:
    """Fluent builder for DAG construction."""

    def __init__(self, name: str = "Workflow"):
        self._dag = DAG(name=name)
        self._last_node_id: Optional[str] = None

    def add_task(
        self,
        task: Task,
        node_id: Optional[str] = None,
        **metadata: Any,
    ) -> "DAGBuilder":
        """Add a task node."""
        node_id = self._dag.add_node(task, node_id, **metadata)
        self._last_node_id = node_id
        return self

    def add_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        node_id: Optional[str] = None,
        config: Optional[TaskConfig] = None,
    ) -> "DAGBuilder":
        """Add a function as a task node."""
        task = FunctionTask(func, name=name, config=config)
        return self.add_task(task, node_id)

    def depends_on(self, *dependency_ids: str) -> "DAGBuilder":
        """Add dependencies to the last added node."""
        if not self._last_node_id:
            raise ValueError("No node to add dependencies to")

        for dep_id in dependency_ids:
            self._dag.add_dependency(self._last_node_id, dep_id)

        return self

    def then(self, task: Task, node_id: Optional[str] = None) -> "DAGBuilder":
        """Add a task that depends on the last added node."""
        prev_node_id = self._last_node_id
        self.add_task(task, node_id)

        if prev_node_id:
            self._dag.add_dependency(self._last_node_id, prev_node_id)

        return self

    def parallel(self, *tasks: Task) -> "DAGBuilder":
        """Add multiple tasks in parallel, all depending on last node."""
        prev_node_id = self._last_node_id

        for task in tasks:
            self.add_task(task)
            if prev_node_id:
                self._dag.add_dependency(self._last_node_id, prev_node_id)

        return self

    def build(self) -> DAG:
        """Build and return the DAG."""
        return self._dag
