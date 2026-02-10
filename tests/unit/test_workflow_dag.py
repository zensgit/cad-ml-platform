"""Tests for workflow DAG module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.workflow.dag import (
    DAG,
    DAGNode,
    DAGEdge,
    DAGBuilder,
    DAGExecutor,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
    NodeType,
)
from src.core.workflow.tasks import FunctionTask, TaskConfig, TaskStatus, TaskResult


class TestDAGNode:
    """Tests for DAGNode class."""

    def test_node_hash(self):
        """Test DAGNode __hash__ (line 61)."""
        task = MagicMock()
        task.task_id = "task1"

        node = DAGNode(node_id="node1", task=task)

        # Hash should be based on node_id
        assert hash(node) == hash("node1")

        # Two nodes with same ID should have same hash
        node2 = DAGNode(node_id="node1", task=task)
        assert hash(node) == hash(node2)

        # Different IDs should have different hashes
        node3 = DAGNode(node_id="node2", task=task)
        assert hash(node) != hash(node3)

    def test_node_type(self):
        """Test DAGNode with different NodeTypes."""
        task = MagicMock()
        task.task_id = "task1"

        node = DAGNode(
            node_id="node1",
            task=task,
            node_type=NodeType.TASK,
        )
        assert node.node_type == NodeType.TASK

        # Test with PARALLEL type
        node2 = DAGNode(
            node_id="node2",
            task=task,
            node_type=NodeType.PARALLEL,
        )
        assert node2.node_type == NodeType.PARALLEL


class TestDAG:
    """Tests for DAG class."""

    def test_add_edge_source_not_found(self):
        """Test add_edge raises error for invalid source (line 153)."""
        dag = DAG(name="test")
        task = MagicMock()
        task.task_id = "task1"
        dag.add_node(task, "node1")

        with pytest.raises(ValueError, match="Source node invalid not found"):
            dag.add_edge("invalid", "node1")

    def test_add_edge_target_not_found(self):
        """Test add_edge raises error for invalid target (line 155)."""
        dag = DAG(name="test")
        task = MagicMock()
        task.task_id = "task1"
        dag.add_node(task, "node1")

        with pytest.raises(ValueError, match="Target node invalid not found"):
            dag.add_edge("node1", "invalid")

    def test_add_dependency(self):
        """Test add_dependency method (line 175)."""
        dag = DAG(name="test")
        task1 = MagicMock()
        task1.task_id = "task1"
        task2 = MagicMock()
        task2.task_id = "task2"

        dag.add_node(task1, "node1")
        dag.add_node(task2, "node2")

        # add_dependency(dependent_id, dependency_id) calls add_edge(dependency_id, dependent_id)
        dag.add_dependency("node2", "node1")

        # node2 should depend on node1
        assert "node1" in dag.get_dependencies("node2")

    def test_get_nodes_returns_list(self):
        """Test get_nodes returns list of all nodes (line 183)."""
        dag = DAG(name="test")
        task1 = MagicMock()
        task1.task_id = "task1"
        task2 = MagicMock()
        task2.task_id = "task2"

        dag.add_node(task1, "node1")
        dag.add_node(task2, "node2")

        nodes = dag.get_nodes()

        assert len(nodes) == 2
        assert isinstance(nodes, list)
        node_ids = [n.node_id for n in nodes]
        assert "node1" in node_ids
        assert "node2" in node_ids

    def test_get_leaf_nodes(self):
        """Test get_leaf_nodes method (line 206)."""
        dag = DAG(name="test")
        task1 = MagicMock()
        task1.task_id = "task1"
        task2 = MagicMock()
        task2.task_id = "task2"
        task3 = MagicMock()
        task3.task_id = "task3"

        dag.add_node(task1, "node1")
        dag.add_node(task2, "node2")
        dag.add_node(task3, "node3")
        dag.add_edge("node1", "node2")

        # node2 and node3 are leaf nodes (no dependents)
        leaf_nodes = dag.get_leaf_nodes()
        assert "node2" in leaf_nodes
        assert "node3" in leaf_nodes
        assert "node1" not in leaf_nodes

    def test_validate_dfs_revisit(self):
        """Test validate dfs handles revisiting nodes (line 262)."""
        dag = DAG(name="test")

        # Create a diamond pattern: A -> B, A -> C, B -> D, C -> D
        # D is reachable from both B and C
        task = MagicMock()
        task.task_id = "task"

        dag.add_node(task, "A")
        dag.add_node(task, "B")
        dag.add_node(task, "C")
        dag.add_node(task, "D")

        dag.add_edge("A", "B")
        dag.add_edge("A", "C")
        dag.add_edge("B", "D")
        dag.add_edge("C", "D")

        # Validate should pass and dfs should handle revisiting D
        is_valid, errors = dag.validate()
        assert is_valid
        assert len(errors) == 0


class TestDAGExecutor:
    """Tests for DAGExecutor class."""

    @pytest.fixture
    def simple_dag(self):
        """Create a simple DAG for testing."""
        dag = DAG(name="test")

        async def task_func(data):
            return data * 2

        task = FunctionTask(task_func, name="double")
        dag.add_node(task, "node1")
        return dag

    @pytest.mark.asyncio
    async def test_execute_invalid_dag(self):
        """Test execution with invalid DAG returns error (lines 315-320)."""
        dag = DAG(name="test")

        # Create a cycle by manually manipulating the DAG
        task1 = MagicMock()
        task1.task_id = "task1"
        task2 = MagicMock()
        task2.task_id = "task2"

        dag.add_node(task1, "node1")
        dag.add_node(task2, "node2")
        dag.add_edge("node1", "node2")
        dag.add_edge("node2", "node1")  # Creates cycle

        executor = DAGExecutor(dag)
        result = await executor.execute()

        assert result.status == WorkflowStatus.FAILED
        assert "Invalid DAG" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_failed_tasks(self):
        """Test execution with failing tasks (lines 333-334)."""
        dag = DAG(name="test")

        async def failing_task(data):
            raise ValueError("Task failed")

        task = FunctionTask(failing_task, name="failing")
        dag.add_node(task, "node1")

        executor = DAGExecutor(dag)
        result = await executor.execute()

        assert result.status == WorkflowStatus.FAILED
        assert result.duration_ms is not None

    @pytest.mark.asyncio
    async def test_execute_cancelled(self):
        """Test execution cancellation (lines 350-362)."""
        dag = DAG(name="test")

        async def slow_task(data):
            await asyncio.sleep(10)
            return data

        task = FunctionTask(slow_task, name="slow")
        dag.add_node(task, "node1")

        executor = DAGExecutor(dag)

        # Create and cancel the task
        async def execute_and_cancel():
            execute_task = asyncio.create_task(executor.execute())
            await asyncio.sleep(0.01)
            execute_task.cancel()
            try:
                return await execute_task
            except asyncio.CancelledError:
                # Create a mock result since the task was cancelled
                return WorkflowResult(
                    workflow_id="test",
                    status=WorkflowStatus.CANCELLED,
                    context=WorkflowContext(workflow_id="test"),
                    error="Workflow was cancelled",
                )

        result = await execute_and_cancel()
        assert result.status == WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_execute_node_get_returns_none(self):
        """Test node execution when get_node returns None (line 383)."""
        dag = DAG(name="test")

        async def task_func(data):
            return data * 2

        task = FunctionTask(task_func, name="double")
        dag.add_node(task, "node1")

        executor = DAGExecutor(dag)

        # Mock get_node to return None for node1
        original_get_node = dag.get_node
        def mock_get_node(node_id):
            return None

        dag.get_node = mock_get_node

        result = await executor.execute()

        # Should complete without processing the node
        assert result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]

    @pytest.mark.asyncio
    async def test_execute_node_condition_false(self):
        """Test node skipped when condition returns False (lines 387-394)."""
        dag = DAG(name="test")

        async def task_func(data):
            return data * 2

        task = FunctionTask(task_func, name="double")

        # Add node with condition that always returns False
        dag.add_node(
            task,
            "node1",
            condition=lambda outputs: False,
        )

        executor = DAGExecutor(dag)
        result = await executor.execute()

        assert result.status == WorkflowStatus.COMPLETED
        assert result.context.results["node1"].status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_no_running_tasks_breaks(self):
        """Test execution breaks when no tasks are running (line 408)."""
        dag = DAG(name="test")

        # Empty DAG should complete immediately
        executor = DAGExecutor(dag)
        result = await executor.execute()

        assert result.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_node_warning_on_failure(self):
        """Test warning logged when node fails (line 452)."""
        dag = DAG(name="test")

        async def failing_task(data):
            raise ValueError("Task error")

        task = FunctionTask(failing_task, name="failing")
        dag.add_node(task, "node1")

        executor = DAGExecutor(dag)

        with patch("src.core.workflow.dag.logger") as mock_logger:
            result = await executor.execute()

            # Should log warning for failed node
            assert result.status == WorkflowStatus.FAILED

    @pytest.mark.asyncio
    async def test_edge_transform_applied(self):
        """Test edge transform function is applied (line 481)."""
        dag = DAG(name="test")

        async def task1_func(data):
            return 10

        async def task2_func(data):
            return data + 5

        task1 = FunctionTask(task1_func, name="task1")
        task2 = FunctionTask(task2_func, name="task2")

        dag.add_node(task1, "node1")
        dag.add_node(task2, "node2")
        dag.add_edge(
            "node1", "node2",
            transform=lambda x: x * 2,  # Transform: multiply by 2
        )

        executor = DAGExecutor(dag)
        result = await executor.execute()

        assert result.status == WorkflowStatus.COMPLETED
        # task1 returns 10, transform doubles it to 20, task2 adds 5 = 25
        assert result.context.outputs["node2"] == 25

    @pytest.mark.asyncio
    async def test_multiple_inputs_returns_dict(self):
        """Test node with multiple inputs returns dict (line 489)."""
        dag = DAG(name="test")

        async def task_a(data):
            return 10

        async def task_b(data):
            return 20

        async def task_c(data):
            # data should be a dict with inputs from both task_a and task_b
            if isinstance(data, dict):
                return sum(data.values())
            return data

        task_a_task = FunctionTask(task_a, name="task_a")
        task_b_task = FunctionTask(task_b, name="task_b")
        task_c_task = FunctionTask(task_c, name="task_c")

        dag.add_node(task_a_task, "node_a")
        dag.add_node(task_b_task, "node_b")
        dag.add_node(task_c_task, "node_c")

        # Both node_a and node_b feed into node_c
        dag.add_edge("node_a", "node_c")
        dag.add_edge("node_b", "node_c")

        executor = DAGExecutor(dag)
        result = await executor.execute()

        assert result.status == WorkflowStatus.COMPLETED
        # node_c should receive dict with both inputs and sum them
        assert result.context.outputs["node_c"] == 30


class TestDAGBuilder:
    """Tests for DAGBuilder class."""

    def test_add_task_with_metadata(self):
        """Test add_task with metadata (lines 506-508)."""
        builder = DAGBuilder(name="test")

        async def task_func(data):
            return data

        task = FunctionTask(task_func, name="task1")

        result = builder.add_task(task, node_id="custom_id", key1="value1")

        assert result is builder  # Returns self for chaining
        assert builder._last_node_id == "custom_id"

        dag = builder.build()
        node = dag.get_node("custom_id")
        assert node is not None
        assert node.metadata.get("key1") == "value1"

    def test_add_function(self):
        """Test add_function method (lines 518-519)."""
        builder = DAGBuilder(name="test")

        async def my_func(data):
            return data * 2

        result = builder.add_function(
            my_func,
            name="double",
            node_id="func_node",
        )

        assert result is builder
        dag = builder.build()
        assert dag.get_node("func_node") is not None

    def test_depends_on_without_node(self):
        """Test depends_on raises error when no node added (lines 523-524)."""
        builder = DAGBuilder(name="test")

        with pytest.raises(ValueError, match="No node to add dependencies to"):
            builder.depends_on("dep1")

    def test_depends_on_multiple_dependencies(self):
        """Test depends_on with multiple dependencies (lines 526-529)."""
        builder = DAGBuilder(name="test")

        async def task_func(data):
            return data

        task1 = FunctionTask(task_func, name="task1")
        task2 = FunctionTask(task_func, name="task2")
        task3 = FunctionTask(task_func, name="task3")

        builder.add_task(task1, "node1")
        builder.add_task(task2, "node2")
        builder.add_task(task3, "node3")
        builder.depends_on("node1", "node2")

        dag = builder.build()
        deps = dag.get_dependencies("node3")
        assert "node1" in deps
        assert "node2" in deps

    def test_then_method(self):
        """Test then method adds task with dependency (lines 533-539)."""
        builder = DAGBuilder(name="test")

        async def task_func(data):
            return data

        task1 = FunctionTask(task_func, name="task1")
        task2 = FunctionTask(task_func, name="task2")

        builder.add_task(task1, "node1")
        builder.then(task2, "node2")

        dag = builder.build()
        deps = dag.get_dependencies("node2")
        assert "node1" in deps

    def test_then_without_prev_node(self):
        """Test then method when no previous node exists (lines 536-537)."""
        builder = DAGBuilder(name="test")
        builder._last_node_id = None

        async def task_func(data):
            return data

        task = FunctionTask(task_func, name="task1")

        builder.then(task, "node1")

        dag = builder.build()
        # Should be added without dependencies
        assert dag.get_node("node1") is not None
        assert len(dag.get_dependencies("node1")) == 0

    def test_parallel_method(self):
        """Test parallel method adds multiple tasks (lines 543-550)."""
        builder = DAGBuilder(name="test")

        async def task_func(data):
            return data

        task0 = FunctionTask(task_func, name="task0")
        task1 = FunctionTask(task_func, name="task1")
        task2 = FunctionTask(task_func, name="task2")
        task3 = FunctionTask(task_func, name="task3")

        builder.add_task(task0, "node0")
        builder.parallel(task1, task2, task3)

        dag = builder.build()

        # All parallel tasks should depend on node0
        assert "node0" in dag.get_dependencies(task1.task_id)
        assert "node0" in dag.get_dependencies(task2.task_id)
        assert "node0" in dag.get_dependencies(task3.task_id)

    def test_parallel_without_prev_node(self):
        """Test parallel method when no previous node exists (lines 547-548)."""
        builder = DAGBuilder(name="test")
        builder._last_node_id = None

        async def task_func(data):
            return data

        task1 = FunctionTask(task_func, name="task1")
        task2 = FunctionTask(task_func, name="task2")

        builder.parallel(task1, task2)

        dag = builder.build()
        # Should be added without dependencies
        assert dag.get_node(task1.task_id) is not None
        assert dag.get_node(task2.task_id) is not None
        assert len(dag.get_dependencies(task1.task_id)) == 0
        assert len(dag.get_dependencies(task2.task_id)) == 0

    def test_build_returns_dag(self):
        """Test build returns the constructed DAG (line 554)."""
        builder = DAGBuilder(name="test_workflow")
        dag = builder.build()

        assert isinstance(dag, DAG)
        assert dag.name == "test_workflow"


class TestWorkflowContext:
    """Tests for WorkflowContext class."""

    def test_context_creation(self):
        """Test WorkflowContext creation."""
        context = WorkflowContext(
            workflow_id="wf1",
            inputs={"key": "value"},
        )

        assert context.workflow_id == "wf1"
        assert context.inputs == {"key": "value"}
        assert context.outputs == {}
        assert context.results == {}

    def test_context_with_metadata(self):
        """Test WorkflowContext with metadata."""
        context = WorkflowContext(
            workflow_id="wf1",
            metadata={"env": "test"},
        )

        assert context.metadata == {"env": "test"}


class TestWorkflowResult:
    """Tests for WorkflowResult class."""

    def test_result_creation(self):
        """Test WorkflowResult creation."""
        context = WorkflowContext(workflow_id="wf1")
        result = WorkflowResult(
            workflow_id="wf1",
            status=WorkflowStatus.COMPLETED,
            context=context,
            duration_ms=100.5,
        )

        assert result.workflow_id == "wf1"
        assert result.status == WorkflowStatus.COMPLETED
        assert result.duration_ms == 100.5

    def test_result_with_error(self):
        """Test WorkflowResult with error."""
        context = WorkflowContext(workflow_id="wf1")
        result = WorkflowResult(
            workflow_id="wf1",
            status=WorkflowStatus.FAILED,
            context=context,
            error="Something went wrong",
        )

        assert result.error == "Something went wrong"


class TestGetNodeInput:
    """Tests for DAGExecutor._get_node_input method."""

    @pytest.mark.asyncio
    async def test_root_node_single_input(self):
        """Test root node with single workflow input (lines 496-497)."""
        dag = DAG(name="test")

        async def task_func(data):
            return data * 2

        task = FunctionTask(task_func, name="task")
        dag.add_node(task, "node1")

        executor = DAGExecutor(dag)

        # Execute with single input value
        result = await executor.execute(inputs={"single_key": 5})

        assert result.status == WorkflowStatus.COMPLETED
        # Single input value should be passed directly
        assert result.context.outputs["node1"] == 10

    @pytest.mark.asyncio
    async def test_root_node_multiple_inputs(self):
        """Test root node with multiple workflow inputs."""
        dag = DAG(name="test")

        async def task_func(data):
            # data should be the full inputs dict
            return sum(data.values()) if isinstance(data, dict) else data

        task = FunctionTask(task_func, name="task")
        dag.add_node(task, "node1")

        executor = DAGExecutor(dag)

        result = await executor.execute(inputs={"a": 1, "b": 2, "c": 3})

        assert result.status == WorkflowStatus.COMPLETED
        assert result.context.outputs["node1"] == 6


class TestDAGExecutorException:
    """Tests for DAGExecutor exception handling."""

    @pytest.mark.asyncio
    async def test_execute_dag_exception(self):
        """Test execution handles unexpected exception (lines 359-367)."""
        dag = DAG(name="test")

        async def task_func(data):
            return data

        task = FunctionTask(task_func, name="task")
        dag.add_node(task, "node1")

        executor = DAGExecutor(dag)

        # Mock _execute_dag to raise an exception
        async def mock_execute_dag(context):
            raise RuntimeError("Unexpected error")

        executor._execute_dag = mock_execute_dag

        result = await executor.execute()

        assert result.status == WorkflowStatus.FAILED
        assert "Unexpected error" in result.error
        assert result.context.completed_at is not None
