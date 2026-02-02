"""Unit tests for P32-P35 Enterprise Features.

P32: Workflow Engine - Task orchestration, DAG execution, state machines
P33: File Storage - S3-compatible storage, presigned URLs, multipart upload
P34: ETL Pipeline - Data ingestion, transformation, scheduling
P35: Metrics & Analytics - Time-series data, aggregation, dashboards
"""

import asyncio
import io
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# P32: Workflow Engine Tests
# ============================================================================

class TestWorkflowTasks:
    """Tests for workflow task primitives."""

    @pytest.mark.asyncio
    async def test_function_task_execution(self):
        """Test FunctionTask executes async function."""
        from src.core.workflow.tasks import FunctionTask, TaskStatus

        async def my_task(data):
            return data * 2

        task = FunctionTask(my_task, name="double")
        result = await task.run(5)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == 10
        assert result.error is None

    @pytest.mark.asyncio
    async def test_function_task_sync(self):
        """Test FunctionTask with sync function."""
        from src.core.workflow.tasks import FunctionTask, TaskStatus

        def my_task(data):
            return data + 1

        task = FunctionTask(my_task, name="increment")
        result = await task.run(10)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == 11

    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task timeout handling."""
        from src.core.workflow.tasks import FunctionTask, TaskConfig, TaskStatus

        async def slow_task(data):
            await asyncio.sleep(5)
            return data

        config = TaskConfig(timeout=0.1)
        task = FunctionTask(slow_task, config=config)
        result = await task.run("test")

        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_task_retry_policy(self):
        """Test task retry on failure."""
        from src.core.workflow.tasks import (
            FunctionTask, TaskConfig, TaskStatus, RetryPolicy
        )

        attempt_count = 0

        async def flaky_task(data):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Failed")
            return "success"

        config = TaskConfig(
            retry_policy=RetryPolicy(max_retries=3, initial_delay=0.01)
        )
        task = FunctionTask(flaky_task, config=config)
        result = await task.run("test")

        assert result.status == TaskStatus.COMPLETED
        assert result.output == "success"
        assert result.retries == 2

    @pytest.mark.asyncio
    async def test_parallel_task(self):
        """Test ParallelTask executes tasks concurrently."""
        from src.core.workflow.tasks import LambdaTask, ParallelTask, TaskStatus

        task1 = LambdaTask(lambda x: x + 1, name="add1")
        task2 = LambdaTask(lambda x: x * 2, name="mul2")
        task3 = LambdaTask(lambda x: x ** 2, name="square")

        parallel = ParallelTask([task1, task2, task3])
        result = await parallel.run(5)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == [6, 10, 25]

    @pytest.mark.asyncio
    async def test_sequential_task(self):
        """Test SequentialTask chains tasks."""
        from src.core.workflow.tasks import LambdaTask, SequentialTask, TaskStatus

        task1 = LambdaTask(lambda x: x + 1, name="add1")
        task2 = LambdaTask(lambda x: x * 2, name="mul2")

        sequential = SequentialTask([task1, task2])
        result = await sequential.run(5)

        assert result.status == TaskStatus.COMPLETED
        assert result.output == 12  # (5 + 1) * 2


class TestWorkflowDAG:
    """Tests for DAG-based workflow execution."""

    @pytest.mark.asyncio
    async def test_dag_simple_execution(self):
        """Test simple DAG execution."""
        from src.core.workflow.dag import DAG, DAGExecutor, WorkflowStatus
        from src.core.workflow.tasks import LambdaTask

        dag = DAG(name="simple")
        task1 = LambdaTask(lambda x: x + 1, name="add1")
        task2 = LambdaTask(lambda x: x * 2, name="mul2")

        dag.add_node(task1, node_id="step1")
        dag.add_node(task2, node_id="step2")
        dag.add_edge("step1", "step2")

        executor = DAGExecutor(dag)
        result = await executor.execute({"value": 5})

        assert result.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dag_parallel_execution(self):
        """Test DAG with parallel branches."""
        from src.core.workflow.dag import DAG, DAGExecutor, WorkflowStatus
        from src.core.workflow.tasks import LambdaTask

        dag = DAG(name="parallel")
        root = LambdaTask(lambda x: x, name="root")
        branch1 = LambdaTask(lambda x: x + 1, name="branch1")
        branch2 = LambdaTask(lambda x: x * 2, name="branch2")
        merge = LambdaTask(lambda x: x, name="merge")

        dag.add_node(root, node_id="root")
        dag.add_node(branch1, node_id="b1")
        dag.add_node(branch2, node_id="b2")
        dag.add_node(merge, node_id="merge")

        dag.add_edge("root", "b1")
        dag.add_edge("root", "b2")
        dag.add_edge("b1", "merge")
        dag.add_edge("b2", "merge")

        executor = DAGExecutor(dag)
        result = await executor.execute({"value": 5})

        assert result.status == WorkflowStatus.COMPLETED

    def test_dag_cycle_detection(self):
        """Test DAG detects cycles."""
        from src.core.workflow.dag import DAG
        from src.core.workflow.tasks import NoOpTask

        dag = DAG(name="cyclic")
        dag.add_node(NoOpTask(), node_id="a")
        dag.add_node(NoOpTask(), node_id="b")
        dag.add_node(NoOpTask(), node_id="c")

        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", "a")  # Creates cycle

        is_valid, errors = dag.validate()
        assert not is_valid
        assert "cycles" in errors[0].lower()

    def test_dag_topological_sort(self):
        """Test topological sorting of DAG."""
        from src.core.workflow.dag import DAG
        from src.core.workflow.tasks import NoOpTask

        dag = DAG(name="topo")
        dag.add_node(NoOpTask(), node_id="a")
        dag.add_node(NoOpTask(), node_id="b")
        dag.add_node(NoOpTask(), node_id="c")

        dag.add_edge("a", "b")
        dag.add_edge("a", "c")
        dag.add_edge("b", "c")

        sorted_nodes = dag.topological_sort()
        assert sorted_nodes.index("a") < sorted_nodes.index("b")
        assert sorted_nodes.index("b") < sorted_nodes.index("c")


class TestStateMachine:
    """Tests for state machine functionality."""

    def test_state_machine_basic_transition(self):
        """Test basic state transitions."""
        from src.core.workflow.state_machine import StateMachineBuilder

        sm = (
            StateMachineBuilder("test")
            .initial_state("draft")
            .state("published")
            .final_state("archived")
            .transition("publish", "draft", "published")
            .transition("archive", "published", "archived")
            .build()
        )

        ctx = sm.initialize()
        assert sm.current_state == "draft"

        result = sm.trigger("publish")
        assert result is True
        assert sm.current_state == "published"

        result = sm.trigger("archive")
        assert result is True
        assert sm.current_state == "archived"
        assert sm.is_final()

    def test_state_machine_invalid_transition(self):
        """Test invalid transition returns False."""
        from src.core.workflow.state_machine import StateMachineBuilder

        sm = (
            StateMachineBuilder("test")
            .initial_state("draft")
            .state("published")
            .transition("publish", "draft", "published")
            .build()
        )

        sm.initialize()
        result = sm.trigger("archive")  # Invalid event
        assert result is False
        assert sm.current_state == "draft"

    def test_state_machine_guard_condition(self):
        """Test transition guard conditions."""
        from src.core.workflow.state_machine import StateMachineBuilder

        sm = (
            StateMachineBuilder("test")
            .initial_state("draft")
            .state("published")
            .transition(
                "publish", "draft", "published",
                guard=lambda ctx: ctx.get("approved", False)
            )
            .build()
        )

        ctx = sm.initialize({"approved": False})
        result = sm.trigger("publish")
        assert result is False
        assert sm.current_state == "draft"

        ctx.set("approved", True)
        result = sm.trigger("publish")
        assert result is True
        assert sm.current_state == "published"

    def test_approval_workflow_preset(self):
        """Test predefined approval workflow."""
        from src.core.workflow.state_machine import create_approval_workflow

        sm = create_approval_workflow()
        ctx = sm.initialize()

        assert sm.current_state == "draft"
        sm.trigger("submit")
        assert sm.current_state == "pending_approval"
        sm.trigger("approve")
        assert sm.current_state == "approved"
        sm.trigger("start")
        assert sm.current_state == "in_progress"
        sm.trigger("complete")
        assert sm.current_state == "completed"
        assert sm.is_final()


class TestScheduler:
    """Tests for workflow scheduler."""

    def test_cron_parser_every_minute(self):
        """Test cron parser for every minute."""
        from src.core.workflow.scheduler import CronParser

        parsed = CronParser.parse("* * * * *")
        assert parsed["minute"] == list(range(60))
        assert parsed["hour"] == list(range(24))

    def test_cron_parser_specific_time(self):
        """Test cron parser for specific time."""
        from src.core.workflow.scheduler import CronParser

        parsed = CronParser.parse("30 9 * * 1-5")
        assert parsed["minute"] == [30]
        assert parsed["hour"] == [9]
        assert parsed["weekday"] == [1, 2, 3, 4, 5]

    def test_cron_parser_step(self):
        """Test cron parser with step."""
        from src.core.workflow.scheduler import CronParser

        parsed = CronParser.parse("*/15 * * * *")
        assert parsed["minute"] == [0, 15, 30, 45]

    @pytest.mark.asyncio
    async def test_scheduler_add_job(self):
        """Test adding a scheduled job."""
        from src.core.workflow.scheduler import Scheduler, ScheduleType, Schedule

        scheduler = Scheduler()
        executed = []

        def job_func():
            executed.append(datetime.utcnow())

        schedule = Schedule(
            schedule_id="test",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=0.1,
            max_runs=2,
        )

        job_id = scheduler.add_job(job_func, schedule, name="test_job")
        job = scheduler.get_job(job_id)

        assert job is not None
        assert job.name == "test_job"


# ============================================================================
# P33: File Storage Tests
# ============================================================================

class TestInMemoryStorage:
    """Tests for in-memory storage backend."""

    @pytest.mark.asyncio
    async def test_put_and_get_object(self):
        """Test storing and retrieving objects."""
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )

        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)

        metadata = await storage.put_object("test.txt", b"Hello, World!")
        assert metadata.key == "test.txt"
        assert metadata.size == 13

        data = await storage.get_object("test.txt")
        assert data == b"Hello, World!"

    @pytest.mark.asyncio
    async def test_delete_object(self):
        """Test deleting objects."""
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )

        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)

        await storage.put_object("test.txt", b"data")
        result = await storage.delete_object("test.txt")
        assert result is True

        with pytest.raises(FileNotFoundError):
            await storage.get_object("test.txt")

    @pytest.mark.asyncio
    async def test_list_objects(self):
        """Test listing objects with prefix."""
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )

        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)

        await storage.put_object("docs/a.txt", b"a")
        await storage.put_object("docs/b.txt", b"b")
        await storage.put_object("images/c.png", b"c")

        objects, token = await storage.list_objects(prefix="docs/")
        assert len(objects) == 2
        assert all(o.key.startswith("docs/") for o in objects)

    @pytest.mark.asyncio
    async def test_copy_object(self):
        """Test copying objects."""
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )

        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)

        await storage.put_object("source.txt", b"data")
        metadata = await storage.copy_object("source.txt", "dest.txt")

        assert metadata.key == "dest.txt"
        data = await storage.get_object("dest.txt")
        assert data == b"data"

    @pytest.mark.asyncio
    async def test_head_object(self):
        """Test getting object metadata."""
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )

        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)

        await storage.put_object(
            "test.json",
            b'{"key": "value"}',
            content_type="application/json",
            metadata={"author": "test"},
        )

        meta = await storage.head_object("test.json")
        assert meta is not None
        assert meta.content_type == "application/json"
        assert meta.metadata["author"] == "test"


class TestPresignedURLs:
    """Tests for presigned URL generation."""

    def test_url_signer_download_url(self):
        """Test generating download URLs."""
        from src.core.storage.presigned import URLSigner

        signer = URLSigner(
            secret_key="test-secret",
            base_url="http://localhost:8000",
        )

        url = signer.generate_download_url(
            bucket="mybucket",
            key="myfile.txt",
            expires_in=3600,
        )

        assert url.url.startswith("http://localhost:8000/storage/mybucket/myfile.txt")
        assert "signature=" in url.url
        assert url.method.value == "GET"

    def test_url_signer_upload_url(self):
        """Test generating upload URLs."""
        from src.core.storage.presigned import URLSigner

        signer = URLSigner(
            secret_key="test-secret",
            base_url="http://localhost:8000",
        )

        url = signer.generate_upload_url(
            bucket="mybucket",
            key="upload.txt",
            content_type="text/plain",
        )

        assert url.method.value == "PUT"
        assert url.headers["Content-Type"] == "text/plain"

    def test_url_signature_validation(self):
        """Test signature validation."""
        from src.core.storage.presigned import URLSigner
        import time

        signer = URLSigner(secret_key="test-secret", base_url="http://localhost:8000")

        # Generate and validate signature
        expires = int(time.time()) + 3600
        signature = signer._compute_signature("GET", "test.txt", expires)

        is_valid = signer.validate_signature("GET", "test.txt", expires, signature)
        assert is_valid is True

        # Invalid signature
        is_valid = signer.validate_signature("GET", "test.txt", expires, "invalid")
        assert is_valid is False


class TestMultipartUpload:
    """Tests for multipart upload functionality."""

    @pytest.mark.asyncio
    async def test_multipart_upload_initiate(self):
        """Test initiating multipart upload."""
        from src.core.storage.multipart import MultipartUploadManager, UploadStatus
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )

        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)
        manager = MultipartUploadManager(storage)

        state = await manager.initiate_upload(
            key="large-file.bin",
            bucket="test",
            total_size=10 * 1024 * 1024,  # 10MB
        )

        assert state.status == UploadStatus.PENDING
        assert state.total_parts > 0

    @pytest.mark.asyncio
    async def test_multipart_upload_complete(self):
        """Test completing multipart upload."""
        from src.core.storage.multipart import MultipartUploadManager, UploadStatus
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )

        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)
        manager = MultipartUploadManager(storage, part_size=1024)

        # Small file for testing
        state = await manager.initiate_upload(
            key="test.bin",
            bucket="test",
            total_size=2048,
        )

        # Upload parts
        await manager.upload_part(state.upload_id, 1, b"A" * 1024)
        await manager.upload_part(state.upload_id, 2, b"B" * 1024)

        # Complete
        metadata = await manager.complete_upload(state.upload_id)
        assert metadata.key == "test.bin"
        assert metadata.size == 2048

    @pytest.mark.asyncio
    async def test_multipart_upload_progress(self):
        """Test upload progress tracking."""
        from src.core.storage.multipart import MultipartUploadManager
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )

        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)
        manager = MultipartUploadManager(storage, part_size=1024)

        state = await manager.initiate_upload(
            key="test.bin",
            bucket="test",
            total_size=3072,
        )

        await manager.upload_part(state.upload_id, 1, b"A" * 1024)
        progress = manager.get_progress(state.upload_id)

        assert progress is not None
        assert progress.completed_parts == 1
        assert progress.total_parts == 3
        assert progress.progress_percent == pytest.approx(33.33, rel=0.1)


# ============================================================================
# P34: ETL Pipeline Tests
# ============================================================================

class TestETLSources:
    """Tests for ETL data sources."""

    @pytest.mark.asyncio
    async def test_memory_source(self):
        """Test in-memory data source."""
        from src.core.etl.sources import MemorySource, SourceConfig, SourceType

        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]

        config = SourceConfig(source_type=SourceType.MEMORY, name="test", batch_size=2)
        source = MemorySource(config, data)

        async with source:
            batch1 = await source.read_batch()
            assert len(batch1) == 2
            assert batch1[0].data["name"] == "Alice"

            batch2 = await source.read_batch()
            assert len(batch2) == 1
            assert batch2[0].data["name"] == "Charlie"

            batch3 = await source.read_batch()
            assert len(batch3) == 0


class TestETLTransforms:
    """Tests for ETL transformations."""

    @pytest.mark.asyncio
    async def test_map_transform(self):
        """Test field mapping transform."""
        from src.core.etl.transforms import MapTransform
        from src.core.etl.sources import Record

        records = [
            Record(data={"name": "Alice", "age": 30}, source="test", offset=0),
            Record(data={"name": "Bob", "age": 25}, source="test", offset=1),
        ]

        transform = MapTransform(
            name="rename",
            mapping={
                "name": "full_name",
                "age": lambda x: x * 2,
            }
        )

        result = await transform.apply(records)
        assert len(result.records) == 2
        assert result.records[0].data["full_name"] == "Alice"
        assert result.records[0].data["age"] == 60

    @pytest.mark.asyncio
    async def test_filter_transform(self):
        """Test filter transform."""
        from src.core.etl.transforms import FilterTransform
        from src.core.etl.sources import Record

        records = [
            Record(data={"name": "Alice", "age": 30}, source="test", offset=0),
            Record(data={"name": "Bob", "age": 25}, source="test", offset=1),
            Record(data={"name": "Charlie", "age": 35}, source="test", offset=2),
        ]

        transform = FilterTransform(
            name="age_filter",
            condition=lambda d: d["age"] >= 30,
        )

        result = await transform.apply(records)
        assert len(result.records) == 2
        assert result.dropped == 1

    @pytest.mark.asyncio
    async def test_validate_transform(self):
        """Test validation transform."""
        from src.core.etl.transforms import ValidateTransform
        from src.core.etl.sources import Record

        records = [
            Record(data={"name": "Alice", "age": 30}, source="test", offset=0),
            Record(data={"name": None, "age": 25}, source="test", offset=1),
        ]

        transform = ValidateTransform(
            name="validate",
            required_fields=["name", "age"],
            field_types={"age": int},
        )

        result = await transform.apply(records)
        assert len(result.records) == 1
        assert result.dropped == 1

    @pytest.mark.asyncio
    async def test_chain_transform(self):
        """Test chaining multiple transforms."""
        from src.core.etl.transforms import (
            ChainTransform, MapTransform, FilterTransform
        )
        from src.core.etl.sources import Record

        records = [
            Record(data={"value": 10}, source="test", offset=0),
            Record(data={"value": 20}, source="test", offset=1),
            Record(data={"value": 30}, source="test", offset=2),
        ]

        transform = ChainTransform(
            name="chain",
            transforms=[
                MapTransform("double", {"value": lambda x: x * 2}),
                FilterTransform("filter", lambda d: d["value"] >= 40),
            ]
        )

        result = await transform.apply(records)
        assert len(result.records) == 2
        assert result.records[0].data["value"] == 40


class TestETLSinks:
    """Tests for ETL data sinks."""

    @pytest.mark.asyncio
    async def test_memory_sink(self):
        """Test in-memory data sink."""
        from src.core.etl.sinks import MemorySink, SinkConfig, SinkType
        from src.core.etl.sources import Record

        config = SinkConfig(sink_type=SinkType.MEMORY, name="test")
        sink = MemorySink(config)

        records = [
            Record(data={"id": 1}, source="test", offset=0),
            Record(data={"id": 2}, source="test", offset=1),
        ]

        async with sink:
            result = await sink.write(records)
            assert result.written == 2
            assert result.failed == 0

            data = sink.get_data()
            assert len(data) == 2


class TestETLPipeline:
    """Tests for complete ETL pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """Test full pipeline execution."""
        from src.core.etl.pipeline import Pipeline, PipelineStatus, PipelineConfig
        from src.core.etl.sources import MemorySource, SourceConfig, SourceType
        from src.core.etl.transforms import MapTransform, FilterTransform
        from src.core.etl.sinks import MemorySink, SinkConfig

        # Source data
        data = [
            {"name": "Alice", "score": 85},
            {"name": "Bob", "score": 65},
            {"name": "Charlie", "score": 92},
        ]

        source = MemorySource(
            SourceConfig(source_type=SourceType.MEMORY, name="input"),
            data
        )

        transforms = [
            FilterTransform("pass_filter", lambda d: d["score"] >= 70),
            MapTransform("grade", {"score": lambda s: "A" if s >= 90 else "B"}),
        ]

        sink = MemorySink(SinkConfig(sink_type=SinkType.MEMORY, name="output"))

        pipeline = Pipeline(
            source=source,
            transforms=transforms,
            sink=sink,
            config=PipelineConfig(name="test_pipeline"),
        )

        result = await pipeline.run()

        assert result.status == PipelineStatus.COMPLETED
        assert result.metrics.records_read == 3
        assert result.metrics.records_written == 2
        assert result.metrics.records_dropped == 1


# ============================================================================
# P35: Analytics Tests
# ============================================================================

class TestTimeSeriesStore:
    """Tests for time-series data storage."""

    @pytest.mark.asyncio
    async def test_write_and_query(self):
        """Test writing and querying time series data."""
        from src.core.analytics.timeseries import TimeSeriesStore
        from datetime import datetime, timedelta

        store = TimeSeriesStore()

        # Write data
        now = datetime.utcnow()
        for i in range(10):
            await store.write(
                "cpu_usage",
                value=50 + i,
                timestamp=now - timedelta(minutes=10 - i),
                tags={"host": "server1"},
            )

        # Query
        result = await store.query(
            "cpu_usage",
            start_time=now - timedelta(minutes=15),
            tags={"host": "server1"},
        )

        assert len(result.series) == 1
        assert result.total_points == 10

    @pytest.mark.asyncio
    async def test_aggregation(self):
        """Test time series aggregation."""
        from src.core.analytics.timeseries import (
            TimeSeriesStore, AggregationType, Resolution
        )
        from datetime import datetime, timedelta

        store = TimeSeriesStore()

        # Write data
        now = datetime.utcnow()
        for i in range(60):
            await store.write(
                "requests",
                value=100 + (i % 10),
                timestamp=now - timedelta(seconds=60 - i),
            )

        # Query with aggregation
        result = await store.query(
            "requests",
            start_time=now - timedelta(minutes=2),
            aggregation=AggregationType.AVG,
            resolution=Resolution.MINUTE,
        )

        assert len(result.series) == 1
        # Should have 1-2 aggregated points for 1 minute of data

    @pytest.mark.asyncio
    async def test_get_metrics_list(self):
        """Test listing all metrics."""
        from src.core.analytics.timeseries import TimeSeriesStore

        store = TimeSeriesStore()

        await store.write("metric1", 10)
        await store.write("metric2", 20)
        await store.write("metric3", 30)

        metrics = await store.get_metrics()
        assert len(metrics) == 3
        assert "metric1" in metrics


class TestMetricsAggregation:
    """Tests for metrics aggregation."""

    def test_counter(self):
        """Test counter metric."""
        from src.core.analytics.metrics import Counter

        counter = Counter("requests_total", labels=["method"])
        counter.inc(1, {"method": "GET"})
        counter.inc(5, {"method": "POST"})
        counter.inc(2, {"method": "GET"})

        values = counter.collect()
        assert len(values) == 2

        get_value = next(v for v in values if v.labels.get("method") == "GET")
        assert get_value.value == 3

    def test_gauge(self):
        """Test gauge metric."""
        from src.core.analytics.metrics import Gauge

        gauge = Gauge("temperature", labels=["location"])
        gauge.set(25.5, {"location": "office"})
        gauge.inc(2.0, {"location": "office"})
        gauge.dec(1.0, {"location": "office"})

        values = gauge.collect()
        assert len(values) == 1
        assert values[0].value == 26.5

    def test_histogram(self):
        """Test histogram metric."""
        from src.core.analytics.metrics import Histogram

        hist = Histogram(
            "request_duration",
            buckets=[0.1, 0.5, 1.0, 5.0],
        )

        hist.observe(0.05)
        hist.observe(0.3)
        hist.observe(0.8)
        hist.observe(2.0)

        values = hist.collect()
        assert len(values) == 1
        assert values[0].count == 4
        assert values[0].sum == pytest.approx(3.15, rel=0.01)

    def test_registry_prometheus_export(self):
        """Test Prometheus format export."""
        from src.core.analytics.metrics import MetricRegistry

        registry = MetricRegistry()
        counter = registry.counter("http_requests", "Total requests", ["method"])
        counter.inc(10, {"method": "GET"})

        output = registry.export_prometheus()
        assert "http_requests" in output
        assert "counter" in output.lower()


class TestDashboardProvider:
    """Tests for dashboard data provider."""

    @pytest.mark.asyncio
    async def test_widget_registration(self):
        """Test registering dashboard widgets."""
        from src.core.analytics.dashboard import (
            DashboardProvider, WidgetConfig, WidgetType
        )
        from src.core.analytics.timeseries import AggregationType

        provider = DashboardProvider()

        config = WidgetConfig(
            widget_id="cpu_chart",
            widget_type=WidgetType.LINE_CHART,
            title="CPU Usage",
            metric_name="cpu_usage",
            aggregation=AggregationType.AVG,
        )

        provider.register_widget(config)
        retrieved = provider.get_widget_config("cpu_chart")

        assert retrieved is not None
        assert retrieved.title == "CPU Usage"

    @pytest.mark.asyncio
    async def test_get_widget_data(self):
        """Test getting widget data."""
        from src.core.analytics.dashboard import (
            DashboardProvider, WidgetConfig, WidgetType, TimeRange
        )
        from src.core.analytics.timeseries import TimeSeriesStore, AggregationType
        from datetime import datetime, timedelta

        store = TimeSeriesStore()
        provider = DashboardProvider(ts_store=store)

        # Add some data
        now = datetime.utcnow()
        for i in range(10):
            await store.write(
                "test_metric",
                value=i * 10,
                timestamp=now - timedelta(minutes=10 - i),
            )

        # Register widget
        config = WidgetConfig(
            widget_id="test_widget",
            widget_type=WidgetType.STAT,
            title="Test Stat",
            metric_name="test_metric",
            aggregation=AggregationType.LAST,
        )
        provider.register_widget(config)

        # Get data
        data = await provider.get_widget_data("test_widget", TimeRange.LAST_1H)

        assert data is not None
        assert data.widget_type == WidgetType.STAT


# ============================================================================
# Integration Tests
# ============================================================================

class TestWorkflowETLIntegration:
    """Integration test combining workflow and ETL."""

    @pytest.mark.asyncio
    async def test_workflow_orchestrated_etl(self):
        """Test ETL pipeline orchestrated by workflow engine."""
        from src.core.workflow.tasks import FunctionTask, TaskStatus
        from src.core.workflow.dag import DAG, DAGExecutor, WorkflowStatus
        from src.core.etl.pipeline import Pipeline, PipelineConfig
        from src.core.etl.sources import MemorySource, SourceConfig, SourceType
        from src.core.etl.transforms import MapTransform
        from src.core.etl.sinks import MemorySink, SinkConfig

        # ETL pipeline function
        async def run_etl(input_data):
            data = [{"value": i} for i in range(5)]
            source = MemorySource(
                SourceConfig(source_type=SourceType.MEMORY, name="in"),
                data
            )
            transform = MapTransform("double", {"value": lambda x: x * 2})
            sink = MemorySink(SinkConfig(sink_type=SinkType.MEMORY, name="out"))

            pipeline = Pipeline(
                source=source,
                transforms=[transform],
                sink=sink,
                config=PipelineConfig(name="test"),
            )

            result = await pipeline.run()
            return sink.get_data()

        # Create workflow
        etl_task = FunctionTask(run_etl, name="etl")
        dag = DAG(name="etl_workflow")
        dag.add_node(etl_task, node_id="etl")

        executor = DAGExecutor(dag)
        result = await executor.execute({})

        assert result.status == WorkflowStatus.COMPLETED
        etl_output = result.context.outputs.get("etl")
        assert etl_output is not None
        assert len(etl_output) == 5
        assert etl_output[0]["value"] == 0
        assert etl_output[4]["value"] == 8


class TestStorageAnalyticsIntegration:
    """Integration test combining storage and analytics."""

    @pytest.mark.asyncio
    async def test_storage_metrics_tracking(self):
        """Test tracking storage metrics with analytics."""
        from src.core.storage.object_store import (
            InMemoryStorage, StorageConfig, StorageBackend
        )
        from src.core.analytics.metrics import Counter, Histogram

        # Create metrics
        upload_counter = Counter("storage_uploads_total", labels=["bucket"])
        upload_size = Histogram(
            "storage_upload_bytes",
            buckets=[1024, 10240, 102400, 1048576],
        )

        # Create storage
        config = StorageConfig(backend=StorageBackend.MEMORY, bucket="test")
        storage = InMemoryStorage(config)

        # Upload files and track metrics
        for i in range(5):
            data = b"X" * (i + 1) * 1000
            await storage.put_object(f"file{i}.txt", data)
            upload_counter.inc(1, {"bucket": "test"})
            upload_size.observe(len(data))

        # Check metrics
        counter_values = upload_counter.collect()
        assert counter_values[0].value == 5

        hist_values = upload_size.collect()
        assert hist_values[0].count == 5
