"""Unit tests for P24-P27 Enterprise Features.

P24: Configuration Center (Consul/etcd)
P25: Service Mesh (Istio) - K8s manifests only, no code tests
P26: Message Queue (Kafka/RabbitMQ)
P27: Batch Processing Framework (Scheduler/Workflows)

These tests bypass the conftest fixtures by importing directly.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# P24: Configuration Center Tests
# ============================================================================

class TestConfigManager:
    """Tests for configuration manager."""

    def test_config_manager_initialization(self):
        """Test config manager can be created."""
        from src.core.config.manager import ConfigManager

        manager = ConfigManager()
        assert manager is not None

    def test_add_source(self):
        """Test adding config sources."""
        from src.core.config.manager import ConfigManager
        from src.core.config.sources import EnvConfigSource

        manager = ConfigManager()
        source = EnvConfigSource(prefix="TEST")
        manager.add_source(source)

        assert len(manager._sources) == 1

    @pytest.mark.asyncio
    async def test_get_config_value(self):
        """Test getting config values."""
        from src.core.config.manager import ConfigManager, ConfigSource, ConfigPriority

        class MockSource(ConfigSource):
            def __init__(self, name, values, priority):
                super().__init__(name, priority)
                self._values = values

            async def get(self, key):
                return self._values.get(key)

            async def get_all(self, prefix=""):
                return self._values

        manager = ConfigManager()
        source = MockSource("mock", {"key": "value"}, ConfigPriority.FILE)
        manager.add_source(source)

        value = await manager.get("key")
        assert value == "value"

        # Should return default for non-existent key
        value = await manager.get("nonexistent", default="default_value")
        assert value == "default_value"

    @pytest.mark.asyncio
    async def test_config_priority(self):
        """Test that higher priority sources take precedence."""
        from src.core.config.manager import ConfigManager, ConfigSource, ConfigPriority

        class MockSource(ConfigSource):
            def __init__(self, name, values, priority):
                super().__init__(name, priority)
                self._values = values

            async def get(self, key):
                return self._values.get(key)

            async def get_all(self, prefix=""):
                return self._values

        manager = ConfigManager()
        low_source = MockSource("low", {"key": "low_value"}, ConfigPriority.FILE)
        high_source = MockSource("high", {"key": "high_value"}, ConfigPriority.ENVIRONMENT)

        manager.add_source(low_source)
        manager.add_source(high_source)

        value = await manager.get("key")
        assert value == "high_value"

    @pytest.mark.asyncio
    async def test_get_all_configs(self):
        """Test getting all config values."""
        from src.core.config.manager import ConfigManager, ConfigSource, ConfigPriority

        class MockSource(ConfigSource):
            def __init__(self, values):
                super().__init__("mock", ConfigPriority.FILE)
                self._values = values

            async def get(self, key):
                return self._values.get(key)

            async def get_all(self, prefix=""):
                return self._values

        manager = ConfigManager()
        source = MockSource({"a": "1", "b": "2"})
        manager.add_source(source)

        all_config = await manager.get_all()
        assert "a" in all_config
        assert "b" in all_config


class TestConfigSources:
    """Tests for config sources."""

    @pytest.mark.asyncio
    async def test_env_config_source(self):
        """Test environment variable config source."""
        from src.core.config.sources import EnvConfigSource

        os.environ["TEST_MY_KEY"] = "test_value"
        source = EnvConfigSource(prefix="TEST")
        value = await source.get("MY_KEY")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_file_config_source_json(self):
        """Test JSON file config source."""
        from src.core.config.sources import FileConfigSource

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            source = FileConfigSource(f.name)
            value = await source.get("key")
            assert value == "value"
            os.unlink(f.name)

    def test_env_source_name(self):
        """Test env source name property."""
        from src.core.config.sources import EnvConfigSource

        source = EnvConfigSource()
        assert source.name == "env"


class TestConfigWatcher:
    """Tests for config watcher."""

    def test_config_watcher_creation(self):
        """Test config watcher initialization."""
        from src.core.config.watcher import ConfigWatcher

        watcher = ConfigWatcher()
        assert watcher is not None

    def test_watch_subscription(self):
        """Test registering watch subscriptions."""
        from src.core.config.watcher import ConfigWatcher

        watcher = ConfigWatcher()

        def callback(event):
            pass

        subscription = watcher.watch("test.key", callback)
        assert subscription is not None
        assert "test.key" in watcher._subscriptions


# ============================================================================
# P26: Message Queue Tests
# ============================================================================

class TestMessageProducer:
    """Tests for message producers."""

    def test_message_creation(self):
        """Test message object creation."""
        from src.core.messaging.producer import Message

        msg = Message(
            topic="test-topic",
            payload={"data": "value"},
            key="partition-key",
        )

        assert msg.topic == "test-topic"
        assert msg.payload == {"data": "value"}
        assert msg.key == "partition-key"
        assert msg.message_id is not None

    def test_message_serialization(self):
        """Test message serialization."""
        import json

        from src.core.messaging.producer import Message

        msg = Message(
            topic="test-topic",
            payload={"data": "value"},
        )

        serialized = msg.serialize()
        assert isinstance(serialized, bytes)

        data = json.loads(serialized.decode("utf-8"))
        assert data["topic"] == "test-topic"
        assert data["payload"] == {"data": "value"}

    @pytest.mark.asyncio
    async def test_in_memory_producer(self):
        """Test in-memory producer for testing."""
        from src.core.messaging.producer import InMemoryProducer, Message

        producer = InMemoryProducer()

        msg = Message(topic="test", payload="data")
        result = await producer.send(msg)

        assert result.success is True
        assert result.topic == "test"

        messages = producer.get_messages("test")
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_in_memory_producer_batch(self):
        """Test batch sending with in-memory producer."""
        from src.core.messaging.producer import InMemoryProducer, Message

        producer = InMemoryProducer()

        messages = [
            Message(topic="test", payload=f"data{i}")
            for i in range(5)
        ]

        results = await producer.send_batch(messages)

        assert len(results) == 5
        assert all(r.success for r in results)


class TestMessageConsumer:
    """Tests for message consumers."""

    def test_received_message_from_bytes(self):
        """Test parsing received messages."""
        import json

        from src.core.messaging.consumer import ReceivedMessage

        data = json.dumps({
            "payload": {"key": "value"},
            "message_id": "123",
        }).encode("utf-8")

        msg = ReceivedMessage.from_bytes("topic", data)

        assert msg.topic == "topic"
        assert msg.payload == {"key": "value"}

    @pytest.mark.asyncio
    async def test_in_memory_consumer_subscribe(self):
        """Test in-memory consumer subscription."""
        from src.core.messaging.consumer import InMemoryConsumer

        consumer = InMemoryConsumer()
        await consumer.subscribe(["topic1", "topic2"])

        assert "topic1" in consumer._subscribed_topics
        assert "topic2" in consumer._subscribed_topics


class TestEventBus:
    """Tests for event bus."""

    def test_event_creation(self):
        """Test event object creation."""
        from src.core.messaging.events import Event

        event = Event(
            event_type="test.event",
            data={"key": "value"},
        )

        assert event.event_type == "test.event"
        assert event.data == {"key": "value"}
        assert event.event_id is not None

    def test_event_serialization(self):
        """Test event to dict conversion."""
        from src.core.messaging.events import Event

        event = Event(
            event_type="test.event",
            data={"key": "value"},
            user_id="user123",
        )

        data = event.to_dict()

        assert data["event_type"] == "test.event"
        assert data["data"] == {"key": "value"}
        assert data["user_id"] == "user123"

    def test_event_from_dict(self):
        """Test event creation from dict."""
        from src.core.messaging.events import Event

        data = {
            "event_type": "test.event",
            "data": {"key": "value"},
            "timestamp": "2024-01-15T10:00:00",
        }

        event = Event.from_dict(data)

        assert event.event_type == "test.event"
        assert event.data == {"key": "value"}

    def test_event_bus_subscribe(self):
        """Test event bus subscription."""
        from src.core.messaging.events import EventBus

        bus = EventBus()

        def handler(event):
            pass

        bus.subscribe("test.event", handler)

        assert "test.event" in bus._handlers
        assert handler in bus._handlers["test.event"]

    def test_event_bus_unsubscribe(self):
        """Test event bus unsubscription."""
        from src.core.messaging.events import EventBus

        bus = EventBus()

        def handler(event):
            pass

        bus.subscribe("test.event", handler)
        bus.unsubscribe("test.event", handler)

        assert len(bus._handlers.get("test.event", [])) == 0

    def test_event_bus_middleware(self):
        """Test event bus middleware."""
        from src.core.messaging.events import Event, EventBus

        bus = EventBus()

        def add_metadata(event):
            event.metadata["processed"] = True
            return event

        bus.use(add_metadata)

        # Middleware is applied during publish
        assert len(bus._middleware) == 1

    def test_event_factory_functions(self):
        """Test event factory functions."""
        from src.core.messaging.events import (
            EventType,
            document_event,
            job_event,
            model_event,
            user_event,
        )

        doc_event = document_event(EventType.DOCUMENT_CREATED, "doc123")
        assert doc_event.data["document_id"] == "doc123"

        model_evt = model_event(EventType.MODEL_TRAINED, "model456")
        assert model_evt.data["model_id"] == "model456"

        user_evt = user_event(EventType.USER_LOGIN, "user789")
        assert user_evt.data["user_id"] == "user789"
        assert user_evt.user_id == "user789"

        job_evt = job_event(EventType.JOB_PROGRESS, "job111", progress=0.5)
        assert job_evt.data["progress"] == 0.5


# ============================================================================
# P27: Batch Processing Tests
# ============================================================================

class TestCronSchedule:
    """Tests for cron schedule."""

    def test_cron_schedule_creation(self):
        """Test cron schedule object creation."""
        from src.core.tasks.scheduler import CronSchedule

        schedule = CronSchedule(
            minute="0",
            hour="9",
            day_of_month="*",
            month="*",
            day_of_week="1-5",
        )

        assert schedule.minute == "0"
        assert schedule.hour == "9"

    def test_cron_matches_exact(self):
        """Test cron matching with exact values."""
        from src.core.tasks.scheduler import CronSchedule

        schedule = CronSchedule(minute="30", hour="9")
        dt = datetime(2024, 1, 15, 9, 30)

        assert schedule.matches(dt) is True

    def test_cron_matches_range(self):
        """Test cron matching with ranges."""
        from src.core.tasks.scheduler import CronSchedule

        schedule = CronSchedule(hour="9-17")
        dt_in = datetime(2024, 1, 15, 12, 0)
        dt_out = datetime(2024, 1, 15, 20, 0)

        assert schedule.matches(dt_in) is True
        assert schedule.matches(dt_out) is False

    def test_cron_matches_step(self):
        """Test cron matching with steps."""
        from src.core.tasks.scheduler import CronSchedule

        schedule = CronSchedule(minute="*/15")
        dt_match = datetime(2024, 1, 15, 9, 0)
        dt_no_match = datetime(2024, 1, 15, 9, 7)

        assert schedule.matches(dt_match) is True
        assert schedule.matches(dt_no_match) is False

    def test_cron_matches_list(self):
        """Test cron matching with lists."""
        from src.core.tasks.scheduler import CronSchedule

        schedule = CronSchedule(hour="9,12,15")
        dt_match = datetime(2024, 1, 15, 12, 0)
        dt_no_match = datetime(2024, 1, 15, 10, 0)

        assert schedule.matches(dt_match) is True
        assert schedule.matches(dt_no_match) is False

    def test_cron_factory_methods(self):
        """Test cron factory methods."""
        from src.core.tasks.scheduler import CronSchedule

        every_min = CronSchedule.every_minute()
        assert every_min.minute == "*"

        every_hour = CronSchedule.every_hour()
        assert every_hour.minute == "0"

        daily = CronSchedule.daily(hour=9, minute=30)
        assert daily.hour == "9"
        assert daily.minute == "30"

        weekly = CronSchedule.weekly(day_of_week=1, hour=10)
        assert weekly.day_of_week == "1"


class TestIntervalSchedule:
    """Tests for interval schedule."""

    def test_interval_schedule_creation(self):
        """Test interval schedule object creation."""
        from src.core.tasks.scheduler import IntervalSchedule

        schedule = IntervalSchedule(minutes=5, seconds=30)

        assert schedule.minutes == 5
        assert schedule.seconds == 30

    def test_interval_total_seconds(self):
        """Test total seconds calculation."""
        from src.core.tasks.scheduler import IntervalSchedule

        schedule = IntervalSchedule(hours=1, minutes=30, seconds=45)

        # 1*3600 + 30*60 + 45 = 3600 + 1800 + 45 = 5445
        assert schedule.total_seconds == 5445

    def test_interval_factory_method(self):
        """Test interval factory method."""
        from src.core.tasks.scheduler import IntervalSchedule

        schedule = IntervalSchedule.every(minutes=10)
        assert schedule.minutes == 10


class TestTaskScheduler:
    """Tests for task scheduler."""

    def test_scheduler_creation(self):
        """Test scheduler initialization."""
        from src.core.tasks.scheduler import TaskScheduler

        scheduler = TaskScheduler()
        assert scheduler is not None

    def test_schedule_task(self):
        """Test scheduling a task."""
        from src.core.tasks.scheduler import IntervalSchedule, TaskScheduler

        scheduler = TaskScheduler()

        def my_task():
            pass

        task_id = scheduler.schedule(
            my_task,
            IntervalSchedule.every(minutes=5),
            name="test_task",
        )

        assert task_id is not None
        task = scheduler.get_task(task_id)
        assert task is not None
        assert task.name == "test_task"

    def test_unschedule_task(self):
        """Test unscheduling a task."""
        from src.core.tasks.scheduler import IntervalSchedule, TaskScheduler

        scheduler = TaskScheduler()

        def my_task():
            pass

        task_id = scheduler.schedule(my_task, IntervalSchedule.every(minutes=5))
        result = scheduler.unschedule(task_id)

        assert result is True
        assert scheduler.get_task(task_id) is None

    def test_enable_disable_task(self):
        """Test enabling and disabling tasks."""
        from src.core.tasks.scheduler import IntervalSchedule, TaskScheduler

        scheduler = TaskScheduler()

        def my_task():
            pass

        task_id = scheduler.schedule(my_task, IntervalSchedule.every(minutes=5))

        scheduler.disable_task(task_id)
        assert scheduler.get_task(task_id).enabled is False

        scheduler.enable_task(task_id)
        assert scheduler.get_task(task_id).enabled is True

    def test_list_tasks(self):
        """Test listing all tasks."""
        from src.core.tasks.scheduler import IntervalSchedule, TaskScheduler

        scheduler = TaskScheduler()

        def task1():
            pass

        def task2():
            pass

        scheduler.schedule(task1, IntervalSchedule.every(minutes=5))
        scheduler.schedule(task2, IntervalSchedule.every(minutes=10))

        tasks = scheduler.list_tasks()
        assert len(tasks) == 2


class TestWorkflowStep:
    """Tests for workflow steps."""

    def test_workflow_step_creation(self):
        """Test workflow step creation."""
        from src.core.tasks.workflows import StepType, WorkflowStep

        step = WorkflowStep(
            step_id="step1",
            name="test_step",
            step_type=StepType.TASK,
        )

        assert step.step_id == "step1"
        assert step.name == "test_step"

    def test_workflow_step_task_factory(self):
        """Test task factory method."""
        from src.core.tasks.workflows import StepType, WorkflowStep

        def my_func(x):
            return x * 2

        step = WorkflowStep.task(my_func, 5, name="double")

        assert step.name == "double"
        assert step.step_type == StepType.TASK
        assert step.args == (5,)


class TestWorkflowBuilderFunctions:
    """Tests for workflow builder functions."""

    def test_chain_creation(self):
        """Test chain workflow creation."""
        from src.core.tasks.workflows import StepType, chain

        def task1():
            pass

        def task2():
            pass

        workflow = chain(task1, task2)

        assert workflow.step_type == StepType.CHAIN
        assert len(workflow.children) == 2

    def test_group_creation(self):
        """Test group workflow creation."""
        from src.core.tasks.workflows import StepType, group

        def task1():
            pass

        def task2():
            pass

        workflow = group(task1, task2)

        assert workflow.step_type == StepType.GROUP
        assert len(workflow.children) == 2

    def test_chord_creation(self):
        """Test chord workflow creation."""
        from src.core.tasks.workflows import StepType, chord

        def task1():
            pass

        def task2():
            pass

        def callback(results):
            return sum(results)

        workflow = chord([task1, task2], callback)

        assert workflow.step_type == StepType.CHORD
        assert workflow.callback == callback

    def test_conditional_creation(self):
        """Test conditional workflow creation."""
        from src.core.tasks.workflows import StepType, conditional

        def if_true():
            pass

        def if_false():
            pass

        workflow = conditional(lambda x: x > 0, if_true, if_false)

        assert workflow.step_type == StepType.CONDITIONAL
        assert len(workflow.children) == 2


class TestWorkflowEngine:
    """Tests for workflow engine."""

    @pytest.mark.asyncio
    async def test_workflow_engine_creation(self):
        """Test workflow engine initialization."""
        from src.core.tasks.workflows import WorkflowEngine

        engine = WorkflowEngine()
        assert engine is not None

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self):
        """Test executing a simple workflow."""
        from src.core.tasks.workflows import (
            StepType,
            Workflow,
            WorkflowEngine,
            WorkflowStep,
        )

        engine = WorkflowEngine()

        def add_one(context=None, **kwargs):
            return 1

        step = WorkflowStep.task(add_one)
        workflow = Workflow(
            workflow_id="test-wf",
            name="test",
            steps=[step],
        )

        result = await engine.execute(workflow)
        assert result == 1

    @pytest.mark.asyncio
    async def test_execute_chain_workflow(self):
        """Test executing a chain workflow."""
        from src.core.tasks.workflows import Workflow, WorkflowEngine, chain

        engine = WorkflowEngine()

        def step1(context=None, **kwargs):
            return 1

        def step2(prev_result=None, context=None, **kwargs):
            return prev_result + 1

        def step3(prev_result=None, context=None, **kwargs):
            return prev_result + 1

        chain_step = chain(step1, step2, step3)
        workflow = Workflow(
            workflow_id="test-chain",
            name="test_chain",
            steps=[chain_step],
        )

        result = await engine.execute(workflow)
        assert result == 3

    @pytest.mark.asyncio
    async def test_execute_group_workflow(self):
        """Test executing a group (parallel) workflow."""
        from src.core.tasks.workflows import Workflow, WorkflowEngine, group

        engine = WorkflowEngine()

        def task_a(context=None, **kwargs):
            return "A"

        def task_b(context=None, **kwargs):
            return "B"

        def task_c(context=None, **kwargs):
            return "C"

        group_step = group(task_a, task_b, task_c)
        workflow = Workflow(
            workflow_id="test-group",
            name="test_group",
            steps=[group_step],
        )

        result = await engine.execute(workflow)
        assert set(result) == {"A", "B", "C"}

    @pytest.mark.asyncio
    async def test_execute_chord_workflow(self):
        """Test executing a chord workflow."""
        from src.core.tasks.workflows import Workflow, WorkflowEngine, chord

        engine = WorkflowEngine()

        def get_num(n):
            def func(context=None, **kwargs):
                return n
            func.__name__ = f"get_{n}"
            return func

        def sum_all(results, context=None):
            return sum(results)

        chord_step = chord([get_num(1), get_num(2), get_num(3)], sum_all)
        workflow = Workflow(
            workflow_id="test-chord",
            name="test_chord",
            steps=[chord_step],
        )

        result = await engine.execute(workflow)
        assert result == 6


class TestWorkflowBuilder:
    """Tests for workflow builder."""

    def test_workflow_builder_creation(self):
        """Test workflow builder initialization."""
        from src.core.tasks.workflows import WorkflowBuilder

        builder = WorkflowBuilder("test_workflow")
        assert builder.name == "test_workflow"

    def test_workflow_builder_add_step(self):
        """Test adding steps to builder."""
        from src.core.tasks.workflows import WorkflowBuilder

        def my_task():
            pass

        builder = WorkflowBuilder("test")
        builder.add_step(my_task)

        assert len(builder._steps) == 1

    def test_workflow_builder_chain(self):
        """Test adding chain to builder."""
        from src.core.tasks.workflows import StepType, WorkflowBuilder

        def task1():
            pass

        def task2():
            pass

        builder = WorkflowBuilder("test")
        builder.chain(task1, task2)

        assert len(builder._steps) == 1
        assert builder._steps[0].step_type == StepType.CHAIN

    def test_workflow_builder_build(self):
        """Test building workflow."""
        from src.core.tasks.workflows import WorkflowBuilder

        def my_task():
            pass

        builder = WorkflowBuilder("test")
        builder.add_step(my_task)
        builder.metadata(priority=1)

        workflow = builder.build()

        assert workflow.name == "test"
        assert len(workflow.steps) == 1
        assert workflow.metadata["priority"] == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestGlobalAccessors:
    """Test global accessor functions."""

    def test_get_scheduler(self):
        """Test global scheduler accessor."""
        from src.core.tasks.scheduler import get_scheduler

        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()

        assert scheduler1 is scheduler2

    def test_get_workflow_engine(self):
        """Test global workflow engine accessor."""
        from src.core.tasks.workflows import get_workflow_engine

        engine1 = get_workflow_engine()
        engine2 = get_workflow_engine()

        assert engine1 is engine2

    def test_get_event_bus(self):
        """Test global event bus accessor."""
        from src.core.messaging.events import get_event_bus

        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2
