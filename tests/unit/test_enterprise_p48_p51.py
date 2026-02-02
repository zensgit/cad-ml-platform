"""Unit tests for Enterprise Features P48-P51.

P48: Distributed Lock
P49: Job Queue
P50: API Gateway Enhanced
P51: Saga Pattern
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List


# ============================================================================
# P48: Distributed Lock Tests
# ============================================================================

class TestDistributedLock:
    """Tests for Distributed Lock module."""

    def test_lock_status_enum(self):
        """Test LockStatus enum."""
        from src.core.distributed_lock import LockStatus

        assert LockStatus.ACQUIRED.value == "acquired"
        assert LockStatus.RELEASED.value == "released"

    def test_lock_info(self):
        """Test LockInfo dataclass."""
        from src.core.distributed_lock import LockInfo

        info = LockInfo(
            name="test_lock",
            owner="owner1",
            acquired_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=30),
            fencing_token=1,
        )

        assert info.name == "test_lock"
        assert info.is_expired is False
        assert info.ttl_seconds > 0

    def test_fencing_token_generator(self):
        """Test FencingTokenGenerator."""
        from src.core.distributed_lock import FencingTokenGenerator

        gen = FencingTokenGenerator()
        t1 = gen.next()
        t2 = gen.next()
        t3 = gen.next()

        assert t2 > t1
        assert t3 > t2

    @pytest.mark.asyncio
    async def test_in_memory_lock_acquire_release(self):
        """Test InMemoryLock acquire and release."""
        from src.core.distributed_lock import InMemoryLock

        lock = InMemoryLock()

        # Acquire lock
        result = await lock.acquire("test", "owner1", ttl_seconds=10)
        assert result.success is True
        assert result.lock_info is not None
        assert result.lock_info.fencing_token > 0

        # Check if locked
        assert await lock.is_locked("test") is True

        # Release lock
        released = await lock.release("test", "owner1")
        assert released is True

        # Check if unlocked
        assert await lock.is_locked("test") is False

    @pytest.mark.asyncio
    async def test_lock_exclusive(self):
        """Test lock exclusivity."""
        from src.core.distributed_lock import InMemoryLock

        lock = InMemoryLock()

        # First owner acquires
        result1 = await lock.acquire("test", "owner1", ttl_seconds=10)
        assert result1.success is True

        # Second owner cannot acquire (no wait)
        result2 = await lock.acquire("test", "owner2", ttl_seconds=10)
        assert result2.success is False

        # Release
        await lock.release("test", "owner1")

        # Now second owner can acquire
        result3 = await lock.acquire("test", "owner2", ttl_seconds=10)
        assert result3.success is True

    @pytest.mark.asyncio
    async def test_lock_extend(self):
        """Test lock extension."""
        from src.core.distributed_lock import InMemoryLock

        lock = InMemoryLock()

        # Acquire
        await lock.acquire("test", "owner1", ttl_seconds=5)

        # Extend
        extended = await lock.extend("test", "owner1", additional_seconds=10)
        assert extended is True

        # Check TTL increased
        info = await lock.get_info("test")
        assert info is not None
        assert info.ttl_seconds > 5

    @pytest.mark.asyncio
    async def test_lock_manager(self):
        """Test LockManager."""
        from src.core.distributed_lock import InMemoryLock, LockManager

        lock = InMemoryLock()
        manager = LockManager(lock)

        # Acquire through manager
        result = await manager.acquire("test", ttl_seconds=10)
        assert result.success is True

        # Release through manager
        released = await manager.release("test")
        assert released is True

    @pytest.mark.asyncio
    async def test_lock_context(self):
        """Test LockContext."""
        from src.core.distributed_lock import InMemoryLock, LockManager, LockContext

        lock = InMemoryLock()
        manager = LockManager(lock)

        async with LockContext(manager, "test", ttl_seconds=10) as ctx:
            assert ctx.fencing_token is not None
            assert await lock.is_locked("test") is True

        # Lock released after context
        assert await lock.is_locked("test") is False

    @pytest.mark.asyncio
    async def test_multi_lock(self):
        """Test MultiLock for multiple locks."""
        from src.core.distributed_lock import InMemoryLock, MultiLock

        lock = InMemoryLock()
        multi = MultiLock(lock, "owner1")

        # Acquire multiple locks
        acquired = await multi.acquire_all(["lock1", "lock2", "lock3"], ttl_seconds=10)
        assert acquired is True

        # All should be locked
        assert await lock.is_locked("lock1") is True
        assert await lock.is_locked("lock2") is True
        assert await lock.is_locked("lock3") is True

        # Release all
        await multi.release_all()

        # All should be unlocked
        assert await lock.is_locked("lock1") is False

    @pytest.mark.asyncio
    async def test_leader_election(self):
        """Test LeaderElection."""
        from src.core.distributed_lock import InMemoryLock, LeaderElection

        lock = InMemoryLock()
        election = LeaderElection(
            lock=lock,
            election_name="leader",
            node_id="node1",
            lease_duration_seconds=5,
            renew_interval_seconds=1,
        )

        # Track election
        elected_count = 0

        def on_elected():
            nonlocal elected_count
            elected_count += 1

        election.on_elected(on_elected)

        # Start election
        task = asyncio.create_task(election.start())
        await asyncio.sleep(0.5)

        assert election.is_leader is True
        assert elected_count == 1

        # Stop
        await election.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ============================================================================
# P49: Job Queue Tests
# ============================================================================

class TestJobQueue:
    """Tests for Job Queue module."""

    def test_job_state_enum(self):
        """Test JobState enum."""
        from src.core.job_queue import JobState

        assert JobState.PENDING.value == "pending"
        assert JobState.COMPLETED.value == "completed"
        assert JobState.DEAD.value == "dead"

    def test_job_priority_enum(self):
        """Test JobPriority enum."""
        from src.core.job_queue import JobPriority

        assert JobPriority.LOW.value < JobPriority.NORMAL.value
        assert JobPriority.NORMAL.value < JobPriority.HIGH.value
        assert JobPriority.HIGH.value < JobPriority.CRITICAL.value

    def test_create_job(self):
        """Test create_job function."""
        from src.core.job_queue import create_job, JobPriority, JobOptions

        job = create_job(
            queue_name="test_queue",
            handler="process_data",
            payload={"key": "value"},
            options=JobOptions(priority=JobPriority.HIGH),
        )

        assert job.queue_name == "test_queue"
        assert job.handler == "process_data"
        assert job.payload["key"] == "value"
        assert job.options.priority == JobPriority.HIGH

    def test_job_serialization(self):
        """Test job to_dict and from_dict."""
        from src.core.job_queue import create_job

        job = create_job(
            queue_name="test",
            handler="handler",
            payload={"data": 123},
        )

        data = job.to_dict()
        restored = type(job).from_dict(data)

        assert restored.id == job.id
        assert restored.handler == job.handler
        assert restored.payload == job.payload

    @pytest.mark.asyncio
    async def test_in_memory_queue_enqueue_dequeue(self):
        """Test InMemoryJobQueue enqueue and dequeue."""
        from src.core.job_queue import InMemoryJobQueue, create_job

        queue = InMemoryJobQueue()

        # Enqueue
        job = create_job("test_queue", "handler", {"x": 1})
        result = await queue.enqueue(job)
        assert result is True

        # Check size
        size = await queue.get_queue_size("test_queue")
        assert size == 1

        # Dequeue
        dequeued = await queue.dequeue("test_queue")
        assert dequeued is not None
        assert dequeued.id == job.id

    @pytest.mark.asyncio
    async def test_job_priority_ordering(self):
        """Test job priority ordering."""
        from src.core.job_queue import InMemoryJobQueue, create_job, JobOptions, JobPriority

        queue = InMemoryJobQueue()

        # Enqueue jobs with different priorities
        low_job = create_job("q", "h", {}, JobOptions(priority=JobPriority.LOW))
        high_job = create_job("q", "h", {}, JobOptions(priority=JobPriority.HIGH))
        normal_job = create_job("q", "h", {}, JobOptions(priority=JobPriority.NORMAL))

        await queue.enqueue(low_job)
        await queue.enqueue(high_job)
        await queue.enqueue(normal_job)

        # Dequeue should return highest priority first
        first = await queue.dequeue("q")
        assert first.id == high_job.id

        second = await queue.dequeue("q")
        assert second.id == normal_job.id

        third = await queue.dequeue("q")
        assert third.id == low_job.id

    @pytest.mark.asyncio
    async def test_job_complete(self):
        """Test job completion."""
        from src.core.job_queue import InMemoryJobQueue, create_job, JobState

        queue = InMemoryJobQueue()

        job = create_job("q", "h", {})
        await queue.enqueue(job)
        job = await queue.dequeue("q")

        # Complete
        await queue.complete(job, result={"output": "done"})

        # Check state
        loaded = await queue.get_job(job.id)
        assert loaded.state == JobState.COMPLETED
        assert loaded.result["output"] == "done"

    @pytest.mark.asyncio
    async def test_job_fail_retry(self):
        """Test job failure and retry."""
        from src.core.job_queue import InMemoryJobQueue, create_job, JobOptions, JobState

        queue = InMemoryJobQueue()

        job = create_job("q", "h", {}, JobOptions(max_retries=2))
        await queue.enqueue(job)
        job = await queue.dequeue("q")

        # Fail
        await queue.fail(job, "Error occurred")

        # Check state (should be retrying)
        loaded = await queue.get_job(job.id)
        assert loaded.state == JobState.RETRYING
        assert loaded.last_error == "Error occurred"

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self):
        """Test dead letter queue."""
        from src.core.job_queue import InMemoryJobQueue, create_job, JobOptions, JobState

        queue = InMemoryJobQueue()

        job = create_job("q", "h", {}, JobOptions(max_retries=0))
        await queue.enqueue(job)
        job = await queue.dequeue("q")

        # Fail with no retries
        await queue.fail(job, "Fatal error")

        # Should be in dead letter
        loaded = await queue.get_job(job.id)
        assert loaded.state == JobState.DEAD

        dead_jobs = await queue.get_dead_letter_jobs("q")
        assert len(dead_jobs) == 1

    @pytest.mark.asyncio
    async def test_handler_registry(self):
        """Test HandlerRegistry."""
        from src.core.job_queue import HandlerRegistry, create_job

        registry = HandlerRegistry()

        # Register handler
        @registry.handler("test_handler")
        async def handle_test(job):
            return job.payload["x"] * 2

        # Get handler
        handler = registry.get("test_handler")
        assert handler is not None

        # Execute
        job = create_job("q", "test_handler", {"x": 21})
        result = await handler.handle(job)
        assert result == 42


# ============================================================================
# P50: API Gateway Tests
# ============================================================================

class TestAPIGateway:
    """Tests for API Gateway module."""

    def test_match_type_enum(self):
        """Test MatchType enum."""
        from src.core.api_gateway import MatchType

        assert MatchType.EXACT.value == "exact"
        assert MatchType.PREFIX.value == "prefix"
        assert MatchType.REGEX.value == "regex"

    def test_route_creation(self):
        """Test Route creation."""
        from src.core.api_gateway import Route, MatchType

        route = Route(
            name="api_v1",
            path="/api/v1",
            backend="api_service",
            methods=["GET", "POST"],
            match_type=MatchType.PREFIX,
        )

        assert route.name == "api_v1"
        assert route.path == "/api/v1"
        assert "GET" in route.methods

    def test_request_creation(self):
        """Test Request creation."""
        from src.core.api_gateway import Request

        request = Request(
            method="GET",
            path="/api/users",
            headers={"Authorization": "Bearer token"},
            query_params={"page": "1"},
        )

        assert request.method == "GET"
        assert request.headers["Authorization"] == "Bearer token"

    def test_path_matcher(self):
        """Test PathMatcher."""
        from src.core.api_gateway import PathMatcher, Route, Request, MatchType

        matcher = PathMatcher()

        # Prefix match
        route = Route(name="api", path="/api", backend="svc", match_type=MatchType.PREFIX)
        request = Request(method="GET", path="/api/users")
        result = matcher.match(request, route)
        assert result.matched is True

        # Exact match
        route = Route(name="health", path="/health", backend="svc", match_type=MatchType.EXACT)
        request = Request(method="GET", path="/health")
        result = matcher.match(request, route)
        assert result.matched is True

        # Non-match
        request = Request(method="GET", path="/other")
        result = matcher.match(request, route)
        assert result.matched is False

    def test_router(self):
        """Test Router."""
        from src.core.api_gateway import Router, Route, Request, MatchType

        router = Router()

        router.add_route(Route(
            name="api",
            path="/api",
            backend="api_service",
            match_type=MatchType.PREFIX,
        ))
        router.add_route(Route(
            name="health",
            path="/health",
            backend="health_service",
            match_type=MatchType.EXACT,
        ))

        # Match API route
        request = Request(method="GET", path="/api/users")
        match = router.match(request)
        assert match.matched is True
        assert match.route.name == "api"

        # Match health route
        request = Request(method="GET", path="/health")
        match = router.match(request)
        assert match.matched is True
        assert match.route.name == "health"

    def test_backend_pool(self):
        """Test BackendPool."""
        from src.core.api_gateway import Backend, BackendPool, BackendState

        pool = BackendPool(name="api")

        pool.add(Backend(id="b1", host="host1", port=8080, weight=1))
        pool.add(Backend(id="b2", host="host2", port=8080, weight=2))

        assert len(pool.backends) == 2
        assert len(pool.get_healthy()) == 2

        # Mark one unhealthy
        pool.backends[0].state = BackendState.UNHEALTHY
        assert len(pool.get_healthy()) == 1

    def test_round_robin_balancer(self):
        """Test RoundRobinBalancer."""
        from src.core.api_gateway import Backend, BackendPool, RoundRobinBalancer

        pool = BackendPool(name="test")
        pool.add(Backend(id="b1", host="h1", port=80))
        pool.add(Backend(id="b2", host="h2", port=80))
        pool.add(Backend(id="b3", host="h3", port=80))

        balancer = RoundRobinBalancer()

        # Should cycle through backends
        ids = [balancer.select(pool).id for _ in range(6)]
        assert ids == ["b1", "b2", "b3", "b1", "b2", "b3"]

    def test_least_connections_balancer(self):
        """Test LeastConnectionsBalancer."""
        from src.core.api_gateway import Backend, BackendPool, LeastConnectionsBalancer

        pool = BackendPool(name="test")
        b1 = Backend(id="b1", host="h1", port=80)
        b2 = Backend(id="b2", host="h2", port=80)
        b1.active_connections = 5
        b2.active_connections = 2

        pool.add(b1)
        pool.add(b2)

        balancer = LeastConnectionsBalancer()
        selected = balancer.select(pool)

        # Should select b2 (fewer connections)
        assert selected.id == "b2"

    @pytest.mark.asyncio
    async def test_gateway_routing(self):
        """Test Gateway routing."""
        from src.core.api_gateway import (
            Gateway, Router, Route, Request, MatchType,
            BackendPool, Backend
        )

        router = Router()
        router.add_route(Route(
            name="api",
            path="/api",
            backend="api_pool",
            match_type=MatchType.PREFIX,
        ))

        gateway = Gateway(router)

        pool = BackendPool(name="api_pool")
        pool.add(Backend(id="b1", host="localhost", port=8080))
        gateway.register_backend_pool("api_pool", pool)

        # Handle request
        request = Request(method="GET", path="/api/users")
        response = await gateway.handle(request)

        assert response.status_code == 200
        assert response.backend == "b1"

    @pytest.mark.asyncio
    async def test_gateway_no_route(self):
        """Test Gateway with no matching route."""
        from src.core.api_gateway import Gateway, Router, Request, GatewayErrorCode

        router = Router()
        gateway = Gateway(router)

        request = Request(method="GET", path="/unknown")
        response = await gateway.handle(request)

        assert response.status_code == 404
        assert response.error_code == GatewayErrorCode.NO_ROUTE


# ============================================================================
# P51: Saga Pattern Tests
# ============================================================================

class TestSagaPattern:
    """Tests for Saga Pattern module."""

    def test_saga_state_enum(self):
        """Test SagaState enum."""
        from src.core.saga import SagaState

        assert SagaState.PENDING.value == "pending"
        assert SagaState.COMPLETED.value == "completed"
        assert SagaState.COMPENSATED.value == "compensated"

    def test_step_result(self):
        """Test StepResult."""
        from src.core.saga import StepResult

        result = StepResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data["key"] == "value"

    def test_saga_context(self):
        """Test SagaContext."""
        from src.core.saga import SagaContext, SagaState

        context = SagaContext(saga_id="saga_123")

        context.set("order_id", "order_456")
        assert context.get("order_id") == "order_456"

        data = context.to_dict()
        assert data["saga_id"] == "saga_123"

    def test_saga_builder(self):
        """Test SagaBuilder."""
        from src.core.saga import SagaBuilder

        def create_order(ctx):
            ctx.set("order_id", "123")
            return {"created": True}

        def cancel_order(ctx):
            return {"cancelled": True}

        definition = (
            SagaBuilder("order_saga")
            .step("create_order", create_order, cancel_order)
            .with_timeout(60)
            .build()
        )

        assert definition.name == "order_saga"
        assert len(definition.steps) == 1
        assert definition.timeout_seconds == 60

    @pytest.mark.asyncio
    async def test_saga_execution_success(self):
        """Test successful saga execution."""
        from src.core.saga import SagaBuilder, SagaOrchestrator

        executed_steps = []
        compensated_steps = []

        def step1_exec(ctx):
            executed_steps.append("step1")
            ctx.set("step1_done", True)

        def step1_comp(ctx):
            compensated_steps.append("step1")

        def step2_exec(ctx):
            executed_steps.append("step2")
            ctx.set("step2_done", True)

        def step2_comp(ctx):
            compensated_steps.append("step2")

        definition = (
            SagaBuilder("test_saga")
            .step("step1", step1_exec, step1_comp)
            .step("step2", step2_exec, step2_comp)
            .build()
        )

        orchestrator = SagaOrchestrator()
        orchestrator.register(definition)

        result = await orchestrator.execute("test_saga")

        assert result.success is True
        assert executed_steps == ["step1", "step2"]
        assert compensated_steps == []  # No compensation needed

    @pytest.mark.asyncio
    async def test_saga_compensation_on_failure(self):
        """Test saga compensation when step fails."""
        from src.core.saga import SagaBuilder, SagaOrchestrator, SagaState

        executed_steps = []
        compensated_steps = []

        def step1_exec(ctx):
            executed_steps.append("step1")

        def step1_comp(ctx):
            compensated_steps.append("step1")

        def step2_exec(ctx):
            executed_steps.append("step2")
            raise ValueError("Step 2 failed!")

        def step2_comp(ctx):
            compensated_steps.append("step2")

        definition = (
            SagaBuilder("failing_saga")
            .step("step1", step1_exec, step1_comp)
            .step("step2", step2_exec, step2_comp)
            .build()
        )

        orchestrator = SagaOrchestrator()
        orchestrator.register(definition)

        result = await orchestrator.execute("failing_saga")

        assert result.success is False
        assert result.state == SagaState.COMPENSATED
        assert executed_steps == ["step1", "step2"]
        assert "step1" in compensated_steps  # step1 should be compensated

    @pytest.mark.asyncio
    async def test_saga_with_initial_data(self):
        """Test saga with initial data."""
        from src.core.saga import SagaBuilder, SagaOrchestrator

        captured_order_id = None

        def process_order(ctx):
            nonlocal captured_order_id
            captured_order_id = ctx.get("order_id")

        def cancel_order(ctx):
            pass

        definition = (
            SagaBuilder("order_saga")
            .step("process", process_order, cancel_order)
            .build()
        )

        orchestrator = SagaOrchestrator()
        orchestrator.register(definition)

        await orchestrator.execute("order_saga", initial_data={"order_id": "ORD-123"})

        assert captured_order_id == "ORD-123"

    @pytest.mark.asyncio
    async def test_saga_step_result_passing(self):
        """Test passing data between saga steps."""
        from src.core.saga import SagaBuilder, SagaOrchestrator

        def create_order(ctx):
            return {"order_id": "ORD-456"}

        def process_payment(ctx):
            order_data = ctx.get_step_result("create_order")
            ctx.set("processed_order", order_data["order_id"])

        def cancel_order(ctx):
            pass

        def refund_payment(ctx):
            pass

        definition = (
            SagaBuilder("order_payment_saga")
            .step("create_order", create_order, cancel_order)
            .step("process_payment", process_payment, refund_payment)
            .build()
        )

        orchestrator = SagaOrchestrator()
        orchestrator.register(definition)

        result = await orchestrator.execute("order_payment_saga")

        assert result.success is True
        assert result.context.get("processed_order") == "ORD-456"

    @pytest.mark.asyncio
    async def test_saga_store(self):
        """Test InMemorySagaStore."""
        from src.core.saga import InMemorySagaStore, SagaContext, SagaState

        store = InMemorySagaStore()

        context = SagaContext(saga_id="saga_1", state=SagaState.RUNNING)

        await store.save(context)
        loaded = await store.load("saga_1")

        assert loaded is not None
        assert loaded.saga_id == "saga_1"

        pending = await store.list_pending()
        assert "saga_1" in pending


# ============================================================================
# Integration Tests
# ============================================================================

class TestP48P51Integration:
    """Integration tests for P48-P51 modules."""

    @pytest.mark.asyncio
    async def test_saga_with_distributed_lock(self):
        """Test saga using distributed lock for coordination."""
        from src.core.distributed_lock import InMemoryLock, LockManager
        from src.core.saga import SagaBuilder, SagaOrchestrator

        lock = InMemoryLock()
        manager = LockManager(lock)

        async def acquire_resource(ctx):
            result = await manager.acquire("resource_lock", ttl_seconds=30)
            ctx.set("lock_acquired", result.success)
            return result.success

        async def release_resource(ctx):
            await manager.release("resource_lock")

        definition = (
            SagaBuilder("locked_saga")
            .step("acquire", acquire_resource, release_resource)
            .build()
        )

        orchestrator = SagaOrchestrator()
        orchestrator.register(definition)

        result = await orchestrator.execute("locked_saga")

        assert result.success is True
        assert result.context.get("lock_acquired") is True

    @pytest.mark.asyncio
    async def test_job_queue_with_gateway_routing(self):
        """Test job processing triggered by gateway."""
        from src.core.job_queue import InMemoryJobQueue, create_job, HandlerRegistry
        from src.core.api_gateway import Router, Route, Request, MatchType

        queue = InMemoryJobQueue()
        registry = HandlerRegistry()

        processed_jobs = []

        @registry.handler("process_request")
        async def process_request(job):
            processed_jobs.append(job.payload)
            return {"processed": True}

        # Simulate gateway creating job
        router = Router()
        router.add_route(Route(name="async", path="/async", backend="queue"))

        request = Request(method="POST", path="/async/task")
        match = router.match(request)

        if match.matched:
            job = create_job(
                queue_name="tasks",
                handler="process_request",
                payload={"path": request.path},
            )
            await queue.enqueue(job)

        # Process job
        job = await queue.dequeue("tasks")
        handler = registry.get(job.handler)
        result = await handler.handle(job)

        assert result["processed"] is True
        assert len(processed_jobs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
