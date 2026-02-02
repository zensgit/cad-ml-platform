"""Unit tests for P40-P43: Tracing, Enhanced Resilience, Service Mesh, Schema Registry."""

import asyncio
import json
import time
import pytest
from typing import Any, Dict, List


# ============================================================================
# P40: OpenTelemetry Tracing Tests
# ============================================================================

class TestTraceContext:
    """Test trace context management."""

    def test_trace_id_generation(self):
        """Test trace ID generation."""
        from src.core.tracing import TraceId

        trace_id = TraceId.generate()
        assert trace_id.is_valid()
        assert len(trace_id.to_hex()) == 32

    def test_trace_id_parsing(self):
        """Test trace ID parsing from hex."""
        from src.core.tracing import TraceId

        hex_str = "0123456789abcdef0123456789abcdef"
        trace_id = TraceId.from_hex(hex_str)
        assert trace_id.to_hex() == hex_str

    def test_span_id_generation(self):
        """Test span ID generation."""
        from src.core.tracing import SpanId

        span_id = SpanId.generate()
        assert span_id.is_valid()
        assert len(span_id.to_hex()) == 16

    def test_span_context_creation(self):
        """Test span context creation."""
        from src.core.tracing import SpanContext, TraceFlags

        context = SpanContext.create_root()
        assert context.is_valid()
        assert context.is_sampled()

    def test_span_context_child(self):
        """Test creating child context."""
        from src.core.tracing import SpanContext

        parent = SpanContext.create_root()
        child = parent.create_child()

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id


class TestContextPropagation:
    """Test trace context propagation."""

    def test_w3c_inject_extract(self):
        """Test W3C trace context injection and extraction."""
        from src.core.tracing import SpanContext, W3CTraceContextPropagator

        propagator = W3CTraceContextPropagator()
        context = SpanContext.create_root()
        carrier: Dict[str, str] = {}

        propagator.inject(context, carrier)
        assert "traceparent" in carrier

        extracted = propagator.extract(carrier)
        assert extracted is not None
        assert extracted.trace_id == context.trace_id
        assert extracted.span_id == context.span_id

    def test_b3_inject_extract(self):
        """Test B3 trace context injection and extraction."""
        from src.core.tracing import SpanContext, B3Propagator

        propagator = B3Propagator()
        context = SpanContext.create_root()
        carrier: Dict[str, str] = {}

        propagator.inject(context, carrier)
        assert "X-B3-TraceId" in carrier
        assert "X-B3-SpanId" in carrier

        extracted = propagator.extract(carrier)
        assert extracted is not None
        assert extracted.trace_id == context.trace_id

    def test_composite_propagator(self):
        """Test composite propagator."""
        from src.core.tracing import (
            SpanContext,
            CompositePropagator,
            W3CTraceContextPropagator,
            B3Propagator,
        )

        propagator = CompositePropagator([
            W3CTraceContextPropagator(),
            B3Propagator(),
        ])
        context = SpanContext.create_root()
        carrier: Dict[str, str] = {}

        propagator.inject(context, carrier)
        # Should have both formats
        assert "traceparent" in carrier
        assert "X-B3-TraceId" in carrier


class TestSpan:
    """Test span creation and management."""

    def test_span_creation(self):
        """Test creating a span."""
        from src.core.tracing import Tracer, SpanKind

        tracer = Tracer("test-service")
        span = tracer.start_span("test-operation", kind=SpanKind.SERVER)

        assert span.name == "test-operation"
        assert span.kind == SpanKind.SERVER
        assert span.is_recording()

    def test_span_attributes(self):
        """Test setting span attributes."""
        from src.core.tracing import Tracer

        tracer = Tracer("test-service")
        span = tracer.start_span("test-operation")

        span.set_attribute("key1", "value1")
        span.set_attributes({"key2": "value2", "key3": 123})

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"
        assert span.attributes["key3"] == 123

    def test_span_events(self):
        """Test adding span events."""
        from src.core.tracing import Tracer

        tracer = Tracer("test-service")
        span = tracer.start_span("test-operation")

        span.add_event("event1", {"attr": "value"})
        span.add_event("event2")

        assert len(span.events) == 2
        assert span.events[0].name == "event1"

    def test_span_context_manager(self):
        """Test span as context manager."""
        from src.core.tracing import Tracer, StatusCode

        tracer = Tracer("test-service")

        with tracer.start_as_current_span("test-operation") as span:
            span.set_attribute("inside", True)

        assert not span.is_recording()
        assert span.end_time is not None

    def test_span_exception_recording(self):
        """Test recording exceptions."""
        from src.core.tracing import Tracer, StatusCode

        tracer = Tracer("test-service")

        try:
            with tracer.start_as_current_span("failing-operation") as span:
                raise ValueError("test error")
        except ValueError:
            pass

        assert span.status.code == StatusCode.ERROR
        assert len(span.events) == 1
        assert span.events[0].name == "exception"


class TestSampler:
    """Test sampling strategies."""

    def test_always_on_sampler(self):
        """Test always-on sampler."""
        from src.core.tracing import AlwaysOnSampler, TraceId

        sampler = AlwaysOnSampler()
        result = sampler.should_sample(None, TraceId.generate(), "test", None)
        assert result.decision is True

    def test_always_off_sampler(self):
        """Test always-off sampler."""
        from src.core.tracing import AlwaysOffSampler, TraceId

        sampler = AlwaysOffSampler()
        result = sampler.should_sample(None, TraceId.generate(), "test", None)
        assert result.decision is False

    def test_trace_id_ratio_sampler(self):
        """Test trace ID ratio sampler."""
        from src.core.tracing import TraceIdRatioSampler, TraceId

        sampler = TraceIdRatioSampler(0.5)

        # Test multiple traces - should have roughly 50% sampled
        sampled = 0
        for _ in range(100):
            result = sampler.should_sample(None, TraceId.generate(), "test", None)
            if result.decision:
                sampled += 1

        # Allow some variance (30-70%)
        assert 30 <= sampled <= 70

    def test_parent_based_sampler(self):
        """Test parent-based sampler."""
        from src.core.tracing import (
            ParentBasedSampler,
            AlwaysOnSampler,
            AlwaysOffSampler,
            SpanContext,
            TraceId,
            TraceFlags,
        )

        sampler = ParentBasedSampler(AlwaysOnSampler())

        # Root span should be sampled
        result = sampler.should_sample(None, TraceId.generate(), "test", None)
        assert result.decision is True

        # Child of sampled parent should be sampled
        sampled_parent = SpanContext.create_root()
        result = sampler.should_sample(sampled_parent, TraceId.generate(), "test", None)
        assert result.decision is True


# ============================================================================
# P41: Enhanced Resilience Tests
# ============================================================================

class TestCircuitBreaker:
    """Test enhanced circuit breaker."""

    def test_initial_state(self):
        """Test circuit breaker starts closed."""
        from src.core.resilience_enhanced import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_success_recording(self):
        """Test recording successes."""
        from src.core.resilience_enhanced import CircuitBreaker

        cb = CircuitBreaker("test")
        cb.record_success(0.1)
        cb.record_success(0.2)

        metrics = cb.get_metrics()
        assert metrics.success_count == 2

    def test_failure_opens_circuit(self):
        """Test failures open the circuit."""
        from src.core.resilience_enhanced import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_rate_threshold=0.5,
            sliding_window_size=4,
        )
        cb = CircuitBreaker("test", config)

        # Record enough failures to trigger open
        cb.record_success(0.1)
        cb.record_success(0.1)
        cb.record_failure(Exception("fail"))
        cb.record_failure(Exception("fail"))

        # Should be open now (50% failure rate)
        assert cb.state == CircuitState.OPEN

    def test_allow_request(self):
        """Test request allowance logic."""
        from src.core.resilience_enhanced import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        cb = CircuitBreaker("test")
        assert cb.allow_request() is True

        # Manually force open
        cb._transition_to(CircuitState.OPEN)
        assert cb.allow_request() is False


class TestBulkhead:
    """Test bulkhead pattern."""

    @pytest.mark.asyncio
    async def test_semaphore_bulkhead(self):
        """Test semaphore-based bulkhead."""
        from src.core.resilience_enhanced import SemaphoreBulkhead, BulkheadConfig

        config = BulkheadConfig(max_concurrent_calls=2)
        bulkhead = SemaphoreBulkhead(config)

        async with bulkhead:
            metrics = bulkhead.get_metrics()
            assert metrics.active_calls == 1

        metrics = bulkhead.get_metrics()
        assert metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_bulkhead_rejection(self):
        """Test bulkhead rejects when full."""
        from src.core.resilience_enhanced import (
            SemaphoreBulkhead,
            BulkheadConfig,
            BulkheadError,
        )

        config = BulkheadConfig(max_concurrent_calls=1, max_wait_time=0)
        bulkhead = SemaphoreBulkhead(config)

        async with bulkhead:
            # Try to acquire again - should be rejected
            acquired = await bulkhead.acquire()
            assert acquired is False


class TestTimeout:
    """Test timeout policies."""

    @pytest.mark.asyncio
    async def test_simple_timeout_success(self):
        """Test successful operation within timeout."""
        from src.core.resilience_enhanced import SimpleTimeout, TimeoutConfig

        timeout = SimpleTimeout(TimeoutConfig(timeout=5.0))

        async def fast_operation():
            return "done"

        result = await timeout.execute(fast_operation)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_simple_timeout_exceeds(self):
        """Test operation exceeding timeout."""
        from src.core.resilience_enhanced import SimpleTimeout, TimeoutConfig, TimeoutError

        timeout = SimpleTimeout(TimeoutConfig(timeout=0.1))

        async def slow_operation():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(TimeoutError):
            await timeout.execute(slow_operation)


class TestFallback:
    """Test fallback strategies."""

    @pytest.mark.asyncio
    async def test_static_fallback(self):
        """Test static fallback value."""
        from src.core.resilience_enhanced import StaticFallback

        fallback = StaticFallback("default")
        result = await fallback.execute(Exception("error"))
        assert result == "default"

    @pytest.mark.asyncio
    async def test_function_fallback(self):
        """Test function-based fallback."""
        from src.core.resilience_enhanced import FunctionFallback

        fallback = FunctionFallback(lambda e: f"error: {e}")
        result = await fallback.execute(Exception("test"))
        assert result == "error: test"

    @pytest.mark.asyncio
    async def test_fallback_chain(self):
        """Test fallback chain."""
        from src.core.resilience_enhanced import (
            FallbackChain,
            FunctionFallback,
            StaticFallback,
        )

        def failing_fallback(e):
            raise Exception("also failed")

        chain = FallbackChain([
            FunctionFallback(failing_fallback),
            StaticFallback("final fallback"),
        ])

        result = await chain.execute(Exception("original"))
        assert result == "final fallback"


# ============================================================================
# P42: Service Mesh Tests
# ============================================================================

class TestServiceDiscovery:
    """Test service discovery."""

    @pytest.mark.asyncio
    async def test_register_service(self):
        """Test service registration."""
        from src.core.service_mesh import (
            ServiceInstance,
            InMemoryServiceRegistry,
        )

        registry = InMemoryServiceRegistry()
        instance = ServiceInstance(
            service_name="api-service",
            instance_id="instance-1",
            host="localhost",
            port=8080,
        )

        result = await registry.register(instance)
        assert result is True

        instances = await registry.get_instances("api-service")
        assert len(instances) == 1

    @pytest.mark.asyncio
    async def test_deregister_service(self):
        """Test service deregistration."""
        from src.core.service_mesh import (
            ServiceInstance,
            InMemoryServiceRegistry,
        )

        registry = InMemoryServiceRegistry()
        instance = ServiceInstance(
            service_name="api-service",
            instance_id="instance-1",
            host="localhost",
            port=8080,
        )

        await registry.register(instance)
        await registry.deregister("api-service", "instance-1")

        instances = await registry.get_instances("api-service")
        assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        """Test heartbeat updates."""
        from src.core.service_mesh import (
            ServiceInstance,
            InMemoryServiceRegistry,
        )

        registry = InMemoryServiceRegistry()
        instance = ServiceInstance(
            service_name="api-service",
            instance_id="instance-1",
            host="localhost",
            port=8080,
        )

        await registry.register(instance)
        initial_heartbeat = instance.last_heartbeat

        await asyncio.sleep(0.1)
        await registry.heartbeat("api-service", "instance-1")

        instances = await registry.get_instances("api-service")
        assert instances[0].last_heartbeat > initial_heartbeat


class TestLoadBalancer:
    """Test load balancing strategies."""

    def test_round_robin(self):
        """Test round-robin load balancing."""
        from src.core.service_mesh import (
            ServiceInstance,
            ServiceStatus,
            RoundRobinBalancer,
        )

        instances = [
            ServiceInstance("svc", "i1", "h1", 80, status=ServiceStatus.HEALTHY),
            ServiceInstance("svc", "i2", "h2", 80, status=ServiceStatus.HEALTHY),
        ]

        balancer = RoundRobinBalancer()

        selections = [balancer.select(instances) for _ in range(4)]
        instance_ids = [s.instance_id for s in selections if s]

        assert "i1" in instance_ids
        assert "i2" in instance_ids

    def test_least_connections(self):
        """Test least connections load balancing."""
        from src.core.service_mesh import (
            ServiceInstance,
            ServiceStatus,
            LeastConnectionsBalancer,
        )

        instances = [
            ServiceInstance("svc", "i1", "h1", 80, status=ServiceStatus.HEALTHY),
            ServiceInstance("svc", "i2", "h2", 80, status=ServiceStatus.HEALTHY),
        ]

        balancer = LeastConnectionsBalancer()

        # First selection
        selected1 = balancer.select(instances)
        assert selected1 is not None

        # Second should get different instance (both have 0 connections)
        selected2 = balancer.select(instances)
        assert selected2 is not None

    def test_consistent_hash(self):
        """Test consistent hash load balancing."""
        from src.core.service_mesh import (
            ServiceInstance,
            ServiceStatus,
            ConsistentHashBalancer,
        )

        instances = [
            ServiceInstance("svc", "i1", "h1", 80, status=ServiceStatus.HEALTHY),
            ServiceInstance("svc", "i2", "h2", 80, status=ServiceStatus.HEALTHY),
        ]

        balancer = ConsistentHashBalancer()

        # Same key should always get same instance
        context = {"user_id": "user-123"}
        selected1 = balancer.select(instances, context)
        selected2 = balancer.select(instances, context)

        assert selected1.instance_id == selected2.instance_id


class TestHealthAggregation:
    """Test health aggregation."""

    @pytest.mark.asyncio
    async def test_function_health_check(self):
        """Test function-based health check."""
        from src.core.service_mesh import (
            FunctionHealthCheck,
            HealthStatus,
        )

        check = FunctionHealthCheck("test", lambda: True)
        result = await check.check()

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_aggregator(self):
        """Test health aggregation."""
        from src.core.service_mesh import (
            HealthAggregator,
            FunctionHealthCheck,
            HealthStatus,
        )

        aggregator = HealthAggregator()
        aggregator.add_check(FunctionHealthCheck("check1", lambda: True))
        aggregator.add_check(FunctionHealthCheck("check2", lambda: True))

        result = await aggregator.check_all()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.checks) == 2


# ============================================================================
# P43: Schema Registry Tests
# ============================================================================

class TestSchemaDefinition:
    """Test schema definition and parsing."""

    def test_json_schema_parsing(self):
        """Test JSON schema parsing."""
        from src.core.schema_registry import JSONSchemaParser

        parser = JSONSchemaParser()
        schema_str = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        })

        errors = parser.validate(schema_str)
        assert len(errors) == 0

    def test_json_schema_invalid(self):
        """Test invalid JSON schema detection."""
        from src.core.schema_registry import JSONSchemaParser

        parser = JSONSchemaParser()
        schema_str = json.dumps({
            "type": "invalid_type",
        })

        errors = parser.validate(schema_str)
        assert len(errors) > 0

    def test_avro_schema_parsing(self):
        """Test Avro schema parsing."""
        from src.core.schema_registry import AvroSchemaParser

        parser = AvroSchemaParser()
        schema_str = json.dumps({
            "type": "record",
            "name": "User",
            "fields": [
                {"name": "name", "type": "string"},
                {"name": "age", "type": "int"},
            ],
        })

        errors = parser.validate(schema_str)
        assert len(errors) == 0


class TestSchemaCompatibility:
    """Test schema compatibility checking."""

    def test_backward_compatible_add_optional(self):
        """Test backward compatibility - adding optional field."""
        from src.core.schema_registry import (
            Schema,
            SchemaType,
            CompatibilityMode,
            get_compatibility_checker,
        )

        checker = get_compatibility_checker(SchemaType.JSON)

        old_schema = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }))

        new_schema = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }))

        result = checker.check(new_schema, [old_schema], CompatibilityMode.BACKWARD)
        assert result.compatible

    def test_backward_incompatible_add_required(self):
        """Test backward incompatibility - adding required field without default."""
        from src.core.schema_registry import (
            Schema,
            SchemaType,
            CompatibilityMode,
            get_compatibility_checker,
        )

        checker = get_compatibility_checker(SchemaType.JSON)

        old_schema = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }))

        new_schema = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }))

        result = checker.check(new_schema, [old_schema], CompatibilityMode.BACKWARD)
        assert not result.compatible


class TestSchemaRegistry:
    """Test schema registry."""

    @pytest.mark.asyncio
    async def test_register_schema(self):
        """Test schema registration."""
        from src.core.schema_registry import (
            Schema,
            SchemaType,
            InMemorySchemaRegistry,
        )

        registry = InMemorySchemaRegistry()
        schema = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }))

        schema_id = await registry.register_schema("user-value", schema)
        assert schema_id >= 1

    @pytest.mark.asyncio
    async def test_get_schema_versions(self):
        """Test getting schema versions."""
        from src.core.schema_registry import (
            Schema,
            SchemaType,
            InMemorySchemaRegistry,
        )

        registry = InMemorySchemaRegistry()

        schema1 = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }))

        schema2 = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }))

        await registry.register_schema("user-value", schema1)
        await registry.register_schema("user-value", schema2)

        versions = await registry.get_versions("user-value")
        assert len(versions) == 2
        assert 1 in versions
        assert 2 in versions

    @pytest.mark.asyncio
    async def test_get_latest_schema(self):
        """Test getting latest schema."""
        from src.core.schema_registry import (
            Schema,
            SchemaType,
            InMemorySchemaRegistry,
        )

        registry = InMemorySchemaRegistry()

        schema1 = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {"v": {"type": "integer", "enum": [1]}},
        }))

        schema2 = Schema(SchemaType.JSON, json.dumps({
            "type": "object",
            "properties": {"v": {"type": "integer", "enum": [1, 2]}},
        }))

        await registry.register_schema("test", schema1)
        await registry.register_schema("test", schema2)

        latest = await registry.get_latest_schema("test")
        assert latest is not None
        assert latest.metadata.version == 2

    @pytest.mark.asyncio
    async def test_delete_schema(self):
        """Test deleting schema version."""
        from src.core.schema_registry import (
            Schema,
            SchemaType,
            InMemorySchemaRegistry,
        )

        registry = InMemorySchemaRegistry()
        schema = Schema(SchemaType.JSON, json.dumps({"type": "object"}))

        await registry.register_schema("test", schema)
        result = await registry.delete_schema("test", 1)

        assert result is True
        versions = await registry.get_versions("test")
        assert len(versions) == 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
