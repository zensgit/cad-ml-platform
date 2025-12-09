"""Tests for Vision Provider Phase 8 features.

Tests cover:
- Service mesh integration
- Distributed tracing
- Configuration management
- Feature flags
- Graceful degradation
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.vision.base import VisionDescription, VisionProvider


# ============================================================================
# Mock Provider for Testing
# ============================================================================


class MockVisionProvider(VisionProvider):
    """Mock vision provider for testing."""

    def __init__(self, name: str = "mock", fail: bool = False):
        self._name = name
        self._fail = fail
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        self._call_count += 1
        if self._fail:
            raise RuntimeError("Mock provider failure")
        return VisionDescription(
            summary=f"Mock analysis by {self._name}",
            details=["Detailed mock description"],
            confidence=0.85,
        )


# ============================================================================
# Service Mesh Tests
# ============================================================================


class TestServiceMesh:
    """Tests for service mesh module."""

    def test_service_status_enum(self):
        """Test ServiceStatus enum values."""
        from src.core.vision.service_mesh import ServiceStatus

        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"
        assert ServiceStatus.DRAINING.value == "draining"

    def test_load_balancer_policy_enum(self):
        """Test LoadBalancerPolicy enum values."""
        from src.core.vision.service_mesh import LoadBalancerPolicy

        assert LoadBalancerPolicy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalancerPolicy.LEAST_CONNECTIONS.value == "least_connections"
        assert LoadBalancerPolicy.RANDOM.value == "random"

    def test_service_instance_creation(self):
        """Test ServiceInstance creation."""
        from src.core.vision.service_mesh import ServiceInstance, ServiceStatus

        instance = ServiceInstance(
            instance_id="inst-1",
            service_name="vision",
            host="localhost",
            port=8080,
        )

        assert instance.instance_id == "inst-1"
        assert instance.service_name == "vision"
        assert instance.address == "localhost:8080"
        assert instance.status == ServiceStatus.HEALTHY
        assert instance.success_rate == 1.0

    def test_service_instance_stats(self):
        """Test ServiceInstance stats tracking."""
        from src.core.vision.service_mesh import ServiceInstance

        instance = ServiceInstance(
            instance_id="inst-1",
            service_name="vision",
            host="localhost",
            port=8080,
        )

        instance.total_requests = 10
        instance.failed_requests = 2

        assert instance.success_rate == 0.8

    def test_service_registry_register(self):
        """Test ServiceRegistry registration."""
        from src.core.vision.service_mesh import ServiceRegistry, ServiceInstance

        registry = ServiceRegistry()

        instance = ServiceInstance(
            instance_id="inst-1",
            service_name="vision",
            host="localhost",
            port=8080,
        )

        registry.register("vision", instance)

        instances = registry.get_instances("vision")
        assert len(instances) == 1
        assert instances[0].instance_id == "inst-1"

    def test_service_registry_deregister(self):
        """Test ServiceRegistry deregistration."""
        from src.core.vision.service_mesh import ServiceRegistry, ServiceInstance

        registry = ServiceRegistry()

        instance = ServiceInstance(
            instance_id="inst-1",
            service_name="vision",
            host="localhost",
            port=8080,
        )

        registry.register("vision", instance)
        assert len(registry.get_instances("vision")) == 1

        result = registry.deregister("vision", "inst-1")
        assert result is True
        assert len(registry.get_instances("vision")) == 0

    def test_service_registry_heartbeat(self):
        """Test ServiceRegistry heartbeat."""
        from src.core.vision.service_mesh import ServiceRegistry, ServiceInstance

        registry = ServiceRegistry()

        instance = ServiceInstance(
            instance_id="inst-1",
            service_name="vision",
            host="localhost",
            port=8080,
        )

        registry.register("vision", instance)

        result = registry.heartbeat("vision", "inst-1")
        assert result is True

    def test_load_balancer_round_robin(self):
        """Test LoadBalancer round robin."""
        from src.core.vision.service_mesh import (
            LoadBalancer, LoadBalancerPolicy, ServiceInstance
        )

        lb = LoadBalancer(policy=LoadBalancerPolicy.ROUND_ROBIN)

        instances = [
            ServiceInstance(f"inst-{i}", "vision", "localhost", 8080 + i)
            for i in range(3)
        ]

        selected = []
        for _ in range(6):
            inst = lb.select("vision", instances)
            selected.append(inst.instance_id)

        # Should cycle through instances
        assert selected == ["inst-0", "inst-1", "inst-2", "inst-0", "inst-1", "inst-2"]

    def test_load_balancer_least_connections(self):
        """Test LoadBalancer least connections."""
        from src.core.vision.service_mesh import (
            LoadBalancer, LoadBalancerPolicy, ServiceInstance
        )

        lb = LoadBalancer(policy=LoadBalancerPolicy.LEAST_CONNECTIONS)

        instances = [
            ServiceInstance(f"inst-{i}", "vision", "localhost", 8080 + i)
            for i in range(3)
        ]

        instances[0].active_connections = 5
        instances[1].active_connections = 2
        instances[2].active_connections = 10

        selected = lb.select("vision", instances)
        assert selected.instance_id == "inst-1"  # Least connections

    def test_traffic_manager_add_rule(self):
        """Test TrafficManager add rule."""
        from src.core.vision.service_mesh import TrafficManager, TrafficRule

        manager = TrafficManager()

        rule = TrafficRule(
            rule_id="rule-1",
            service_name="vision",
            match_headers={"x-version": "v2"},
            destination_service="vision-v2",
        )

        manager.add_rule(rule)

        found = manager.get_rule("rule-1")
        assert found is not None
        assert found.destination_service == "vision-v2"

    def test_traffic_manager_match_rule(self):
        """Test TrafficManager rule matching."""
        from src.core.vision.service_mesh import TrafficManager, TrafficRule

        manager = TrafficManager()

        rule = TrafficRule(
            rule_id="rule-1",
            service_name="vision",
            match_headers={"x-version": "v2"},
            destination_service="vision-v2",
        )

        manager.add_rule(rule)

        matched = manager.match_rule("vision", {"x-version": "v2"})
        assert matched is not None
        assert matched.rule_id == "rule-1"

        not_matched = manager.match_rule("vision", {"x-version": "v1"})
        assert not_matched is None

    def test_service_mesh_register(self):
        """Test ServiceMesh service registration."""
        from src.core.vision.service_mesh import ServiceMesh

        mesh = ServiceMesh()

        instance = mesh.register_service("vision", "localhost", 8080)

        assert instance is not None
        assert instance.service_name == "vision"
        assert instance.host == "localhost"
        assert instance.port == 8080

    def test_service_mesh_select_instance(self):
        """Test ServiceMesh instance selection."""
        from src.core.vision.service_mesh import ServiceMesh

        mesh = ServiceMesh()

        mesh.register_service("vision", "localhost", 8080)
        mesh.register_service("vision", "localhost", 8081)

        selected = mesh.select_instance("vision")
        assert selected is not None

    @pytest.mark.asyncio
    async def test_mesh_vision_provider(self):
        """Test MeshVisionProvider."""
        from src.core.vision.service_mesh import ServiceMesh, create_mesh_provider

        mesh = ServiceMesh()
        mock_provider = MockVisionProvider()

        provider = create_mesh_provider(mock_provider, mesh, "vision")
        provider.register("localhost", 8080)

        result = await provider.analyze_image(b"test image")

        assert result.summary == "Mock analysis by mock"
        assert mock_provider._call_count == 1


# ============================================================================
# Distributed Tracing Tests
# ============================================================================


class TestDistributedTracing:
    """Tests for distributed tracing module."""

    def test_span_kind_enum(self):
        """Test SpanKind enum values."""
        from src.core.vision.distributed_tracing import SpanKind

        assert SpanKind.INTERNAL.value == "internal"
        assert SpanKind.SERVER.value == "server"
        assert SpanKind.CLIENT.value == "client"

    def test_span_status_enum(self):
        """Test SpanStatus enum values."""
        from src.core.vision.distributed_tracing import SpanStatus

        assert SpanStatus.UNSET.value == "unset"
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"

    def test_trace_id_creation(self):
        """Test TraceId creation."""
        from src.core.vision.distributed_tracing import TraceId

        trace_id = TraceId()
        assert trace_id.value is not None
        assert len(trace_id.value) == 32  # UUID hex

    def test_span_id_creation(self):
        """Test SpanId creation."""
        from src.core.vision.distributed_tracing import SpanId

        span_id = SpanId()
        assert span_id.value is not None
        assert len(span_id.value) == 16

    def test_span_context_creation(self):
        """Test SpanContext creation."""
        from src.core.vision.distributed_tracing import SpanContext, TraceId, SpanId

        context = SpanContext(
            trace_id=TraceId(),
            span_id=SpanId(),
        )

        assert context.is_valid
        assert context.is_sampled

    def test_tracing_span_creation(self):
        """Test TracingSpan creation."""
        from src.core.vision.distributed_tracing import (
            TracingSpan, SpanContext, TraceId, SpanId, SpanKind
        )

        context = SpanContext(trace_id=TraceId(), span_id=SpanId())
        span = TracingSpan(
            name="test-span",
            context=context,
            kind=SpanKind.CLIENT,
        )

        assert span.name == "test-span"
        assert span.kind == SpanKind.CLIENT

    def test_tracing_span_attributes(self):
        """Test TracingSpan attributes."""
        from src.core.vision.distributed_tracing import (
            TracingSpan, SpanContext, TraceId, SpanId
        )

        context = SpanContext(trace_id=TraceId(), span_id=SpanId())
        span = TracingSpan(name="test-span", context=context)

        span.set_attribute("key1", "value1")
        span.set_attributes({"key2": "value2", "key3": 123})

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"
        assert span.attributes["key3"] == 123

    def test_tracing_span_events(self):
        """Test TracingSpan events."""
        from src.core.vision.distributed_tracing import (
            TracingSpan, SpanContext, TraceId, SpanId
        )

        context = SpanContext(trace_id=TraceId(), span_id=SpanId())
        span = TracingSpan(name="test-span", context=context)

        span.add_event("event1", {"detail": "value"})

        assert len(span.events) == 1
        assert span.events[0].name == "event1"

    def test_tracing_span_end(self):
        """Test TracingSpan end."""
        from src.core.vision.distributed_tracing import (
            TracingSpan, SpanContext, TraceId, SpanId
        )

        context = SpanContext(trace_id=TraceId(), span_id=SpanId())
        span = TracingSpan(name="test-span", context=context)

        assert span.end_time is None
        span.end()
        assert span.end_time is not None
        assert span.duration_ms is not None

    def test_always_on_sampler(self):
        """Test AlwaysOnSampler."""
        from src.core.vision.distributed_tracing import (
            AlwaysOnSampler, TraceId, SamplingDecision
        )

        sampler = AlwaysOnSampler()
        decision = sampler.should_sample(None, TraceId(), "test", {})

        assert decision == SamplingDecision.RECORD_AND_SAMPLE

    def test_always_off_sampler(self):
        """Test AlwaysOffSampler."""
        from src.core.vision.distributed_tracing import (
            AlwaysOffSampler, TraceId, SamplingDecision
        )

        sampler = AlwaysOffSampler()
        decision = sampler.should_sample(None, TraceId(), "test", {})

        assert decision == SamplingDecision.DROP

    def test_trace_id_ratio_sampler(self):
        """Test TraceIdRatioSampler."""
        from src.core.vision.distributed_tracing import (
            TraceIdRatioSampler, TraceId, SamplingDecision
        )

        sampler = TraceIdRatioSampler(ratio=1.0)
        decision = sampler.should_sample(None, TraceId(), "test", {})
        assert decision == SamplingDecision.RECORD_AND_SAMPLE

        sampler = TraceIdRatioSampler(ratio=0.0)
        decision = sampler.should_sample(None, TraceId(), "test", {})
        assert decision == SamplingDecision.DROP

    def test_in_memory_span_exporter(self):
        """Test InMemorySpanExporter."""
        from src.core.vision.distributed_tracing import (
            InMemorySpanExporter, TracingSpan, SpanContext, TraceId, SpanId
        )

        exporter = InMemorySpanExporter()

        context = SpanContext(trace_id=TraceId(), span_id=SpanId())
        span = TracingSpan(name="test-span", context=context)

        result = exporter.export([span])
        assert result is True

        spans = exporter.get_spans()
        assert len(spans) == 1

    def test_w3c_trace_context_propagator(self):
        """Test W3CTraceContextPropagator."""
        from src.core.vision.distributed_tracing import (
            W3CTraceContextPropagator, SpanContext, TraceId, SpanId
        )

        propagator = W3CTraceContextPropagator()

        context = SpanContext(
            trace_id=TraceId(value="a" * 32),
            span_id=SpanId(value="b" * 16),
            trace_flags=1,
        )

        carrier = {}
        propagator.inject(context, carrier)

        assert "traceparent" in carrier

        extracted = propagator.extract(carrier)
        assert extracted is not None
        assert extracted.trace_id.value == "a" * 32
        assert extracted.span_id.value == "b" * 16

    def test_b3_propagator(self):
        """Test B3Propagator."""
        from src.core.vision.distributed_tracing import (
            B3Propagator, SpanContext, TraceId, SpanId
        )

        propagator = B3Propagator()

        context = SpanContext(
            trace_id=TraceId(value="a" * 32),
            span_id=SpanId(value="b" * 16),
            trace_flags=1,
        )

        carrier = {}
        propagator.inject(context, carrier)

        assert "X-B3-TraceId" in carrier
        assert "X-B3-SpanId" in carrier

        extracted = propagator.extract(carrier)
        assert extracted is not None

    def test_tracer_provider(self):
        """Test TracerProvider."""
        from src.core.vision.distributed_tracing import TracerProvider, TracerConfig

        config = TracerConfig(service_name="test-service")
        provider = TracerProvider(config=config)

        tracer = provider.get_tracer("test-tracer")
        assert tracer is not None
        assert tracer.name == "test-tracer"

    def test_distributed_tracer_start_span(self):
        """Test DistributedTracer start span."""
        from src.core.vision.distributed_tracing import (
            TracerProvider, SpanKind
        )

        provider = TracerProvider()
        tracer = provider.get_tracer("test")

        span = tracer.start_span("test-operation", kind=SpanKind.CLIENT)
        assert span is not None
        assert span.name == "test-operation"
        assert span.kind == SpanKind.CLIENT

    @pytest.mark.asyncio
    async def test_tracing_vision_provider(self):
        """Test TracingVisionProvider."""
        from src.core.vision.distributed_tracing import create_tracing_provider

        mock_provider = MockVisionProvider()
        provider = create_tracing_provider(mock_provider, service_name="vision")

        result = await provider.analyze_image(b"test image")

        assert result.summary == "Mock analysis by mock"
        assert provider.stats.total_spans == 1


# ============================================================================
# Configuration Management Tests
# ============================================================================


class TestConfigurationManagement:
    """Tests for configuration management module."""

    def test_config_source_enum(self):
        """Test ConfigSource enum values."""
        from src.core.vision.config_management import ConfigSource

        assert ConfigSource.DEFAULT.value == "default"
        assert ConfigSource.FILE.value == "file"
        assert ConfigSource.ENVIRONMENT.value == "environment"

    def test_validation_level_enum(self):
        """Test ValidationLevel enum values."""
        from src.core.vision.config_management import ValidationLevel

        assert ValidationLevel.NONE.value == "none"
        assert ValidationLevel.WARN.value == "warn"
        assert ValidationLevel.STRICT.value == "strict"

    def test_config_value_creation(self):
        """Test ConfigValue creation."""
        from src.core.vision.config_management import ConfigValue, ConfigSource

        value = ConfigValue(
            key="test.key",
            value="test_value",
            source=ConfigSource.DEFAULT,
        )

        assert value.key == "test.key"
        assert value.value == "test_value"
        assert value.source == ConfigSource.DEFAULT

    def test_config_schema_validation(self):
        """Test ConfigSchema validation."""
        from src.core.vision.config_management import ConfigSchema

        schema = ConfigSchema(
            key="test.key",
            value_type=str,
            required=True,
        )

        # Valid value
        errors = schema.validate("valid")
        assert len(errors) == 0

        # Invalid type
        errors = schema.validate(123)
        assert len(errors) > 0

        # Missing required
        errors = schema.validate(None)
        assert len(errors) > 0

    def test_config_schema_choices(self):
        """Test ConfigSchema with choices."""
        from src.core.vision.config_management import ConfigSchema

        schema = ConfigSchema(
            key="test.key",
            value_type=str,
            choices=["a", "b", "c"],
        )

        errors = schema.validate("a")
        assert len(errors) == 0

        errors = schema.validate("d")
        assert len(errors) > 0

    def test_config_schema_range(self):
        """Test ConfigSchema with range."""
        from src.core.vision.config_management import ConfigSchema

        schema = ConfigSchema(
            key="test.key",
            value_type=int,
            min_value=0,
            max_value=100,
        )

        errors = schema.validate(50)
        assert len(errors) == 0

        errors = schema.validate(-1)
        assert len(errors) > 0

        errors = schema.validate(101)
        assert len(errors) > 0

    def test_dict_config_provider(self):
        """Test DictConfigProvider."""
        from src.core.vision.config_management import DictConfigProvider

        config = {
            "level1": {
                "level2": "value"
            },
            "simple": "direct"
        }

        provider = DictConfigProvider(config)

        assert provider.get("simple") == "direct"
        assert provider.get("level1.level2") == "value"
        assert provider.get("missing") is None

    def test_configuration_manager_get_set(self):
        """Test ConfigurationManager get/set."""
        from src.core.vision.config_management import ConfigurationManager

        manager = ConfigurationManager()

        manager.set("test.key", "value")
        result = manager.get("test.key")

        assert result == "value"

    def test_configuration_manager_with_provider(self):
        """Test ConfigurationManager with provider."""
        from src.core.vision.config_management import (
            ConfigurationManager, DictConfigProvider
        )

        manager = ConfigurationManager()
        manager.add_provider(DictConfigProvider({"key1": "value1"}))

        result = manager.get("key1")
        assert result == "value1"

    def test_configuration_manager_override(self):
        """Test ConfigurationManager override."""
        from src.core.vision.config_management import (
            ConfigurationManager, DictConfigProvider
        )

        manager = ConfigurationManager()
        manager.add_provider(DictConfigProvider({"key1": "value1"}))

        # Override
        manager.set("key1", "override")

        result = manager.get("key1")
        assert result == "override"

    def test_configuration_manager_delete(self):
        """Test ConfigurationManager delete."""
        from src.core.vision.config_management import ConfigurationManager

        manager = ConfigurationManager()
        manager.set("key1", "value1")

        result = manager.delete("key1")
        assert result is True

        value = manager.get("key1")
        assert value is None

    def test_config_watcher(self):
        """Test ConfigWatcher."""
        from src.core.vision.config_management import ConfigurationManager

        manager = ConfigurationManager()
        changes = []

        def on_change(change):
            changes.append(change)

        manager.watcher.watch("test.key", on_change)
        manager.set("test.key", "value")

        assert len(changes) == 1
        assert changes[0].new_value == "value"

    def test_config_profile(self):
        """Test ConfigProfile."""
        from src.core.vision.config_management import (
            ConfigurationManager, ConfigProfile, ProfileManager
        )

        manager = ConfigurationManager()
        profile_manager = ProfileManager(manager)

        profile = ConfigProfile(
            name="development",
            config={"debug": True, "log_level": "DEBUG"},
        )

        profile_manager.register_profile(profile)
        result = profile_manager.activate_profile("development")

        assert result is True
        assert manager.get("debug") is True

    def test_config_snapshot(self):
        """Test ConfigSnapshot."""
        from src.core.vision.config_management import (
            ConfigurationManager, ConfigSnapshotManager
        )

        manager = ConfigurationManager()
        manager.set("key1", "value1")

        snapshot_manager = ConfigSnapshotManager(manager)
        snapshot = snapshot_manager.create_snapshot("test-snapshot")

        assert snapshot is not None
        assert snapshot.config["key1"] == "value1"

        # Modify config
        manager.set("key1", "modified")

        # Restore
        result = snapshot_manager.restore_snapshot("test-snapshot")
        assert result is True
        assert manager.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_configurable_vision_provider(self):
        """Test ConfigurableVisionProvider."""
        from src.core.vision.config_management import create_configurable_provider

        mock_provider = MockVisionProvider()
        provider = create_configurable_provider(
            mock_provider,
            config={"vision.enabled": True},
        )

        result = await provider.analyze_image(b"test image")
        assert result.summary == "Mock analysis by mock"


# ============================================================================
# Feature Flags Tests
# ============================================================================


class TestFeatureFlags:
    """Tests for feature flags module."""

    def test_flag_type_enum(self):
        """Test FlagType enum values."""
        from src.core.vision.feature_flags import FlagType

        assert FlagType.BOOLEAN.value == "boolean"
        assert FlagType.STRING.value == "string"
        assert FlagType.NUMBER.value == "number"

    def test_flag_status_enum(self):
        """Test FlagStatus enum values."""
        from src.core.vision.feature_flags import FlagStatus

        assert FlagStatus.ACTIVE.value == "active"
        assert FlagStatus.INACTIVE.value == "inactive"
        assert FlagStatus.ARCHIVED.value == "archived"

    def test_targeting_operator_enum(self):
        """Test TargetingOperator enum values."""
        from src.core.vision.feature_flags import TargetingOperator

        assert TargetingOperator.EQUALS.value == "equals"
        assert TargetingOperator.IN.value == "in"
        assert TargetingOperator.GREATER_THAN.value == "greater_than"

    def test_flag_variation_creation(self):
        """Test FlagVariation creation."""
        from src.core.vision.feature_flags import FlagVariation

        variation = FlagVariation(
            variation_id="v1",
            value=True,
            name="Enabled",
        )

        assert variation.variation_id == "v1"
        assert variation.value is True

    def test_targeting_rule_creation(self):
        """Test TargetingRule creation."""
        from src.core.vision.feature_flags import TargetingRule, TargetingOperator

        rule = TargetingRule(
            rule_id="r1",
            attribute="user_id",
            operator=TargetingOperator.EQUALS,
            values=["user-123"],
            variation_id="v1",
        )

        assert rule.rule_id == "r1"
        assert rule.attribute == "user_id"

    def test_targeting_rule_evaluation(self):
        """Test TargetingRule evaluation."""
        from src.core.vision.feature_flags import TargetingRule, TargetingOperator

        rule = TargetingRule(
            rule_id="r1",
            attribute="user_id",
            operator=TargetingOperator.EQUALS,
            values=["user-123"],
            variation_id="v1",
        )

        assert rule.evaluate({"user_id": "user-123"}) is True
        assert rule.evaluate({"user_id": "user-456"}) is False

    def test_targeting_rule_in_operator(self):
        """Test TargetingRule IN operator."""
        from src.core.vision.feature_flags import TargetingRule, TargetingOperator

        rule = TargetingRule(
            rule_id="r1",
            attribute="country",
            operator=TargetingOperator.IN,
            values=["US", "CA", "UK"],
            variation_id="v1",
        )

        assert rule.evaluate({"country": "US"}) is True
        assert rule.evaluate({"country": "FR"}) is False

    def test_feature_flag_creation(self):
        """Test FeatureFlag creation."""
        from src.core.vision.feature_flags import (
            FeatureFlag, FlagType, FlagVariation, FlagStatus
        )

        flag = FeatureFlag(
            flag_key="test-flag",
            flag_type=FlagType.BOOLEAN,
            variations=[
                FlagVariation("true", True, "Enabled"),
                FlagVariation("false", False, "Disabled"),
            ],
            default_variation_id="false",
        )

        assert flag.flag_key == "test-flag"
        assert flag.flag_type == FlagType.BOOLEAN
        assert flag.status == FlagStatus.ACTIVE

    def test_in_memory_flag_store(self):
        """Test InMemoryFlagStore."""
        from src.core.vision.feature_flags import (
            InMemoryFlagStore, FeatureFlag, FlagType, FlagVariation
        )

        store = InMemoryFlagStore()

        flag = FeatureFlag(
            flag_key="test-flag",
            flag_type=FlagType.BOOLEAN,
            variations=[
                FlagVariation("true", True),
                FlagVariation("false", False),
            ],
            default_variation_id="false",
        )

        store.save_flag(flag)

        retrieved = store.get_flag("test-flag")
        assert retrieved is not None
        assert retrieved.flag_key == "test-flag"

    def test_flag_evaluator_basic(self):
        """Test FlagEvaluator basic evaluation."""
        from src.core.vision.feature_flags import (
            FlagEvaluator, InMemoryFlagStore, FeatureFlag, FlagType, FlagVariation
        )

        store = InMemoryFlagStore()
        flag = FeatureFlag(
            flag_key="test-flag",
            flag_type=FlagType.BOOLEAN,
            variations=[
                FlagVariation("true", True),
                FlagVariation("false", False),
            ],
            default_variation_id="true",
        )
        store.save_flag(flag)

        evaluator = FlagEvaluator(store)
        result = evaluator.evaluate("test-flag")

        assert result.value is True
        assert result.variation_id == "true"

    def test_flag_evaluator_with_targeting(self):
        """Test FlagEvaluator with targeting rules."""
        from src.core.vision.feature_flags import (
            FlagEvaluator, InMemoryFlagStore, FeatureFlag, FlagType,
            FlagVariation, TargetingRule, TargetingOperator, EvaluationContext
        )

        store = InMemoryFlagStore()
        flag = FeatureFlag(
            flag_key="test-flag",
            flag_type=FlagType.BOOLEAN,
            variations=[
                FlagVariation("true", True),
                FlagVariation("false", False),
            ],
            default_variation_id="false",
            targeting_rules=[
                TargetingRule(
                    rule_id="r1",
                    attribute="user_id",
                    operator=TargetingOperator.EQUALS,
                    values=["vip-user"],
                    variation_id="true",
                )
            ],
        )
        store.save_flag(flag)

        evaluator = FlagEvaluator(store)

        # Without matching context
        result = evaluator.evaluate("test-flag")
        assert result.value is False

        # With matching context
        context = EvaluationContext(attributes={"user_id": "vip-user"})
        result = evaluator.evaluate("test-flag", context)
        assert result.value is True

    def test_feature_flag_manager_create_boolean(self):
        """Test FeatureFlagManager create boolean flag."""
        from src.core.vision.feature_flags import FeatureFlagManager

        manager = FeatureFlagManager()

        flag = manager.create_boolean_flag("test-flag", default_value=True)

        assert flag is not None
        assert flag.flag_key == "test-flag"
        assert manager.is_enabled("test-flag") is True

    def test_feature_flag_manager_is_enabled(self):
        """Test FeatureFlagManager is_enabled."""
        from src.core.vision.feature_flags import FeatureFlagManager

        manager = FeatureFlagManager()

        manager.create_boolean_flag("enabled-flag", default_value=True)
        manager.create_boolean_flag("disabled-flag", default_value=False)

        assert manager.is_enabled("enabled-flag") is True
        assert manager.is_enabled("disabled-flag") is False
        assert manager.is_enabled("missing-flag") is False

    @pytest.mark.asyncio
    async def test_feature_flag_vision_provider_enabled(self):
        """Test FeatureFlagVisionProvider when enabled."""
        from src.core.vision.feature_flags import create_feature_flag_provider

        mock_provider = MockVisionProvider()
        provider = create_feature_flag_provider(
            mock_provider,
            feature_flag_key="vision.enabled",
            default_enabled=True,
        )

        result = await provider.analyze_image(b"test image")
        assert result.summary == "Mock analysis by mock"

    @pytest.mark.asyncio
    async def test_feature_flag_vision_provider_disabled(self):
        """Test FeatureFlagVisionProvider when disabled."""
        from src.core.vision.feature_flags import (
            create_feature_flag_provider, FeatureFlagManager
        )

        manager = FeatureFlagManager()
        manager.create_boolean_flag("vision.enabled", default_value=False)

        mock_provider = MockVisionProvider()
        provider = create_feature_flag_provider(
            mock_provider,
            flag_manager=manager,
            feature_flag_key="vision.enabled",
        )

        with pytest.raises(RuntimeError, match="disabled by feature flag"):
            await provider.analyze_image(b"test image")


# ============================================================================
# Graceful Degradation Tests
# ============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation module."""

    def test_degradation_level_enum(self):
        """Test DegradationLevel enum values."""
        from src.core.vision.graceful_degradation import DegradationLevel

        assert DegradationLevel.NORMAL.value == "normal"
        assert DegradationLevel.REDUCED.value == "reduced"
        assert DegradationLevel.MINIMAL.value == "minimal"
        assert DegradationLevel.OFFLINE.value == "offline"

    def test_degradation_reason_enum(self):
        """Test DegradationReason enum values."""
        from src.core.vision.graceful_degradation import DegradationReason

        assert DegradationReason.NONE.value == "none"
        assert DegradationReason.HIGH_LATENCY.value == "high_latency"
        assert DegradationReason.HIGH_ERROR_RATE.value == "high_error_rate"

    def test_recovery_strategy_enum(self):
        """Test RecoveryStrategy enum values."""
        from src.core.vision.graceful_degradation import RecoveryStrategy

        assert RecoveryStrategy.IMMEDIATE.value == "immediate"
        assert RecoveryStrategy.GRADUAL.value == "gradual"
        assert RecoveryStrategy.MANUAL.value == "manual"

    def test_degradation_metrics_initial(self):
        """Test DegradationMetrics initial state."""
        from src.core.vision.graceful_degradation import DegradationMetrics

        metrics = DegradationMetrics()

        assert metrics.total_requests == 0
        assert metrics.error_rate == 0.0
        assert metrics.success_rate == 1.0

    def test_degradation_metrics_record_success(self):
        """Test DegradationMetrics record success."""
        from src.core.vision.graceful_degradation import DegradationMetrics

        metrics = DegradationMetrics()
        metrics.record_success(100.0)

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.success_rate == 1.0
        assert metrics.average_latency_ms == 100.0

    def test_degradation_metrics_record_failure(self):
        """Test DegradationMetrics record failure."""
        from src.core.vision.graceful_degradation import DegradationMetrics

        metrics = DegradationMetrics()
        metrics.record_failure()

        assert metrics.total_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.consecutive_failures == 1
        assert metrics.error_rate == 1.0

    def test_degradation_thresholds_defaults(self):
        """Test DegradationThresholds defaults."""
        from src.core.vision.graceful_degradation import DegradationThresholds

        thresholds = DegradationThresholds()

        assert thresholds.max_error_rate == 0.3
        assert thresholds.consecutive_failures == 5

    def test_degradation_policy_normal(self):
        """Test DegradationPolicy returns normal state."""
        from src.core.vision.graceful_degradation import (
            DegradationPolicy, DegradationMetrics, DegradationState, DegradationLevel
        )

        policy = DegradationPolicy()
        metrics = DegradationMetrics()
        state = DegradationState()

        # Record some successes
        for _ in range(10):
            metrics.record_success(100.0)

        new_state = policy.evaluate(metrics, state)
        assert new_state.level == DegradationLevel.NORMAL

    def test_degradation_policy_consecutive_failures(self):
        """Test DegradationPolicy with consecutive failures."""
        from src.core.vision.graceful_degradation import (
            DegradationPolicy, DegradationMetrics, DegradationState,
            DegradationLevel, DegradationThresholds
        )

        thresholds = DegradationThresholds(consecutive_failures=3)
        policy = DegradationPolicy(thresholds)
        metrics = DegradationMetrics()
        state = DegradationState()

        # Record consecutive failures
        for _ in range(5):
            metrics.record_failure()

        new_state = policy.evaluate(metrics, state)
        assert new_state.level == DegradationLevel.OFFLINE

    def test_static_fallback_provider(self):
        """Test StaticFallbackProvider."""
        from src.core.vision.graceful_degradation import (
            StaticFallbackProvider, DegradationLevel
        )

        provider = StaticFallbackProvider(
            default_summary="Fallback response",
            default_details="Service unavailable",
        )

        response = provider.get_fallback(b"test", DegradationLevel.OFFLINE)

        assert response.summary == "Fallback response"
        assert response.is_fallback is True

    def test_degradation_manager_record_success(self):
        """Test DegradationManager record success."""
        from src.core.vision.graceful_degradation import (
            DegradationManager, DegradationLevel
        )

        manager = DegradationManager()
        manager.record_success(100.0)

        assert manager.metrics.successful_requests == 1
        assert manager.state.level == DegradationLevel.NORMAL

    def test_degradation_manager_record_failure(self):
        """Test DegradationManager record failure."""
        from src.core.vision.graceful_degradation import DegradationManager

        manager = DegradationManager()

        for _ in range(10):
            manager.record_failure()

        assert manager.metrics.failed_requests == 10

    def test_degradation_manager_force_degradation(self):
        """Test DegradationManager force degradation."""
        from src.core.vision.graceful_degradation import (
            DegradationManager, DegradationLevel
        )

        manager = DegradationManager()
        manager.force_degradation(DegradationLevel.OFFLINE, "Manual maintenance")

        assert manager.state.level == DegradationLevel.OFFLINE

    def test_degradation_manager_force_recovery(self):
        """Test DegradationManager force recovery."""
        from src.core.vision.graceful_degradation import (
            DegradationManager, DegradationLevel
        )

        manager = DegradationManager()
        manager.force_degradation(DegradationLevel.OFFLINE)
        manager.force_recovery()

        assert manager.state.level == DegradationLevel.NORMAL

    def test_fallback_chain_creation(self):
        """Test FallbackChain creation."""
        from src.core.vision.graceful_degradation import FallbackChain

        chain = FallbackChain()
        provider1 = MockVisionProvider("provider1")
        provider2 = MockVisionProvider("provider2")

        chain.add_provider(provider1).add_provider(provider2)

        assert len(chain._providers) == 2

    @pytest.mark.asyncio
    async def test_fallback_chain_success(self):
        """Test FallbackChain with successful provider."""
        from src.core.vision.graceful_degradation import FallbackChain

        chain = FallbackChain()
        chain.add_provider(MockVisionProvider("provider1"))

        result = await chain.analyze(b"test image")
        assert result.summary == "Mock analysis by provider1"

    @pytest.mark.asyncio
    async def test_fallback_chain_fallback(self):
        """Test FallbackChain with fallback."""
        from src.core.vision.graceful_degradation import (
            FallbackChain, StaticFallbackProvider
        )

        chain = FallbackChain()
        chain.add_provider(MockVisionProvider("provider1", fail=True))
        chain.add_provider(MockVisionProvider("provider2"))

        result = await chain.analyze(b"test image")
        assert result.summary == "Mock analysis by provider2"

    @pytest.mark.asyncio
    async def test_graceful_degradation_provider_normal(self):
        """Test GracefulDegradationVisionProvider in normal mode."""
        from src.core.vision.graceful_degradation import create_graceful_provider

        mock_provider = MockVisionProvider()
        provider = create_graceful_provider(mock_provider)

        result = await provider.analyze_image(b"test image")
        assert result.summary == "Mock analysis by mock"

    @pytest.mark.asyncio
    async def test_graceful_degradation_provider_degraded(self):
        """Test GracefulDegradationVisionProvider in degraded mode."""
        from src.core.vision.graceful_degradation import (
            create_graceful_provider, DegradationLevel
        )

        mock_provider = MockVisionProvider(fail=True)
        provider = create_graceful_provider(mock_provider)

        # Force degradation
        provider.degradation_manager.force_degradation(DegradationLevel.OFFLINE)

        # Should return fallback
        result = await provider.analyze_image(b"test image")
        assert "unavailable" in result.summary.lower() or mock_provider._call_count == 0


# ============================================================================
# Phase 8 Integration Tests
# ============================================================================


class TestPhase8Integration:
    """Integration tests for Phase 8 features."""

    @pytest.mark.asyncio
    async def test_mesh_with_tracing(self):
        """Test service mesh with distributed tracing."""
        from src.core.vision.service_mesh import ServiceMesh, create_mesh_provider
        from src.core.vision.distributed_tracing import (
            TracerProvider, create_tracing_provider
        )

        mock_provider = MockVisionProvider()
        mesh = ServiceMesh()

        mesh_provider = create_mesh_provider(mock_provider, mesh, "vision")
        mesh_provider.register("localhost", 8080)

        tracer_provider = TracerProvider()
        traced_provider = create_tracing_provider(mesh_provider, tracer_provider)

        result = await traced_provider.analyze_image(b"test image")

        assert result.summary == "Mock analysis by mock"
        assert traced_provider.stats.total_spans >= 1

    @pytest.mark.asyncio
    async def test_config_with_feature_flags(self):
        """Test configuration with feature flags."""
        from src.core.vision.config_management import (
            ConfigurationManager, create_configurable_provider
        )
        from src.core.vision.feature_flags import (
            FeatureFlagManager, create_feature_flag_provider
        )

        mock_provider = MockVisionProvider()

        # Create configurable provider
        config_provider = create_configurable_provider(
            mock_provider,
            config={"vision.timeout": 30},
        )

        # Add feature flags
        flag_manager = FeatureFlagManager()
        flag_manager.create_boolean_flag("vision.enabled", default_value=True)

        final_provider = create_feature_flag_provider(
            config_provider,
            flag_manager=flag_manager,
        )

        result = await final_provider.analyze_image(b"test image")
        assert result.summary == "Mock analysis by mock"

    @pytest.mark.asyncio
    async def test_graceful_degradation_with_fallback_chain(self):
        """Test graceful degradation with fallback chain."""
        from src.core.vision.graceful_degradation import create_fallback_chain

        provider1 = MockVisionProvider("primary", fail=True)
        provider2 = MockVisionProvider("secondary", fail=True)
        provider3 = MockVisionProvider("tertiary")

        fallback_provider = create_fallback_chain(
            [provider1, provider2, provider3],
            fallback_message="All providers failed",
        )

        result = await fallback_provider.analyze_image(b"test image")
        assert result.summary == "Mock analysis by tertiary"

    @pytest.mark.asyncio
    async def test_full_phase8_stack(self):
        """Test full Phase 8 stack integration."""
        from src.core.vision.service_mesh import ServiceMesh, create_mesh_provider
        from src.core.vision.distributed_tracing import create_tracing_provider
        from src.core.vision.config_management import create_configurable_provider
        from src.core.vision.feature_flags import (
            FeatureFlagManager, create_feature_flag_provider
        )
        from src.core.vision.graceful_degradation import create_graceful_provider

        # Base provider
        mock_provider = MockVisionProvider()

        # Layer 1: Service Mesh
        mesh = ServiceMesh()
        mesh_provider = create_mesh_provider(mock_provider, mesh, "vision")
        mesh_provider.register("localhost", 8080)

        # Layer 2: Distributed Tracing
        traced_provider = create_tracing_provider(mesh_provider)

        # Layer 3: Configuration
        config_provider = create_configurable_provider(traced_provider)

        # Layer 4: Feature Flags
        flag_manager = FeatureFlagManager()
        flag_manager.create_boolean_flag("vision.enabled", True)
        flag_provider = create_feature_flag_provider(
            config_provider, flag_manager=flag_manager
        )

        # Layer 5: Graceful Degradation
        final_provider = create_graceful_provider(flag_provider)

        result = await final_provider.analyze_image(b"test image")

        assert result.summary == "Mock analysis by mock"
        assert result.confidence == 0.85
