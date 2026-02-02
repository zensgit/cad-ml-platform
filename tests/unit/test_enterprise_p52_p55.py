"""Unit tests for Enterprise Features P52-P55.

P52: Configuration Management
P53: Service Registry
P54: Metrics Aggregator
P55: Request Context
"""

import asyncio
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ==============================================================================
# P52: Configuration Management Tests
# ==============================================================================


class TestConfigSource:
    """Tests for configuration sources."""

    def test_dict_config_source(self):
        """Test DictConfigSource basic operations."""
        from src.core.config_management import DictConfigSource

        # DictConfigSource uses nested structure for dot notation
        source = DictConfigSource({"app": {"name": "test", "debug": True}})

        assert source.get("app.name") == "test"
        assert source.get("app.debug") is True
        assert source.get("app.missing") is None

    def test_dict_config_source_nested(self):
        """Test DictConfigSource with nested keys."""
        from src.core.config_management import DictConfigSource

        source = DictConfigSource({
            "database": {
                "host": "localhost",
                "port": 5432,
            }
        })

        assert source.get("database.host") == "localhost"
        assert source.get("database.port") == 5432

    def test_environment_config_source(self):
        """Test EnvironmentConfigSource."""
        from src.core.config_management import EnvironmentConfigSource

        # Set test environment variable
        os.environ["TEST_CONFIG_VAR"] = "test_value"
        os.environ["TEST_PREFIX_NAME"] = "prefixed"

        try:
            # Without prefix - key is converted to uppercase
            source = EnvironmentConfigSource()
            assert source.get("test_config_var") == "test_value"

            # With prefix
            prefixed_source = EnvironmentConfigSource(prefix="TEST_PREFIX")
            assert prefixed_source.get("name") == "prefixed"
        finally:
            del os.environ["TEST_CONFIG_VAR"]
            del os.environ["TEST_PREFIX_NAME"]

    def test_file_config_source_json(self):
        """Test FileConfigSource with JSON."""
        from src.core.config_management import FileConfigSource

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write('{"app": {"name": "test", "version": "1.0"}}')
            f.flush()

            try:
                source = FileConfigSource(f.name)
                assert source.get("app.name") == "test"
                assert source.get("app.version") == "1.0"
            finally:
                os.unlink(f.name)

    def test_file_config_source_yaml(self):
        """Test FileConfigSource with YAML."""
        from src.core.config_management import FileConfigSource

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("app:\n  name: test\n  debug: true\n")
            f.flush()

            try:
                source = FileConfigSource(f.name)
                assert source.get("app.name") == "test"
                assert source.get("app.debug") is True
            finally:
                os.unlink(f.name)


class TestConfigSchema:
    """Tests for configuration schema validation."""

    def test_config_schema_type_validation(self):
        """Test type validation in schema."""
        from src.core.config_management import (
            ConfigSchema,
            ConfigValidator,
            ConfigValueType,
        )

        validator = ConfigValidator()
        validator.add_schema(ConfigSchema(
            key="port",
            value_type=ConfigValueType.INTEGER,
            required=True,
        ))
        validator.add_schema(ConfigSchema(
            key="host",
            value_type=ConfigValueType.STRING,
            default="localhost",
        ))

        # Valid config
        result = validator.validate({"port": 8080, "host": "0.0.0.0"})
        assert result.valid

        # Missing required field
        result = validator.validate({"host": "localhost"})
        assert not result.valid
        assert any("port" in e.lower() for e in result.errors)

    def test_config_schema_custom_validator(self):
        """Test custom validators in schema."""
        from src.core.config_management import (
            ConfigSchema,
            ConfigValidator,
            ConfigValueType,
        )

        validator = ConfigValidator()
        validator.add_schema(ConfigSchema(
            key="port",
            value_type=ConfigValueType.INTEGER,
            required=True,
            validators=[lambda x: 1 <= x <= 65535],
        ))

        # Valid port
        result = validator.validate({"port": 8080})
        assert result.valid

        # Invalid port
        result = validator.validate({"port": 70000})
        assert not result.valid


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_config_manager_basic(self):
        """Test basic ConfigManager operations."""
        from src.core.config_management import (
            ConfigManager,
            DictConfigSource,
        )

        source = DictConfigSource({
            "app": {
                "name": "TestApp",
                "port": 8080,
                "debug": True,
            }
        }, source_priority=10)

        manager = ConfigManager()
        manager.add_source(source)

        assert manager.get("app.name") == "TestApp"
        assert manager.get("app.port") == 8080
        assert manager.get("app.debug") is True
        assert manager.get("app.missing", "default") == "default"

    def test_config_manager_source_priority(self):
        """Test source priority (higher priority wins)."""
        from src.core.config_management import (
            ConfigManager,
            DictConfigSource,
        )

        source1 = DictConfigSource(
            {"key": "value1", "only_in_1": "first"},
            source_priority=1,
        )
        source2 = DictConfigSource(
            {"key": "value2", "only_in_2": "second"},
            source_priority=2,
        )

        manager = ConfigManager()
        manager.add_source(source1)
        manager.add_source(source2)

        # Higher priority wins
        assert manager.get("key") == "value2"
        assert manager.get("only_in_1") == "first"
        assert manager.get("only_in_2") == "second"

    def test_config_manager_type_coercion(self):
        """Test type coercion methods."""
        from src.core.config_management import (
            ConfigManager,
            DictConfigSource,
        )

        source = DictConfigSource({
            "string_port": "8080",
            "int_value": 42,
            "bool_true": "true",
            "bool_false": "false",
            "float_value": "3.14",
        })

        manager = ConfigManager()
        manager.add_source(source)

        assert manager.get_int("string_port") == 8080
        assert manager.get_int("int_value") == 42
        assert manager.get_bool("bool_true") is True
        assert manager.get_bool("bool_false") is False
        assert manager.get_float("float_value") == 3.14

    def test_config_manager_reload(self):
        """Test configuration reload."""
        from src.core.config_management import (
            ConfigManager,
            DictConfigSource,
        )

        source = DictConfigSource({"key": "original"})
        manager = ConfigManager()
        manager.add_source(source)

        assert manager.get("key") == "original"

        # Modify source data and reload
        source._data["key"] = "updated"
        manager.reload()

        assert manager.get("key") == "updated"

    def test_config_manager_listener(self):
        """Test ConfigManager change listener."""
        from src.core.config_management import ConfigChange, ConfigManager

        changes_received = []

        def on_change(change: ConfigChange):
            changes_received.append(change)

        manager = ConfigManager()
        manager.add_listener(on_change)

        # Set a value at runtime
        manager.set("test.key", "test_value")

        assert len(changes_received) == 1
        assert changes_received[0].key == "test.key"
        assert changes_received[0].new_value == "test_value"


# ==============================================================================
# P53: Service Registry Tests
# ==============================================================================


class TestServiceInstance:
    """Tests for ServiceInstance."""

    def test_service_instance_creation(self):
        """Test ServiceInstance creation."""
        from src.core.service_registry import ServiceInstance, ServiceStatus

        instance = ServiceInstance(
            instance_id="test-1",
            service_name="test-service",
            host="localhost",
            port=8080,
            status=ServiceStatus.HEALTHY,
        )

        assert instance.instance_id == "test-1"
        assert instance.service_name == "test-service"
        assert instance.address == "localhost:8080"
        assert instance.status == ServiceStatus.HEALTHY

    def test_service_instance_metadata(self):
        """Test ServiceInstance with metadata."""
        from src.core.service_registry import ServiceInstance

        instance = ServiceInstance(
            instance_id="test-1",
            service_name="test-service",
            host="localhost",
            port=8080,
            metadata={"version": "1.0", "region": "us-east"},
        )

        assert instance.metadata["version"] == "1.0"
        assert instance.metadata["region"] == "us-east"


class TestInMemoryServiceRegistry:
    """Tests for InMemoryServiceRegistry."""

    @pytest.mark.asyncio
    async def test_service_registration(self):
        """Test service registration."""
        from src.core.service_registry import (
            InMemoryServiceRegistry,
            ServiceInstance,
            ServiceStatus,
        )

        registry = InMemoryServiceRegistry()

        instance = ServiceInstance(
            instance_id="api-1",
            service_name="api",
            host="localhost",
            port=8080,
            status=ServiceStatus.HEALTHY,
        )

        await registry.register(instance)

        # Lookup
        service = await registry.get_service("api")
        assert service is not None
        assert len(service.instances) == 1

    @pytest.mark.asyncio
    async def test_service_deregistration(self):
        """Test service deregistration."""
        from src.core.service_registry import (
            InMemoryServiceRegistry,
            ServiceInstance,
            ServiceStatus,
        )

        registry = InMemoryServiceRegistry()

        instance = ServiceInstance(
            instance_id="api-1",
            service_name="api",
            host="localhost",
            port=8080,
        )

        await registry.register(instance)
        service = await registry.get_service("api")
        assert len(service.instances) == 1

        await registry.deregister("api-1")
        service = await registry.get_service("api")
        assert len(service.instances) == 0

    @pytest.mark.asyncio
    async def test_get_instance(self):
        """Test getting instance by ID."""
        from src.core.service_registry import (
            InMemoryServiceRegistry,
            ServiceInstance,
            ServiceStatus,
        )

        registry = InMemoryServiceRegistry()

        instance = ServiceInstance(
            instance_id="api-1",
            service_name="api",
            host="localhost",
            port=8080,
            status=ServiceStatus.HEALTHY,
        )

        await registry.register(instance)

        # Get by ID
        inst = await registry.get_instance("api-1")
        assert inst is not None
        assert inst.instance_id == "api-1"

    @pytest.mark.asyncio
    async def test_list_services(self):
        """Test listing all services."""
        from src.core.service_registry import (
            InMemoryServiceRegistry,
            ServiceInstance,
        )

        registry = InMemoryServiceRegistry()

        await registry.register(ServiceInstance(
            instance_id="api-1",
            service_name="api",
            host="localhost",
            port=8080,
        ))
        await registry.register(ServiceInstance(
            instance_id="db-1",
            service_name="database",
            host="localhost",
            port=5432,
        ))

        services = await registry.list_services()
        assert "api" in services
        assert "database" in services


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_health_check_config(self):
        """Test health check configuration."""
        from src.core.service_registry import HealthCheckConfig, HealthCheckType

        config = HealthCheckConfig(
            check_type=HealthCheckType.HTTP,
            endpoint="/health",
            interval_seconds=10,
            timeout_seconds=5,
            healthy_threshold=2,
            unhealthy_threshold=3,
        )

        assert config.check_type == HealthCheckType.HTTP
        assert config.endpoint == "/health"
        assert config.interval_seconds == 10

    def test_health_check_result(self):
        """Test health check result."""
        from src.core.service_registry import HealthCheckResult

        result = HealthCheckResult(
            healthy=True,
            message="OK",
            latency_ms=50.0,
        )

        assert result.healthy is True
        assert result.message == "OK"
        assert result.latency_ms == 50.0


# ==============================================================================
# P54: Metrics Aggregator Tests
# ==============================================================================


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_increment(self):
        """Test counter increment."""
        from src.core.metrics_aggregator import Counter

        counter = Counter("test_counter", "Test counter description")

        counter.inc()
        counter.inc(5)

        values = counter.collect()
        assert len(values) == 1
        assert values[0].value == 6

    def test_counter_with_labels(self):
        """Test counter with labels."""
        from src.core.metrics_aggregator import Counter

        counter = Counter(
            "http_requests",
            "HTTP request count",
            label_names=["method", "status"],
        )

        counter.labels(method="GET", status="200").inc()
        counter.labels(method="GET", status="200").inc()
        counter.labels(method="POST", status="201").inc()

        values = counter.collect()
        assert len(values) == 2

    def test_counter_cannot_decrement(self):
        """Test that counter cannot be decremented."""
        from src.core.metrics_aggregator import Counter

        counter = Counter("test_counter")

        with pytest.raises(ValueError):
            counter.inc(-1)


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_set(self):
        """Test gauge set."""
        from src.core.metrics_aggregator import Gauge

        gauge = Gauge("temperature", "Current temperature")

        gauge.set(25.5)
        values = gauge.collect()
        assert values[0].value == 25.5

        gauge.set(30.0)
        values = gauge.collect()
        assert values[0].value == 30.0

    def test_gauge_inc_dec(self):
        """Test gauge increment and decrement."""
        from src.core.metrics_aggregator import Gauge

        gauge = Gauge("connections", "Active connections")

        gauge.inc()
        gauge.inc()
        gauge.dec()

        values = gauge.collect()
        assert values[0].value == 1

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        from src.core.metrics_aggregator import Gauge

        gauge = Gauge(
            "cpu_usage",
            "CPU usage percentage",
            label_names=["core"],
        )

        gauge.labels(core="0").set(45.5)
        gauge.labels(core="1").set(32.0)

        values = gauge.collect()
        assert len(values) == 2


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_observe(self):
        """Test histogram observation."""
        from src.core.metrics_aggregator import Histogram

        histogram = Histogram(
            "request_duration",
            "Request duration in seconds",
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )

        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)

        values = histogram.collect()

        # Should have bucket values + sum + count
        bucket_values = [v for v in values if "le" in v.labels.labels]
        sum_value = [v for v in values if v.labels.labels.get("_type") == "sum"]
        count_value = [v for v in values if v.labels.labels.get("_type") == "count"]

        assert len(bucket_values) == 4  # 4 buckets
        assert len(sum_value) == 1
        assert len(count_value) == 1
        assert count_value[0].value == 3

    def test_histogram_timer(self):
        """Test histogram timer context manager."""
        from src.core.metrics_aggregator import Histogram

        histogram = Histogram("operation_duration")

        with histogram.time():
            time.sleep(0.01)  # 10ms

        values = histogram.collect()
        count_value = [v for v in values if v.labels.labels.get("_type") == "count"]
        assert count_value[0].value == 1


class TestSummary:
    """Tests for Summary metric."""

    def test_summary_observe(self):
        """Test summary observation."""
        from src.core.metrics_aggregator import Summary

        summary = Summary(
            "response_size",
            "Response size in bytes",
            quantiles=(0.5, 0.9, 0.99),
        )

        for i in range(100):
            summary.observe(i)

        values = summary.collect()

        # Should have quantile values + sum + count
        quantile_values = [v for v in values if "quantile" in v.labels.labels]
        assert len(quantile_values) == 3  # 3 quantiles


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_registry_counter(self):
        """Test registry counter creation."""
        from src.core.metrics_aggregator import MetricsRegistry

        registry = MetricsRegistry(prefix="myapp")

        counter = registry.counter("requests_total", "Total requests")

        assert counter.name == "myapp_requests_total"

    def test_registry_get_or_create(self):
        """Test registry returns existing metric."""
        from src.core.metrics_aggregator import MetricsRegistry

        registry = MetricsRegistry()

        counter1 = registry.counter("test_counter")
        counter2 = registry.counter("test_counter")

        assert counter1 is counter2

    def test_registry_type_mismatch(self):
        """Test registry raises on type mismatch."""
        from src.core.metrics_aggregator import MetricsRegistry

        registry = MetricsRegistry()

        registry.counter("my_metric")

        with pytest.raises(ValueError):
            registry.gauge("my_metric")

    def test_registry_collect(self):
        """Test registry collect all metrics."""
        from src.core.metrics_aggregator import MetricsRegistry

        registry = MetricsRegistry()

        counter = registry.counter("requests", "Request count")
        gauge = registry.gauge("connections", "Active connections")

        counter.inc(10)
        gauge.set(5)

        samples = registry.collect()
        assert len(samples) == 2


class TestMetricsExporter:
    """Tests for metrics exporters."""

    def test_prometheus_exporter(self):
        """Test Prometheus format export."""
        from src.core.metrics_aggregator import (
            MetricsRegistry,
            PrometheusExporter,
        )

        registry = MetricsRegistry()
        counter = registry.counter("http_requests", "HTTP request count")
        counter.inc(42)

        exporter = PrometheusExporter()
        output = exporter.export(registry.collect())

        assert "# HELP http_requests HTTP request count" in output
        assert "# TYPE http_requests counter" in output
        assert "http_requests" in output
        assert "42" in output

    def test_json_exporter(self):
        """Test JSON format export."""
        import json

        from src.core.metrics_aggregator import JSONExporter, MetricsRegistry

        registry = MetricsRegistry()
        gauge = registry.gauge("temperature")
        gauge.set(25.5)

        exporter = JSONExporter()
        output = exporter.export(registry.collect())

        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["name"] == "temperature"
        assert data[0]["type"] == "gauge"

    def test_statsd_exporter(self):
        """Test StatsD format export."""
        from src.core.metrics_aggregator import MetricsRegistry, StatsDExporter

        registry = MetricsRegistry()
        counter = registry.counter("events")
        counter.inc(100)

        exporter = StatsDExporter()
        output = exporter.export(registry.collect())

        assert "events:100|c" in output


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    def test_aggregator_multiple_registries(self):
        """Test aggregating from multiple registries."""
        from src.core.metrics_aggregator import MetricsAggregator, MetricsRegistry

        registry1 = MetricsRegistry(prefix="service1")
        registry2 = MetricsRegistry(prefix="service2")

        registry1.counter("requests").inc(10)
        registry2.counter("requests").inc(20)

        aggregator = MetricsAggregator()
        aggregator.add_registry(registry1)
        aggregator.add_registry(registry2)

        samples = aggregator.collect_all()
        assert len(samples) == 2

    def test_aggregator_custom_collector(self):
        """Test custom collector."""
        from src.core.metrics_aggregator import (
            MetricSample,
            MetricsAggregator,
            MetricType,
            MetricValue,
        )

        def custom_collector():
            return [
                MetricSample(
                    name="custom_metric",
                    metric_type=MetricType.GAUGE,
                    description="Custom metric",
                    values=[MetricValue(value=42)],
                )
            ]

        aggregator = MetricsAggregator()
        aggregator.add_collector(custom_collector)

        samples = aggregator.collect_all()
        assert len(samples) == 1
        assert samples[0].name == "custom_metric"


# ==============================================================================
# P55: Request Context Tests
# ==============================================================================


class TestBaggage:
    """Tests for Baggage."""

    def test_baggage_set_get(self):
        """Test baggage set and get."""
        from src.core.request_context import Baggage

        baggage = Baggage()
        baggage.set("user_id", "123")
        baggage.set("tenant", "acme")

        assert baggage.get("user_id") == "123"
        assert baggage.get("tenant") == "acme"
        assert baggage.get("missing") is None
        assert baggage.get("missing", "default") == "default"

    def test_baggage_remove(self):
        """Test baggage remove."""
        from src.core.request_context import Baggage

        baggage = Baggage()
        baggage.set("key", "value")
        baggage.remove("key")

        assert baggage.get("key") is None

    def test_baggage_to_header(self):
        """Test baggage to HTTP header."""
        from src.core.request_context import Baggage

        baggage = Baggage()
        baggage.set("user", "123")
        baggage.set("tenant", "acme")

        header = baggage.to_header()
        assert "user=123" in header
        assert "tenant=acme" in header

    def test_baggage_from_header(self):
        """Test baggage from HTTP header."""
        from src.core.request_context import Baggage

        header = "user=123,tenant=acme"
        baggage = Baggage.from_header(header)

        assert baggage.get("user") == "123"
        assert baggage.get("tenant") == "acme"

    def test_baggage_copy(self):
        """Test baggage copy."""
        from src.core.request_context import Baggage

        original = Baggage()
        original.set("key", "value")

        copy = original.copy()
        copy.set("key", "modified")

        assert original.get("key") == "value"
        assert copy.get("key") == "modified"


class TestRequestContext:
    """Tests for RequestContext."""

    def test_request_context_creation(self):
        """Test RequestContext creation."""
        from src.core.request_context import RequestContext

        ctx = RequestContext()

        assert ctx.request_id is not None
        assert ctx.correlation_id == ctx.request_id  # Defaults to request_id
        assert ctx.start_time is not None

    def test_request_context_custom_ids(self):
        """Test RequestContext with custom IDs."""
        from src.core.request_context import RequestContext

        ctx = RequestContext(
            request_id="req-123",
            correlation_id="corr-456",
            trace_id="trace-789",
        )

        assert ctx.request_id == "req-123"
        assert ctx.correlation_id == "corr-456"
        assert ctx.trace_id == "trace-789"

    def test_request_context_attributes(self):
        """Test RequestContext attributes."""
        from src.core.request_context import RequestContext

        ctx = RequestContext()
        ctx.set_attribute("custom_key", "custom_value")

        assert ctx.get_attribute("custom_key") == "custom_value"
        assert ctx.get_attribute("missing") is None
        assert ctx.get_attribute("missing", "default") == "default"

    def test_request_context_elapsed(self):
        """Test RequestContext elapsed time."""
        from src.core.request_context import RequestContext

        ctx = RequestContext()
        time.sleep(0.01)  # 10ms

        elapsed = ctx.elapsed_ms
        assert elapsed >= 10

    def test_request_context_deadline(self):
        """Test RequestContext deadline."""
        from src.core.request_context import RequestContext

        # No deadline
        ctx = RequestContext()
        assert not ctx.is_expired
        assert ctx.remaining_ms is None

        # With future deadline
        ctx = RequestContext(deadline=datetime.utcnow() + timedelta(seconds=10))
        assert not ctx.is_expired
        assert ctx.remaining_ms > 0

        # With past deadline
        ctx = RequestContext(deadline=datetime.utcnow() - timedelta(seconds=10))
        assert ctx.is_expired
        assert ctx.remaining_ms == 0

    def test_request_context_child(self):
        """Test child context creation."""
        from src.core.request_context import RequestContext

        parent = RequestContext(
            trace_id="trace-123",
            user_id="user-456",
        )
        parent.baggage.set("tenant", "acme")

        child = parent.child_context(operation_name="sub_operation")

        # Different request_id but same correlation
        assert child.request_id != parent.request_id
        assert child.correlation_id == parent.correlation_id
        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id
        assert child.user_id == parent.user_id
        assert child.baggage.get("tenant") == "acme"
        assert child.operation_name == "sub_operation"

    def test_request_context_to_dict(self):
        """Test RequestContext to dict."""
        from src.core.request_context import RequestContext

        ctx = RequestContext(
            request_id="req-123",
            user_id="user-456",
        )

        data = ctx.to_dict()
        assert data["request_id"] == "req-123"
        assert data["user_id"] == "user-456"
        assert "elapsed_ms" in data

    def test_request_context_to_headers(self):
        """Test RequestContext to HTTP headers."""
        from src.core.request_context import RequestContext

        ctx = RequestContext(
            request_id="req-123",
            trace_id="trace-456",
            user_id="user-789",
        )
        ctx.baggage.set("tenant", "acme")

        headers = ctx.to_headers()
        assert headers["X-Request-ID"] == "req-123"
        assert headers["X-Trace-ID"] == "trace-456"
        assert headers["X-User-ID"] == "user-789"
        assert "Baggage" in headers

    def test_request_context_from_headers(self):
        """Test RequestContext from HTTP headers."""
        from src.core.request_context import RequestContext

        headers = {
            "X-Request-ID": "req-123",
            "X-Correlation-ID": "corr-456",
            "X-Trace-ID": "trace-789",
            "X-User-ID": "user-abc",
            "Baggage": "tenant=acme",
        }

        ctx = RequestContext.from_headers(
            headers,
            service_name="api",
            operation_name="handle_request",
        )

        assert ctx.request_id == "req-123"
        assert ctx.correlation_id == "corr-456"
        assert ctx.trace_id == "trace-789"
        assert ctx.user_id == "user-abc"
        assert ctx.service_name == "api"
        assert ctx.baggage.get("tenant") == "acme"


class TestContextPropagation:
    """Tests for context propagation."""

    def test_context_var_propagation(self):
        """Test context variable propagation."""
        from src.core.request_context import (
            RequestContext,
            get_current_context,
            reset_context,
            set_current_context,
        )

        ctx = RequestContext(request_id="test-123")
        token = set_current_context(ctx)

        try:
            current = get_current_context()
            assert current is not None
            assert current.request_id == "test-123"
        finally:
            reset_context(token)

        # After reset, no context
        assert get_current_context() is None

    def test_context_scope(self):
        """Test context_scope context manager."""
        from src.core.request_context import (
            context_scope,
            get_current_context,
        )

        with context_scope(service_name="api", operation_name="test"):
            ctx = get_current_context()
            assert ctx is not None
            assert ctx.service_name == "api"
            assert ctx.operation_name == "test"

        # Outside scope
        assert get_current_context() is None

    def test_child_scope(self):
        """Test child_scope context manager."""
        from src.core.request_context import (
            child_scope,
            context_scope,
            get_current_context,
        )

        with context_scope(service_name="api") as parent_ctx:
            parent_id = parent_ctx.request_id

            with child_scope(operation_name="child_op"):
                child_ctx = get_current_context()
                assert child_ctx.correlation_id == parent_ctx.correlation_id
                assert child_ctx.request_id != parent_id
                assert child_ctx.operation_name == "child_op"

    def test_convenience_functions(self):
        """Test convenience functions."""
        from src.core.request_context import (
            context_scope,
            get_attribute,
            get_correlation_id,
            get_request_id,
            get_trace_id,
            get_user_id,
            set_attribute,
        )

        with context_scope(user_id="user-123", trace_id="trace-456"):
            assert get_user_id() == "user-123"
            assert get_trace_id() == "trace-456"
            assert get_request_id() is not None
            assert get_correlation_id() is not None

            set_attribute("custom", "value")
            assert get_attribute("custom") == "value"

    def test_baggage_convenience(self):
        """Test baggage convenience functions."""
        from src.core.request_context import (
            context_scope,
            get_baggage,
            set_baggage,
        )

        with context_scope():
            set_baggage("tenant", "acme")
            assert get_baggage("tenant") == "acme"


class TestContextDecorators:
    """Tests for context decorators."""

    def test_with_request_context_sync(self):
        """Test @with_request_context on sync function."""
        from src.core.request_context import (
            get_current_context,
            with_request_context,
        )

        @with_request_context(service_name="api")
        def my_function():
            ctx = get_current_context()
            return ctx.service_name

        result = my_function()
        assert result == "api"

        # Outside function, no context
        assert get_current_context() is None

    def test_with_request_context_async(self):
        """Test @with_request_context on async function."""
        from src.core.request_context import (
            get_current_context,
            with_request_context,
        )

        @with_request_context(service_name="async_api")
        async def my_async_function():
            ctx = get_current_context()
            return ctx.service_name

        result = asyncio.run(my_async_function())
        assert result == "async_api"

    def test_propagate_context_sync(self):
        """Test @propagate_context decorator."""
        from src.core.request_context import (
            context_scope,
            get_current_context,
            propagate_context,
        )

        @propagate_context()
        def child_operation():
            ctx = get_current_context()
            return ctx.operation_name

        with context_scope(service_name="api") as parent:
            result = child_operation()
            assert result == "child_operation"

    def test_propagate_context_async(self):
        """Test @propagate_context on async function."""
        from src.core.request_context import (
            context_scope,
            get_current_context,
            propagate_context,
        )

        @propagate_context()
        async def async_child():
            ctx = get_current_context()
            return ctx.operation_name

        async def run_test():
            with context_scope(service_name="api"):
                return await async_child()

        result = asyncio.run(run_test())
        assert result == "async_child"


class TestContextThreadSafety:
    """Tests for context thread safety."""

    def test_context_isolation_between_threads(self):
        """Test context isolation between threads."""
        from src.core.request_context import (
            RequestContext,
            get_current_context,
            with_context,
        )

        results = {}

        def thread_func(thread_id):
            ctx = RequestContext(request_id=f"thread-{thread_id}")
            with with_context(ctx):
                time.sleep(0.01)  # Simulate work
                current = get_current_context()
                results[thread_id] = current.request_id if current else None

        threads = []
        for i in range(5):
            t = threading.Thread(target=thread_func, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Each thread should have its own context
        for i in range(5):
            assert results[i] == f"thread-{i}"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests across P52-P55."""

    def test_config_with_metrics(self):
        """Test configuration driving metrics."""
        from src.core.config_management import ConfigManager, DictConfigSource
        from src.core.metrics_aggregator import MetricsRegistry

        # Config determines metric prefix
        config = ConfigManager()
        config.add_source(DictConfigSource({"metrics": {"prefix": "myapp"}}))

        prefix = config.get("metrics.prefix", "default")
        registry = MetricsRegistry(prefix=prefix)

        counter = registry.counter("requests")
        assert counter.name == "myapp_requests"

    @pytest.mark.asyncio
    async def test_service_registry_with_context(self):
        """Test service registry with request context."""
        from src.core.request_context import context_scope, get_current_context
        from src.core.service_registry import (
            InMemoryServiceRegistry,
            ServiceInstance,
        )

        registry = InMemoryServiceRegistry()

        with context_scope(service_name="api", operation_name="register"):
            ctx = get_current_context()

            instance = ServiceInstance(
                instance_id=f"api-{ctx.request_id[:8]}",
                service_name="api",
                host="localhost",
                port=8080,
                metadata={"trace_id": ctx.trace_id},
            )
            await registry.register(instance)

        service = await registry.get_service("api")
        assert service is not None
        assert len(service.instances) == 1
        for inst in service.instances.values():
            assert "trace_id" in inst.metadata

    def test_metrics_with_context_labels(self):
        """Test metrics with context-based labels."""
        from src.core.metrics_aggregator import MetricsRegistry
        from src.core.request_context import context_scope, get_current_context

        registry = MetricsRegistry()
        counter = registry.counter(
            "requests",
            label_names=["service", "operation"],
        )

        with context_scope(service_name="api", operation_name="get_user"):
            ctx = get_current_context()
            counter.labels(
                service=ctx.service_name,
                operation=ctx.operation_name,
            ).inc()

        values = counter.collect()
        assert len(values) == 1
        assert values[0].labels.labels["service"] == "api"
        assert values[0].labels.labels["operation"] == "get_user"


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
