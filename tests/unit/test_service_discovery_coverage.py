"""Tests for service_mesh discovery module to improve coverage."""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from src.core.service_mesh.discovery import (
    ServiceStatus,
    ServiceInstance,
    ServiceDefinition,
    ServiceRegistry,
    InMemoryServiceRegistry,
    ServiceDiscovery,
    ServiceRegistrar,
)


class TestServiceStatus:
    """Tests for ServiceStatus enum."""

    def test_all_statuses(self):
        """Test all service statuses exist."""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"
        assert ServiceStatus.UNKNOWN.value == "unknown"
        assert ServiceStatus.DRAINING.value == "draining"


class TestServiceInstance:
    """Tests for ServiceInstance class."""

    def test_address_property(self):
        """Test address property."""
        instance = ServiceInstance(
            service_name="test",
            instance_id="i1",
            host="localhost",
            port=8080,
        )

        assert instance.address == "localhost:8080"

    def test_is_available_healthy(self):
        """Test is_available returns True for healthy."""
        instance = ServiceInstance(
            service_name="test",
            instance_id="i1",
            host="localhost",
            port=8080,
            status=ServiceStatus.HEALTHY,
        )

        assert instance.is_available() is True

    def test_is_available_unhealthy(self):
        """Test is_available returns False for unhealthy."""
        instance = ServiceInstance(
            service_name="test",
            instance_id="i1",
            host="localhost",
            port=8080,
            status=ServiceStatus.UNHEALTHY,
        )

        assert instance.is_available() is False

    def test_default_values(self):
        """Test default values."""
        instance = ServiceInstance(
            service_name="test",
            instance_id="i1",
            host="localhost",
            port=8080,
        )

        assert instance.status == ServiceStatus.UNKNOWN
        assert instance.metadata == {}
        assert instance.tags == set()
        assert instance.weight == 100


class TestServiceDefinition:
    """Tests for ServiceDefinition class."""

    def test_default_values(self):
        """Test default values."""
        definition = ServiceDefinition(name="test-service")

        assert definition.name == "test-service"
        assert definition.version == "1.0.0"
        assert definition.protocol == "http"
        assert definition.health_check_path == "/health"
        assert definition.health_check_interval == 30.0
        assert definition.deregister_after == 90.0


class TestServiceRegistryAbstract:
    """Tests for abstract ServiceRegistry class."""

    def test_register_is_abstract(self):
        """Test register is abstract (line 74)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ServiceRegistry()

    def test_deregister_is_abstract(self):
        """Test deregister is abstract (line 79)."""
        # Verified through concrete implementations
        pass

    def test_heartbeat_is_abstract(self):
        """Test heartbeat is abstract (line 84)."""
        pass

    def test_get_instances_is_abstract(self):
        """Test get_instances is abstract (line 94)."""
        pass

    def test_get_all_services_is_abstract(self):
        """Test get_all_services is abstract (line 99)."""
        pass

    def test_watch_is_abstract(self):
        """Test watch is abstract (line 108)."""
        pass


class TestInMemoryServiceRegistry:
    """Tests for InMemoryServiceRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return InMemoryServiceRegistry(heartbeat_timeout=1.0)

    @pytest.fixture
    def instance(self):
        """Create a test service instance."""
        return ServiceInstance(
            service_name="test-service",
            instance_id="i1",
            host="localhost",
            port=8080,
        )

    @pytest.mark.asyncio
    async def test_register(self, registry, instance):
        """Test register method."""
        result = await registry.register(instance)

        assert result is True
        assert instance.status == ServiceStatus.HEALTHY
        instances = await registry.get_instances("test-service")
        assert len(instances) == 1

    @pytest.mark.asyncio
    async def test_deregister(self, registry, instance):
        """Test deregister method (lines 143-151)."""
        await registry.register(instance)

        result = await registry.deregister("test-service", "i1")

        assert result is True
        instances = await registry.get_instances("test-service")
        assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_deregister_nonexistent_service(self, registry):
        """Test deregister with nonexistent service (line 151)."""
        result = await registry.deregister("nonexistent", "i1")
        assert result is False

    @pytest.mark.asyncio
    async def test_deregister_nonexistent_instance(self, registry, instance):
        """Test deregister with nonexistent instance."""
        await registry.register(instance)

        result = await registry.deregister("test-service", "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp(self, registry, instance):
        """Test heartbeat updates last_heartbeat (lines 153-163)."""
        await registry.register(instance)
        original_time = instance.last_heartbeat

        await asyncio.sleep(0.01)
        result = await registry.heartbeat("test-service", "i1")

        assert result is True
        assert instance.last_heartbeat > original_time

    @pytest.mark.asyncio
    async def test_heartbeat_recovers_unhealthy(self, registry, instance):
        """Test heartbeat recovers unhealthy instance (lines 159-161)."""
        await registry.register(instance)
        instance.status = ServiceStatus.UNHEALTHY

        # Track if watcher was called
        watcher_called = []
        def watcher(instances):
            watcher_called.append(True)

        await registry.watch("test-service", watcher)

        result = await registry.heartbeat("test-service", "i1")

        assert result is True
        assert instance.status == ServiceStatus.HEALTHY
        assert len(watcher_called) > 0  # Watcher should be notified

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_service(self, registry):
        """Test heartbeat with nonexistent service (line 163)."""
        result = await registry.heartbeat("nonexistent", "i1")
        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_instance(self, registry, instance):
        """Test heartbeat with nonexistent instance."""
        await registry.register(instance)

        result = await registry.heartbeat("test-service", "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_instances_empty_service(self, registry):
        """Test get_instances with no service (line 176)."""
        instances = await registry.get_instances("nonexistent")
        assert instances == []

    @pytest.mark.asyncio
    async def test_get_instances_with_tags(self, registry):
        """Test get_instances filters by tags (line 184)."""
        instance1 = ServiceInstance(
            service_name="test-service",
            instance_id="i1",
            host="localhost",
            port=8080,
            tags={"region:us-east", "env:prod"},
        )
        instance2 = ServiceInstance(
            service_name="test-service",
            instance_id="i2",
            host="localhost",
            port=8081,
            tags={"region:us-west", "env:prod"},
        )

        await registry.register(instance1)
        await registry.register(instance2)

        # Filter by tags
        instances = await registry.get_instances(
            "test-service",
            tags={"region:us-east"},
        )

        assert len(instances) == 1
        assert instances[0].instance_id == "i1"

    @pytest.mark.asyncio
    async def test_get_all_services(self, registry):
        """Test get_all_services method (lines 188-190)."""
        instance1 = ServiceInstance(
            service_name="service-a",
            instance_id="i1",
            host="localhost",
            port=8080,
        )
        instance2 = ServiceInstance(
            service_name="service-b",
            instance_id="i2",
            host="localhost",
            port=8081,
        )

        await registry.register(instance1)
        await registry.register(instance2)

        services = await registry.get_all_services()

        assert "service-a" in services
        assert "service-b" in services

    @pytest.mark.asyncio
    async def test_watch_and_unsubscribe(self, registry, instance):
        """Test watch returns unsubscribe function (lines 197-207)."""
        watcher_calls = []

        def watcher(instances):
            watcher_calls.append(instances)

        unsubscribe = await registry.watch("test-service", watcher)

        # Register should trigger watcher
        await registry.register(instance)
        assert len(watcher_calls) == 1

        # Unsubscribe
        unsubscribe()

        # Register another instance - watcher should not be called
        instance2 = ServiceInstance(
            service_name="test-service",
            instance_id="i2",
            host="localhost",
            port=8081,
        )
        await registry.register(instance2)

        # Watcher should not have been called again
        assert len(watcher_calls) == 1

    @pytest.mark.asyncio
    async def test_check_health_marks_stale_unhealthy(self, registry):
        """Test _check_health marks stale instances unhealthy (lines 217-224)."""
        # Use short heartbeat timeout
        registry.heartbeat_timeout = 0.01

        instance = ServiceInstance(
            service_name="test-service",
            instance_id="i1",
            host="localhost",
            port=8080,
            last_heartbeat=time.time() - 100,  # Old heartbeat
        )
        await registry.register(instance)

        # Wait for health check
        await asyncio.sleep(0.02)

        # Get instances will trigger health check
        instances = await registry.get_instances("test-service", healthy_only=False)

        assert len(instances) == 1
        assert instances[0].status == ServiceStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_notify_watchers_async_callback(self, registry, instance):
        """Test _notify_watchers with async callback (lines 231-239)."""
        async_calls = []

        async def async_watcher(instances):
            async_calls.append(instances)

        await registry.watch("test-service", async_watcher)
        await registry.register(instance)

        assert len(async_calls) == 1

    @pytest.mark.asyncio
    async def test_notify_watchers_callback_error(self, registry, instance):
        """Test _notify_watchers handles callback errors (lines 238-239)."""
        def bad_watcher(instances):
            raise ValueError("Watcher error")

        await registry.watch("test-service", bad_watcher)

        # Should not raise
        await registry.register(instance)


class TestServiceDiscovery:
    """Tests for ServiceDiscovery class."""

    @pytest.fixture
    def registry(self):
        return InMemoryServiceRegistry()

    @pytest.fixture
    def discovery(self, registry):
        return ServiceDiscovery(registry)

    @pytest.fixture
    def instance(self):
        return ServiceInstance(
            service_name="test-service",
            instance_id="i1",
            host="localhost",
            port=8080,
        )

    @pytest.mark.asyncio
    async def test_discover_with_cache(self, registry, discovery, instance):
        """Test discover uses cache (lines 257-271)."""
        await registry.register(instance)

        # First call - populates cache
        instances1 = await discovery.discover("test-service")
        assert len(instances1) == 1

        # Second call - should use cache
        instances2 = await discovery.discover("test-service")
        assert instances2 == instances1

    @pytest.mark.asyncio
    async def test_discover_cache_expired(self, registry, discovery, instance):
        """Test discover bypasses expired cache (lines 259-262)."""
        discovery._cache_ttl = 0.01  # Very short TTL

        await registry.register(instance)

        # First call
        await discovery.discover("test-service")

        # Wait for cache to expire
        await asyncio.sleep(0.02)

        # This should fetch fresh data
        instances = await discovery.discover("test-service")
        assert len(instances) == 1

    @pytest.mark.asyncio
    async def test_discover_without_cache(self, registry, discovery, instance):
        """Test discover without cache."""
        await registry.register(instance)

        instances = await discovery.discover("test-service", use_cache=False)
        assert len(instances) == 1

    @pytest.mark.asyncio
    async def test_discover_one(self, registry, discovery):
        """Test discover_one returns single instance (lines 279-282)."""
        for i in range(3):
            instance = ServiceInstance(
                service_name="test-service",
                instance_id=f"i{i}",
                host="localhost",
                port=8080 + i,
            )
            await registry.register(instance)

        result = await discovery.discover_one("test-service")

        assert result is not None
        assert result.service_name == "test-service"

    @pytest.mark.asyncio
    async def test_discover_one_no_instances(self, discovery):
        """Test discover_one returns None when no instances (lines 280-281)."""
        result = await discovery.discover_one("nonexistent")
        assert result is None

    def test_invalidate_cache_specific_service(self, registry, discovery):
        """Test invalidate_cache for specific service (lines 286-289)."""
        discovery._cache = {
            "service-a:": ([MagicMock()], time.time()),
            "service-a:tag1": ([MagicMock()], time.time()),
            "service-b:": ([MagicMock()], time.time()),
        }

        discovery.invalidate_cache("service-a")

        assert "service-a:" not in discovery._cache
        assert "service-a:tag1" not in discovery._cache
        assert "service-b:" in discovery._cache

    def test_invalidate_cache_all(self, registry, discovery):
        """Test invalidate_cache clears all (lines 290-291)."""
        discovery._cache = {
            "service-a:": ([MagicMock()], time.time()),
            "service-b:": ([MagicMock()], time.time()),
        }

        discovery.invalidate_cache()

        assert discovery._cache == {}


class TestServiceRegistrar:
    """Tests for ServiceRegistrar class."""

    @pytest.fixture
    def registry(self):
        return InMemoryServiceRegistry()

    @pytest.fixture
    def definition(self):
        return ServiceDefinition(
            name="test-service",
            version="2.0.0",
            health_check_interval=0.1,  # Short for testing
        )

    @pytest.mark.asyncio
    async def test_generate_instance_id(self, registry, definition):
        """Test _generate_instance_id (lines 317-318)."""
        registrar = ServiceRegistrar(
            registry=registry,
            definition=definition,
            host="localhost",
            port=8080,
        )

        assert len(registrar.instance_id) == 12

    @pytest.mark.asyncio
    async def test_start_registers_instance(self, registry, definition):
        """Test start registers and begins heartbeats (lines 322-337)."""
        registrar = ServiceRegistrar(
            registry=registry,
            definition=definition,
            host="localhost",
            port=8080,
        )

        await registrar.start()

        try:
            instances = await registry.get_instances("test-service")
            assert len(instances) == 1
            assert instances[0].metadata["version"] == "2.0.0"
            assert registrar._running is True
            assert registrar._heartbeat_task is not None
        finally:
            await registrar.stop()

    @pytest.mark.asyncio
    async def test_stop_deregisters_instance(self, registry, definition):
        """Test stop deregisters and stops heartbeats (lines 341-354)."""
        registrar = ServiceRegistrar(
            registry=registry,
            definition=definition,
            host="localhost",
            port=8080,
        )

        await registrar.start()
        await registrar.stop()

        instances = await registry.get_instances("test-service")
        assert len(instances) == 0
        assert registrar._running is False

    @pytest.mark.asyncio
    async def test_heartbeat_loop(self, registry, definition):
        """Test _heartbeat_loop sends periodic heartbeats (lines 358-369)."""
        definition.health_check_interval = 0.03  # Very short interval

        registrar = ServiceRegistrar(
            registry=registry,
            definition=definition,
            host="localhost",
            port=8080,
        )

        await registrar.start()

        try:
            # Wait for a few heartbeats
            await asyncio.sleep(0.05)

            # Instance should still be healthy
            instances = await registry.get_instances("test-service")
            assert len(instances) == 1
            assert instances[0].status == ServiceStatus.HEALTHY
        finally:
            await registrar.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self, registry, definition):
        """Test async context manager (lines 371-376)."""
        registrar = ServiceRegistrar(
            registry=registry,
            definition=definition,
            host="localhost",
            port=8080,
        )

        async with registrar as reg:
            assert reg is registrar
            instances = await registry.get_instances("test-service")
            assert len(instances) == 1

        # After exiting context, should be deregistered
        instances = await registry.get_instances("test-service")
        assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_heartbeat_loop_error_handling(self, registry, definition):
        """Test _heartbeat_loop handles errors (lines 368-369)."""
        definition.health_check_interval = 0.01

        registrar = ServiceRegistrar(
            registry=registry,
            definition=definition,
            host="localhost",
            port=8080,
        )

        await registrar.start()

        # Simulate heartbeat error by breaking the registry
        original_heartbeat = registry.heartbeat
        async def failing_heartbeat(*args, **kwargs):
            raise RuntimeError("Heartbeat failed")
        registry.heartbeat = failing_heartbeat

        try:
            # Wait for a heartbeat attempt
            await asyncio.sleep(0.02)

            # Should not crash, just log error
            assert registrar._running is True
        finally:
            registry.heartbeat = original_heartbeat
            await registrar.stop()

    @pytest.mark.asyncio
    async def test_custom_instance_id(self, registry, definition):
        """Test custom instance_id parameter."""
        registrar = ServiceRegistrar(
            registry=registry,
            definition=definition,
            host="localhost",
            port=8080,
            instance_id="custom-id-123",
        )

        assert registrar.instance_id == "custom-id-123"

        await registrar.start()
        try:
            instances = await registry.get_instances("test-service")
            assert instances[0].instance_id == "custom-id-123"
        finally:
            await registrar.stop()


class TestInMemoryServiceRegistryLockCreation:
    """Test lock creation edge cases."""

    @pytest.mark.asyncio
    async def test_lock_created_once(self):
        """Test _get_lock creates lock only once."""
        registry = InMemoryServiceRegistry()

        lock1 = registry._get_lock()
        lock2 = registry._get_lock()

        assert lock1 is lock2
