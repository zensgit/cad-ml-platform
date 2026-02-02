"""Service Registry Implementation.

Provides service registration and discovery:
- In-memory registry
- Health checking
- Event notifications
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.service_registry.core import (
    HealthCheckConfig,
    HealthCheckResult,
    HealthCheckType,
    Service,
    ServiceDefinition,
    ServiceEvent,
    ServiceInstance,
    ServiceQuery,
    ServiceStatus,
)

logger = logging.getLogger(__name__)


class ServiceRegistry(ABC):
    """Abstract service registry."""

    @abstractmethod
    async def register(self, instance: ServiceInstance) -> bool:
        """Register a service instance."""
        pass

    @abstractmethod
    async def deregister(self, instance_id: str) -> bool:
        """Deregister a service instance."""
        pass

    @abstractmethod
    async def heartbeat(self, instance_id: str) -> bool:
        """Send heartbeat for an instance."""
        pass

    @abstractmethod
    async def get_service(self, name: str) -> Optional[Service]:
        """Get service by name."""
        pass

    @abstractmethod
    async def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """Get instance by ID."""
        pass

    @abstractmethod
    async def query(self, query: ServiceQuery) -> List[ServiceInstance]:
        """Query for service instances."""
        pass

    @abstractmethod
    async def list_services(self) -> List[str]:
        """List all service names."""
        pass


class InMemoryServiceRegistry(ServiceRegistry):
    """In-memory service registry."""

    def __init__(self):
        self._services: Dict[str, Service] = {}
        self._instances: Dict[str, ServiceInstance] = {}  # instance_id -> instance
        self._listeners: List[Callable[[ServiceEvent], None]] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def register(self, instance: ServiceInstance) -> bool:
        async with self._get_lock():
            # Create service if not exists
            if instance.service_name not in self._services:
                self._services[instance.service_name] = Service(
                    definition=ServiceDefinition(name=instance.service_name)
                )

            service = self._services[instance.service_name]
            service.add_instance(instance)
            self._instances[instance.instance_id] = instance

            # Set initial status
            if instance.status == ServiceStatus.UNKNOWN:
                instance.status = ServiceStatus.STARTING

            logger.info(
                f"Registered instance {instance.instance_id} "
                f"for service {instance.service_name}"
            )

            # Notify listeners
            await self._notify(ServiceEvent(
                event_type="registered",
                service_name=instance.service_name,
                instance_id=instance.instance_id,
                instance=instance,
            ))

            return True

    async def deregister(self, instance_id: str) -> bool:
        async with self._get_lock():
            instance = self._instances.pop(instance_id, None)
            if not instance:
                return False

            if instance.service_name in self._services:
                self._services[instance.service_name].remove_instance(instance_id)

            instance.status = ServiceStatus.STOPPED

            logger.info(f"Deregistered instance {instance_id}")

            await self._notify(ServiceEvent(
                event_type="deregistered",
                service_name=instance.service_name,
                instance_id=instance_id,
                instance=instance,
            ))

            return True

    async def heartbeat(self, instance_id: str) -> bool:
        async with self._get_lock():
            instance = self._instances.get(instance_id)
            if not instance:
                return False

            instance.heartbeat()
            return True

    async def get_service(self, name: str) -> Optional[Service]:
        async with self._get_lock():
            return self._services.get(name)

    async def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        async with self._get_lock():
            return self._instances.get(instance_id)

    async def query(self, query: ServiceQuery) -> List[ServiceInstance]:
        async with self._get_lock():
            service = self._services.get(query.service_name)
            if not service:
                return []

            return [
                instance for instance in service.instances.values()
                if query.matches(instance)
            ]

    async def list_services(self) -> List[str]:
        async with self._get_lock():
            return list(self._services.keys())

    def add_listener(self, listener: Callable[[ServiceEvent], None]) -> None:
        """Add event listener."""
        self._listeners.append(listener)

    async def _notify(self, event: ServiceEvent) -> None:
        """Notify listeners of event."""
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                logger.error(f"Event listener error: {e}")


class HealthChecker:
    """Health checker for service instances."""

    def __init__(self, registry: ServiceRegistry):
        self._registry = registry
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._check_interval = 10.0

    async def start(self) -> None:
        """Start health checking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info("Health checker started")

    async def stop(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _check_loop(self) -> None:
        """Health check loop."""
        while self._running:
            try:
                services = await self._registry.list_services()

                for service_name in services:
                    service = await self._registry.get_service(service_name)
                    if not service:
                        continue

                    for instance in service.instances.values():
                        await self._check_instance(instance)

            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(self._check_interval)

    async def _check_instance(self, instance: ServiceInstance) -> None:
        """Check health of an instance."""
        if not instance.health_config:
            # No health config, assume healthy after registration
            if instance.status == ServiceStatus.STARTING:
                instance.status = ServiceStatus.HEALTHY
            return

        result = await self._perform_check(instance)
        old_status = instance.status
        instance.record_health_check(result)

        # Log status change
        if instance.status != old_status:
            logger.info(
                f"Instance {instance.instance_id} status changed: "
                f"{old_status.value} -> {instance.status.value}"
            )

    async def _perform_check(
        self,
        instance: ServiceInstance,
    ) -> HealthCheckResult:
        """Perform actual health check."""
        config = instance.health_config
        if not config:
            return HealthCheckResult(healthy=True, message="No health check configured")

        start_time = asyncio.get_event_loop().time()

        try:
            if config.check_type == HealthCheckType.HTTP:
                result = await self._http_check(instance, config)
            elif config.check_type == HealthCheckType.TCP:
                result = await self._tcp_check(instance, config)
            else:
                result = HealthCheckResult(
                    healthy=True,
                    message="Check type not implemented",
                )

            result.latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            return result

        except asyncio.TimeoutError:
            return HealthCheckResult(
                healthy=False,
                message="Health check timed out",
                latency_ms=config.timeout_seconds * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=str(e),
            )

    async def _http_check(
        self,
        instance: ServiceInstance,
        config: HealthCheckConfig,
    ) -> HealthCheckResult:
        """Perform HTTP health check."""
        port = config.port or instance.port
        url = f"http://{instance.host}:{port}{config.endpoint}"

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
                    headers=config.headers,
                ) as response:
                    healthy = 200 <= response.status < 300
                    return HealthCheckResult(
                        healthy=healthy,
                        message=f"HTTP {response.status}",
                        details={"status_code": response.status},
                    )
        except ImportError:
            # aiohttp not installed, simulate success
            return HealthCheckResult(healthy=True, message="HTTP check simulated")

    async def _tcp_check(
        self,
        instance: ServiceInstance,
        config: HealthCheckConfig,
    ) -> HealthCheckResult:
        """Perform TCP health check."""
        port = config.port or instance.port

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(instance.host, port),
                timeout=config.timeout_seconds,
            )
            writer.close()
            await writer.wait_closed()

            return HealthCheckResult(
                healthy=True,
                message="TCP connection successful",
            )
        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                message=f"TCP connection failed: {e}",
            )


class HeartbeatManager:
    """Manages heartbeat expiration."""

    def __init__(
        self,
        registry: InMemoryServiceRegistry,
        check_interval: float = 5.0,
        default_ttl: float = 30.0,
    ):
        self._registry = registry
        self._check_interval = check_interval
        self._default_ttl = default_ttl
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start heartbeat checking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info("Heartbeat manager started")

    async def stop(self) -> None:
        """Stop heartbeat checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _check_loop(self) -> None:
        """Check for expired instances."""
        while self._running:
            try:
                now = datetime.utcnow()
                services = await self._registry.list_services()

                for service_name in services:
                    service = await self._registry.get_service(service_name)
                    if not service:
                        continue

                    ttl = service.definition.ttl_seconds or self._default_ttl
                    cutoff = now - timedelta(seconds=ttl)

                    expired = [
                        instance.instance_id
                        for instance in service.instances.values()
                        if instance.last_heartbeat < cutoff
                    ]

                    for instance_id in expired:
                        logger.warning(f"Instance {instance_id} expired (no heartbeat)")
                        await self._registry.deregister(instance_id)

            except Exception as e:
                logger.error(f"Heartbeat check error: {e}")

            await asyncio.sleep(self._check_interval)


class ServiceClient:
    """Client for interacting with service registry."""

    def __init__(
        self,
        registry: ServiceRegistry,
        instance: ServiceInstance,
        heartbeat_interval: float = 10.0,
    ):
        self._registry = registry
        self._instance = instance
        self._heartbeat_interval = heartbeat_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def instance(self) -> ServiceInstance:
        return self._instance

    async def register(self) -> bool:
        """Register with the registry."""
        success = await self._registry.register(self._instance)
        if success:
            self._running = True
            self._task = asyncio.create_task(self._heartbeat_loop())
        return success

    async def deregister(self) -> bool:
        """Deregister from the registry."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        return await self._registry.deregister(self._instance.instance_id)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            try:
                await self._registry.heartbeat(self._instance.instance_id)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            await asyncio.sleep(self._heartbeat_interval)

    async def __aenter__(self) -> "ServiceClient":
        await self.register()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.deregister()
