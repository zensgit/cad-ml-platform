"""Service Discovery.

Provides service discovery capabilities:
- Service registration
- Service lookup
- Health-aware discovery
- Multiple backends (in-memory, consul, etcd)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service instance health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DRAINING = "draining"  # Not accepting new connections


@dataclass
class ServiceInstance:
    """A service instance."""
    service_name: str
    instance_id: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    weight: int = 100
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def is_available(self) -> bool:
        return self.status == ServiceStatus.HEALTHY


@dataclass
class ServiceDefinition:
    """Definition of a service."""
    name: str
    version: str = "1.0.0"
    protocol: str = "http"
    health_check_path: str = "/health"
    health_check_interval: float = 30.0
    deregister_after: float = 90.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, str] = field(default_factory=dict)


class ServiceRegistry(ABC):
    """Abstract base class for service registries."""

    @abstractmethod
    async def register(self, instance: ServiceInstance) -> bool:
        """Register a service instance."""
        pass

    @abstractmethod
    async def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance."""
        pass

    @abstractmethod
    async def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Send heartbeat for a service instance."""
        pass

    @abstractmethod
    async def get_instances(
        self,
        service_name: str,
        healthy_only: bool = True,
        tags: Optional[Set[str]] = None,
    ) -> List[ServiceInstance]:
        """Get instances of a service."""
        pass

    @abstractmethod
    async def get_all_services(self) -> List[str]:
        """Get all registered service names."""
        pass

    @abstractmethod
    async def watch(
        self,
        service_name: str,
        callback: Callable[[List[ServiceInstance]], None],
    ) -> Callable[[], None]:
        """Watch for changes to service instances. Returns unsubscribe function."""
        pass


class InMemoryServiceRegistry(ServiceRegistry):
    """In-memory service registry for testing and development."""

    def __init__(self, heartbeat_timeout: float = 60.0):
        self.heartbeat_timeout = heartbeat_timeout
        # {service_name: {instance_id: ServiceInstance}}
        self._services: Dict[str, Dict[str, ServiceInstance]] = {}
        self._watchers: Dict[str, List[Callable[[List[ServiceInstance]], None]]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def register(self, instance: ServiceInstance) -> bool:
        async with self._get_lock():
            if instance.service_name not in self._services:
                self._services[instance.service_name] = {}

            instance.status = ServiceStatus.HEALTHY
            instance.last_heartbeat = time.time()
            self._services[instance.service_name][instance.instance_id] = instance

            logger.info(
                f"Registered service instance: {instance.service_name}/{instance.instance_id} "
                f"at {instance.address}"
            )

            await self._notify_watchers(instance.service_name)
            return True

    async def deregister(self, service_name: str, instance_id: str) -> bool:
        async with self._get_lock():
            if service_name in self._services:
                if instance_id in self._services[service_name]:
                    del self._services[service_name][instance_id]
                    logger.info(f"Deregistered: {service_name}/{instance_id}")
                    await self._notify_watchers(service_name)
                    return True
            return False

    async def heartbeat(self, service_name: str, instance_id: str) -> bool:
        async with self._get_lock():
            if service_name in self._services:
                if instance_id in self._services[service_name]:
                    instance = self._services[service_name][instance_id]
                    instance.last_heartbeat = time.time()
                    if instance.status == ServiceStatus.UNHEALTHY:
                        instance.status = ServiceStatus.HEALTHY
                        await self._notify_watchers(service_name)
                    return True
            return False

    async def get_instances(
        self,
        service_name: str,
        healthy_only: bool = True,
        tags: Optional[Set[str]] = None,
    ) -> List[ServiceInstance]:
        async with self._get_lock():
            # Check for stale instances
            await self._check_health()

            if service_name not in self._services:
                return []

            instances = list(self._services[service_name].values())

            if healthy_only:
                instances = [i for i in instances if i.is_available()]

            if tags:
                instances = [i for i in instances if tags.issubset(i.tags)]

            return instances

    async def get_all_services(self) -> List[str]:
        async with self._get_lock():
            return list(self._services.keys())

    async def watch(
        self,
        service_name: str,
        callback: Callable[[List[ServiceInstance]], None],
    ) -> Callable[[], None]:
        async with self._get_lock():
            if service_name not in self._watchers:
                self._watchers[service_name] = []
            self._watchers[service_name].append(callback)

            def unsubscribe() -> None:
                if service_name in self._watchers:
                    if callback in self._watchers[service_name]:
                        self._watchers[service_name].remove(callback)

            return unsubscribe

    async def _check_health(self) -> None:
        """Mark stale instances as unhealthy."""
        now = time.time()
        for service_name, instances in self._services.items():
            changed = False
            for instance in instances.values():
                if instance.status == ServiceStatus.HEALTHY:
                    if now - instance.last_heartbeat > self.heartbeat_timeout:
                        instance.status = ServiceStatus.UNHEALTHY
                        changed = True
                        logger.warning(
                            f"Instance {service_name}/{instance.instance_id} "
                            f"marked unhealthy (no heartbeat)"
                        )
            if changed:
                await self._notify_watchers(service_name)

    async def _notify_watchers(self, service_name: str) -> None:
        """Notify watchers of changes."""
        if service_name not in self._watchers:
            return

        instances = list(self._services.get(service_name, {}).values())
        for callback in self._watchers[service_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(instances)
                else:
                    callback(instances)
            except Exception as e:
                logger.error(f"Watcher callback error: {e}")


class ServiceDiscovery:
    """High-level service discovery client."""

    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self._cache: Dict[str, Tuple[List[ServiceInstance], float]] = {}
        self._cache_ttl = 5.0  # seconds

    async def discover(
        self,
        service_name: str,
        tags: Optional[Set[str]] = None,
        use_cache: bool = True,
    ) -> List[ServiceInstance]:
        """Discover service instances."""
        cache_key = f"{service_name}:{','.join(sorted(tags or []))}"

        if use_cache and cache_key in self._cache:
            instances, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self._cache_ttl:
                return instances

        instances = await self.registry.get_instances(
            service_name,
            healthy_only=True,
            tags=tags,
        )

        self._cache[cache_key] = (instances, time.time())
        return instances

    async def discover_one(
        self,
        service_name: str,
        tags: Optional[Set[str]] = None,
    ) -> Optional[ServiceInstance]:
        """Discover a single service instance (random selection)."""
        instances = await self.discover(service_name, tags)
        if not instances:
            return None
        return random.choice(instances)

    def invalidate_cache(self, service_name: Optional[str] = None) -> None:
        """Invalidate discovery cache."""
        if service_name:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{service_name}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


class ServiceRegistrar:
    """Handles service registration lifecycle."""

    def __init__(
        self,
        registry: ServiceRegistry,
        definition: ServiceDefinition,
        host: str,
        port: int,
        instance_id: Optional[str] = None,
    ):
        self.registry = registry
        self.definition = definition
        self.host = host
        self.port = port
        self.instance_id = instance_id or self._generate_instance_id()

        self._instance: Optional[ServiceInstance] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    def _generate_instance_id(self) -> str:
        """Generate a unique instance ID."""
        data = f"{self.definition.name}:{self.host}:{self.port}:{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    async def start(self) -> None:
        """Start the registrar (register and begin heartbeats)."""
        self._instance = ServiceInstance(
            service_name=self.definition.name,
            instance_id=self.instance_id,
            host=self.host,
            port=self.port,
            metadata={
                "version": self.definition.version,
                "protocol": self.definition.protocol,
                **self.definition.metadata,
            },
            tags=self.definition.tags,
        )

        await self.registry.register(self._instance)
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop the registrar (deregister and stop heartbeats)."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._instance:
            await self.registry.deregister(
                self._instance.service_name,
                self._instance.instance_id,
            )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(self.definition.health_check_interval / 3)
                if self._instance and self._running:
                    await self.registry.heartbeat(
                        self._instance.service_name,
                        self._instance.instance_id,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def __aenter__(self) -> "ServiceRegistrar":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
