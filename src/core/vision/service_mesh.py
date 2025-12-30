"""Service mesh integration patterns for Vision Provider system.

This module provides service mesh capabilities including:
- Service discovery and registration
- Traffic management (routing, load balancing)
- Sidecar proxy patterns
- Service-to-service communication
- mTLS and security policies
"""

import asyncio
import hashlib
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .base import VisionDescription, VisionProvider


class ServiceStatus(Enum):
    """Status of a service instance."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    STARTING = "starting"
    STOPPING = "stopping"


class LoadBalancerPolicy(Enum):
    """Load balancer policy for service mesh."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"


class RetryPolicy(Enum):
    """Retry policy for service mesh."""

    NONE = "none"
    SIMPLE = "simple"
    EXPONENTIAL = "exponential"


class CircuitState(Enum):
    """Circuit breaker state."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ServiceInstance:
    """Service instance in the mesh."""

    instance_id: str
    service_name: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.HEALTHY
    weight: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0

    @property
    def address(self) -> str:
        """Get service address."""
        return f"{self.host}:{self.port}"

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    def update_heartbeat(self) -> None:
        """Update last heartbeat time."""
        self.last_heartbeat = datetime.now()

    def is_stale(self, timeout_seconds: float = 30.0) -> bool:
        """Check if heartbeat is stale."""
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return elapsed > timeout_seconds


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""

    service_name: str
    instances: List[ServiceInstance] = field(default_factory=list)
    load_balancer_policy: LoadBalancerPolicy = LoadBalancerPolicy.ROUND_ROBIN
    retry_policy: RetryPolicy = RetryPolicy.SIMPLE
    max_retries: int = 3
    timeout_seconds: float = 30.0
    circuit_breaker_enabled: bool = True
    circuit_threshold: int = 5
    circuit_timeout_seconds: float = 30.0

    def add_instance(self, instance: ServiceInstance) -> None:
        """Add service instance."""
        self.instances.append(instance)

    def remove_instance(self, instance_id: str) -> bool:
        """Remove service instance."""
        for i, inst in enumerate(self.instances):
            if inst.instance_id == instance_id:
                self.instances.pop(i)
                return True
        return False

    def get_healthy_instances(self) -> List[ServiceInstance]:
        """Get healthy instances."""
        return [inst for inst in self.instances if inst.status == ServiceStatus.HEALTHY]


@dataclass
class TrafficRule:
    """Traffic routing rule."""

    rule_id: str
    service_name: str
    match_headers: Dict[str, str] = field(default_factory=dict)
    match_path: Optional[str] = None
    destination_service: Optional[str] = None
    weight_distribution: Dict[str, int] = field(default_factory=dict)
    retry_on: List[str] = field(default_factory=list)
    timeout_seconds: Optional[float] = None
    priority: int = 0
    enabled: bool = True

    def matches(self, headers: Dict[str, str], path: Optional[str] = None) -> bool:
        """Check if rule matches request."""
        if not self.enabled:
            return False

        for key, value in self.match_headers.items():
            if headers.get(key) != value:
                return False

        if self.match_path and path:
            if not path.startswith(self.match_path):
                return False

        return True


@dataclass
class MeshConfig:
    """Service mesh configuration."""

    mesh_name: str = "default"
    heartbeat_interval_seconds: float = 10.0
    heartbeat_timeout_seconds: float = 30.0
    default_timeout_seconds: float = 30.0
    default_retries: int = 3
    circuit_breaker_enabled: bool = True
    mtls_enabled: bool = False
    tracing_enabled: bool = True
    metrics_enabled: bool = True


class ServiceRegistry:
    """Service registry for service discovery."""

    def __init__(self, config: Optional[MeshConfig] = None) -> None:
        """Initialize service registry.

        Args:
            config: Mesh configuration
        """
        self._config = config or MeshConfig()
        self._services: Dict[str, ServiceEndpoint] = {}
        self._lock = threading.Lock()
        self._watchers: Dict[str, List[Callable[[str, ServiceInstance, str], None]]] = {}

    @property
    def config(self) -> MeshConfig:
        """Return mesh configuration."""
        return self._config

    def register(
        self,
        service_name: str,
        instance: ServiceInstance,
    ) -> None:
        """Register a service instance.

        Args:
            service_name: Name of service
            instance: Service instance
        """
        with self._lock:
            if service_name not in self._services:
                self._services[service_name] = ServiceEndpoint(service_name=service_name)
            self._services[service_name].add_instance(instance)

        self._notify_watchers(service_name, instance, "register")

    def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance.

        Args:
            service_name: Name of service
            instance_id: Instance ID

        Returns:
            True if deregistered
        """
        with self._lock:
            if service_name not in self._services:
                return False

            endpoint = self._services[service_name]
            for inst in endpoint.instances:
                if inst.instance_id == instance_id:
                    endpoint.remove_instance(instance_id)
                    self._notify_watchers(service_name, inst, "deregister")
                    return True

        return False

    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Update heartbeat for instance.

        Args:
            service_name: Name of service
            instance_id: Instance ID

        Returns:
            True if updated
        """
        with self._lock:
            if service_name not in self._services:
                return False

            for inst in self._services[service_name].instances:
                if inst.instance_id == instance_id:
                    inst.update_heartbeat()
                    return True

        return False

    def get_service(self, service_name: str) -> Optional[ServiceEndpoint]:
        """Get service endpoint.

        Args:
            service_name: Name of service

        Returns:
            Service endpoint or None
        """
        with self._lock:
            return self._services.get(service_name)

    def get_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get service instances.

        Args:
            service_name: Name of service

        Returns:
            List of instances
        """
        with self._lock:
            endpoint = self._services.get(service_name)
            if endpoint:
                return list(endpoint.instances)
            return []

    def get_healthy_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get healthy service instances.

        Args:
            service_name: Name of service

        Returns:
            List of healthy instances
        """
        with self._lock:
            endpoint = self._services.get(service_name)
            if endpoint:
                return endpoint.get_healthy_instances()
            return []

    def list_services(self) -> List[str]:
        """List all registered services.

        Returns:
            List of service names
        """
        with self._lock:
            return list(self._services.keys())

    def watch(
        self,
        service_name: str,
        callback: Callable[[str, ServiceInstance, str], None],
    ) -> None:
        """Watch for service changes.

        Args:
            service_name: Service to watch
            callback: Callback function(service_name, instance, event_type)
        """
        with self._lock:
            if service_name not in self._watchers:
                self._watchers[service_name] = []
            self._watchers[service_name].append(callback)

    def unwatch(
        self,
        service_name: str,
        callback: Callable[[str, ServiceInstance, str], None],
    ) -> bool:
        """Remove watcher.

        Args:
            service_name: Service name
            callback: Callback to remove

        Returns:
            True if removed
        """
        with self._lock:
            if service_name in self._watchers:
                try:
                    self._watchers[service_name].remove(callback)
                    return True
                except ValueError:
                    pass
        return False

    def _notify_watchers(
        self,
        service_name: str,
        instance: ServiceInstance,
        event_type: str,
    ) -> None:
        """Notify watchers of service change."""
        callbacks = self._watchers.get(service_name, [])
        for callback in callbacks:
            try:
                callback(service_name, instance, event_type)
            except Exception:
                pass

    def cleanup_stale(self) -> int:
        """Remove stale instances.

        Returns:
            Number of removed instances
        """
        removed = 0
        timeout = self._config.heartbeat_timeout_seconds

        with self._lock:
            for service_name, endpoint in self._services.items():
                stale = [inst for inst in endpoint.instances if inst.is_stale(timeout)]
                for inst in stale:
                    endpoint.remove_instance(inst.instance_id)
                    inst.status = ServiceStatus.UNHEALTHY
                    removed += 1

        return removed


class LoadBalancer:
    """Load balancer for service mesh."""

    def __init__(
        self,
        policy: LoadBalancerPolicy = LoadBalancerPolicy.ROUND_ROBIN,
    ) -> None:
        """Initialize load balancer.

        Args:
            policy: Load balancer policy
        """
        self._policy = policy
        self._round_robin_index: Dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def policy(self) -> LoadBalancerPolicy:
        """Return load balancer policy."""
        return self._policy

    def select(
        self,
        service_name: str,
        instances: List[ServiceInstance],
        request_key: Optional[str] = None,
    ) -> Optional[ServiceInstance]:
        """Select an instance based on policy.

        Args:
            service_name: Name of service
            instances: Available instances
            request_key: Optional key for consistent hashing

        Returns:
            Selected instance or None
        """
        if not instances:
            return None

        healthy = [i for i in instances if i.status == ServiceStatus.HEALTHY]
        if not healthy:
            return None

        if self._policy == LoadBalancerPolicy.ROUND_ROBIN:
            return self._round_robin(service_name, healthy)
        elif self._policy == LoadBalancerPolicy.LEAST_CONNECTIONS:
            return self._least_connections(healthy)
        elif self._policy == LoadBalancerPolicy.RANDOM:
            return self._random(healthy)
        elif self._policy == LoadBalancerPolicy.WEIGHTED:
            return self._weighted(healthy)
        elif self._policy == LoadBalancerPolicy.CONSISTENT_HASH:
            return self._consistent_hash(healthy, request_key)
        else:
            return healthy[0]

    def _round_robin(
        self,
        service_name: str,
        instances: List[ServiceInstance],
    ) -> ServiceInstance:
        """Round robin selection."""
        with self._lock:
            idx = self._round_robin_index.get(service_name, 0)
            selected = instances[idx % len(instances)]
            self._round_robin_index[service_name] = idx + 1
            return selected

    def _least_connections(
        self,
        instances: List[ServiceInstance],
    ) -> ServiceInstance:
        """Least connections selection."""
        return min(instances, key=lambda i: i.active_connections)

    def _random(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random selection."""
        return random.choice(instances)

    def _weighted(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted selection."""
        total_weight = sum(i.weight for i in instances)
        if total_weight == 0:
            return instances[0]

        pick = random.uniform(0, total_weight)
        current = 0
        for inst in instances:
            current += inst.weight
            if current >= pick:
                return inst

        return instances[-1]

    def _consistent_hash(
        self,
        instances: List[ServiceInstance],
        request_key: Optional[str],
    ) -> ServiceInstance:
        """Consistent hash selection."""
        if not request_key:
            return self._random(instances)

        hash_val = int(hashlib.sha256(request_key.encode()).hexdigest(), 16)
        idx = hash_val % len(instances)
        return instances[idx]


class TrafficManager:
    """Traffic management for service mesh."""

    def __init__(self) -> None:
        """Initialize traffic manager."""
        self._rules: List[TrafficRule] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: TrafficRule) -> None:
        """Add traffic rule.

        Args:
            rule: Traffic rule
        """
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove traffic rule.

        Args:
            rule_id: Rule ID

        Returns:
            True if removed
        """
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.rule_id == rule_id:
                    self._rules.pop(i)
                    return True
        return False

    def get_rule(self, rule_id: str) -> Optional[TrafficRule]:
        """Get traffic rule.

        Args:
            rule_id: Rule ID

        Returns:
            Traffic rule or None
        """
        with self._lock:
            for rule in self._rules:
                if rule.rule_id == rule_id:
                    return rule
        return None

    def match_rule(
        self,
        service_name: str,
        headers: Dict[str, str],
        path: Optional[str] = None,
    ) -> Optional[TrafficRule]:
        """Find matching rule for request.

        Args:
            service_name: Service name
            headers: Request headers
            path: Request path

        Returns:
            Matching rule or None
        """
        with self._lock:
            for rule in self._rules:
                if rule.service_name == service_name:
                    if rule.matches(headers, path):
                        return rule
        return None

    def get_destination(
        self,
        rule: TrafficRule,
    ) -> Optional[str]:
        """Get destination service based on rule.

        Args:
            rule: Traffic rule

        Returns:
            Destination service name
        """
        if rule.destination_service:
            return rule.destination_service

        if rule.weight_distribution:
            total = sum(rule.weight_distribution.values())
            pick = random.uniform(0, total)
            current = 0
            for service, weight in rule.weight_distribution.items():
                current += weight
                if current >= pick:
                    return service

        return rule.service_name

    def list_rules(self, service_name: Optional[str] = None) -> List[TrafficRule]:
        """List traffic rules.

        Args:
            service_name: Optional filter by service

        Returns:
            List of rules
        """
        with self._lock:
            if service_name:
                return [r for r in self._rules if r.service_name == service_name]
            return list(self._rules)


@dataclass
class SidecarConfig:
    """Sidecar proxy configuration."""

    proxy_port: int = 15001
    admin_port: int = 15000
    inbound_port: int = 15006
    outbound_port: int = 15001
    enable_access_log: bool = True
    enable_tracing: bool = True
    tracing_sample_rate: float = 0.1
    connection_timeout_seconds: float = 5.0
    idle_timeout_seconds: float = 300.0


@dataclass
class SidecarStats:
    """Sidecar proxy statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_connections: int = 0
    active_connections: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    total_latency_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


class Sidecar:
    """Sidecar proxy for service mesh."""

    def __init__(
        self,
        service_name: str,
        config: Optional[SidecarConfig] = None,
    ) -> None:
        """Initialize sidecar.

        Args:
            service_name: Name of associated service
            config: Sidecar configuration
        """
        self._service_name = service_name
        self._config = config or SidecarConfig()
        self._stats = SidecarStats()
        self._running = False

    @property
    def service_name(self) -> str:
        """Return service name."""
        return self._service_name

    @property
    def config(self) -> SidecarConfig:
        """Return sidecar configuration."""
        return self._config

    @property
    def stats(self) -> SidecarStats:
        """Return sidecar statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Check if sidecar is running."""
        return self._running

    def start(self) -> None:
        """Start sidecar."""
        self._running = True
        self._stats.started_at = datetime.now()

    def stop(self) -> None:
        """Stop sidecar."""
        self._running = False

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        bytes_sent: int = 0,
        bytes_received: int = 0,
    ) -> None:
        """Record request statistics.

        Args:
            success: Whether request succeeded
            latency_ms: Request latency
            bytes_sent: Bytes sent
            bytes_received: Bytes received
        """
        self._stats.total_requests += 1
        if success:
            self._stats.successful_requests += 1
        else:
            self._stats.failed_requests += 1

        self._stats.total_latency_ms += latency_ms
        self._stats.bytes_sent += bytes_sent
        self._stats.bytes_received += bytes_received

    def record_connection(self, opened: bool) -> None:
        """Record connection event.

        Args:
            opened: Whether connection was opened
        """
        self._stats.total_connections += 1
        if opened:
            self._stats.active_connections += 1
        else:
            self._stats.active_connections = max(0, self._stats.active_connections - 1)


class ServiceMesh:
    """Service mesh coordinator."""

    def __init__(self, config: Optional[MeshConfig] = None) -> None:
        """Initialize service mesh.

        Args:
            config: Mesh configuration
        """
        self._config = config or MeshConfig()
        self._registry = ServiceRegistry(self._config)
        self._load_balancer = LoadBalancer()
        self._traffic_manager = TrafficManager()
        self._sidecars: Dict[str, Sidecar] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> MeshConfig:
        """Return mesh configuration."""
        return self._config

    @property
    def registry(self) -> ServiceRegistry:
        """Return service registry."""
        return self._registry

    @property
    def load_balancer(self) -> LoadBalancer:
        """Return load balancer."""
        return self._load_balancer

    @property
    def traffic_manager(self) -> TrafficManager:
        """Return traffic manager."""
        return self._traffic_manager

    def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ServiceInstance:
        """Register a service.

        Args:
            service_name: Name of service
            host: Service host
            port: Service port
            metadata: Optional metadata

        Returns:
            Service instance
        """
        instance = ServiceInstance(
            instance_id=f"{service_name}_{host}_{port}",
            service_name=service_name,
            host=host,
            port=port,
            metadata=metadata or {},
        )

        self._registry.register(service_name, instance)

        # Create sidecar
        with self._lock:
            sidecar = Sidecar(service_name)
            sidecar.start()
            self._sidecars[instance.instance_id] = sidecar

        return instance

    def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service.

        Args:
            service_name: Name of service
            instance_id: Instance ID

        Returns:
            True if deregistered
        """
        result = self._registry.deregister(service_name, instance_id)

        if result:
            with self._lock:
                if instance_id in self._sidecars:
                    self._sidecars[instance_id].stop()
                    del self._sidecars[instance_id]

        return result

    def get_sidecar(self, instance_id: str) -> Optional[Sidecar]:
        """Get sidecar for instance.

        Args:
            instance_id: Instance ID

        Returns:
            Sidecar or None
        """
        with self._lock:
            return self._sidecars.get(instance_id)

    def select_instance(
        self,
        service_name: str,
        headers: Optional[Dict[str, str]] = None,
        request_key: Optional[str] = None,
    ) -> Optional[ServiceInstance]:
        """Select service instance for request.

        Args:
            service_name: Target service
            headers: Request headers for routing
            request_key: Key for consistent hashing

        Returns:
            Selected instance or None
        """
        # Check traffic rules
        if headers:
            rule = self._traffic_manager.match_rule(service_name, headers)
            if rule:
                service_name = self._traffic_manager.get_destination(rule) or service_name

        instances = self._registry.get_healthy_instances(service_name)
        return self._load_balancer.select(service_name, instances, request_key)

    def get_mesh_stats(self) -> Dict[str, Any]:
        """Get mesh statistics.

        Returns:
            Mesh statistics
        """
        services = self._registry.list_services()
        total_instances = 0
        healthy_instances = 0

        for service in services:
            instances = self._registry.get_instances(service)
            total_instances += len(instances)
            healthy_instances += len([i for i in instances if i.status == ServiceStatus.HEALTHY])

        sidecar_stats = {}
        with self._lock:
            for instance_id, sidecar in self._sidecars.items():
                sidecar_stats[instance_id] = {
                    "total_requests": sidecar.stats.total_requests,
                    "success_rate": sidecar.stats.success_rate,
                    "average_latency_ms": sidecar.stats.average_latency_ms,
                }

        return {
            "mesh_name": self._config.mesh_name,
            "total_services": len(services),
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "traffic_rules": len(self._traffic_manager.list_rules()),
            "sidecars": sidecar_stats,
        }


class MeshVisionProvider(VisionProvider):
    """Vision provider with service mesh integration."""

    def __init__(
        self,
        provider: VisionProvider,
        mesh: ServiceMesh,
        service_name: str,
    ) -> None:
        """Initialize mesh provider.

        Args:
            provider: Underlying vision provider
            mesh: Service mesh
            service_name: Service name for this provider
        """
        self._provider = provider
        self._mesh = mesh
        self._service_name = service_name
        self._instance: Optional[ServiceInstance] = None

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"mesh_{self._provider.provider_name}"

    @property
    def mesh(self) -> ServiceMesh:
        """Return service mesh."""
        return self._mesh

    @property
    def service_name(self) -> str:
        """Return service name."""
        return self._service_name

    def register(self, host: str, port: int) -> ServiceInstance:
        """Register this provider in mesh.

        Args:
            host: Service host
            port: Service port

        Returns:
            Service instance
        """
        self._instance = self._mesh.register_service(
            self._service_name,
            host,
            port,
            metadata={"provider": self._provider.provider_name},
        )
        return self._instance

    def deregister(self) -> bool:
        """Deregister from mesh.

        Returns:
            True if deregistered
        """
        if self._instance:
            return self._mesh.deregister_service(
                self._service_name,
                self._instance.instance_id,
            )
        return False

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with mesh integration.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        start_time = time.time()
        success = False

        try:
            result = await self._provider.analyze_image(image_data, include_description)
            success = True

            # Update instance stats
            if self._instance:
                self._instance.total_requests += 1

                # Update sidecar stats
                sidecar = self._mesh.get_sidecar(self._instance.instance_id)
                if sidecar:
                    latency_ms = (time.time() - start_time) * 1000
                    sidecar.record_request(
                        success=True,
                        latency_ms=latency_ms,
                        bytes_received=len(image_data),
                    )

            return result

        except Exception as e:
            if self._instance:
                self._instance.total_requests += 1
                self._instance.failed_requests += 1

                sidecar = self._mesh.get_sidecar(self._instance.instance_id)
                if sidecar:
                    latency_ms = (time.time() - start_time) * 1000
                    sidecar.record_request(
                        success=False,
                        latency_ms=latency_ms,
                    )

            raise


def create_mesh_provider(
    provider: VisionProvider,
    mesh: Optional[ServiceMesh] = None,
    service_name: str = "vision",
) -> MeshVisionProvider:
    """Create a mesh vision provider.

    Args:
        provider: Underlying vision provider
        mesh: Optional service mesh
        service_name: Service name

    Returns:
        MeshVisionProvider instance
    """
    if mesh is None:
        mesh = ServiceMesh()

    return MeshVisionProvider(
        provider=provider,
        mesh=mesh,
        service_name=service_name,
    )
