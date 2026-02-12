"""Service Registry Core.

Provides service discovery primitives:
- Service definitions
- Instance management
- Health status
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ServiceStatus(Enum):
    """Status of a service instance."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    STOPPED = "stopped"


class HealthCheckType(Enum):
    """Types of health checks."""
    HTTP = "http"
    TCP = "tcp"
    GRPC = "grpc"
    SCRIPT = "script"


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    check_type: HealthCheckType = HealthCheckType.HTTP
    endpoint: str = "/health"
    port: Optional[int] = None
    interval_seconds: float = 10.0
    timeout_seconds: float = 5.0
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceInstance:
    """An instance of a service."""
    instance_id: str
    service_name: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, str] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    weight: int = 1
    zone: Optional[str] = None
    region: Optional[str] = None

    # Health check
    health_config: Optional[HealthCheckConfig] = None
    last_health_check: Optional[HealthCheckResult] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Timestamps
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        return self.status in (ServiceStatus.HEALTHY, ServiceStatus.STARTING)

    def record_health_check(self, result: HealthCheckResult) -> None:
        """Record health check result."""
        self.last_health_check = result

        if result.healthy:
            self.consecutive_failures = 0
            self.consecutive_successes += 1

            if self.health_config:
                if self.consecutive_successes >= self.health_config.healthy_threshold:
                    self.status = ServiceStatus.HEALTHY
        else:
            self.consecutive_successes = 0
            self.consecutive_failures += 1

            if self.health_config:
                if self.consecutive_failures >= self.health_config.unhealthy_threshold:
                    self.status = ServiceStatus.UNHEALTHY

    def heartbeat(self) -> None:
        """Record heartbeat."""
        self.last_heartbeat = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "service_name": self.service_name,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "metadata": self.metadata,
            "tags": list(self.tags),
            "weight": self.weight,
            "zone": self.zone,
            "region": self.region,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
        }


@dataclass
class ServiceDefinition:
    """Definition of a service."""
    name: str
    description: str = ""
    version: str = "1.0.0"
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, str] = field(default_factory=dict)
    health_config: Optional[HealthCheckConfig] = None
    ttl_seconds: float = 30.0  # Instance TTL without heartbeat

    def create_instance(
        self,
        host: str,
        port: int,
        instance_id: Optional[str] = None,
        **kwargs,
    ) -> ServiceInstance:
        """Create a new instance of this service."""
        instance_id = instance_id or generate_instance_id(self.name)

        return ServiceInstance(
            instance_id=instance_id,
            service_name=self.name,
            host=host,
            port=port,
            health_config=self.health_config,
            tags=self.tags.copy(),
            **kwargs,
        )


@dataclass
class Service:
    """A service with its instances."""
    definition: ServiceDefinition
    instances: Dict[str, ServiceInstance] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def healthy_instances(self) -> List[ServiceInstance]:
        return [i for i in self.instances.values() if i.is_healthy]

    @property
    def available_instances(self) -> List[ServiceInstance]:
        return [i for i in self.instances.values() if i.is_available]

    def add_instance(self, instance: ServiceInstance) -> None:
        """Add an instance."""
        self.instances[instance.instance_id] = instance

    def remove_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """Remove an instance."""
        return self.instances.pop(instance_id, None)

    def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """Get an instance by ID."""
        return self.instances.get(instance_id)


def generate_instance_id(service_name: str) -> str:
    """Generate unique instance ID."""
    timestamp = int(time.time() * 1000)
    random_part = secrets.token_hex(4)
    return f"{service_name}_{timestamp}_{random_part}"


@dataclass
class ServiceQuery:
    """Query for finding services."""
    service_name: str
    tags: Optional[Set[str]] = None
    zone: Optional[str] = None
    region: Optional[str] = None
    healthy_only: bool = True
    metadata_filters: Dict[str, str] = field(default_factory=dict)

    def matches(self, instance: ServiceInstance) -> bool:
        """Check if instance matches query."""
        if instance.service_name != self.service_name:
            return False

        if self.healthy_only and not instance.is_healthy:
            return False

        if self.tags and not self.tags.issubset(instance.tags):
            return False

        if self.zone and instance.zone != self.zone:
            return False

        if self.region and instance.region != self.region:
            return False

        for key, value in self.metadata_filters.items():
            if instance.metadata.get(key) != value:
                return False

        return True


@dataclass
class ServiceEvent:
    """Event for service changes."""
    event_type: str  # registered, deregistered, health_changed
    service_name: str
    instance_id: str
    instance: Optional[ServiceInstance] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
