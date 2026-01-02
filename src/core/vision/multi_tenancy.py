"""Multi-tenancy support for Vision Provider system.

This module provides multi-tenant features including:
- Tenant isolation
- Resource quotas
- Tenant-specific configurations
- Usage tracking per tenant
- Tenant routing
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

from .base import VisionDescription, VisionProvider


class TenantStatus(Enum):
    """Tenant status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DELETED = "deleted"


class TenantTier(Enum):
    """Tenant service tier."""

    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class IsolationLevel(Enum):
    """Tenant isolation level."""

    SHARED = "shared"  # Shared resources
    DEDICATED = "dedicated"  # Dedicated resources
    ISOLATED = "isolated"  # Complete isolation


class QuotaType(Enum):
    """Quota type."""

    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"
    DATA_SIZE_MB = "data_size_mb"
    STORAGE_MB = "storage_mb"


@dataclass
class ResourceQuota:
    """Resource quota definition."""

    quota_type: QuotaType
    limit: float
    current_usage: float = 0.0
    reset_interval: Optional[timedelta] = None
    last_reset: datetime = field(default_factory=datetime.now)

    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.current_usage >= self.limit

    def remaining(self) -> float:
        """Get remaining quota."""
        return max(0, self.limit - self.current_usage)

    def use(self, amount: float = 1.0) -> bool:
        """Use quota.

        Args:
            amount: Amount to use

        Returns:
            True if quota was available
        """
        if self.current_usage + amount > self.limit:
            return False
        self.current_usage += amount
        return True

    def check_and_reset(self) -> None:
        """Check and reset quota if interval elapsed."""
        if self.reset_interval:
            elapsed = datetime.now() - self.last_reset
            if elapsed >= self.reset_interval:
                self.current_usage = 0.0
                self.last_reset = datetime.now()


@dataclass
class TenantConfig:
    """Tenant-specific configuration."""

    tenant_id: str
    tier: TenantTier = TenantTier.FREE
    isolation_level: IsolationLevel = IsolationLevel.SHARED
    max_concurrent_requests: int = 10
    priority: int = 0  # Higher is more priority
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    allowed_providers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "tier": self.tier.value,
            "isolation_level": self.isolation_level.value,
            "max_concurrent_requests": self.max_concurrent_requests,
            "priority": self.priority,
            "custom_settings": dict(self.custom_settings),
            "allowed_providers": list(self.allowed_providers),
        }


@dataclass
class Tenant:
    """Tenant entity."""

    tenant_id: str
    name: str
    status: TenantStatus = TenantStatus.ACTIVE
    config: Optional[TenantConfig] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post initialization."""
        if self.config is None:
            self.config = TenantConfig(tenant_id=self.tenant_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status.value,
            "config": self.config.to_dict() if self.config else None,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass
class TenantUsage:
    """Tenant usage statistics."""

    tenant_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    data_processed_mb: float = 0.0
    period_start: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        data_size_mb: float = 0.0,
    ) -> None:
        """Record a request.

        Args:
            success: Whether request succeeded
            latency_ms: Request latency
            data_size_mb: Data size processed
        """
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.data_processed_mb += data_size_mb

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1


class TenantStore(ABC):
    """Abstract tenant store."""

    @abstractmethod
    def get(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        pass

    @abstractmethod
    def save(self, tenant: Tenant) -> None:
        """Save tenant."""
        pass

    @abstractmethod
    def delete(self, tenant_id: str) -> None:
        """Delete tenant."""
        pass

    @abstractmethod
    def list_all(self) -> List[Tenant]:
        """List all tenants."""
        pass


class InMemoryTenantStore(TenantStore):
    """In-memory tenant store."""

    def __init__(self) -> None:
        """Initialize store."""
        self._tenants: Dict[str, Tenant] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        with self._lock:
            return self._tenants.get(tenant_id)

    def save(self, tenant: Tenant) -> None:
        """Save tenant."""
        with self._lock:
            self._tenants[tenant.tenant_id] = tenant

    def delete(self, tenant_id: str) -> None:
        """Delete tenant."""
        with self._lock:
            self._tenants.pop(tenant_id, None)

    def list_all(self) -> List[Tenant]:
        """List all tenants."""
        with self._lock:
            return list(self._tenants.values())


class QuotaManager:
    """Manages tenant quotas."""

    def __init__(self) -> None:
        """Initialize manager."""
        self._quotas: Dict[str, Dict[QuotaType, ResourceQuota]] = {}
        self._lock = threading.Lock()

    def set_quota(
        self,
        tenant_id: str,
        quota: ResourceQuota,
    ) -> None:
        """Set quota for tenant.

        Args:
            tenant_id: Tenant ID
            quota: Resource quota
        """
        with self._lock:
            if tenant_id not in self._quotas:
                self._quotas[tenant_id] = {}
            self._quotas[tenant_id][quota.quota_type] = quota

    def get_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> Optional[ResourceQuota]:
        """Get quota for tenant.

        Args:
            tenant_id: Tenant ID
            quota_type: Quota type

        Returns:
            Resource quota or None
        """
        with self._lock:
            tenant_quotas = self._quotas.get(tenant_id, {})
            return tenant_quotas.get(quota_type)

    def check_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: float = 1.0,
    ) -> bool:
        """Check if quota allows operation.

        Args:
            tenant_id: Tenant ID
            quota_type: Quota type
            amount: Amount to check

        Returns:
            True if quota available
        """
        quota = self.get_quota(tenant_id, quota_type)
        if not quota:
            return True  # No quota set = unlimited

        quota.check_and_reset()
        return quota.current_usage + amount <= quota.limit

    def use_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: float = 1.0,
    ) -> bool:
        """Use quota.

        Args:
            tenant_id: Tenant ID
            quota_type: Quota type
            amount: Amount to use

        Returns:
            True if quota was used
        """
        with self._lock:
            quota = self._quotas.get(tenant_id, {}).get(quota_type)
            if not quota:
                return True

            quota.check_and_reset()
            return quota.use(amount)

    def get_all_quotas(self, tenant_id: str) -> Dict[QuotaType, ResourceQuota]:
        """Get all quotas for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Dictionary of quotas
        """
        with self._lock:
            return dict(self._quotas.get(tenant_id, {}))

    def reset_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
    ) -> None:
        """Reset quota for tenant.

        Args:
            tenant_id: Tenant ID
            quota_type: Quota type
        """
        with self._lock:
            if tenant_id in self._quotas and quota_type in self._quotas[tenant_id]:
                self._quotas[tenant_id][quota_type].current_usage = 0.0
                self._quotas[tenant_id][quota_type].last_reset = datetime.now()


class UsageTracker:
    """Tracks tenant usage."""

    def __init__(self) -> None:
        """Initialize tracker."""
        self._usage: Dict[str, TenantUsage] = {}
        self._lock = threading.Lock()

    def get_usage(self, tenant_id: str) -> TenantUsage:
        """Get usage for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tenant usage
        """
        with self._lock:
            if tenant_id not in self._usage:
                self._usage[tenant_id] = TenantUsage(tenant_id=tenant_id)
            return self._usage[tenant_id]

    def record(
        self,
        tenant_id: str,
        success: bool,
        latency_ms: float,
        data_size_mb: float = 0.0,
    ) -> None:
        """Record request.

        Args:
            tenant_id: Tenant ID
            success: Whether request succeeded
            latency_ms: Request latency
            data_size_mb: Data size processed
        """
        usage = self.get_usage(tenant_id)
        usage.record_request(success, latency_ms, data_size_mb)

    def reset_usage(self, tenant_id: str) -> None:
        """Reset usage for tenant.

        Args:
            tenant_id: Tenant ID
        """
        with self._lock:
            if tenant_id in self._usage:
                self._usage[tenant_id] = TenantUsage(tenant_id=tenant_id)


class TenantContext:
    """Thread-local tenant context."""

    _local = threading.local()

    @classmethod
    def set_tenant(cls, tenant_id: str) -> None:
        """Set current tenant.

        Args:
            tenant_id: Tenant ID
        """
        cls._local.tenant_id = tenant_id

    @classmethod
    def get_tenant(cls) -> Optional[str]:
        """Get current tenant.

        Returns:
            Current tenant ID or None
        """
        return getattr(cls._local, "tenant_id", None)

    @classmethod
    def clear(cls) -> None:
        """Clear tenant context."""
        cls._local.tenant_id = None


class TenantManager:
    """Manages tenants."""

    def __init__(
        self,
        store: Optional[TenantStore] = None,
    ) -> None:
        """Initialize manager.

        Args:
            store: Tenant store
        """
        self._store = store or InMemoryTenantStore()
        self._quotas = QuotaManager()
        self._usage = UsageTracker()
        self._providers: Dict[str, VisionProvider] = {}
        self._default_provider: Optional[VisionProvider] = None

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        config: Optional[TenantConfig] = None,
    ) -> Tenant:
        """Create a new tenant.

        Args:
            tenant_id: Tenant ID
            name: Tenant name
            tier: Service tier
            config: Tenant configuration

        Returns:
            Created tenant
        """
        if config is None:
            config = TenantConfig(tenant_id=tenant_id, tier=tier)

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            config=config,
        )

        self._store.save(tenant)
        self._setup_default_quotas(tenant)

        return tenant

    def _setup_default_quotas(self, tenant: Tenant) -> None:
        """Set up default quotas based on tier."""
        tier_quotas = {
            TenantTier.FREE: {
                QuotaType.REQUESTS_PER_MINUTE: 10,
                QuotaType.REQUESTS_PER_DAY: 100,
                QuotaType.CONCURRENT_REQUESTS: 2,
            },
            TenantTier.BASIC: {
                QuotaType.REQUESTS_PER_MINUTE: 60,
                QuotaType.REQUESTS_PER_DAY: 1000,
                QuotaType.CONCURRENT_REQUESTS: 5,
            },
            TenantTier.PROFESSIONAL: {
                QuotaType.REQUESTS_PER_MINUTE: 300,
                QuotaType.REQUESTS_PER_DAY: 10000,
                QuotaType.CONCURRENT_REQUESTS: 20,
            },
            TenantTier.ENTERPRISE: {
                QuotaType.REQUESTS_PER_MINUTE: 1000,
                QuotaType.REQUESTS_PER_DAY: 100000,
                QuotaType.CONCURRENT_REQUESTS: 100,
            },
        }

        tier = tenant.config.tier if tenant.config else TenantTier.FREE
        quotas = tier_quotas.get(tier, tier_quotas[TenantTier.FREE])

        for quota_type, limit in quotas.items():
            reset_interval = None
            if quota_type == QuotaType.REQUESTS_PER_MINUTE:
                reset_interval = timedelta(minutes=1)
            elif quota_type == QuotaType.REQUESTS_PER_HOUR:
                reset_interval = timedelta(hours=1)
            elif quota_type == QuotaType.REQUESTS_PER_DAY:
                reset_interval = timedelta(days=1)

            self._quotas.set_quota(
                tenant.tenant_id,
                ResourceQuota(
                    quota_type=quota_type,
                    limit=limit,
                    reset_interval=reset_interval,
                ),
            )

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tenant or None
        """
        return self._store.get(tenant_id)

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
    ) -> Optional[Tenant]:
        """Update tenant.

        Args:
            tenant_id: Tenant ID
            name: New name
            status: New status
            tier: New tier

        Returns:
            Updated tenant or None
        """
        tenant = self._store.get(tenant_id)
        if not tenant:
            return None

        if name:
            tenant.name = name
        if status:
            tenant.status = status
        if tier and tenant.config:
            tenant.config.tier = tier
            self._setup_default_quotas(tenant)

        self._store.save(tenant)
        return tenant

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if deleted
        """
        tenant = self._store.get(tenant_id)
        if not tenant:
            return False

        tenant.status = TenantStatus.DELETED
        self._store.save(tenant)
        return True

    def suspend_tenant(self, tenant_id: str) -> bool:
        """Suspend tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if suspended
        """
        return self.update_tenant(tenant_id, status=TenantStatus.SUSPENDED) is not None

    def activate_tenant(self, tenant_id: str) -> bool:
        """Activate tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if activated
        """
        return self.update_tenant(tenant_id, status=TenantStatus.ACTIVE) is not None

    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
    ) -> List[Tenant]:
        """List tenants.

        Args:
            status: Filter by status
            tier: Filter by tier

        Returns:
            List of tenants
        """
        tenants = self._store.list_all()

        if status:
            tenants = [t for t in tenants if t.status == status]
        if tier:
            tenants = [t for t in tenants if t.config and t.config.tier == tier]

        return tenants

    def check_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: float = 1.0,
    ) -> bool:
        """Check tenant quota.

        Args:
            tenant_id: Tenant ID
            quota_type: Quota type
            amount: Amount to check

        Returns:
            True if quota available
        """
        return self._quotas.check_quota(tenant_id, quota_type, amount)

    def use_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: float = 1.0,
    ) -> bool:
        """Use tenant quota.

        Args:
            tenant_id: Tenant ID
            quota_type: Quota type
            amount: Amount to use

        Returns:
            True if quota used
        """
        return self._quotas.use_quota(tenant_id, quota_type, amount)

    def get_usage(self, tenant_id: str) -> TenantUsage:
        """Get tenant usage.

        Args:
            tenant_id: Tenant ID

        Returns:
            Tenant usage
        """
        return self._usage.get_usage(tenant_id)

    def record_usage(
        self,
        tenant_id: str,
        success: bool,
        latency_ms: float,
        data_size_mb: float = 0.0,
    ) -> None:
        """Record tenant usage.

        Args:
            tenant_id: Tenant ID
            success: Whether request succeeded
            latency_ms: Request latency
            data_size_mb: Data size processed
        """
        self._usage.record(tenant_id, success, latency_ms, data_size_mb)

    def register_provider(
        self,
        tenant_id: str,
        provider: VisionProvider,
    ) -> None:
        """Register dedicated provider for tenant.

        Args:
            tenant_id: Tenant ID
            provider: Vision provider
        """
        self._providers[tenant_id] = provider

    def set_default_provider(self, provider: VisionProvider) -> None:
        """Set default provider for shared tenants.

        Args:
            provider: Vision provider
        """
        self._default_provider = provider

    def get_provider(self, tenant_id: str) -> Optional[VisionProvider]:
        """Get provider for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Vision provider or None
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant or tenant.status != TenantStatus.ACTIVE:
            return None

        # Check for dedicated provider
        if tenant_id in self._providers:
            return self._providers[tenant_id]

        # Check isolation level
        if tenant.config and tenant.config.isolation_level == IsolationLevel.DEDICATED:
            return self._providers.get(tenant_id)

        return self._default_provider


class MultiTenantVisionProvider(VisionProvider):
    """Multi-tenant vision provider."""

    def __init__(
        self,
        manager: TenantManager,
        default_provider: Optional[VisionProvider] = None,
    ) -> None:
        """Initialize provider.

        Args:
            manager: Tenant manager
            default_provider: Default provider for shared tenants
        """
        self._manager = manager
        if default_provider:
            self._manager.set_default_provider(default_provider)

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "multi_tenant"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image for current tenant.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        tenant_id = TenantContext.get_tenant()
        if not tenant_id:
            raise RuntimeError("No tenant context set")

        return await self.analyze_for_tenant(tenant_id, image_data, include_description)

    async def analyze_for_tenant(
        self,
        tenant_id: str,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image for specific tenant.

        Args:
            tenant_id: Tenant ID
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        # Check tenant status
        tenant = self._manager.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")

        if tenant.status != TenantStatus.ACTIVE:
            raise RuntimeError(f"Tenant is not active: {tenant.status.value}")

        # Check quotas
        if not self._manager.check_quota(tenant_id, QuotaType.REQUESTS_PER_MINUTE):
            raise RuntimeError("Rate limit exceeded")

        if not self._manager.check_quota(tenant_id, QuotaType.REQUESTS_PER_DAY):
            raise RuntimeError("Daily quota exceeded")

        # Get provider
        provider = self._manager.get_provider(tenant_id)
        if not provider:
            raise RuntimeError("No provider available for tenant")

        # Use quotas
        self._manager.use_quota(tenant_id, QuotaType.REQUESTS_PER_MINUTE)
        self._manager.use_quota(tenant_id, QuotaType.REQUESTS_PER_DAY)

        start_time = time.time()

        try:
            result = await provider.analyze_image(image_data, include_description)
            latency_ms = (time.time() - start_time) * 1000
            data_size_mb = len(image_data) / (1024 * 1024)

            self._manager.record_usage(tenant_id, True, latency_ms, data_size_mb)

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._manager.record_usage(tenant_id, False, latency_ms)
            raise


def create_multi_tenant_provider(
    default_provider: VisionProvider,
) -> MultiTenantVisionProvider:
    """Create multi-tenant provider.

    Args:
        default_provider: Default provider

    Returns:
        Multi-tenant provider
    """
    manager = TenantManager()
    return MultiTenantVisionProvider(manager, default_provider)
