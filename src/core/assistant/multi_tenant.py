"""
Multi-Tenant Support Module for CAD Assistant.

Provides tenant isolation, resource management, and tenant-specific configuration.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class TenantStatus(Enum):
    """Tenant status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DELETED = "deleted"


class TenantTier(Enum):
    """Tenant subscription tiers."""

    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuota:
    """Resource quotas for a tenant."""

    max_conversations: int = 100
    max_messages_per_day: int = 1000
    max_knowledge_items: int = 500
    max_api_calls_per_minute: int = 60
    max_storage_mb: int = 100
    allowed_models: List[str] = field(default_factory=lambda: ["offline"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_conversations": self.max_conversations,
            "max_messages_per_day": self.max_messages_per_day,
            "max_knowledge_items": self.max_knowledge_items,
            "max_api_calls_per_minute": self.max_api_calls_per_minute,
            "max_storage_mb": self.max_storage_mb,
            "allowed_models": self.allowed_models,
        }

    @classmethod
    def for_tier(cls, tier: TenantTier) -> "TenantQuota":
        """Get default quotas for a tier."""
        quotas = {
            TenantTier.FREE: cls(
                max_conversations=10,
                max_messages_per_day=100,
                max_knowledge_items=50,
                max_api_calls_per_minute=10,
                max_storage_mb=10,
                allowed_models=["offline"],
            ),
            TenantTier.BASIC: cls(
                max_conversations=100,
                max_messages_per_day=1000,
                max_knowledge_items=500,
                max_api_calls_per_minute=30,
                max_storage_mb=100,
                allowed_models=["offline", "qwen"],
            ),
            TenantTier.PROFESSIONAL: cls(
                max_conversations=1000,
                max_messages_per_day=10000,
                max_knowledge_items=5000,
                max_api_calls_per_minute=100,
                max_storage_mb=1000,
                allowed_models=["offline", "qwen", "openai"],
            ),
            TenantTier.ENTERPRISE: cls(
                max_conversations=-1,  # Unlimited
                max_messages_per_day=-1,
                max_knowledge_items=-1,
                max_api_calls_per_minute=500,
                max_storage_mb=10000,
                allowed_models=["offline", "qwen", "openai", "claude"],
            ),
        }
        return quotas.get(tier, cls())


@dataclass
class TenantUsage:
    """Current resource usage for a tenant."""

    conversations: int = 0
    messages_today: int = 0
    knowledge_items: int = 0
    api_calls_this_minute: int = 0
    storage_used_mb: float = 0
    last_reset: float = field(default_factory=time.time)

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.messages_today = 0
        self.last_reset = time.time()

    def reset_minute(self) -> None:
        """Reset per-minute counters."""
        self.api_calls_this_minute = 0


@dataclass
class Tenant:
    """A tenant (organization/company)."""

    id: str
    name: str
    status: TenantStatus = TenantStatus.ACTIVE
    tier: TenantTier = TenantTier.FREE
    quota: TenantQuota = field(default_factory=TenantQuota)
    usage: TenantUsage = field(default_factory=TenantUsage)
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "tier": self.tier.value,
            "quota": self.quota.to_dict(),
            "usage": {
                "conversations": self.usage.conversations,
                "messages_today": self.usage.messages_today,
                "knowledge_items": self.usage.knowledge_items,
                "storage_used_mb": self.usage.storage_used_mb,
            },
            "settings": self.settings,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def check_quota(self, resource: str, amount: int = 1) -> bool:
        """
        Check if operation is within quota.

        Args:
            resource: Resource type
            amount: Amount to use

        Returns:
            True if within quota
        """
        if self.status != TenantStatus.ACTIVE:
            return False

        checks = {
            "conversations": (
                self.quota.max_conversations < 0 or
                self.usage.conversations + amount <= self.quota.max_conversations
            ),
            "messages": (
                self.quota.max_messages_per_day < 0 or
                self.usage.messages_today + amount <= self.quota.max_messages_per_day
            ),
            "knowledge": (
                self.quota.max_knowledge_items < 0 or
                self.usage.knowledge_items + amount <= self.quota.max_knowledge_items
            ),
            "api_calls": (
                self.quota.max_api_calls_per_minute < 0 or
                self.usage.api_calls_this_minute + amount <= self.quota.max_api_calls_per_minute
            ),
        }

        return checks.get(resource, True)

    def use_quota(self, resource: str, amount: int = 1) -> bool:
        """
        Use quota for a resource.

        Args:
            resource: Resource type
            amount: Amount to use

        Returns:
            True if successful
        """
        if not self.check_quota(resource, amount):
            return False

        if resource == "conversations":
            self.usage.conversations += amount
        elif resource == "messages":
            self.usage.messages_today += amount
        elif resource == "knowledge":
            self.usage.knowledge_items += amount
        elif resource == "api_calls":
            self.usage.api_calls_this_minute += amount

        return True


class TenantManager:
    """
    Manages tenants and their resources.

    Provides CRUD operations and quota management.

    Example:
        >>> manager = TenantManager()
        >>> tenant_id = manager.create_tenant("ACME Corp", TenantTier.PROFESSIONAL)
        >>> tenant = manager.get_tenant(tenant_id)
        >>> if tenant.check_quota("messages"):
        ...     tenant.use_quota("messages")
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize tenant manager.

        Args:
            storage_path: Path for tenant data persistence
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._tenants: Dict[str, Tenant] = {}
        self._user_tenants: Dict[str, str] = {}  # user_id -> tenant_id

        if self.storage_path and self.storage_path.exists():
            self._load()

    def create_tenant(
        self,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        settings: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new tenant.

        Args:
            name: Tenant name
            tier: Subscription tier
            settings: Initial settings

        Returns:
            Tenant ID
        """
        tenant_id = str(uuid.uuid4())[:8]

        tenant = Tenant(
            id=tenant_id,
            name=name,
            tier=tier,
            quota=TenantQuota.for_tier(tier),
            settings=settings or {},
        )

        self._tenants[tenant_id] = tenant
        return tenant_id

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)

    def get_tenant_for_user(self, user_id: str) -> Optional[Tenant]:
        """Get tenant for a user."""
        tenant_id = self._user_tenants.get(user_id)
        if tenant_id:
            return self.get_tenant(tenant_id)
        return None

    def assign_user_to_tenant(self, user_id: str, tenant_id: str) -> bool:
        """Assign a user to a tenant."""
        if tenant_id not in self._tenants:
            return False
        self._user_tenants[user_id] = tenant_id
        return True

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        tier: Optional[TenantTier] = None,
        status: Optional[TenantStatus] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update tenant properties."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        if name:
            tenant.name = name
        if tier:
            tenant.tier = tier
            tenant.quota = TenantQuota.for_tier(tier)
        if status:
            tenant.status = status
        if settings:
            tenant.settings.update(settings)

        tenant.updated_at = time.time()
        return True

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant (soft delete)."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.status = TenantStatus.DELETED
        tenant.updated_at = time.time()
        return True

    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
    ) -> List[Dict[str, Any]]:
        """List tenants with optional filters."""
        tenants = []
        for tenant in self._tenants.values():
            if status and tenant.status != status:
                continue
            if tier and tenant.tier != tier:
                continue
            tenants.append(tenant.to_dict())
        return tenants

    def reset_usage_counters(self) -> None:
        """Reset daily usage counters for all tenants."""
        for tenant in self._tenants.values():
            tenant.usage.reset_daily()

    def save(self) -> bool:
        """Save tenant data to disk."""
        if not self.storage_path:
            return False

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "tenants": {tid: t.to_dict() for tid, t in self._tenants.items()},
                "user_tenants": self._user_tenants,
                "saved_at": time.time(),
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except IOError:
            return False

    def _load(self) -> bool:
        """Load tenant data from disk."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for tid, tdata in data.get("tenants", {}).items():
                tenant = Tenant(
                    id=tdata["id"],
                    name=tdata["name"],
                    status=TenantStatus(tdata["status"]),
                    tier=TenantTier(tdata["tier"]),
                    settings=tdata.get("settings", {}),
                    metadata=tdata.get("metadata", {}),
                    created_at=tdata.get("created_at", time.time()),
                    updated_at=tdata.get("updated_at", time.time()),
                )
                tenant.quota = TenantQuota.for_tier(tenant.tier)
                self._tenants[tid] = tenant

            self._user_tenants = data.get("user_tenants", {})
            return True
        except (IOError, json.JSONDecodeError):
            return False


class TenantContext:
    """
    Context manager for tenant-scoped operations.

    Ensures proper tenant isolation during request processing.

    Example:
        >>> with TenantContext(tenant) as ctx:
        ...     result = assistant.ask(query)
    """

    _current_tenant: Optional[Tenant] = None

    def __init__(self, tenant: Tenant):
        """
        Initialize tenant context.

        Args:
            tenant: Tenant for this context
        """
        self.tenant = tenant
        self._previous_tenant: Optional[Tenant] = None

    def __enter__(self) -> "TenantContext":
        """Enter tenant context."""
        self._previous_tenant = TenantContext._current_tenant
        TenantContext._current_tenant = self.tenant
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit tenant context."""
        TenantContext._current_tenant = self._previous_tenant

    @classmethod
    def get_current(cls) -> Optional[Tenant]:
        """Get current tenant from context."""
        return cls._current_tenant

    def check_quota(self, resource: str, amount: int = 1) -> bool:
        """Check quota in current context."""
        return self.tenant.check_quota(resource, amount)

    def use_quota(self, resource: str, amount: int = 1) -> bool:
        """Use quota in current context."""
        return self.tenant.use_quota(resource, amount)
