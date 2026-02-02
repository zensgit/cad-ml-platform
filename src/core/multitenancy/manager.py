"""Tenant Manager for Multi-tenancy.

Provides tenant lifecycle management, provisioning, and administration.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.multitenancy.context import TenantContext

logger = logging.getLogger(__name__)


class TenantStatus(str, Enum):
    """Tenant lifecycle status."""

    PENDING = "pending"          # Created but not provisioned
    PROVISIONING = "provisioning"  # Being set up
    ACTIVE = "active"            # Fully operational
    SUSPENDED = "suspended"      # Temporarily disabled
    DEACTIVATED = "deactivated"  # Marked for deletion
    DELETED = "deleted"          # Soft-deleted


class TenantTier(str, Enum):
    """Tenant subscription tier."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuotas:
    """Resource quotas for a tenant."""

    max_storage_bytes: int = 1 * 1024 * 1024 * 1024  # 1 GB
    max_documents: int = 1000
    max_users: int = 5
    max_api_calls_per_hour: int = 1000
    max_models: int = 3
    max_concurrent_jobs: int = 2

    @classmethod
    def for_tier(cls, tier: TenantTier) -> "TenantQuotas":
        """Get quotas for a subscription tier."""
        quotas = {
            TenantTier.FREE: cls(
                max_storage_bytes=1 * 1024 * 1024 * 1024,  # 1 GB
                max_documents=100,
                max_users=2,
                max_api_calls_per_hour=100,
                max_models=1,
                max_concurrent_jobs=1,
            ),
            TenantTier.STARTER: cls(
                max_storage_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
                max_documents=1000,
                max_users=10,
                max_api_calls_per_hour=1000,
                max_models=5,
                max_concurrent_jobs=3,
            ),
            TenantTier.PROFESSIONAL: cls(
                max_storage_bytes=100 * 1024 * 1024 * 1024,  # 100 GB
                max_documents=10000,
                max_users=50,
                max_api_calls_per_hour=10000,
                max_models=20,
                max_concurrent_jobs=10,
            ),
            TenantTier.ENTERPRISE: cls(
                max_storage_bytes=1024 * 1024 * 1024 * 1024,  # 1 TB
                max_documents=100000,
                max_users=500,
                max_api_calls_per_hour=100000,
                max_models=100,
                max_concurrent_jobs=50,
            ),
        }
        return quotas.get(tier, cls())


@dataclass
class Tenant:
    """Tenant entity."""

    tenant_id: str
    name: str
    slug: str  # URL-safe identifier
    status: TenantStatus = TenantStatus.PENDING
    tier: TenantTier = TenantTier.FREE
    quotas: TenantQuotas = field(default_factory=TenantQuotas)

    # Ownership
    owner_user_id: Optional[str] = None
    admin_emails: List[str] = field(default_factory=list)

    # Isolation settings
    schema_name: Optional[str] = None
    database_name: Optional[str] = None

    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)
    features: Set[str] = field(default_factory=set)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    provisioned_at: Optional[datetime] = None
    suspended_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

    # Usage tracking
    current_storage_bytes: int = 0
    current_documents: int = 0
    current_users: int = 0

    # API access
    api_key_hash: Optional[str] = None

    def to_context(self) -> TenantContext:
        """Convert to TenantContext."""
        return TenantContext(
            tenant_id=self.tenant_id,
            tenant_name=self.name,
            schema_name=self.schema_name,
            database_name=self.database_name,
            settings=self.settings,
            max_storage_bytes=self.quotas.max_storage_bytes,
            max_documents=self.quotas.max_documents,
            max_users=self.quotas.max_users,
            max_api_calls_per_hour=self.quotas.max_api_calls_per_hour,
            current_storage_bytes=self.current_storage_bytes,
            current_documents=self.current_documents,
            current_users=self.current_users,
        )

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE

    def can_create_document(self) -> bool:
        """Check if tenant can create more documents."""
        return self.current_documents < self.quotas.max_documents

    def can_add_user(self) -> bool:
        """Check if tenant can add more users."""
        return self.current_users < self.quotas.max_users

    def has_feature(self, feature: str) -> bool:
        """Check if tenant has a feature enabled."""
        return feature in self.features

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "slug": self.slug,
            "status": self.status.value,
            "tier": self.tier.value,
            "owner_user_id": self.owner_user_id,
            "created_at": self.created_at.isoformat(),
            "quotas": {
                "max_storage_bytes": self.quotas.max_storage_bytes,
                "max_documents": self.quotas.max_documents,
                "max_users": self.quotas.max_users,
            },
            "usage": {
                "storage_bytes": self.current_storage_bytes,
                "documents": self.current_documents,
                "users": self.current_users,
            },
            "features": list(self.features),
        }


# Provisioning hook type
ProvisioningHook = Callable[[Tenant], bool]


class TenantManager:
    """Manager for tenant lifecycle and operations."""

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._slug_to_id: Dict[str, str] = {}
        self._provisioning_hooks: List[ProvisioningHook] = []
        self._deprovisioning_hooks: List[ProvisioningHook] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def add_provisioning_hook(self, hook: ProvisioningHook) -> None:
        """Add a hook called during tenant provisioning."""
        self._provisioning_hooks.append(hook)

    def add_deprovisioning_hook(self, hook: ProvisioningHook) -> None:
        """Add a hook called during tenant deprovisioning."""
        self._deprovisioning_hooks.append(hook)

    async def create_tenant(
        self,
        name: str,
        slug: str,
        owner_user_id: Optional[str] = None,
        tier: TenantTier = TenantTier.FREE,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Create a new tenant.

        Args:
            name: Tenant display name
            slug: URL-safe identifier
            owner_user_id: Owner's user ID
            tier: Subscription tier
            settings: Initial settings

        Returns:
            Created Tenant

        Raises:
            ValueError: If slug already exists
        """
        async with self._get_lock():
            # Check slug uniqueness
            if slug in self._slug_to_id:
                raise ValueError(f"Tenant slug '{slug}' already exists")

            tenant_id = str(uuid.uuid4())

            tenant = Tenant(
                tenant_id=tenant_id,
                name=name,
                slug=slug,
                owner_user_id=owner_user_id,
                tier=tier,
                quotas=TenantQuotas.for_tier(tier),
                settings=settings or {},
                schema_name=f"tenant_{slug}",
            )

            self._tenants[tenant_id] = tenant
            self._slug_to_id[slug] = tenant_id

            logger.info(f"Created tenant: {name} ({tenant_id})")
            return tenant

    async def provision_tenant(self, tenant_id: str) -> bool:
        """Provision tenant resources.

        This runs all provisioning hooks to set up:
        - Database schema/tables
        - Storage buckets
        - Default settings
        - etc.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if provisioning succeeded
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            logger.error(f"Tenant not found: {tenant_id}")
            return False

        if tenant.status not in (TenantStatus.PENDING, TenantStatus.DEACTIVATED):
            logger.warning(f"Tenant {tenant_id} is not in provisionable state")
            return False

        tenant.status = TenantStatus.PROVISIONING
        tenant.updated_at = datetime.utcnow()

        try:
            # Run provisioning hooks
            for hook in self._provisioning_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        success = await hook(tenant)
                    else:
                        success = hook(tenant)

                    if not success:
                        raise Exception("Provisioning hook returned False")

                except Exception as e:
                    logger.error(f"Provisioning hook failed: {e}")
                    tenant.status = TenantStatus.PENDING
                    return False

            tenant.status = TenantStatus.ACTIVE
            tenant.provisioned_at = datetime.utcnow()
            tenant.updated_at = datetime.utcnow()

            logger.info(f"Provisioned tenant: {tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to provision tenant {tenant_id}: {e}")
            tenant.status = TenantStatus.PENDING
            return False

    async def suspend_tenant(self, tenant_id: str, reason: str = "") -> bool:
        """Suspend a tenant.

        Args:
            tenant_id: Tenant ID
            reason: Suspension reason

        Returns:
            True if suspended successfully
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        tenant.status = TenantStatus.SUSPENDED
        tenant.suspended_at = datetime.utcnow()
        tenant.updated_at = datetime.utcnow()
        tenant.settings["suspension_reason"] = reason

        logger.info(f"Suspended tenant: {tenant_id}, reason: {reason}")
        return True

    async def reactivate_tenant(self, tenant_id: str) -> bool:
        """Reactivate a suspended tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            True if reactivated successfully
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant or tenant.status != TenantStatus.SUSPENDED:
            return False

        tenant.status = TenantStatus.ACTIVE
        tenant.suspended_at = None
        tenant.updated_at = datetime.utcnow()
        tenant.settings.pop("suspension_reason", None)

        logger.info(f"Reactivated tenant: {tenant_id}")
        return True

    async def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate tenant (mark for deletion).

        Args:
            tenant_id: Tenant ID

        Returns:
            True if deactivated
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False

        # Run deprovisioning hooks
        for hook in self._deprovisioning_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(tenant)
                else:
                    hook(tenant)
            except Exception as e:
                logger.error(f"Deprovisioning hook error: {e}")

        tenant.status = TenantStatus.DEACTIVATED
        tenant.updated_at = datetime.utcnow()

        logger.info(f"Deactivated tenant: {tenant_id}")
        return True

    async def delete_tenant(self, tenant_id: str, hard_delete: bool = False) -> bool:
        """Delete a tenant.

        Args:
            tenant_id: Tenant ID
            hard_delete: If True, permanently remove; else soft-delete

        Returns:
            True if deleted
        """
        async with self._get_lock():
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                return False

            if hard_delete:
                # Permanent deletion
                self._slug_to_id.pop(tenant.slug, None)
                del self._tenants[tenant_id]
                logger.info(f"Hard-deleted tenant: {tenant_id}")
            else:
                # Soft delete
                tenant.status = TenantStatus.DELETED
                tenant.deleted_at = datetime.utcnow()
                tenant.updated_at = datetime.utcnow()
                logger.info(f"Soft-deleted tenant: {tenant_id}")

            return True

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)

    def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        tenant_id = self._slug_to_id.get(slug)
        if tenant_id:
            return self._tenants.get(tenant_id)
        return None

    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
    ) -> List[Tenant]:
        """List tenants with optional filtering.

        Args:
            status: Filter by status
            tier: Filter by tier

        Returns:
            List of matching tenants
        """
        tenants = list(self._tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]

        if tier:
            tenants = [t for t in tenants if t.tier == tier]

        return tenants

    async def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        features: Optional[Set[str]] = None,
    ) -> Optional[Tenant]:
        """Update tenant properties.

        Args:
            tenant_id: Tenant ID
            name: New name
            settings: Settings to update
            features: Features to set

        Returns:
            Updated tenant or None
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        if name:
            tenant.name = name

        if settings:
            tenant.settings.update(settings)

        if features is not None:
            tenant.features = features

        tenant.updated_at = datetime.utcnow()
        return tenant

    async def update_tier(
        self,
        tenant_id: str,
        new_tier: TenantTier,
    ) -> Optional[Tenant]:
        """Update tenant subscription tier.

        Args:
            tenant_id: Tenant ID
            new_tier: New tier

        Returns:
            Updated tenant
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        tenant.tier = new_tier
        tenant.quotas = TenantQuotas.for_tier(new_tier)
        tenant.updated_at = datetime.utcnow()

        logger.info(f"Updated tenant {tenant_id} tier to {new_tier.value}")
        return tenant

    async def generate_api_key(self, tenant_id: str) -> Optional[str]:
        """Generate a new API key for tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            New API key (only shown once)
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        # Generate secure random key
        api_key = f"tnt_{secrets.token_urlsafe(32)}"

        # Store hash only
        tenant.api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        tenant.updated_at = datetime.utcnow()

        logger.info(f"Generated new API key for tenant {tenant_id}")
        return api_key

    def validate_api_key(self, tenant_id: str, api_key: str) -> bool:
        """Validate an API key.

        Args:
            tenant_id: Tenant ID
            api_key: API key to validate

        Returns:
            True if valid
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant or not tenant.api_key_hash:
            return False

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return key_hash == tenant.api_key_hash

    async def update_usage(
        self,
        tenant_id: str,
        storage_delta: int = 0,
        documents_delta: int = 0,
        users_delta: int = 0,
    ) -> Optional[Tenant]:
        """Update tenant resource usage.

        Args:
            tenant_id: Tenant ID
            storage_delta: Change in storage bytes
            documents_delta: Change in document count
            users_delta: Change in user count

        Returns:
            Updated tenant
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None

        tenant.current_storage_bytes = max(0, tenant.current_storage_bytes + storage_delta)
        tenant.current_documents = max(0, tenant.current_documents + documents_delta)
        tenant.current_users = max(0, tenant.current_users + users_delta)
        tenant.updated_at = datetime.utcnow()

        return tenant


# Global tenant manager
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get global tenant manager."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager
