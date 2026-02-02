"""API Key Management.

Provides secure API key generation, validation, and scope management.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class APIKeyScope(str, Enum):
    """API key permission scopes."""

    # Read operations
    READ = "read"
    READ_DOCUMENTS = "read:documents"
    READ_MODELS = "read:models"
    READ_ANALYTICS = "read:analytics"

    # Write operations
    WRITE = "write"
    WRITE_DOCUMENTS = "write:documents"
    WRITE_MODELS = "write:models"

    # Special operations
    PREDICT = "predict"
    TRAIN = "train"
    DEPLOY = "deploy"

    # Admin
    ADMIN = "admin"
    ADMIN_KEYS = "admin:keys"
    ADMIN_USERS = "admin:users"


# Scope hierarchy - higher scopes include lower ones
SCOPE_HIERARCHY: Dict[APIKeyScope, Set[APIKeyScope]] = {
    APIKeyScope.READ: {
        APIKeyScope.READ_DOCUMENTS,
        APIKeyScope.READ_MODELS,
        APIKeyScope.READ_ANALYTICS,
    },
    APIKeyScope.WRITE: {
        APIKeyScope.WRITE_DOCUMENTS,
        APIKeyScope.WRITE_MODELS,
    },
    APIKeyScope.ADMIN: {
        APIKeyScope.READ,
        APIKeyScope.WRITE,
        APIKeyScope.PREDICT,
        APIKeyScope.TRAIN,
        APIKeyScope.DEPLOY,
        APIKeyScope.ADMIN_KEYS,
        APIKeyScope.ADMIN_USERS,
    },
}


@dataclass
class APIKey:
    """API key data model."""

    key_id: str
    key_hash: str  # Hashed key value
    name: str
    owner_id: str
    scopes: Set[APIKeyScope] = field(default_factory=set)
    tenant_id: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    use_count: int = 0

    # Restrictions
    allowed_ips: Set[str] = field(default_factory=set)
    allowed_origins: Set[str] = field(default_factory=set)
    rate_limit_override: Optional[int] = None

    # Status
    is_active: bool = True
    revoked_at: Optional[datetime] = None
    revoked_reason: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if key is valid for use."""
        return self.is_active and not self.is_expired

    def has_scope(self, scope: APIKeyScope) -> bool:
        """Check if key has a specific scope."""
        if scope in self.scopes:
            return True

        # Check hierarchy
        for parent_scope, child_scopes in SCOPE_HIERARCHY.items():
            if parent_scope in self.scopes and scope in child_scopes:
                return True

        return False

    def has_any_scope(self, scopes: Set[APIKeyScope]) -> bool:
        """Check if key has any of the specified scopes."""
        return any(self.has_scope(s) for s in scopes)

    def get_effective_scopes(self) -> Set[APIKeyScope]:
        """Get all effective scopes including inherited."""
        effective = set(self.scopes)

        for scope in list(self.scopes):
            if scope in SCOPE_HIERARCHY:
                effective.update(SCOPE_HIERARCHY[scope])

        return effective

    def is_allowed_ip(self, ip: str) -> bool:
        """Check if IP is allowed."""
        if not self.allowed_ips:
            return True  # No restriction
        return ip in self.allowed_ips

    def is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if not self.allowed_origins:
            return True  # No restriction
        return origin in self.allowed_origins


@dataclass
class APIKeyCreateResult:
    """Result of API key creation."""

    key: APIKey
    raw_key: str  # Only returned once during creation


class APIKeyManager:
    """Manages API key lifecycle."""

    KEY_PREFIX = "cad_"
    KEY_LENGTH = 32

    def __init__(self, secret_key: Optional[str] = None):
        """Initialize manager.

        Args:
            secret_key: Secret for HMAC hashing (defaults to env var or random)
        """
        self._secret_key = (
            secret_key
            or os.getenv("API_KEY_SECRET")
            or secrets.token_hex(32)
        )
        self._keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self._hash_to_id: Dict[str, str] = {}  # key_hash -> key_id
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _generate_key(self) -> str:
        """Generate a new API key."""
        random_part = secrets.token_urlsafe(self.KEY_LENGTH)
        return f"{self.KEY_PREFIX}{random_part}"

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for storage."""
        return hmac.new(
            self._secret_key.encode(),
            raw_key.encode(),
            hashlib.sha256
        ).hexdigest()

    async def create_key(
        self,
        name: str,
        owner_id: str,
        scopes: Optional[Set[APIKeyScope]] = None,
        tenant_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        allowed_ips: Optional[Set[str]] = None,
        allowed_origins: Optional[Set[str]] = None,
        rate_limit_override: Optional[int] = None,
    ) -> APIKeyCreateResult:
        """Create a new API key.

        Args:
            name: Human-readable name
            owner_id: Owner identifier
            scopes: Permission scopes
            tenant_id: Tenant identifier
            expires_in_days: Days until expiration
            allowed_ips: Allowed IP addresses
            allowed_origins: Allowed CORS origins
            rate_limit_override: Custom rate limit

        Returns:
            APIKeyCreateResult with key and raw value
        """
        async with self._get_lock():
            # Generate key
            raw_key = self._generate_key()
            key_hash = self._hash_key(raw_key)
            key_id = f"key_{secrets.token_hex(8)}"

            # Calculate expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

            # Create key object
            api_key = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                name=name,
                owner_id=owner_id,
                scopes=scopes or {APIKeyScope.READ},
                tenant_id=tenant_id,
                expires_at=expires_at,
                allowed_ips=allowed_ips or set(),
                allowed_origins=allowed_origins or set(),
                rate_limit_override=rate_limit_override,
            )

            # Store
            self._keys[key_id] = api_key
            self._hash_to_id[key_hash] = key_id

            logger.info(f"Created API key {key_id} for owner {owner_id}")

            return APIKeyCreateResult(key=api_key, raw_key=raw_key)

    async def validate_key(
        self,
        raw_key: str,
        required_scopes: Optional[Set[APIKeyScope]] = None,
        ip_address: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> Optional[APIKey]:
        """Validate an API key.

        Args:
            raw_key: The raw API key to validate
            required_scopes: Scopes required for the operation
            ip_address: Client IP address
            origin: Request origin

        Returns:
            APIKey if valid, None otherwise
        """
        async with self._get_lock():
            # Hash the provided key
            key_hash = self._hash_key(raw_key)

            # Look up by hash
            key_id = self._hash_to_id.get(key_hash)
            if not key_id:
                logger.warning(f"Invalid API key attempted")
                return None

            api_key = self._keys.get(key_id)
            if not api_key:
                return None

            # Check validity
            if not api_key.is_valid:
                logger.warning(f"API key {key_id} is not valid (expired or revoked)")
                return None

            # Check IP restriction
            if ip_address and not api_key.is_allowed_ip(ip_address):
                logger.warning(f"API key {key_id} used from disallowed IP {ip_address}")
                return None

            # Check origin restriction
            if origin and not api_key.is_allowed_origin(origin):
                logger.warning(f"API key {key_id} used from disallowed origin {origin}")
                return None

            # Check scopes
            if required_scopes:
                if not api_key.has_any_scope(required_scopes):
                    logger.warning(f"API key {key_id} lacks required scopes")
                    return None

            # Update usage stats
            api_key.last_used_at = datetime.utcnow()
            api_key.use_count += 1

            return api_key

    async def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self._keys.get(key_id)

    async def list_keys(
        self,
        owner_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        include_revoked: bool = False,
    ) -> List[APIKey]:
        """List API keys with filters."""
        keys = list(self._keys.values())

        if owner_id:
            keys = [k for k in keys if k.owner_id == owner_id]

        if tenant_id:
            keys = [k for k in keys if k.tenant_id == tenant_id]

        if not include_revoked:
            keys = [k for k in keys if k.is_active]

        return keys

    async def revoke_key(self, key_id: str, reason: Optional[str] = None) -> bool:
        """Revoke an API key.

        Args:
            key_id: Key to revoke
            reason: Optional revocation reason

        Returns:
            True if revoked successfully
        """
        async with self._get_lock():
            api_key = self._keys.get(key_id)
            if not api_key:
                return False

            api_key.is_active = False
            api_key.revoked_at = datetime.utcnow()
            api_key.revoked_reason = reason

            logger.info(f"Revoked API key {key_id}: {reason}")
            return True

    async def rotate_key(self, key_id: str) -> Optional[APIKeyCreateResult]:
        """Rotate an API key (create new, revoke old).

        Args:
            key_id: Key to rotate

        Returns:
            New API key result if successful
        """
        async with self._get_lock():
            old_key = self._keys.get(key_id)
            if not old_key:
                return None

            # Create new key with same settings
            result = await self.create_key(
                name=f"{old_key.name} (rotated)",
                owner_id=old_key.owner_id,
                scopes=old_key.scopes,
                tenant_id=old_key.tenant_id,
                allowed_ips=old_key.allowed_ips,
                allowed_origins=old_key.allowed_origins,
                rate_limit_override=old_key.rate_limit_override,
            )

            # Revoke old key
            await self.revoke_key(key_id, "Rotated")

            logger.info(f"Rotated API key {key_id} -> {result.key.key_id}")
            return result

    async def update_scopes(self, key_id: str, scopes: Set[APIKeyScope]) -> bool:
        """Update API key scopes.

        Args:
            key_id: Key to update
            scopes: New scopes

        Returns:
            True if updated successfully
        """
        async with self._get_lock():
            api_key = self._keys.get(key_id)
            if not api_key:
                return False

            api_key.scopes = scopes
            logger.info(f"Updated scopes for API key {key_id}")
            return True

    async def delete_key(self, key_id: str) -> bool:
        """Permanently delete an API key.

        Args:
            key_id: Key to delete

        Returns:
            True if deleted successfully
        """
        async with self._get_lock():
            api_key = self._keys.pop(key_id, None)
            if api_key:
                self._hash_to_id.pop(api_key.key_hash, None)
                logger.info(f"Deleted API key {key_id}")
                return True
            return False


# Global API key manager
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def validate_api_key(
    required_scopes: Optional[Set[APIKeyScope]] = None,
    key_header: str = "X-API-Key",
) -> Callable[[F], F]:
    """Decorator for API key validation.

    Args:
        required_scopes: Required scopes for the endpoint
        key_header: Header containing the API key

    Example:
        @validate_api_key(required_scopes={APIKeyScope.READ})
        async def get_data(request: Request):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract request
            request = kwargs.get("request") or (args[0] if args else None)
            if not request:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="Request not found")

            # Get API key from header
            api_key_value = request.headers.get(key_header)
            if not api_key_value:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=401,
                    detail="API key required",
                    headers={"WWW-Authenticate": "ApiKey"},
                )

            # Get client info
            ip_address = request.client.host if request.client else None
            origin = request.headers.get("origin")

            # Validate key
            manager = get_api_key_manager()
            api_key = await manager.validate_key(
                api_key_value,
                required_scopes=required_scopes,
                ip_address=ip_address,
                origin=origin,
            )

            if not api_key:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired API key",
                )

            # Attach key to request state
            request.state.api_key = api_key

            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # For sync functions
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            finally:
                loop.close()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class APIKeyMiddleware:
    """FastAPI middleware for API key authentication."""

    def __init__(
        self,
        app: Any,
        key_header: str = "X-API-Key",
        exclude_paths: Optional[Set[str]] = None,
    ):
        self.app = app
        self.key_header = key_header.lower().encode()
        self.exclude_paths = exclude_paths or {"/health", "/metrics", "/docs", "/openapi.json"}
        self.manager = get_api_key_manager()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check if path is excluded
        path = scope.get("path", "")
        if path in self.exclude_paths:
            await self.app(scope, receive, send)
            return

        # Get API key from headers
        headers = dict(scope.get("headers", []))
        api_key_value = headers.get(self.key_header, b"").decode()

        if not api_key_value:
            await self._send_error(send, 401, "API key required")
            return

        # Validate key
        client = scope.get("client")
        ip_address = client[0] if client else None

        api_key = await self.manager.validate_key(
            api_key_value,
            ip_address=ip_address,
        )

        if not api_key:
            await self._send_error(send, 401, "Invalid API key")
            return

        # Add key info to scope
        scope["state"] = scope.get("state", {})
        scope["state"]["api_key"] = api_key

        await self.app(scope, receive, send)

    async def _send_error(self, send: Any, status: int, message: str) -> None:
        """Send error response."""
        import json
        body = json.dumps({"error": message}).encode()

        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b"ApiKey"),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })
