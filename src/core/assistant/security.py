"""
Security and Authentication Module for CAD Assistant API.

Provides authentication, authorization, and security utilities.
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from functools import wraps


class Permission(Enum):
    """API permissions."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    EVALUATE = "evaluate"
    MANAGE_KNOWLEDGE = "manage_knowledge"


class AuthError(Exception):
    """Authentication error."""

    def __init__(self, message: str, code: str = "AUTH_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


class UnauthorizedError(AuthError):
    """Unauthorized access error."""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(message, code="UNAUTHORIZED")


class ForbiddenError(AuthError):
    """Forbidden access error."""

    def __init__(self, message: str = "Forbidden"):
        super().__init__(message, code="FORBIDDEN")


class InvalidTokenError(AuthError):
    """Invalid token error."""

    def __init__(self, message: str = "Invalid token"):
        super().__init__(message, code="INVALID_TOKEN")


class TokenExpiredError(AuthError):
    """Token expired error."""

    def __init__(self, message: str = "Token expired"):
        super().__init__(message, code="TOKEN_EXPIRED")


@dataclass
class APIKey:
    """API key configuration."""

    key_id: str
    key_hash: str  # Hashed key
    name: str
    permissions: Set[Permission]
    rate_limit: int = 60  # requests per minute
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    is_active: bool = True

    def has_permission(self, permission: Permission) -> bool:
        """Check if key has a permission."""
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions

    def is_expired(self) -> bool:
        """Check if key has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "permissions": [p.value for p in self.permissions],
            "rate_limit": self.rate_limit,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used": self.last_used,
            "is_active": self.is_active,
        }


class APIKeyManager:
    """
    Manages API keys for authentication.

    Example:
        >>> manager = APIKeyManager()
        >>> key_id, secret = manager.create_key("my-app", {Permission.READ})
        >>> api_key = manager.validate_key(secret)
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize API key manager.

        Args:
            secret_key: Secret for key hashing (generates if not provided)
        """
        self._secret_key = secret_key or secrets.token_hex(32)
        self._keys: Dict[str, APIKey] = {}

    def create_key(
        self,
        name: str,
        permissions: Set[Permission],
        rate_limit: int = 60,
        expires_in_days: Optional[int] = None,
    ) -> tuple:
        """
        Create a new API key.

        Args:
            name: Key name/description
            permissions: Set of permissions
            rate_limit: Rate limit per minute
            expires_in_days: Days until expiration (None for no expiration)

        Returns:
            Tuple of (key_id, secret_key)
        """
        key_id = secrets.token_hex(8)
        secret = f"cad_{secrets.token_hex(32)}"

        key_hash = self._hash_key(secret)
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 86400)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            expires_at=expires_at,
        )

        self._keys[key_id] = api_key
        return key_id, secret

    def validate_key(self, secret: str) -> Optional[APIKey]:
        """
        Validate an API key and return its configuration.

        Args:
            secret: The API key secret

        Returns:
            APIKey if valid, None otherwise

        Raises:
            InvalidTokenError: If key format is invalid
            TokenExpiredError: If key has expired
            UnauthorizedError: If key is inactive
        """
        if not secret.startswith("cad_"):
            raise InvalidTokenError("Invalid key format")

        key_hash = self._hash_key(secret)

        # Find matching key
        for api_key in self._keys.values():
            if hmac.compare_digest(api_key.key_hash, key_hash):
                if not api_key.is_active:
                    raise UnauthorizedError("API key is inactive")
                if api_key.is_expired():
                    raise TokenExpiredError()

                api_key.last_used = time.time()
                return api_key

        raise InvalidTokenError("Invalid API key")

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            return True
        return False

    def delete_key(self, key_id: str) -> bool:
        """Delete an API key."""
        if key_id in self._keys:
            del self._keys[key_id]
            return True
        return False

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without secrets)."""
        return [key.to_dict() for key in self._keys.values()]

    def _hash_key(self, secret: str) -> str:
        """Hash an API key."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            secret.encode(),
            self._secret_key.encode(),
            100000,
        ).hex()


class SecurityAuditor:
    """
    Records security-related events for auditing.

    Example:
        >>> auditor = SecurityAuditor()
        >>> auditor.log_authentication("user-123", success=True)
        >>> auditor.log_access("user-123", "knowledge:write", granted=False)
    """

    def __init__(self, max_events: int = 10000):
        """
        Initialize security auditor.

        Args:
            max_events: Maximum events to keep in memory
        """
        self._events: List[Dict[str, Any]] = []
        self._max_events = max_events

    def log_authentication(
        self,
        identifier: str,
        success: bool,
        method: str = "api_key",
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log authentication attempt."""
        self._add_event({
            "type": "authentication",
            "identifier": identifier,
            "success": success,
            "method": method,
            "ip_address": ip_address,
            "details": details or {},
            "timestamp": time.time(),
        })

    def log_access(
        self,
        identifier: str,
        resource: str,
        granted: bool,
        permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log access attempt."""
        self._add_event({
            "type": "access",
            "identifier": identifier,
            "resource": resource,
            "granted": granted,
            "permission": permission,
            "details": details or {},
            "timestamp": time.time(),
        })

    def log_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        current_count: int,
        limit: int,
    ) -> None:
        """Log rate limit event."""
        self._add_event({
            "type": "rate_limit",
            "identifier": identifier,
            "endpoint": endpoint,
            "current_count": current_count,
            "limit": limit,
            "timestamp": time.time(),
        })

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log generic security event."""
        self._add_event({
            "type": "security",
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "details": details or {},
            "timestamp": time.time(),
        })

    def get_events(
        self,
        event_type: Optional[str] = None,
        identifier: Optional[str] = None,
        start_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get security events with optional filters."""
        events = self._events

        if event_type:
            events = [e for e in events if e["type"] == event_type]

        if identifier:
            events = [e for e in events if e.get("identifier") == identifier]

        if start_time:
            events = [e for e in events if e["timestamp"] >= start_time]

        return events[-limit:]

    def get_failed_authentications(
        self,
        window_seconds: int = 3600,
        threshold: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get identifiers with multiple failed authentications."""
        cutoff = time.time() - window_seconds

        failures_by_id: Dict[str, int] = {}
        for event in self._events:
            if (
                event["type"] == "authentication"
                and not event["success"]
                and event["timestamp"] >= cutoff
            ):
                id_ = event["identifier"]
                failures_by_id[id_] = failures_by_id.get(id_, 0) + 1

        return [
            {"identifier": id_, "failures": count}
            for id_, count in failures_by_id.items()
            if count >= threshold
        ]

    def _add_event(self, event: Dict[str, Any]) -> None:
        """Add event and trim if needed."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]


def require_permission(permission: Permission):
    """
    Decorator to require a permission for an endpoint.

    Example:
        >>> @require_permission(Permission.WRITE)
        ... def create_item(request, api_key):
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, api_key: APIKey = None, **kwargs):
            if api_key is None:
                raise UnauthorizedError("API key required")

            if not api_key.has_permission(permission):
                raise ForbiddenError(f"Permission denied: {permission.value}")

            return func(*args, api_key=api_key, **kwargs)

        return wrapper

    return decorator


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    # Truncate
    if len(text) > max_length:
        text = text[:max_length]

    # Remove null bytes
    text = text.replace("\x00", "")

    # Basic XSS prevention (for display purposes)
    # Note: Use proper HTML escaping for web display
    text = text.replace("<script", "&lt;script")
    text = text.replace("</script", "&lt;/script")

    return text


def validate_conversation_id(conversation_id: str) -> bool:
    """Validate conversation ID format."""
    if not conversation_id:
        return False
    if len(conversation_id) > 100:
        return False
    # Allow alphanumeric, hyphens, underscores
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return all(c in allowed for c in conversation_id)
