"""Secrets Manager.

Provides high-level secrets management:
- Secret CRUD operations
- Rotation management
- Access control
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.secrets.core import (
    AccessLog,
    AccessPolicy,
    Encryptor,
    FernetEncryptor,
    KeyDerivation,
    Secret,
    SecretMetadata,
    SecretType,
)
from src.core.secrets.store import InMemorySecretStore, SecretStore

logger = logging.getLogger(__name__)


@dataclass
class RotationResult:
    """Result of secret rotation."""
    success: bool
    old_version: int
    new_version: int
    error: Optional[str] = None


class SecretsManager:
    """High-level secrets management."""

    def __init__(
        self,
        store: Optional[SecretStore] = None,
        encryptor: Optional[Encryptor] = None,
    ):
        self.encryptor = encryptor or FernetEncryptor()
        self.store = store or InMemorySecretStore(self.encryptor)
        self._rotation_callbacks: Dict[str, Callable[[str, str], None]] = {}

    async def create_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.GENERIC,
        description: str = "",
        owner: str = "",
        expires_in_days: Optional[int] = None,
        rotation_interval_days: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        policy: Optional[AccessPolicy] = None,
    ) -> Secret:
        """Create a new secret."""
        # Encrypt the value
        encrypted, nonce, tag = self.encryptor.encrypt(value.encode())

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        metadata = SecretMetadata(
            name=name,
            secret_type=secret_type,
            description=description,
            owner=owner,
            expires_at=expires_at,
            rotation_interval_days=rotation_interval_days,
            last_rotated_at=datetime.utcnow(),
            version=1,
            tags=tags or set(),
        )

        secret = Secret(
            name=name,
            encrypted_value=encrypted,
            metadata=metadata,
            nonce=nonce,
            tag=tag,
        )

        await self.store.put(secret)

        # Set policy if provided
        if policy and hasattr(self.store, 'set_policy'):
            await self.store.set_policy(name, policy)

        logger.info(f"Created secret: {name}")
        return secret

    async def get_secret(
        self,
        name: str,
        version: Optional[int] = None,
        identity: Optional[str] = None,
        service: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Optional[str]:
        """Get decrypted secret value."""
        # Check policy
        if hasattr(self.store, 'get_policy'):
            policy = await self.store.get_policy(name)
            if policy and not policy.is_allowed(identity, service, ip_address):
                await self._log_access(
                    name, "read", identity or "unknown",
                    service, ip_address, success=False
                )
                logger.warning(f"Access denied to secret: {name}")
                return None

        secret = await self.store.get(name, version)
        if not secret:
            return None

        # Check expiration
        if secret.metadata.is_expired:
            logger.warning(f"Secret expired: {name}")
            return None

        # Decrypt
        try:
            decrypted = self.encryptor.decrypt(
                secret.encrypted_value,
                secret.nonce or b"",
                secret.tag or b"",
            )
            await self._log_access(
                name, "read", identity or "unknown",
                service, ip_address, success=True
            )
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret {name}: {e}")
            return None

    async def update_secret(
        self,
        name: str,
        value: str,
    ) -> Optional[Secret]:
        """Update a secret (creates new version)."""
        existing = await self.store.get(name)
        if not existing:
            return None

        # Encrypt new value
        encrypted, nonce, tag = self.encryptor.encrypt(value.encode())

        # Create new version
        new_version = existing.metadata.version + 1
        new_metadata = SecretMetadata(
            name=name,
            secret_type=existing.metadata.secret_type,
            description=existing.metadata.description,
            owner=existing.metadata.owner,
            expires_at=existing.metadata.expires_at,
            rotation_interval_days=existing.metadata.rotation_interval_days,
            last_rotated_at=datetime.utcnow(),
            version=new_version,
            tags=existing.metadata.tags,
        )

        new_secret = Secret(
            name=name,
            encrypted_value=encrypted,
            metadata=new_metadata,
            nonce=nonce,
            tag=tag,
        )

        await self.store.put(new_secret)
        logger.info(f"Updated secret: {name} to v{new_version}")
        return new_secret

    async def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        result = await self.store.delete(name)
        if result:
            logger.info(f"Deleted secret: {name}")
        return result

    async def list_secrets(
        self,
        secret_type: Optional[SecretType] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[str]:
        """List secret names with optional filtering."""
        all_names = await self.store.list_secrets()

        if not secret_type and not tags:
            return all_names

        filtered = []
        for name in all_names:
            secret = await self.store.get(name)
            if not secret:
                continue

            if secret_type and secret.metadata.secret_type != secret_type:
                continue

            if tags and not tags.issubset(secret.metadata.tags):
                continue

            filtered.append(name)

        return filtered

    async def get_metadata(self, name: str) -> Optional[SecretMetadata]:
        """Get secret metadata without decryption."""
        secret = await self.store.get(name)
        return secret.metadata if secret else None

    # Rotation

    async def rotate_secret(
        self,
        name: str,
        new_value: Optional[str] = None,
        generator: Optional[Callable[[], str]] = None,
    ) -> RotationResult:
        """Rotate a secret."""
        existing = await self.store.get(name)
        if not existing:
            return RotationResult(
                success=False,
                old_version=0,
                new_version=0,
                error="Secret not found",
            )

        old_version = existing.metadata.version

        # Generate new value if not provided
        if new_value is None:
            if generator:
                new_value = generator()
            elif existing.metadata.secret_type == SecretType.PASSWORD:
                new_value = KeyDerivation.generate_password()
            elif existing.metadata.secret_type == SecretType.API_KEY:
                new_value = KeyDerivation.generate_api_key()
            else:
                new_value = KeyDerivation.generate_password(48)

        # Update the secret
        new_secret = await self.update_secret(name, new_value)
        if not new_secret:
            return RotationResult(
                success=False,
                old_version=old_version,
                new_version=old_version,
                error="Failed to update secret",
            )

        # Call rotation callback if registered
        if name in self._rotation_callbacks:
            try:
                old_value = await self.get_secret(name, version=old_version)
                self._rotation_callbacks[name](old_value or "", new_value)
            except Exception as e:
                logger.error(f"Rotation callback failed for {name}: {e}")

        logger.info(f"Rotated secret: {name} from v{old_version} to v{new_secret.metadata.version}")
        return RotationResult(
            success=True,
            old_version=old_version,
            new_version=new_secret.metadata.version,
        )

    def register_rotation_callback(
        self,
        name: str,
        callback: Callable[[str, str], None],
    ) -> None:
        """Register callback for secret rotation."""
        self._rotation_callbacks[name] = callback

    async def get_secrets_needing_rotation(self) -> List[str]:
        """Get secrets that need rotation."""
        needs_rotation = []
        for name in await self.store.list_secrets():
            secret = await self.store.get(name)
            if secret and secret.metadata.needs_rotation:
                needs_rotation.append(name)
        return needs_rotation

    async def get_expiring_secrets(
        self,
        within_days: int = 30,
    ) -> List[tuple[str, datetime]]:
        """Get secrets expiring within specified days."""
        expiring = []
        cutoff = datetime.utcnow() + timedelta(days=within_days)

        for name in await self.store.list_secrets():
            secret = await self.store.get(name)
            if secret and secret.metadata.expires_at:
                if secret.metadata.expires_at < cutoff:
                    expiring.append((name, secret.metadata.expires_at))

        return sorted(expiring, key=lambda x: x[1])

    # Access logging

    async def _log_access(
        self,
        name: str,
        action: str,
        identity: str,
        service: Optional[str],
        ip_address: Optional[str],
        success: bool,
    ) -> None:
        """Log secret access."""
        if hasattr(self.store, 'log_access'):
            log = AccessLog(
                secret_name=name,
                action=action,
                identity=identity,
                service=service,
                ip_address=ip_address,
                success=success,
            )
            await self.store.log_access(log)

    async def get_access_logs(
        self,
        name: Optional[str] = None,
        limit: int = 100,
    ) -> List[AccessLog]:
        """Get access logs."""
        if hasattr(self.store, 'get_access_logs'):
            return await self.store.get_access_logs(name, limit)
        return []

    # Utility methods

    async def generate_and_store(
        self,
        name: str,
        secret_type: SecretType = SecretType.GENERIC,
        **kwargs,
    ) -> tuple[Secret, str]:
        """Generate and store a new secret."""
        if secret_type == SecretType.PASSWORD:
            value = KeyDerivation.generate_password()
        elif secret_type == SecretType.API_KEY:
            value = KeyDerivation.generate_api_key()
        else:
            value = KeyDerivation.generate_password(48)

        secret = await self.create_secret(
            name=name,
            value=value,
            secret_type=secret_type,
            **kwargs,
        )
        return secret, value
