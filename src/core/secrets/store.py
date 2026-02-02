"""Secrets Store.

Provides secret storage backends:
- In-memory store
- File-based store
- Vault integration interface
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.core.secrets.core import (
    AccessLog,
    AccessPolicy,
    Encryptor,
    FernetEncryptor,
    Secret,
    SecretMetadata,
    SecretType,
    SecretVersion,
)

logger = logging.getLogger(__name__)


class SecretStore(ABC):
    """Abstract base class for secret storage."""

    @abstractmethod
    async def get(self, name: str, version: Optional[int] = None) -> Optional[Secret]:
        """Get a secret by name and optional version."""
        pass

    @abstractmethod
    async def put(self, secret: Secret) -> None:
        """Store a secret."""
        pass

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """Delete a secret."""
        pass

    @abstractmethod
    async def list_secrets(self) -> List[str]:
        """List all secret names."""
        pass

    @abstractmethod
    async def get_versions(self, name: str) -> List[SecretVersion]:
        """Get all versions of a secret."""
        pass


class InMemorySecretStore(SecretStore):
    """In-memory secret storage."""

    def __init__(self, encryptor: Optional[Encryptor] = None):
        self.encryptor = encryptor or FernetEncryptor()
        # {name: {version: Secret}}
        self._secrets: Dict[str, Dict[int, Secret]] = {}
        self._policies: Dict[str, AccessPolicy] = {}
        self._access_logs: List[AccessLog] = []
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self, name: str, version: Optional[int] = None) -> Optional[Secret]:
        async with self._get_lock():
            if name not in self._secrets:
                return None

            versions = self._secrets[name]
            if not versions:
                return None

            if version is None:
                # Get latest version
                latest_version = max(versions.keys())
                return versions[latest_version]
            else:
                return versions.get(version)

    async def put(self, secret: Secret) -> None:
        async with self._get_lock():
            if secret.name not in self._secrets:
                self._secrets[secret.name] = {}

            version = secret.metadata.version
            self._secrets[secret.name][version] = secret
            logger.info(f"Stored secret: {secret.name} v{version}")

    async def delete(self, name: str) -> bool:
        async with self._get_lock():
            if name in self._secrets:
                del self._secrets[name]
                self._policies.pop(name, None)
                logger.info(f"Deleted secret: {name}")
                return True
            return False

    async def list_secrets(self) -> List[str]:
        async with self._get_lock():
            return list(self._secrets.keys())

    async def get_versions(self, name: str) -> List[SecretVersion]:
        async with self._get_lock():
            if name not in self._secrets:
                return []

            versions = []
            for version, secret in sorted(self._secrets[name].items()):
                versions.append(SecretVersion(
                    version=version,
                    encrypted_value=secret.encrypted_value,
                    created_at=secret.metadata.created_at,
                    nonce=secret.nonce,
                    tag=secret.tag,
                ))
            return versions

    async def set_policy(self, name: str, policy: AccessPolicy) -> None:
        """Set access policy for a secret."""
        async with self._get_lock():
            self._policies[name] = policy

    async def get_policy(self, name: str) -> Optional[AccessPolicy]:
        """Get access policy for a secret."""
        async with self._get_lock():
            return self._policies.get(name)

    async def log_access(self, log: AccessLog) -> None:
        """Log secret access."""
        async with self._get_lock():
            self._access_logs.append(log)
            # Keep only last 10000 logs
            if len(self._access_logs) > 10000:
                self._access_logs = self._access_logs[-10000:]

    async def get_access_logs(
        self,
        secret_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[AccessLog]:
        """Get access logs."""
        async with self._get_lock():
            logs = self._access_logs
            if secret_name:
                logs = [l for l in logs if l.secret_name == secret_name]
            return logs[-limit:]


class EncryptedFileSecretStore(SecretStore):
    """File-based encrypted secret storage."""

    def __init__(
        self,
        directory: str,
        encryptor: Optional[Encryptor] = None,
        master_key: Optional[bytes] = None,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.encryptor = encryptor or FernetEncryptor(master_key)
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _secret_path(self, name: str) -> Path:
        # Sanitize name for filesystem
        safe_name = name.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe_name}.secret"

    async def get(self, name: str, version: Optional[int] = None) -> Optional[Secret]:
        async with self._get_lock():
            path = self._secret_path(name)
            if not path.exists():
                return None

            try:
                data = json.loads(path.read_text())
                versions = data.get("versions", {})

                if version is None:
                    version = max(int(v) for v in versions.keys())

                version_data = versions.get(str(version))
                if not version_data:
                    return None

                metadata = SecretMetadata(
                    name=name,
                    secret_type=SecretType(data.get("type", "generic")),
                    description=data.get("description", ""),
                    version=version,
                    created_at=datetime.fromisoformat(version_data["created_at"]),
                )

                return Secret(
                    name=name,
                    encrypted_value=bytes.fromhex(version_data["encrypted_value"]),
                    metadata=metadata,
                    nonce=bytes.fromhex(version_data["nonce"]) if version_data.get("nonce") else None,
                    tag=bytes.fromhex(version_data["tag"]) if version_data.get("tag") else None,
                )
            except Exception as e:
                logger.error(f"Failed to read secret {name}: {e}")
                return None

    async def put(self, secret: Secret) -> None:
        async with self._get_lock():
            path = self._secret_path(secret.name)

            # Load existing data or create new
            if path.exists():
                data = json.loads(path.read_text())
            else:
                data = {
                    "name": secret.name,
                    "type": secret.metadata.secret_type.value,
                    "description": secret.metadata.description,
                    "versions": {},
                }

            # Add version
            version_data = {
                "encrypted_value": secret.encrypted_value.hex(),
                "nonce": secret.nonce.hex() if secret.nonce else "",
                "tag": secret.tag.hex() if secret.tag else "",
                "created_at": secret.metadata.created_at.isoformat(),
            }
            data["versions"][str(secret.metadata.version)] = version_data
            data["updated_at"] = datetime.utcnow().isoformat()

            path.write_text(json.dumps(data, indent=2))
            logger.info(f"Stored secret to file: {secret.name} v{secret.metadata.version}")

    async def delete(self, name: str) -> bool:
        async with self._get_lock():
            path = self._secret_path(name)
            if path.exists():
                path.unlink()
                logger.info(f"Deleted secret file: {name}")
                return True
            return False

    async def list_secrets(self) -> List[str]:
        async with self._get_lock():
            secrets = []
            for path in self.directory.glob("*.secret"):
                try:
                    data = json.loads(path.read_text())
                    secrets.append(data["name"])
                except Exception:
                    continue
            return secrets

    async def get_versions(self, name: str) -> List[SecretVersion]:
        async with self._get_lock():
            path = self._secret_path(name)
            if not path.exists():
                return []

            try:
                data = json.loads(path.read_text())
                versions = []
                for ver_str, ver_data in data.get("versions", {}).items():
                    versions.append(SecretVersion(
                        version=int(ver_str),
                        encrypted_value=bytes.fromhex(ver_data["encrypted_value"]),
                        created_at=datetime.fromisoformat(ver_data["created_at"]),
                        nonce=bytes.fromhex(ver_data["nonce"]) if ver_data.get("nonce") else None,
                        tag=bytes.fromhex(ver_data["tag"]) if ver_data.get("tag") else None,
                    ))
                return sorted(versions, key=lambda v: v.version)
            except Exception as e:
                logger.error(f"Failed to read versions for {name}: {e}")
                return []


class EnvironmentSecretStore(SecretStore):
    """Secret store backed by environment variables."""

    def __init__(self, prefix: str = "SECRET_"):
        self.prefix = prefix

    async def get(self, name: str, version: Optional[int] = None) -> Optional[Secret]:
        env_name = f"{self.prefix}{name.upper().replace('-', '_')}"
        value = os.environ.get(env_name)

        if value is None:
            return None

        return Secret(
            name=name,
            encrypted_value=value.encode(),  # Not actually encrypted
            metadata=SecretMetadata(name=name),
            nonce=None,
            tag=None,
        )

    async def put(self, secret: Secret) -> None:
        # Cannot write to environment in a meaningful way
        logger.warning("EnvironmentSecretStore does not support writing secrets")

    async def delete(self, name: str) -> bool:
        # Cannot delete environment variables
        logger.warning("EnvironmentSecretStore does not support deleting secrets")
        return False

    async def list_secrets(self) -> List[str]:
        secrets = []
        for key in os.environ:
            if key.startswith(self.prefix):
                name = key[len(self.prefix):].lower().replace('_', '-')
                secrets.append(name)
        return secrets

    async def get_versions(self, name: str) -> List[SecretVersion]:
        # Environment variables don't support versioning
        return []


class CompositeSecretStore(SecretStore):
    """Composite store that searches multiple stores."""

    def __init__(self, stores: List[SecretStore], write_store: Optional[SecretStore] = None):
        self.stores = stores
        self.write_store = write_store or (stores[0] if stores else None)

    async def get(self, name: str, version: Optional[int] = None) -> Optional[Secret]:
        for store in self.stores:
            secret = await store.get(name, version)
            if secret:
                return secret
        return None

    async def put(self, secret: Secret) -> None:
        if self.write_store:
            await self.write_store.put(secret)

    async def delete(self, name: str) -> bool:
        if self.write_store:
            return await self.write_store.delete(name)
        return False

    async def list_secrets(self) -> List[str]:
        all_secrets: Set[str] = set()
        for store in self.stores:
            secrets = await store.list_secrets()
            all_secrets.update(secrets)
        return list(all_secrets)

    async def get_versions(self, name: str) -> List[SecretVersion]:
        for store in self.stores:
            versions = await store.get_versions(name)
            if versions:
                return versions
        return []
