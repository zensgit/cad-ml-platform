"""
Key Management Module for Vision Provider System.

Provides cryptographic key management, key rotation, secrets vault,
and secure key storage for vision analysis operations.
"""

import base64
import hashlib
import hmac
import os
import secrets
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import VisionDescription, VisionProvider

# ============================================================================
# Enums and Types
# ============================================================================


class KeyType(Enum):
    """Types of cryptographic keys."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    API_KEY = "api_key"
    SIGNING_KEY = "signing_key"
    ENCRYPTION_KEY = "encryption_key"
    MASTER_KEY = "master_key"


class KeyAlgorithm(Enum):
    """Key algorithms."""

    AES_256 = "aes_256"
    AES_128 = "aes_128"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"
    HMAC_SHA256 = "hmac_sha256"
    HMAC_SHA512 = "hmac_sha512"


class KeyStatus(Enum):
    """Key status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_DELETION = "pending_deletion"
    DELETED = "deleted"
    COMPROMISED = "compromised"
    EXPIRED = "expired"


class SecretType(Enum):
    """Types of secrets."""

    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    ASYMMETRIC = "asymmetric"
    CONNECTION_STRING = "connection_string"
    CREDENTIAL = "credential"
    CUSTOM = "custom"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CryptoKey:
    """A cryptographic key."""

    key_id: str
    key_type: KeyType
    algorithm: KeyAlgorithm
    key_material: bytes
    status: KeyStatus = KeyStatus.ACTIVE
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotated_from: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def is_active(self) -> bool:
        """Check if key is active."""
        if self.status != KeyStatus.ACTIVE:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def get_key_hash(self) -> str:
        """Get hash of key material for verification."""
        return hashlib.sha256(self.key_material).hexdigest()[:16]


@dataclass
class Secret:
    """A secret stored in the vault."""

    secret_id: str
    name: str
    secret_type: SecretType
    value: bytes
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if secret is expired."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return True
        return False


@dataclass
class KeyRotationPolicy:
    """Key rotation policy."""

    policy_id: str
    name: str
    key_types: List[KeyType]
    rotation_interval_days: int
    auto_rotate: bool = True
    notify_before_days: int = 7
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyAuditEntry:
    """Audit entry for key operations."""

    entry_id: str
    key_id: str
    operation: str
    user_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# Key Generation
# ============================================================================


class KeyGenerator:
    """Generates cryptographic keys."""

    def generate_key(
        self,
        key_type: KeyType,
        algorithm: KeyAlgorithm,
    ) -> bytes:
        """Generate a key based on algorithm."""
        if algorithm == KeyAlgorithm.AES_256:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == KeyAlgorithm.AES_128:
            return secrets.token_bytes(16)  # 128 bits
        elif algorithm in (KeyAlgorithm.HMAC_SHA256, KeyAlgorithm.HMAC_SHA512):
            return secrets.token_bytes(64)  # 512 bits for HMAC
        else:
            # Default to 32 bytes
            return secrets.token_bytes(32)

    def generate_api_key(self, prefix: str = "sk") -> str:
        """Generate an API key."""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"

    def generate_secret(self, length: int = 32) -> str:
        """Generate a random secret."""
        return secrets.token_urlsafe(length)

    def derive_key(
        self,
        master_key: bytes,
        salt: bytes,
        context: str,
        length: int = 32,
    ) -> bytes:
        """Derive a key from master key using HKDF-like approach."""
        # Simple key derivation using HMAC
        info = context.encode()
        prk = hmac.new(salt, master_key, hashlib.sha256).digest()
        okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()
        return okm[:length]


# ============================================================================
# Key Store
# ============================================================================


class KeyStore(ABC):
    """Abstract base for key storage."""

    @abstractmethod
    def store(self, key: CryptoKey) -> None:
        """Store a key."""
        pass

    @abstractmethod
    def get(self, key_id: str) -> Optional[CryptoKey]:
        """Get a key by ID."""
        pass

    @abstractmethod
    def delete(self, key_id: str) -> bool:
        """Delete a key."""
        pass

    @abstractmethod
    def list(self) -> List[CryptoKey]:
        """List all keys."""
        pass


class InMemoryKeyStore(KeyStore):
    """In-memory key storage."""

    def __init__(self) -> None:
        self._keys: Dict[str, CryptoKey] = {}
        self._lock = threading.Lock()

    def store(self, key: CryptoKey) -> None:
        """Store a key."""
        with self._lock:
            self._keys[key.key_id] = key

    def get(self, key_id: str) -> Optional[CryptoKey]:
        """Get a key by ID."""
        return self._keys.get(key_id)

    def delete(self, key_id: str) -> bool:
        """Delete a key."""
        with self._lock:
            if key_id in self._keys:
                del self._keys[key_id]
                return True
            return False

    def list(self) -> List[CryptoKey]:
        """List all keys."""
        return list(self._keys.values())


class EncryptedKeyStore(KeyStore):
    """Encrypted key storage."""

    def __init__(self, master_key: bytes) -> None:
        self._master_key = master_key
        self._inner_store = InMemoryKeyStore()
        self._lock = threading.Lock()

    def store(self, key: CryptoKey) -> None:
        """Store an encrypted key."""
        # Encrypt key material before storing
        encrypted_material = self._encrypt(key.key_material)
        encrypted_key = CryptoKey(
            key_id=key.key_id,
            key_type=key.key_type,
            algorithm=key.algorithm,
            key_material=encrypted_material,
            status=key.status,
            version=key.version,
            created_at=key.created_at,
            expires_at=key.expires_at,
            rotated_from=key.rotated_from,
            metadata=key.metadata,
            tags=key.tags,
        )
        self._inner_store.store(encrypted_key)

    def get(self, key_id: str) -> Optional[CryptoKey]:
        """Get and decrypt a key."""
        encrypted_key = self._inner_store.get(key_id)
        if not encrypted_key:
            return None

        # Decrypt key material
        decrypted_material = self._decrypt(encrypted_key.key_material)
        return CryptoKey(
            key_id=encrypted_key.key_id,
            key_type=encrypted_key.key_type,
            algorithm=encrypted_key.algorithm,
            key_material=decrypted_material,
            status=encrypted_key.status,
            version=encrypted_key.version,
            created_at=encrypted_key.created_at,
            expires_at=encrypted_key.expires_at,
            rotated_from=encrypted_key.rotated_from,
            metadata=encrypted_key.metadata,
            tags=encrypted_key.tags,
        )

    def delete(self, key_id: str) -> bool:
        """Delete a key."""
        return self._inner_store.delete(key_id)

    def list(self) -> List[CryptoKey]:
        """List all keys (without decrypting material)."""
        return self._inner_store.list()

    def _encrypt(self, data: bytes) -> bytes:
        """Simple XOR encryption (use proper encryption in production)."""
        key_extended = (self._master_key * ((len(data) // len(self._master_key)) + 1))[: len(data)]
        return bytes(a ^ b for a, b in zip(data, key_extended))

    def _decrypt(self, data: bytes) -> bytes:
        """Simple XOR decryption."""
        return self._encrypt(data)  # XOR is symmetric


# ============================================================================
# Secrets Vault
# ============================================================================


class SecretsVault:
    """Secure secrets storage vault."""

    def __init__(self, encryption_key: Optional[bytes] = None) -> None:
        self._secrets: Dict[str, Secret] = {}
        self._versions: Dict[str, List[Secret]] = {}
        self._encryption_key = encryption_key or secrets.token_bytes(32)
        self._lock = threading.Lock()

    def put_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.CUSTOM,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Secret:
        """Store a secret."""
        with self._lock:
            # Check for existing secret
            existing = self._secrets.get(name)
            version = 1
            if existing:
                version = existing.version + 1
                # Archive old version
                if name not in self._versions:
                    self._versions[name] = []
                self._versions[name].append(existing)

            # Encrypt value
            encrypted_value = self._encrypt(value.encode())

            secret = Secret(
                secret_id=str(uuid.uuid4()),
                name=name,
                secret_type=secret_type,
                value=encrypted_value,
                version=version,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            self._secrets[name] = secret
            return secret

    def get_secret(self, name: str) -> Optional[str]:
        """Get a secret value."""
        secret = self._secrets.get(name)
        if not secret:
            return None

        if secret.is_expired():
            return None

        # Decrypt value
        return self._decrypt(secret.value).decode()

    def get_secret_metadata(self, name: str) -> Optional[Secret]:
        """Get secret metadata without value."""
        secret = self._secrets.get(name)
        if secret:
            # Return copy without actual value
            return Secret(
                secret_id=secret.secret_id,
                name=secret.name,
                secret_type=secret.secret_type,
                value=b"",  # Don't expose value
                version=secret.version,
                created_at=secret.created_at,
                updated_at=secret.updated_at,
                expires_at=secret.expires_at,
                metadata=secret.metadata,
                tags=secret.tags,
            )
        return None

    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        with self._lock:
            if name in self._secrets:
                del self._secrets[name]
                if name in self._versions:
                    del self._versions[name]
                return True
            return False

    def list_secrets(self) -> List[str]:
        """List all secret names."""
        return list(self._secrets.keys())

    def get_version(self, name: str, version: int) -> Optional[str]:
        """Get a specific version of a secret."""
        if name not in self._versions:
            return None

        for secret in self._versions[name]:
            if secret.version == version:
                return self._decrypt(secret.value).decode()

        return None

    def rotate_secret(
        self,
        name: str,
        new_value: str,
    ) -> Optional[Secret]:
        """Rotate a secret to a new value."""
        if name not in self._secrets:
            return None
        return self.put_secret(
            name=name,
            value=new_value,
            secret_type=self._secrets[name].secret_type,
            metadata=self._secrets[name].metadata,
        )

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        key = (self._encryption_key * ((len(data) // len(self._encryption_key)) + 1))[: len(data)]
        return bytes(a ^ b for a, b in zip(data, key))

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data."""
        return self._encrypt(data)


# ============================================================================
# Key Manager
# ============================================================================


class KeyManager:
    """Comprehensive key management."""

    def __init__(
        self,
        store: Optional[KeyStore] = None,
        master_key: Optional[bytes] = None,
    ) -> None:
        self._master_key = master_key or secrets.token_bytes(32)
        self._store = store or InMemoryKeyStore()
        self._generator = KeyGenerator()
        self._policies: Dict[str, KeyRotationPolicy] = {}
        self._audit_log: List[KeyAuditEntry] = []
        self._lock = threading.Lock()

    def create_key(
        self,
        key_type: KeyType,
        algorithm: KeyAlgorithm,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> CryptoKey:
        """Create a new key."""
        key_material = self._generator.generate_key(key_type, algorithm)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        key = CryptoKey(
            key_id=str(uuid.uuid4()),
            key_type=key_type,
            algorithm=algorithm,
            key_material=key_material,
            expires_at=expires_at,
            metadata=metadata or {},
            tags=tags or [],
        )

        self._store.store(key)
        self._log_operation(key.key_id, "create")

        return key

    def get_key(self, key_id: str) -> Optional[CryptoKey]:
        """Get a key by ID."""
        key = self._store.get(key_id)
        if key:
            self._log_operation(key_id, "access")
        return key

    def delete_key(self, key_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a key."""
        key = self._store.get(key_id)
        if key:
            key.status = KeyStatus.DELETED
            self._store.store(key)
            self._log_operation(key_id, "delete", user_id)
            return True
        return False

    def rotate_key(self, key_id: str) -> Optional[CryptoKey]:
        """Rotate a key to a new version."""
        old_key = self._store.get(key_id)
        if not old_key:
            return None

        # Mark old key as inactive
        old_key.status = KeyStatus.INACTIVE
        self._store.store(old_key)

        # Create new key
        new_key = self.create_key(
            key_type=old_key.key_type,
            algorithm=old_key.algorithm,
            metadata=old_key.metadata,
            tags=old_key.tags,
        )
        new_key.version = old_key.version + 1
        new_key.rotated_from = old_key.key_id
        self._store.store(new_key)

        self._log_operation(old_key.key_id, "rotate", details={"new_key_id": new_key.key_id})

        return new_key

    def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        status: Optional[KeyStatus] = None,
    ) -> List[CryptoKey]:
        """List keys with optional filters."""
        keys = self._store.list()

        if key_type:
            keys = [k for k in keys if k.key_type == key_type]

        if status:
            keys = [k for k in keys if k.status == status]

        return keys

    def get_active_key(
        self,
        key_type: KeyType,
        algorithm: Optional[KeyAlgorithm] = None,
    ) -> Optional[CryptoKey]:
        """Get the active key of a type."""
        keys = self.list_keys(key_type=key_type, status=KeyStatus.ACTIVE)

        if algorithm:
            keys = [k for k in keys if k.algorithm == algorithm]

        if keys:
            # Return latest version
            return max(keys, key=lambda k: k.version)
        return None

    def add_rotation_policy(self, policy: KeyRotationPolicy) -> None:
        """Add a key rotation policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def check_rotation_needed(self) -> List[CryptoKey]:
        """Check which keys need rotation."""
        keys_to_rotate: List[CryptoKey] = []
        now = datetime.utcnow()

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            for key in self._store.list():
                if key.key_type not in policy.key_types:
                    continue

                if key.status != KeyStatus.ACTIVE:
                    continue

                age_days = (now - key.created_at).days
                if age_days >= policy.rotation_interval_days:
                    keys_to_rotate.append(key)

        return keys_to_rotate

    def auto_rotate_keys(self) -> List[CryptoKey]:
        """Automatically rotate keys that need rotation."""
        rotated_keys: List[CryptoKey] = []

        for key in self.check_rotation_needed():
            new_key = self.rotate_key(key.key_id)
            if new_key:
                rotated_keys.append(new_key)

        return rotated_keys

    def derive_key(
        self,
        context: str,
        length: int = 32,
    ) -> bytes:
        """Derive a key from master key."""
        salt = hashlib.sha256(context.encode()).digest()
        return self._generator.derive_key(self._master_key, salt, context, length)

    def generate_api_key(self, prefix: str = "sk") -> str:
        """Generate an API key."""
        return self._generator.generate_api_key(prefix)

    def get_audit_log(self, key_id: Optional[str] = None) -> List[KeyAuditEntry]:
        """Get audit log entries."""
        if key_id:
            return [e for e in self._audit_log if e.key_id == key_id]
        return self._audit_log.copy()

    def _log_operation(
        self,
        key_id: str,
        operation: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a key operation."""
        entry = KeyAuditEntry(
            entry_id=str(uuid.uuid4()),
            key_id=key_id,
            operation=operation,
            user_id=user_id,
            details=details or {},
        )
        with self._lock:
            self._audit_log.append(entry)


# ============================================================================
# Key Management Vision Provider
# ============================================================================


class KeyManagedVisionProvider(VisionProvider):
    """Vision provider with key management integration."""

    def __init__(
        self,
        provider: VisionProvider,
        key_manager: KeyManager,
        encryption_key_id: Optional[str] = None,
    ) -> None:
        self._provider = provider
        self._key_manager = key_manager
        self._encryption_key_id = encryption_key_id

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"keymgmt_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image with key management."""
        # Could use encryption key for signing/verifying results
        result = await self._provider.analyze_image(image_data, include_description)
        return result

    def get_key_manager(self) -> KeyManager:
        """Get the key manager."""
        return self._key_manager

    def get_signing_key(self) -> Optional[CryptoKey]:
        """Get the signing key."""
        return self._key_manager.get_active_key(KeyType.SIGNING_KEY, KeyAlgorithm.HMAC_SHA256)


# ============================================================================
# Factory Functions
# ============================================================================


def create_key_manager(
    master_key: Optional[bytes] = None,
) -> KeyManager:
    """Create a key manager."""
    return KeyManager(master_key=master_key)


def create_secrets_vault(
    encryption_key: Optional[bytes] = None,
) -> SecretsVault:
    """Create a secrets vault."""
    return SecretsVault(encryption_key)


def create_key_generator() -> KeyGenerator:
    """Create a key generator."""
    return KeyGenerator()


def create_in_memory_key_store() -> InMemoryKeyStore:
    """Create an in-memory key store."""
    return InMemoryKeyStore()


def create_encrypted_key_store(master_key: bytes) -> EncryptedKeyStore:
    """Create an encrypted key store."""
    return EncryptedKeyStore(master_key)


def create_key_managed_provider(
    provider: VisionProvider,
    key_manager: Optional[KeyManager] = None,
) -> KeyManagedVisionProvider:
    """Create a key-managed vision provider."""
    if key_manager is None:
        key_manager = create_key_manager()
    return KeyManagedVisionProvider(provider, key_manager)


def create_rotation_policy(
    name: str,
    key_types: List[KeyType],
    rotation_interval_days: int = 90,
    auto_rotate: bool = True,
) -> KeyRotationPolicy:
    """Create a key rotation policy."""
    return KeyRotationPolicy(
        policy_id=str(uuid.uuid4()),
        name=name,
        key_types=key_types,
        rotation_interval_days=rotation_interval_days,
        auto_rotate=auto_rotate,
    )


def create_crypto_key(
    key_type: KeyType,
    algorithm: KeyAlgorithm,
    key_material: Optional[bytes] = None,
) -> CryptoKey:
    """Create a crypto key."""
    if key_material is None:
        key_material = KeyGenerator().generate_key(key_type, algorithm)
    return CryptoKey(
        key_id=str(uuid.uuid4()),
        key_type=key_type,
        algorithm=algorithm,
        key_material=key_material,
    )
