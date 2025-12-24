"""Encryption Manager Module for Vision System.

This module provides encryption and key management capabilities including:
- Data encryption and decryption
- Key generation and rotation
- Key storage and retrieval
- Certificate management
- Secure data handling
- Encryption at rest and in transit

Phase 19: Advanced Security & Compliance
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithms."""

    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    AES_128_GCM = "aes-128-gcm"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    RSA_OAEP = "rsa-oaep"


class KeyType(str, Enum):
    """Types of encryption keys."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HMAC = "hmac"
    DERIVED = "derived"


class KeyStatus(str, Enum):
    """Key lifecycle status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ROTATING = "rotating"
    COMPROMISED = "compromised"
    DESTROYED = "destroyed"


class KeyPurpose(str, Enum):
    """Purpose of the key."""

    ENCRYPTION = "encryption"
    SIGNING = "signing"
    KEY_WRAPPING = "key_wrapping"
    AUTHENTICATION = "authentication"


class HashAlgorithm(str, Enum):
    """Hash algorithms."""

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"


# ========================
# Dataclasses
# ========================


@dataclass
class EncryptionKey:
    """An encryption key."""

    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    purpose: KeyPurpose
    key_material: bytes = field(repr=False)  # Never log key material
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    rotated_from: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptedData:
    """Encrypted data with metadata."""

    ciphertext: bytes
    key_id: str
    algorithm: EncryptionAlgorithm
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    aad: Optional[bytes] = None  # Additional authenticated data
    encrypted_at: datetime = field(default_factory=datetime.now)


@dataclass
class KeyPolicy:
    """Policy for key management."""

    policy_id: str
    name: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    rotation_days: int = 90
    max_usages: int = 0  # 0 = unlimited
    allowed_purposes: List[KeyPurpose] = field(default_factory=list)
    auto_rotate: bool = True


@dataclass
class Certificate:
    """A certificate for asymmetric operations."""

    cert_id: str
    subject: str
    issuer: str
    public_key: bytes
    serial_number: str
    valid_from: datetime
    valid_until: datetime
    fingerprint: str
    chain: List[bytes] = field(default_factory=list)


@dataclass
class KeyRotationEvent:
    """Key rotation event."""

    event_id: str
    old_key_id: str
    new_key_id: str
    rotated_at: datetime = field(default_factory=datetime.now)
    reason: str = "scheduled"
    initiated_by: str = "system"


# ========================
# Core Classes
# ========================


class KeyStore(ABC):
    """Abstract base class for key storage."""

    @abstractmethod
    def store_key(self, key: EncryptionKey) -> None:
        """Store an encryption key."""
        pass

    @abstractmethod
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Retrieve a key by ID."""
        pass

    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        pass

    @abstractmethod
    def list_keys(self, status: Optional[KeyStatus] = None) -> List[EncryptionKey]:
        """List keys."""
        pass


class MemoryKeyStore(KeyStore):
    """In-memory key store (for development/testing)."""

    def __init__(self):
        self._keys: Dict[str, EncryptionKey] = {}
        self._lock = threading.RLock()

    def store_key(self, key: EncryptionKey) -> None:
        """Store an encryption key."""
        with self._lock:
            self._keys[key.key_id] = key

    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Retrieve a key by ID."""
        return self._keys.get(key_id)

    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        with self._lock:
            if key_id in self._keys:
                del self._keys[key_id]
                return True
        return False

    def list_keys(self, status: Optional[KeyStatus] = None) -> List[EncryptionKey]:
        """List keys."""
        keys = list(self._keys.values())
        if status:
            keys = [k for k in keys if k.status == status]
        return keys


class KeyGenerator:
    """Generate encryption keys."""

    def generate_symmetric_key(
        self,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        purpose: KeyPurpose = KeyPurpose.ENCRYPTION,
    ) -> EncryptionKey:
        """Generate a symmetric key."""
        # Determine key size based on algorithm
        key_sizes = {
            EncryptionAlgorithm.AES_256_GCM: 32,
            EncryptionAlgorithm.AES_256_CBC: 32,
            EncryptionAlgorithm.AES_128_GCM: 16,
            EncryptionAlgorithm.CHACHA20_POLY1305: 32,
        }
        key_size = key_sizes.get(algorithm, 32)
        key_material = secrets.token_bytes(key_size)

        key_id = hashlib.sha256(f"{time.time()}:{secrets.token_hex(8)}".encode()).hexdigest()[:16]

        return EncryptionKey(
            key_id=key_id,
            key_type=KeyType.SYMMETRIC,
            algorithm=algorithm,
            purpose=purpose,
            key_material=key_material,
        )

    def generate_hmac_key(
        self,
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> EncryptionKey:
        """Generate an HMAC key."""
        # Key size should match hash output size
        key_sizes = {
            HashAlgorithm.SHA256: 32,
            HashAlgorithm.SHA384: 48,
            HashAlgorithm.SHA512: 64,
            HashAlgorithm.BLAKE2B: 64,
        }
        key_size = key_sizes.get(hash_algorithm, 32)
        key_material = secrets.token_bytes(key_size)

        key_id = hashlib.sha256(f"{time.time()}:{secrets.token_hex(8)}".encode()).hexdigest()[:16]

        return EncryptionKey(
            key_id=key_id,
            key_type=KeyType.HMAC,
            algorithm=EncryptionAlgorithm.AES_256_GCM,  # Placeholder
            purpose=KeyPurpose.AUTHENTICATION,
            key_material=key_material,
            metadata={"hash_algorithm": hash_algorithm.value},
        )

    def derive_key(
        self,
        master_key: EncryptionKey,
        context: bytes,
        purpose: KeyPurpose = KeyPurpose.ENCRYPTION,
    ) -> EncryptionKey:
        """Derive a key from a master key."""
        # Simple KDF using HKDF-like construction
        derived_material = hmac.new(
            master_key.key_material,
            context,
            hashlib.sha256,
        ).digest()

        key_id = hashlib.sha256(f"{master_key.key_id}:{context.hex()}".encode()).hexdigest()[:16]

        return EncryptionKey(
            key_id=key_id,
            key_type=KeyType.DERIVED,
            algorithm=master_key.algorithm,
            purpose=purpose,
            key_material=derived_material,
            metadata={"derived_from": master_key.key_id},
        )


class Encryptor:
    """Handle encryption and decryption operations."""

    def __init__(self, key_store: KeyStore):
        self._key_store = key_store

    def encrypt(
        self,
        plaintext: bytes,
        key_id: str,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data using the specified key."""
        key = self._key_store.get_key(key_id)
        if key is None:
            raise ValueError(f"Key not found: {key_id}")

        if key.status != KeyStatus.ACTIVE:
            raise ValueError(f"Key is not active: {key_id}")

        # Generate IV
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM

        # Simplified encryption (in production, use proper crypto library)
        # This is a demonstration - NOT secure for production use
        cipher_data = self._xor_encrypt(plaintext, key.key_material, iv)
        tag = hmac.new(key.key_material, cipher_data + (aad or b""), hashlib.sha256).digest()[:16]

        return EncryptedData(
            ciphertext=cipher_data,
            key_id=key_id,
            algorithm=key.algorithm,
            iv=iv,
            tag=tag,
            aad=aad,
        )

    def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data."""
        key = self._key_store.get_key(encrypted_data.key_id)
        if key is None:
            raise ValueError(f"Key not found: {encrypted_data.key_id}")

        # Verify tag
        expected_tag = hmac.new(
            key.key_material,
            encrypted_data.ciphertext + (encrypted_data.aad or b""),
            hashlib.sha256,
        ).digest()[:16]

        if not hmac.compare_digest(encrypted_data.tag or b"", expected_tag):
            raise ValueError("Authentication failed - data may be tampered")

        # Decrypt
        return self._xor_decrypt(
            encrypted_data.ciphertext,
            key.key_material,
            encrypted_data.iv or b"",
        )

    def _xor_encrypt(self, plaintext: bytes, key: bytes, iv: bytes) -> bytes:
        """Simple XOR encryption (demonstration only)."""
        # Generate keystream
        keystream = self._generate_keystream(key, iv, len(plaintext))
        return bytes(a ^ b for a, b in zip(plaintext, keystream))

    def _xor_decrypt(self, ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
        """Simple XOR decryption (demonstration only)."""
        return self._xor_encrypt(ciphertext, key, iv)

    def _generate_keystream(self, key: bytes, iv: bytes, length: int) -> bytes:
        """Generate keystream for encryption."""
        keystream = b""
        counter = 0
        while len(keystream) < length:
            block = hmac.new(
                key,
                iv + counter.to_bytes(8, "big"),
                hashlib.sha256,
            ).digest()
            keystream += block
            counter += 1
        return keystream[:length]


class KeyRotationManager:
    """Manage key rotation."""

    def __init__(self, key_store: KeyStore, key_generator: KeyGenerator):
        self._key_store = key_store
        self._key_generator = key_generator
        self._rotation_events: List[KeyRotationEvent] = []
        self._lock = threading.RLock()

    def rotate_key(
        self,
        old_key_id: str,
        reason: str = "scheduled",
        initiated_by: str = "system",
    ) -> Optional[EncryptionKey]:
        """Rotate an encryption key."""
        old_key = self._key_store.get_key(old_key_id)
        if old_key is None:
            return None

        # Generate new key
        new_key = self._key_generator.generate_symmetric_key(
            algorithm=old_key.algorithm,
            purpose=old_key.purpose,
        )
        new_key.rotated_from = old_key_id

        # Update old key status
        old_key.status = KeyStatus.INACTIVE

        # Store new key
        self._key_store.store_key(new_key)

        # Record rotation event
        event = KeyRotationEvent(
            event_id=hashlib.md5(f"{old_key_id}:{new_key.key_id}".encode()).hexdigest()[:8],
            old_key_id=old_key_id,
            new_key_id=new_key.key_id,
            reason=reason,
            initiated_by=initiated_by,
        )

        with self._lock:
            self._rotation_events.append(event)

        return new_key

    def get_rotation_history(self, key_id: str) -> List[KeyRotationEvent]:
        """Get rotation history for a key."""
        with self._lock:
            return [
                e for e in self._rotation_events if e.old_key_id == key_id or e.new_key_id == key_id
            ]

    def check_rotation_needed(self, key: EncryptionKey, policy: KeyPolicy) -> bool:
        """Check if key needs rotation based on policy."""
        if not policy.auto_rotate:
            return False

        if key.expires_at and datetime.now() > key.expires_at:
            return True

        age = (datetime.now() - key.created_at).days
        return age >= policy.rotation_days


class EncryptionManager:
    """Main encryption management component."""

    def __init__(self, key_store: Optional[KeyStore] = None):
        self._key_store = key_store or MemoryKeyStore()
        self._key_generator = KeyGenerator()
        self._encryptor = Encryptor(self._key_store)
        self._rotation_manager = KeyRotationManager(self._key_store, self._key_generator)
        self._policies: Dict[str, KeyPolicy] = {}
        self._certificates: Dict[str, Certificate] = {}
        self._lock = threading.RLock()

    def generate_key(
        self,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        purpose: KeyPurpose = KeyPurpose.ENCRYPTION,
    ) -> EncryptionKey:
        """Generate and store a new key."""
        key = self._key_generator.generate_symmetric_key(algorithm, purpose)
        self._key_store.store_key(key)
        return key

    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get a key by ID."""
        return self._key_store.get_key(key_id)

    def list_keys(self, status: Optional[KeyStatus] = None) -> List[EncryptionKey]:
        """List all keys."""
        return self._key_store.list_keys(status)

    def encrypt(
        self,
        plaintext: bytes,
        key_id: str,
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data."""
        return self._encryptor.encrypt(plaintext, key_id, aad)

    def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data."""
        return self._encryptor.decrypt(encrypted_data)

    def rotate_key(self, key_id: str, reason: str = "manual") -> Optional[EncryptionKey]:
        """Rotate a key."""
        return self._rotation_manager.rotate_key(key_id, reason)

    def add_policy(self, policy: KeyPolicy) -> None:
        """Add a key policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def get_policy(self, policy_id: str) -> Optional[KeyPolicy]:
        """Get a key policy."""
        return self._policies.get(policy_id)

    def hash_data(
        self,
        data: bytes,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> bytes:
        """Hash data."""
        hash_functions = {
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA384: hashlib.sha384,
            HashAlgorithm.SHA512: hashlib.sha512,
            HashAlgorithm.BLAKE2B: hashlib.blake2b,
        }
        hash_fn = hash_functions.get(algorithm, hashlib.sha256)
        return hash_fn(data).digest()

    def sign_data(self, data: bytes, key_id: str) -> bytes:
        """Sign data with an HMAC key."""
        key = self._key_store.get_key(key_id)
        if key is None:
            raise ValueError(f"Key not found: {key_id}")
        return hmac.new(key.key_material, data, hashlib.sha256).digest()

    def verify_signature(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify a signature."""
        key = self._key_store.get_key(key_id)
        if key is None:
            return False
        expected = hmac.new(key.key_material, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)


# ========================
# Vision Provider
# ========================


class EncryptionManagerVisionProvider(VisionProvider):
    """Vision provider for encryption capabilities."""

    def __init__(self):
        self._manager: Optional[EncryptionManager] = None

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "encryption_manager"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """Analyze image for encryption context."""
        return self.get_description()

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Encryption Manager Vision Provider",
            version="1.0.0",
            description="Data encryption and key management",
            capabilities=[
                "key_generation",
                "encryption",
                "decryption",
                "key_rotation",
                "signing",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._manager = EncryptionManager()

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._manager = None

    def get_manager(self) -> EncryptionManager:
        """Get the encryption manager."""
        if self._manager is None:
            self.initialize()
        return self._manager


# ========================
# Factory Functions
# ========================


def create_encryption_manager(
    key_store: Optional[KeyStore] = None,
) -> EncryptionManager:
    """Create an encryption manager."""
    return EncryptionManager(key_store=key_store)


def create_memory_key_store() -> MemoryKeyStore:
    """Create a memory key store."""
    return MemoryKeyStore()


def create_key_generator() -> KeyGenerator:
    """Create a key generator."""
    return KeyGenerator()


def create_encryptor(key_store: KeyStore) -> Encryptor:
    """Create an encryptor."""
    return Encryptor(key_store=key_store)


def create_key_policy(
    policy_id: str,
    name: str,
    key_type: KeyType = KeyType.SYMMETRIC,
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
    rotation_days: int = 90,
) -> KeyPolicy:
    """Create a key policy."""
    return KeyPolicy(
        policy_id=policy_id,
        name=name,
        key_type=key_type,
        algorithm=algorithm,
        rotation_days=rotation_days,
    )


def create_encryption_manager_provider() -> EncryptionManagerVisionProvider:
    """Create an encryption manager vision provider."""
    return EncryptionManagerVisionProvider()
