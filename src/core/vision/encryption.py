"""Data encryption and key management for Vision Provider system.

This module provides security features including:
- Symmetric encryption (AES)
- Asymmetric encryption (RSA)
- Key management and rotation
- Secure storage
- Hash functions and HMAC
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar, Union

from .base import VisionDescription, VisionProvider


class EncryptionAlgorithm(Enum):
    """Encryption algorithm types."""

    AES_128_GCM = "aes_128_gcm"
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"


class KeyType(Enum):
    """Key types."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    HMAC = "hmac"
    DERIVED = "derived"


class KeyStatus(Enum):
    """Key status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPROMISED = "compromised"
    EXPIRED = "expired"
    PENDING_ROTATION = "pending_rotation"


class HashAlgorithm(Enum):
    """Hash algorithms."""

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"


@dataclass
class KeyMetadata:
    """Key metadata."""

    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: KeyStatus = KeyStatus.ACTIVE
    version: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    rotation_policy: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_active(self) -> bool:
        """Check if key is active."""
        return self.status == KeyStatus.ACTIVE and not self.is_expired()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key_id": self.key_id,
            "key_type": self.key_type.value,
            "algorithm": self.algorithm.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status.value,
            "version": self.version,
            "tags": dict(self.tags),
            "rotation_policy": self.rotation_policy,
        }


@dataclass
class EncryptedData:
    """Encrypted data container."""

    ciphertext: bytes
    key_id: str
    algorithm: EncryptionAlgorithm
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    associated_data: Optional[bytes] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "nonce": base64.b64encode(self.nonce).decode() if self.nonce else None,
            "tag": base64.b64encode(self.tag).decode() if self.tag else None,
            "associated_data": base64.b64encode(self.associated_data).decode()
            if self.associated_data
            else None,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedData":
        """Create from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            key_id=data["key_id"],
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            nonce=base64.b64decode(data["nonce"]) if data.get("nonce") else None,
            tag=base64.b64decode(data["tag"]) if data.get("tag") else None,
            associated_data=base64.b64decode(data["associated_data"])
            if data.get("associated_data")
            else None,
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class KeyStore(ABC):
    """Abstract key store."""

    @abstractmethod
    def store_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata) -> None:
        """Store a key."""
        pass

    @abstractmethod
    def get_key(self, key_id: str) -> Optional[Tuple[bytes, KeyMetadata]]:
        """Get a key."""
        pass

    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        pass

    @abstractmethod
    def list_keys(self, key_type: Optional[KeyType] = None) -> List[KeyMetadata]:
        """List keys."""
        pass


class InMemoryKeyStore(KeyStore):
    """In-memory key store."""

    def __init__(self) -> None:
        """Initialize store."""
        self._keys: Dict[str, Tuple[bytes, KeyMetadata]] = {}
        self._lock = threading.Lock()

    def store_key(self, key_id: str, key_data: bytes, metadata: KeyMetadata) -> None:
        """Store a key."""
        with self._lock:
            self._keys[key_id] = (key_data, metadata)

    def get_key(self, key_id: str) -> Optional[Tuple[bytes, KeyMetadata]]:
        """Get a key."""
        with self._lock:
            return self._keys.get(key_id)

    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        with self._lock:
            if key_id in self._keys:
                del self._keys[key_id]
                return True
            return False

    def list_keys(self, key_type: Optional[KeyType] = None) -> List[KeyMetadata]:
        """List keys."""
        with self._lock:
            keys = [metadata for _, metadata in self._keys.values()]
            if key_type:
                keys = [k for k in keys if k.key_type == key_type]
            return keys

    def update_status(self, key_id: str, status: KeyStatus) -> bool:
        """Update key status."""
        with self._lock:
            if key_id in self._keys:
                key_data, metadata = self._keys[key_id]
                metadata.status = status
                self._keys[key_id] = (key_data, metadata)
                return True
            return False


class Encryptor(ABC):
    """Abstract encryptor."""

    @abstractmethod
    def encrypt(
        self, plaintext: bytes, key: bytes, associated_data: Optional[bytes] = None
    ) -> EncryptedData:
        """Encrypt data."""
        pass

    @abstractmethod
    def decrypt(self, encrypted: EncryptedData, key: bytes) -> bytes:
        """Decrypt data."""
        pass


class SimpleAESEncryptor(Encryptor):
    """Simple AES encryptor using XOR (for demonstration - use real crypto in production)."""

    def __init__(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM) -> None:
        """Initialize encryptor."""
        self._algorithm = algorithm
        self._key_id = ""

    def set_key_id(self, key_id: str) -> None:
        """Set key ID for encrypted data."""
        self._key_id = key_id

    def encrypt(
        self, plaintext: bytes, key: bytes, associated_data: Optional[bytes] = None
    ) -> EncryptedData:
        """Encrypt data using simple XOR (demonstration only)."""
        # Generate nonce
        nonce = secrets.token_bytes(12)

        # Simple XOR encryption (NOT secure - use real AES in production)
        key_stream = self._generate_key_stream(key, nonce, len(plaintext))
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, key_stream))

        # Generate authentication tag
        tag = self._generate_tag(key, nonce, ciphertext, associated_data)

        return EncryptedData(
            ciphertext=ciphertext,
            key_id=self._key_id,
            algorithm=self._algorithm,
            nonce=nonce,
            tag=tag,
            associated_data=associated_data,
        )

    def decrypt(self, encrypted: EncryptedData, key: bytes) -> bytes:
        """Decrypt data."""
        # Verify tag
        expected_tag = self._generate_tag(
            key, encrypted.nonce, encrypted.ciphertext, encrypted.associated_data
        )
        if encrypted.tag != expected_tag:
            raise ValueError("Authentication failed")

        # Decrypt using XOR
        key_stream = self._generate_key_stream(key, encrypted.nonce, len(encrypted.ciphertext))
        plaintext = bytes(c ^ k for c, k in zip(encrypted.ciphertext, key_stream))

        return plaintext

    def _generate_key_stream(self, key: bytes, nonce: bytes, length: int) -> bytes:
        """Generate key stream."""
        # Simple key stream generation (NOT secure)
        combined = key + nonce
        stream = b""
        counter = 0
        while len(stream) < length:
            block = hashlib.sha256(combined + counter.to_bytes(4, "big")).digest()
            stream += block
            counter += 1
        return stream[:length]

    def _generate_tag(
        self,
        key: bytes,
        nonce: Optional[bytes],
        ciphertext: bytes,
        associated_data: Optional[bytes],
    ) -> bytes:
        """Generate authentication tag."""
        data = (nonce or b"") + ciphertext + (associated_data or b"")
        return hmac.new(key, data, hashlib.sha256).digest()[:16]


class KeyManager:
    """Key manager for handling encryption keys."""

    def __init__(self, key_store: Optional[KeyStore] = None) -> None:
        """Initialize key manager."""
        self._key_store = key_store or InMemoryKeyStore()
        self._active_keys: Dict[str, str] = {}  # purpose -> key_id
        self._lock = threading.Lock()

    def generate_key(
        self,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        key_type: KeyType = KeyType.SYMMETRIC,
        expires_in: Optional[timedelta] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a new key."""
        key_id = str(uuid.uuid4())

        # Determine key size
        if algorithm in (EncryptionAlgorithm.AES_128_GCM,):
            key_size = 16
        elif algorithm in (EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC):
            key_size = 32
        else:
            key_size = 32

        # Generate key
        key_data = secrets.token_bytes(key_size)

        # Create metadata
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            expires_at=datetime.now() + expires_in if expires_in else None,
            tags=tags or {},
        )

        self._key_store.store_key(key_id, key_data, metadata)
        return key_id

    def get_key(self, key_id: str) -> Optional[bytes]:
        """Get key data."""
        result = self._key_store.get_key(key_id)
        if result:
            key_data, metadata = result
            if metadata.is_active():
                return key_data
        return None

    def get_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata."""
        result = self._key_store.get_key(key_id)
        if result:
            _, metadata = result
            return metadata
        return None

    def rotate_key(self, old_key_id: str) -> Optional[str]:
        """Rotate a key."""
        result = self._key_store.get_key(old_key_id)
        if not result:
            return None

        _, old_metadata = result

        # Mark old key for rotation
        if isinstance(self._key_store, InMemoryKeyStore):
            self._key_store.update_status(old_key_id, KeyStatus.PENDING_ROTATION)

        # Generate new key with same parameters
        new_key_id = self.generate_key(
            algorithm=old_metadata.algorithm,
            key_type=old_metadata.key_type,
            tags=old_metadata.tags,
        )

        # Update version
        new_result = self._key_store.get_key(new_key_id)
        if new_result:
            _, new_metadata = new_result
            new_metadata.version = old_metadata.version + 1

        return new_key_id

    def set_active_key(self, purpose: str, key_id: str) -> None:
        """Set active key for a purpose."""
        with self._lock:
            self._active_keys[purpose] = key_id

    def get_active_key(self, purpose: str) -> Optional[str]:
        """Get active key for a purpose."""
        with self._lock:
            return self._active_keys.get(purpose)

    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key."""
        if isinstance(self._key_store, InMemoryKeyStore):
            return self._key_store.update_status(key_id, KeyStatus.COMPROMISED)
        return False

    def list_keys(self, key_type: Optional[KeyType] = None) -> List[KeyMetadata]:
        """List all keys."""
        return self._key_store.list_keys(key_type)

    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        return self._key_store.delete_key(key_id)


class EncryptionService:
    """High-level encryption service."""

    def __init__(
        self,
        key_manager: Optional[KeyManager] = None,
        encryptor: Optional[Encryptor] = None,
    ) -> None:
        """Initialize service."""
        self._key_manager = key_manager or KeyManager()
        self._encryptor = encryptor or SimpleAESEncryptor()

    def encrypt(
        self,
        plaintext: bytes,
        key_id: Optional[str] = None,
        purpose: str = "default",
        associated_data: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data."""
        # Get or create key
        if key_id is None:
            key_id = self._key_manager.get_active_key(purpose)
            if key_id is None:
                key_id = self._key_manager.generate_key()
                self._key_manager.set_active_key(purpose, key_id)

        key = self._key_manager.get_key(key_id)
        if key is None:
            raise ValueError(f"Key not found or inactive: {key_id}")

        # Set key ID for encryptor
        if isinstance(self._encryptor, SimpleAESEncryptor):
            self._encryptor.set_key_id(key_id)

        return self._encryptor.encrypt(plaintext, key, associated_data)

    def decrypt(self, encrypted: EncryptedData) -> bytes:
        """Decrypt data."""
        key = self._key_manager.get_key(encrypted.key_id)
        if key is None:
            raise ValueError(f"Key not found or inactive: {encrypted.key_id}")

        return self._encryptor.decrypt(encrypted, key)

    def encrypt_string(
        self,
        plaintext: str,
        key_id: Optional[str] = None,
        purpose: str = "default",
    ) -> str:
        """Encrypt a string and return base64 encoded result."""
        encrypted = self.encrypt(plaintext.encode(), key_id, purpose)
        return base64.b64encode(json.dumps(encrypted.to_dict()).encode()).decode()

    def decrypt_string(self, encrypted_str: str) -> str:
        """Decrypt a base64 encoded encrypted string."""
        data = json.loads(base64.b64decode(encrypted_str))
        encrypted = EncryptedData.from_dict(data)
        return self.decrypt(encrypted).decode()

    def get_key_manager(self) -> KeyManager:
        """Get key manager."""
        return self._key_manager


class Hasher:
    """Hash functions utility."""

    def __init__(self, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> None:
        """Initialize hasher."""
        self._algorithm = algorithm

    def hash(self, data: bytes) -> bytes:
        """Hash data."""
        if self._algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).digest()
        elif self._algorithm == HashAlgorithm.SHA384:
            return hashlib.sha384(data).digest()
        elif self._algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).digest()
        elif self._algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).digest()
        elif self._algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s(data).digest()
        else:
            return hashlib.sha256(data).digest()

    def hash_hex(self, data: bytes) -> str:
        """Hash data and return hex string."""
        return self.hash(data).hex()

    def hash_string(self, data: str) -> str:
        """Hash string and return hex string."""
        return self.hash_hex(data.encode())

    def verify(self, data: bytes, expected_hash: bytes) -> bool:
        """Verify hash."""
        return hmac.compare_digest(self.hash(data), expected_hash)


class HMACAuthenticator:
    """HMAC authenticator."""

    def __init__(
        self,
        key: bytes,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> None:
        """Initialize authenticator."""
        self._key = key
        self._algorithm = algorithm

    def sign(self, data: bytes) -> bytes:
        """Sign data."""
        hash_name = self._algorithm.value
        return hmac.new(self._key, data, hash_name).digest()

    def sign_hex(self, data: bytes) -> str:
        """Sign data and return hex string."""
        return self.sign(data).hex()

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify signature."""
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)


@dataclass
class SecureValue:
    """Secure value wrapper."""

    encrypted_data: EncryptedData
    hash: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if value is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class SecureStorage:
    """Secure storage for sensitive data."""

    def __init__(self, encryption_service: Optional[EncryptionService] = None) -> None:
        """Initialize storage."""
        self._encryption_service = encryption_service or EncryptionService()
        self._storage: Dict[str, SecureValue] = {}
        self._hasher = Hasher()
        self._lock = threading.Lock()

    def store(
        self,
        key: str,
        value: bytes,
        expires_in: Optional[timedelta] = None,
    ) -> None:
        """Store a value securely."""
        encrypted = self._encryption_service.encrypt(value, purpose="secure_storage")
        value_hash = self._hasher.hash_hex(value)

        secure_value = SecureValue(
            encrypted_data=encrypted,
            hash=value_hash,
            expires_at=datetime.now() + expires_in if expires_in else None,
        )

        with self._lock:
            self._storage[key] = secure_value

    def retrieve(self, key: str) -> Optional[bytes]:
        """Retrieve a value."""
        with self._lock:
            secure_value = self._storage.get(key)

        if secure_value is None:
            return None

        if secure_value.is_expired():
            self.delete(key)
            return None

        decrypted = self._encryption_service.decrypt(secure_value.encrypted_data)

        # Verify integrity
        if self._hasher.hash_hex(decrypted) != secure_value.hash:
            raise ValueError("Data integrity check failed")

        return decrypted

    def delete(self, key: str) -> bool:
        """Delete a value."""
        with self._lock:
            if key in self._storage:
                del self._storage[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            secure_value = self._storage.get(key)
            if secure_value is None:
                return False
            if secure_value.is_expired():
                return False
            return True

    def list_keys(self) -> List[str]:
        """List all keys."""
        with self._lock:
            return [k for k, v in self._storage.items() if not v.is_expired()]


class EncryptedVisionProvider(VisionProvider):
    """Vision provider with encrypted data handling."""

    def __init__(
        self,
        provider: VisionProvider,
        encryption_service: Optional[EncryptionService] = None,
        encrypt_input: bool = True,
        encrypt_output: bool = True,
    ) -> None:
        """Initialize provider."""
        self._provider = provider
        self._encryption_service = encryption_service or EncryptionService()
        self._encrypt_input = encrypt_input
        self._encrypt_output = encrypt_output
        self._secure_storage = SecureStorage(self._encryption_service)

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"encrypted_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with encryption."""
        # Store encrypted input if enabled
        if self._encrypt_input:
            request_id = str(uuid.uuid4())
            self._secure_storage.store(
                f"input_{request_id}",
                image_data,
                expires_in=timedelta(hours=1),
            )

        # Call underlying provider
        result = await self._provider.analyze_image(image_data, include_description)

        # Store encrypted output if enabled
        if self._encrypt_output:
            output_data = json.dumps(
                {
                    "summary": result.summary,
                    "details": result.details,
                    "confidence": result.confidence,
                }
            ).encode()
            self._secure_storage.store(
                f"output_{request_id}" if self._encrypt_input else f"output_{uuid.uuid4()}",
                output_data,
                expires_in=timedelta(hours=1),
            )

        return result

    def get_encryption_service(self) -> EncryptionService:
        """Get encryption service."""
        return self._encryption_service

    def get_secure_storage(self) -> SecureStorage:
        """Get secure storage."""
        return self._secure_storage


def create_encryption_service(
    key_store: Optional[KeyStore] = None,
) -> EncryptionService:
    """Create encryption service.

    Args:
        key_store: Optional key store

    Returns:
        Encryption service
    """
    key_manager = KeyManager(key_store)
    return EncryptionService(key_manager)


def create_encrypted_provider(
    provider: VisionProvider,
    encrypt_input: bool = True,
    encrypt_output: bool = True,
) -> EncryptedVisionProvider:
    """Create encrypted vision provider.

    Args:
        provider: Provider to wrap
        encrypt_input: Whether to encrypt input
        encrypt_output: Whether to encrypt output

    Returns:
        Encrypted provider
    """
    return EncryptedVisionProvider(
        provider,
        encrypt_input=encrypt_input,
        encrypt_output=encrypt_output,
    )
