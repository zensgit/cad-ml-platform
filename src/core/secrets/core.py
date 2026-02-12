"""Secrets Management Core.

Provides secure secret storage:
- Secret encryption
- Secret versioning
- Access control
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets."""
    PASSWORD = "password"
    API_KEY = "api_key"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    DATABASE_CREDENTIAL = "database_credential"
    ENCRYPTION_KEY = "encryption_key"
    GENERIC = "generic"


@dataclass
class SecretMetadata:
    """Metadata for a secret."""
    name: str
    secret_type: SecretType = SecretType.GENERIC
    description: str = ""
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotation_interval_days: Optional[int] = None
    last_rotated_at: Optional[datetime] = None
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    allowed_services: Set[str] = field(default_factory=set)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def needs_rotation(self) -> bool:
        if self.rotation_interval_days is None:
            return False
        if self.last_rotated_at is None:
            return True
        rotation_due = self.last_rotated_at + timedelta(days=self.rotation_interval_days)
        return datetime.utcnow() > rotation_due


@dataclass
class Secret:
    """A secret with encrypted value."""
    name: str
    encrypted_value: bytes
    metadata: SecretMetadata
    nonce: Optional[bytes] = None  # For encryption
    tag: Optional[bytes] = None    # For authentication

    def __repr__(self) -> str:
        return f"Secret(name={self.name}, type={self.metadata.secret_type.value}, version={self.metadata.version})"


@dataclass
class SecretVersion:
    """A version of a secret."""
    version: int
    encrypted_value: bytes
    created_at: datetime
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    deprecated: bool = False


class Encryptor(ABC):
    """Abstract base class for secret encryption."""

    @abstractmethod
    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes, bytes]:
        """Encrypt plaintext. Returns (ciphertext, nonce, tag)."""
        pass

    @abstractmethod
    def decrypt(self, ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
        """Decrypt ciphertext."""
        pass

    @abstractmethod
    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        pass


class AESGCMEncryptor(Encryptor):
    """AES-GCM encryption."""

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or self.generate_key()

    def generate_key(self) -> bytes:
        """Generate a 256-bit AES key."""
        return secrets.token_bytes(32)

    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes, bytes]:
        """Encrypt using AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._key)
            nonce = secrets.token_bytes(12)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            # Tag is appended to ciphertext in AESGCM
            return ciphertext[:-16], nonce, ciphertext[-16:]
        except ImportError:
            # Fallback to simple XOR for demo (NOT SECURE)
            logger.warning("cryptography not installed, using insecure fallback")
            nonce = secrets.token_bytes(12)
            # Simple XOR with key (NOT SECURE - for demo only)
            key_extended = (self._key * (len(plaintext) // len(self._key) + 1))[:len(plaintext)]
            ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_extended))
            tag = hmac.new(self._key, ciphertext, hashlib.sha256).digest()[:16]
            return ciphertext, nonce, tag

    def decrypt(self, ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
        """Decrypt using AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._key)
            # Reconstruct full ciphertext with tag
            full_ciphertext = ciphertext + tag
            return aesgcm.decrypt(nonce, full_ciphertext, None)
        except ImportError:
            # Fallback (NOT SECURE)
            expected_tag = hmac.new(self._key, ciphertext, hashlib.sha256).digest()[:16]
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("Authentication failed")
            key_extended = (self._key * (len(ciphertext) // len(self._key) + 1))[:len(ciphertext)]
            return bytes(a ^ b for a, b in zip(ciphertext, key_extended))


class FernetEncryptor(Encryptor):
    """Fernet encryption (symmetric)."""

    def __init__(self, key: Optional[bytes] = None):
        try:
            from cryptography.fernet import Fernet
            if key:
                self._fernet = Fernet(key)
                self._key = key
            else:
                self._key = Fernet.generate_key()
                self._fernet = Fernet(self._key)
        except ImportError:
            logger.warning("cryptography not installed")
            self._key = secrets.token_bytes(32)
            self._fernet = None

    def generate_key(self) -> bytes:
        try:
            from cryptography.fernet import Fernet
            return Fernet.generate_key()
        except ImportError:
            return base64.urlsafe_b64encode(secrets.token_bytes(32))

    def encrypt(self, plaintext: bytes) -> tuple[bytes, bytes, bytes]:
        if self._fernet:
            token = self._fernet.encrypt(plaintext)
            return token, b"", b""
        else:
            # Fallback
            encryptor = AESGCMEncryptor(self._key[:32])
            return encryptor.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
        if self._fernet:
            return self._fernet.decrypt(ciphertext)
        else:
            encryptor = AESGCMEncryptor(self._key[:32])
            return encryptor.decrypt(ciphertext, nonce, tag)


class KeyDerivation:
    """Key derivation functions."""

    @staticmethod
    def derive_key(
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 100000,
    ) -> tuple[bytes, bytes]:
        """Derive encryption key from password using PBKDF2."""
        salt = salt or secrets.token_bytes(16)
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
            )
            key = kdf.derive(password.encode())
        except ImportError:
            # Fallback using hashlib
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt,
                iterations,
                dklen=32,
            )
        return key, salt

    @staticmethod
    def generate_password(length: int = 32) -> str:
        """Generate a secure random password."""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        """Generate an API key."""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"


@dataclass
class AccessPolicy:
    """Access control policy for secrets."""
    allowed_identities: Set[str] = field(default_factory=set)
    allowed_services: Set[str] = field(default_factory=set)
    allowed_ips: Set[str] = field(default_factory=set)
    require_mfa: bool = False
    max_age_seconds: Optional[int] = None

    def is_allowed(
        self,
        identity: Optional[str] = None,
        service: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Check if access is allowed."""
        if self.allowed_identities and identity not in self.allowed_identities:
            return False
        if self.allowed_services and service not in self.allowed_services:
            return False
        if self.allowed_ips and ip_address not in self.allowed_ips:
            return False
        return True


@dataclass
class AccessLog:
    """Log entry for secret access."""
    secret_name: str
    action: str  # read, write, delete, rotate
    identity: str
    service: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
