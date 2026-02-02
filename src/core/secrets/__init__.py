"""Secrets Management Module.

Provides secure secret management:
- Encrypted storage
- Secret rotation
- Access control
- Multiple backends
"""

from src.core.secrets.core import (
    SecretType,
    SecretMetadata,
    Secret,
    SecretVersion,
    Encryptor,
    AESGCMEncryptor,
    FernetEncryptor,
    KeyDerivation,
    AccessPolicy,
    AccessLog,
)
from src.core.secrets.store import (
    SecretStore,
    InMemorySecretStore,
    EncryptedFileSecretStore,
    EnvironmentSecretStore,
    CompositeSecretStore,
)
from src.core.secrets.manager import (
    RotationResult,
    SecretsManager,
)

__all__ = [
    # Core
    "SecretType",
    "SecretMetadata",
    "Secret",
    "SecretVersion",
    "Encryptor",
    "AESGCMEncryptor",
    "FernetEncryptor",
    "KeyDerivation",
    "AccessPolicy",
    "AccessLog",
    # Store
    "SecretStore",
    "InMemorySecretStore",
    "EncryptedFileSecretStore",
    "EnvironmentSecretStore",
    "CompositeSecretStore",
    # Manager
    "RotationResult",
    "SecretsManager",
]
