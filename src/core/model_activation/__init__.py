"""Phase-A C1 model-activation core.

Reusable controlled-store pin activation: raw-pin domain, store-root-anchored
resolver (openat2 or component-walk), single-file same-fd SHA-256, and
bundle tree-digest-v1 over a service-private freeze.

This package does **not** wire any model family, create production pins,
enable reload/retraining, or implement Track E / Phase B.
"""

from src.core.model_activation.digest import (
    sha256_hex,
    tree_digest_v1,
    tree_digest_v1_from_file_bytes,
)
from src.core.model_activation.pin_domain import validate_raw_pin
from src.core.model_activation.resolver import (
    ResolverImpl,
    default_resolver_impl,
    last_open_impl,
    openat2_available,
)
from src.core.model_activation.store import ControlledStore
from src.core.model_activation.types import (
    ActivationRefusal,
    ArtifactKind,
    BoundPolicy,
    FrozenBundle,
    PinRecord,
    RefusalReason,
    TerminalKind,
)

__all__ = [
    "ActivationRefusal",
    "ArtifactKind",
    "BoundPolicy",
    "ControlledStore",
    "FrozenBundle",
    "PinRecord",
    "RefusalReason",
    "ResolverImpl",
    "TerminalKind",
    "default_resolver_impl",
    "last_open_impl",
    "openat2_available",
    "sha256_hex",
    "tree_digest_v1",
    "tree_digest_v1_from_file_bytes",
    "validate_raw_pin",
]
