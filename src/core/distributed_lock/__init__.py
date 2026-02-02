"""Distributed Lock Module.

Provides distributed locking capabilities:
- Lock primitives with fencing tokens
- In-memory and Redis backends
- Leader election
"""

from src.core.distributed_lock.core import (
    LockStatus,
    LockInfo,
    LockResult,
    FencingTokenGenerator,
    DistributedLock,
    LockManager,
    LockContext,
    generate_owner_id,
)
from src.core.distributed_lock.backends import (
    InMemoryLock,
    RedisLock,
    MultiLock,
)
from src.core.distributed_lock.election import (
    LeaderStatus,
    LeaderInfo,
    LeaderElection,
    LeaderElectionGroup,
)

__all__ = [
    # Core
    "LockStatus",
    "LockInfo",
    "LockResult",
    "FencingTokenGenerator",
    "DistributedLock",
    "LockManager",
    "LockContext",
    "generate_owner_id",
    # Backends
    "InMemoryLock",
    "RedisLock",
    "MultiLock",
    # Election
    "LeaderStatus",
    "LeaderInfo",
    "LeaderElection",
    "LeaderElectionGroup",
]
