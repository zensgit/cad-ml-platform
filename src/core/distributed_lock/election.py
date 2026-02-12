"""Leader Election.

Provides leader election for distributed systems:
- Single leader election
- Leader heartbeat
- Failover handling
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core.distributed_lock.core import DistributedLock, LockInfo

logger = logging.getLogger(__name__)


class LeaderStatus(Enum):
    """Status of a node in leader election."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


@dataclass
class LeaderInfo:
    """Information about the current leader."""
    leader_id: str
    elected_at: datetime
    lease_expires_at: datetime
    term: int
    metadata: Dict[str, Any]

    @property
    def is_lease_valid(self) -> bool:
        return datetime.utcnow() < self.lease_expires_at


class LeaderElection:
    """Leader election using distributed locks."""

    def __init__(
        self,
        lock: DistributedLock,
        election_name: str,
        node_id: Optional[str] = None,
        lease_duration_seconds: float = 30.0,
        renew_interval_seconds: float = 10.0,
    ):
        self._lock = lock
        self._election_name = election_name
        self._node_id = node_id or f"node_{secrets.token_hex(8)}"
        self._lease_duration = lease_duration_seconds
        self._renew_interval = renew_interval_seconds

        self._status = LeaderStatus.UNKNOWN
        self._term = 0
        self._leader_info: Optional[LeaderInfo] = None
        self._running = False
        self._renew_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_elected: Optional[Callable[[], None]] = None
        self._on_demoted: Optional[Callable[[], None]] = None
        self._on_leader_change: Optional[Callable[[Optional[str]], None]] = None

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def status(self) -> LeaderStatus:
        return self._status

    @property
    def is_leader(self) -> bool:
        return self._status == LeaderStatus.LEADER

    @property
    def leader_info(self) -> Optional[LeaderInfo]:
        return self._leader_info

    def on_elected(self, callback: Callable[[], None]) -> "LeaderElection":
        """Set callback for when this node becomes leader."""
        self._on_elected = callback
        return self

    def on_demoted(self, callback: Callable[[], None]) -> "LeaderElection":
        """Set callback for when this node loses leadership."""
        self._on_demoted = callback
        return self

    def on_leader_change(
        self,
        callback: Callable[[Optional[str]], None],
    ) -> "LeaderElection":
        """Set callback for leader changes."""
        self._on_leader_change = callback
        return self

    async def start(self) -> None:
        """Start participating in leader election."""
        if self._running:
            return

        self._running = True
        self._status = LeaderStatus.CANDIDATE
        logger.info(f"Node {self._node_id} starting leader election for '{self._election_name}'")

        # Start election loop
        self._renew_task = asyncio.create_task(self._election_loop())

    async def stop(self) -> None:
        """Stop participating in leader election."""
        self._running = False

        if self._renew_task:
            self._renew_task.cancel()
            try:
                await self._renew_task
            except asyncio.CancelledError:
                pass

        # Release leadership if we have it
        if self._status == LeaderStatus.LEADER:
            await self._release_leadership()

        self._status = LeaderStatus.UNKNOWN
        logger.info(f"Node {self._node_id} stopped leader election")

    async def _election_loop(self) -> None:
        """Main election loop."""
        while self._running:
            try:
                if self._status == LeaderStatus.LEADER:
                    # Try to renew lease
                    renewed = await self._renew_lease()
                    if not renewed:
                        await self._handle_demotion()
                else:
                    # Try to become leader
                    elected = await self._try_become_leader()
                    if elected:
                        await self._handle_election()

                # Update leader info
                await self._update_leader_info()

            except Exception as e:
                logger.error(f"Election loop error: {e}")

            await asyncio.sleep(self._renew_interval)

    async def _try_become_leader(self) -> bool:
        """Try to acquire leadership."""
        result = await self._lock.acquire(
            name=self._election_name,
            owner=self._node_id,
            ttl_seconds=self._lease_duration,
            metadata={"term": self._term + 1},
        )

        return result.success

    async def _renew_lease(self) -> bool:
        """Renew leadership lease."""
        return await self._lock.extend(
            name=self._election_name,
            owner=self._node_id,
            additional_seconds=self._lease_duration,
        )

    async def _release_leadership(self) -> None:
        """Release leadership."""
        await self._lock.release(self._election_name, self._node_id)

    async def _handle_election(self) -> None:
        """Handle becoming leader."""
        self._term += 1
        self._status = LeaderStatus.LEADER
        logger.info(f"Node {self._node_id} elected as leader (term {self._term})")

        if self._on_elected:
            try:
                if asyncio.iscoroutinefunction(self._on_elected):
                    await self._on_elected()
                else:
                    self._on_elected()
            except Exception as e:
                logger.error(f"on_elected callback error: {e}")

    async def _handle_demotion(self) -> None:
        """Handle losing leadership."""
        self._status = LeaderStatus.FOLLOWER
        logger.info(f"Node {self._node_id} demoted from leader")

        if self._on_demoted:
            try:
                if asyncio.iscoroutinefunction(self._on_demoted):
                    await self._on_demoted()
                else:
                    self._on_demoted()
            except Exception as e:
                logger.error(f"on_demoted callback error: {e}")

    async def _update_leader_info(self) -> None:
        """Update information about current leader."""
        lock_info = await self._lock.get_info(self._election_name)

        if lock_info:
            old_leader = self._leader_info.leader_id if self._leader_info else None
            new_leader = lock_info.owner

            self._leader_info = LeaderInfo(
                leader_id=lock_info.owner,
                elected_at=lock_info.acquired_at,
                lease_expires_at=lock_info.expires_at,
                term=lock_info.metadata.get("term", 0),
                metadata=lock_info.metadata,
            )

            if old_leader != new_leader and self._on_leader_change:
                try:
                    if asyncio.iscoroutinefunction(self._on_leader_change):
                        await self._on_leader_change(new_leader)
                    else:
                        self._on_leader_change(new_leader)
                except Exception as e:
                    logger.error(f"on_leader_change callback error: {e}")
        else:
            self._leader_info = None

    async def get_current_leader(self) -> Optional[str]:
        """Get current leader ID."""
        lock_info = await self._lock.get_info(self._election_name)
        return lock_info.owner if lock_info else None


class LeaderElectionGroup:
    """Manage multiple leader elections."""

    def __init__(self, lock: DistributedLock):
        self._lock = lock
        self._elections: Dict[str, LeaderElection] = {}

    def create_election(
        self,
        name: str,
        node_id: Optional[str] = None,
        lease_duration_seconds: float = 30.0,
    ) -> LeaderElection:
        """Create a new leader election."""
        if name in self._elections:
            return self._elections[name]

        election = LeaderElection(
            lock=self._lock,
            election_name=name,
            node_id=node_id,
            lease_duration_seconds=lease_duration_seconds,
        )
        self._elections[name] = election
        return election

    async def start_all(self) -> None:
        """Start all elections."""
        for election in self._elections.values():
            await election.start()

    async def stop_all(self) -> None:
        """Stop all elections."""
        for election in self._elections.values():
            await election.stop()

    def get_leaders(self) -> Dict[str, Optional[str]]:
        """Get all current leaders."""
        return {
            name: election.leader_info.leader_id if election.leader_info else None
            for name, election in self._elections.items()
        }
