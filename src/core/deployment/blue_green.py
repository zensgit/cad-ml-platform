"""Blue-Green Deployment Manager.

Manages two identical production environments (blue and green) for
zero-downtime deployments with instant rollback capability.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SlotStatus(str, Enum):
    """Deployment slot status."""

    ACTIVE = "active"  # Receiving production traffic
    STANDBY = "standby"  # Ready but not receiving traffic
    DEPLOYING = "deploying"  # Deployment in progress
    UNHEALTHY = "unhealthy"  # Health checks failing
    DRAINING = "draining"  # Draining existing connections


@dataclass
class DeploymentSlot:
    """Represents a deployment slot (blue or green)."""

    name: str  # "blue" or "green"
    status: SlotStatus = SlotStatus.STANDBY
    version: str = ""
    deployed_at: Optional[datetime] = None
    health_check_url: str = "/health"
    replicas: int = 1
    ready_replicas: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if slot is healthy."""
        return self.ready_replicas >= self.replicas and self.status not in (
            SlotStatus.UNHEALTHY,
            SlotStatus.DEPLOYING,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "version": self.version,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "health_check_url": self.health_check_url,
            "replicas": self.replicas,
            "ready_replicas": self.ready_replicas,
            "is_healthy": self.is_healthy(),
            "metadata": self.metadata,
        }


@dataclass
class DeploymentEvent:
    """Records a deployment event."""

    timestamp: datetime
    event_type: str  # deploy, switch, rollback, health_check
    slot: str
    version: str
    success: bool
    message: str = ""
    duration_ms: int = 0


class BlueGreenManager:
    """Manages blue-green deployments.

    Features:
    - Zero-downtime deployments
    - Instant rollback
    - Health check validation
    - Traffic switching
    - Deployment history
    """

    def __init__(
        self,
        health_check_fn: Optional[Callable[[str], bool]] = None,
        switch_traffic_fn: Optional[Callable[[str], bool]] = None,
        deploy_fn: Optional[Callable[[str, str], bool]] = None,
    ):
        """Initialize Blue-Green manager.

        Args:
            health_check_fn: Function to check slot health (slot_name) -> healthy
            switch_traffic_fn: Function to switch traffic (slot_name) -> success
            deploy_fn: Function to deploy version (slot_name, version) -> success
        """
        self.blue = DeploymentSlot(name="blue")
        self.green = DeploymentSlot(name="green")
        self._active_slot = "blue"
        self._history: List[DeploymentEvent] = []
        self._lock: Optional[asyncio.Lock] = None  # Lazy init for Python 3.9

        # Callbacks (can be replaced with actual K8s/cloud operations)
        self._health_check_fn = health_check_fn or self._default_health_check
        self._switch_traffic_fn = switch_traffic_fn or self._default_switch_traffic
        self._deploy_fn = deploy_fn or self._default_deploy

    def _get_lock(self) -> asyncio.Lock:
        """Lazy init lock for Python 3.9 compatibility."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def active_slot(self) -> DeploymentSlot:
        """Get the currently active slot."""
        return self.blue if self._active_slot == "blue" else self.green

    @property
    def standby_slot(self) -> DeploymentSlot:
        """Get the standby slot."""
        return self.green if self._active_slot == "blue" else self.blue

    def _get_slot(self, name: str) -> DeploymentSlot:
        """Get slot by name."""
        if name == "blue":
            return self.blue
        elif name == "green":
            return self.green
        raise ValueError(f"Unknown slot: {name}")

    def _default_health_check(self, slot_name: str) -> bool:
        """Default health check (always returns True)."""
        slot = self._get_slot(slot_name)
        return slot.is_healthy()

    def _default_switch_traffic(self, slot_name: str) -> bool:
        """Default traffic switch (just updates internal state)."""
        logger.info(f"Switching traffic to {slot_name}")
        return True

    def _default_deploy(self, slot_name: str, version: str) -> bool:
        """Default deploy (just updates internal state)."""
        logger.info(f"Deploying {version} to {slot_name}")
        return True

    def _record_event(
        self,
        event_type: str,
        slot: str,
        version: str,
        success: bool,
        message: str = "",
        duration_ms: int = 0,
    ) -> None:
        """Record a deployment event."""
        event = DeploymentEvent(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            slot=slot,
            version=version,
            success=success,
            message=message,
            duration_ms=duration_ms,
        )
        self._history.append(event)
        # Keep only last 100 events
        if len(self._history) > 100:
            self._history = self._history[-100:]

    async def deploy(
        self,
        version: str,
        auto_switch: bool = False,
        health_check_retries: int = 3,
        health_check_interval: float = 10.0,
    ) -> bool:
        """Deploy a new version to the standby slot.

        Args:
            version: Version to deploy
            auto_switch: Automatically switch traffic after successful deploy
            health_check_retries: Number of health check retries
            health_check_interval: Seconds between health checks

        Returns:
            True if deployment successful
        """
        async with self._get_lock():
            start_time = time.time()
            target_slot = self.standby_slot

            logger.info(f"Starting deployment of {version} to {target_slot.name}")
            target_slot.status = SlotStatus.DEPLOYING

            try:
                # Perform deployment
                if not self._deploy_fn(target_slot.name, version):
                    target_slot.status = SlotStatus.UNHEALTHY
                    self._record_event(
                        "deploy",
                        target_slot.name,
                        version,
                        False,
                        "Deployment failed",
                    )
                    return False

                target_slot.version = version
                target_slot.deployed_at = datetime.utcnow()

                # Wait for health checks
                healthy = False
                for i in range(health_check_retries):
                    await asyncio.sleep(health_check_interval)
                    if self._health_check_fn(target_slot.name):
                        healthy = True
                        break
                    logger.warning(
                        f"Health check {i+1}/{health_check_retries} failed for {target_slot.name}"
                    )

                if not healthy:
                    target_slot.status = SlotStatus.UNHEALTHY
                    duration = int((time.time() - start_time) * 1000)
                    self._record_event(
                        "deploy",
                        target_slot.name,
                        version,
                        False,
                        "Health checks failed",
                        duration,
                    )
                    return False

                target_slot.status = SlotStatus.STANDBY
                target_slot.ready_replicas = target_slot.replicas
                duration = int((time.time() - start_time) * 1000)
                self._record_event(
                    "deploy",
                    target_slot.name,
                    version,
                    True,
                    "Deployment successful",
                    duration,
                )

                # Auto-switch if requested
                if auto_switch:
                    return await self.switch_traffic()

                return True

            except Exception as e:
                target_slot.status = SlotStatus.UNHEALTHY
                duration = int((time.time() - start_time) * 1000)
                self._record_event(
                    "deploy",
                    target_slot.name,
                    version,
                    False,
                    str(e),
                    duration,
                )
                logger.error(f"Deployment failed: {e}")
                return False

    async def switch_traffic(self) -> bool:
        """Switch traffic from active to standby slot.

        Returns:
            True if switch successful
        """
        async with self._get_lock():
            start_time = time.time()
            current = self.active_slot
            target = self.standby_slot

            logger.info(f"Switching traffic from {current.name} to {target.name}")

            # Verify target is healthy
            if not target.is_healthy():
                self._record_event(
                    "switch",
                    target.name,
                    target.version,
                    False,
                    "Target slot not healthy",
                )
                return False

            try:
                # Drain current slot
                current.status = SlotStatus.DRAINING

                # Switch traffic
                if not self._switch_traffic_fn(target.name):
                    current.status = SlotStatus.ACTIVE
                    self._record_event(
                        "switch",
                        target.name,
                        target.version,
                        False,
                        "Traffic switch failed",
                    )
                    return False

                # Update statuses
                target.status = SlotStatus.ACTIVE
                current.status = SlotStatus.STANDBY
                self._active_slot = target.name

                duration = int((time.time() - start_time) * 1000)
                self._record_event(
                    "switch",
                    target.name,
                    target.version,
                    True,
                    f"Switched from {current.name}",
                    duration,
                )

                logger.info(f"Traffic switched to {target.name} ({target.version})")
                return True

            except Exception as e:
                current.status = SlotStatus.ACTIVE
                self._record_event(
                    "switch",
                    target.name,
                    target.version,
                    False,
                    str(e),
                )
                logger.error(f"Traffic switch failed: {e}")
                return False

    async def rollback(self) -> bool:
        """Rollback to the previous version (standby slot).

        Returns:
            True if rollback successful
        """
        logger.info("Initiating rollback")
        return await self.switch_traffic()

    def get_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "active_slot": self._active_slot,
            "blue": self.blue.to_dict(),
            "green": self.green.to_dict(),
            "can_switch": self.standby_slot.is_healthy(),
            "can_rollback": self.standby_slot.is_healthy(),
        }

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get deployment history."""
        events = self._history[-limit:]
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type,
                "slot": e.slot,
                "version": e.version,
                "success": e.success,
                "message": e.message,
                "duration_ms": e.duration_ms,
            }
            for e in reversed(events)
        ]


# Global manager instance
_manager: Optional[BlueGreenManager] = None


def get_blue_green_manager() -> BlueGreenManager:
    """Get global Blue-Green manager."""
    global _manager
    if _manager is None:
        _manager = BlueGreenManager()
    return _manager
