"""Notification Client for deployment events."""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationEventType(str, Enum):
    """Types of deployment events."""

    DEPLOY_STARTED = "deploy_started"
    DEPLOY_SUCCEEDED = "deploy_succeeded"
    DEPLOY_FAILED = "deploy_failed"
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_COMPLETED = "rollback_completed"
    CANARY_PROMOTED = "canary_promoted"
    CANARY_ROLLED_BACK = "canary_rolled_back"
    HEALTH_DEGRADED = "health_degraded"
    HEALTH_RECOVERED = "health_recovered"
    AB_TEST_STARTED = "ab_test_started"
    AB_TEST_COMPLETED = "ab_test_completed"


@dataclass
class NotificationMessage:
    """A notification message."""

    title: str
    body: str
    event_type: NotificationEventType
    priority: NotificationPriority = NotificationPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Deployment context
    environment: str = ""
    version: str = ""
    service: str = ""
    namespace: str = ""

    # Error context (for failures)
    error_message: str = ""
    error_details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "body": self.body,
            "event_type": self.event_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "environment": self.environment,
            "version": self.version,
            "service": self.service,
            "namespace": self.namespace,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    name: str = "base"

    @abstractmethod
    async def send(self, message: NotificationMessage) -> bool:
        """Send a notification message."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if channel is properly configured."""
        pass


class NotificationClient:
    """Multi-channel notification client."""

    def __init__(self):
        self._channels: Dict[str, NotificationChannel] = {}
        self._event_subscriptions: Dict[NotificationEventType, List[str]] = {}
        self._default_channels: List[str] = []

    def register_channel(
        self,
        channel: NotificationChannel,
        default: bool = False,
    ) -> None:
        """Register a notification channel."""
        if channel.is_configured():
            self._channels[channel.name] = channel
            if default:
                self._default_channels.append(channel.name)
            logger.info(f"Registered notification channel: {channel.name}")
        else:
            logger.warning(f"Channel {channel.name} is not configured, skipping")

    def subscribe(
        self,
        event_type: NotificationEventType,
        channel_name: str,
    ) -> None:
        """Subscribe a channel to an event type."""
        if event_type not in self._event_subscriptions:
            self._event_subscriptions[event_type] = []
        if channel_name not in self._event_subscriptions[event_type]:
            self._event_subscriptions[event_type].append(channel_name)

    def get_channels_for_event(
        self,
        event_type: NotificationEventType,
    ) -> List[NotificationChannel]:
        """Get channels subscribed to an event type."""
        channel_names = self._event_subscriptions.get(event_type, self._default_channels)
        return [
            self._channels[name]
            for name in channel_names
            if name in self._channels
        ]

    async def send(
        self,
        message: NotificationMessage,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Send notification to specified or subscribed channels.

        Args:
            message: Notification message
            channels: Specific channels to use (overrides subscriptions)

        Returns:
            Dict of channel name to success status
        """
        results: Dict[str, bool] = {}

        # Determine which channels to use
        if channels:
            target_channels = [
                self._channels[name]
                for name in channels
                if name in self._channels
            ]
        else:
            target_channels = self.get_channels_for_event(message.event_type)

        if not target_channels:
            logger.warning(f"No channels configured for event {message.event_type}")
            return results

        # Send to all channels concurrently
        tasks = []
        channel_names = []
        for channel in target_channels:
            tasks.append(channel.send(message))
            channel_names.append(channel.name)

        send_results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(channel_names, send_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to send to {name}: {result}")
                results[name] = False
            else:
                results[name] = result

        return results

    async def notify_deploy_started(
        self,
        service: str,
        version: str,
        environment: str,
        **kwargs,
    ) -> Dict[str, bool]:
        """Send deployment started notification."""
        message = NotificationMessage(
            title=f"🚀 Deployment Started: {service}",
            body=f"Deploying version {version} to {environment}",
            event_type=NotificationEventType.DEPLOY_STARTED,
            priority=NotificationPriority.NORMAL,
            service=service,
            version=version,
            environment=environment,
            metadata=kwargs,
        )
        return await self.send(message)

    async def notify_deploy_succeeded(
        self,
        service: str,
        version: str,
        environment: str,
        duration_seconds: int = 0,
        **kwargs,
    ) -> Dict[str, bool]:
        """Send deployment succeeded notification."""
        message = NotificationMessage(
            title=f"✅ Deployment Succeeded: {service}",
            body=f"Version {version} deployed to {environment} in {duration_seconds}s",
            event_type=NotificationEventType.DEPLOY_SUCCEEDED,
            priority=NotificationPriority.NORMAL,
            service=service,
            version=version,
            environment=environment,
            metadata={"duration_seconds": duration_seconds, **kwargs},
        )
        return await self.send(message)

    async def notify_deploy_failed(
        self,
        service: str,
        version: str,
        environment: str,
        error: str,
        **kwargs,
    ) -> Dict[str, bool]:
        """Send deployment failed notification."""
        message = NotificationMessage(
            title=f"❌ Deployment Failed: {service}",
            body=f"Failed to deploy version {version} to {environment}",
            event_type=NotificationEventType.DEPLOY_FAILED,
            priority=NotificationPriority.URGENT,
            service=service,
            version=version,
            environment=environment,
            error_message=error,
            metadata=kwargs,
        )
        return await self.send(message)

    async def notify_rollback(
        self,
        service: str,
        from_version: str,
        to_version: str,
        environment: str,
        reason: str = "",
        **kwargs,
    ) -> Dict[str, bool]:
        """Send rollback notification."""
        message = NotificationMessage(
            title=f"⏪ Rollback: {service}",
            body=f"Rolling back from {from_version} to {to_version} in {environment}. Reason: {reason}",
            event_type=NotificationEventType.ROLLBACK_COMPLETED,
            priority=NotificationPriority.HIGH,
            service=service,
            version=to_version,
            environment=environment,
            metadata={"from_version": from_version, "reason": reason, **kwargs},
        )
        return await self.send(message)

    async def notify_canary_progress(
        self,
        service: str,
        version: str,
        environment: str,
        percentage: int,
        phase: str,
        **kwargs,
    ) -> Dict[str, bool]:
        """Send canary progress notification."""
        message = NotificationMessage(
            title=f"🐤 Canary Progress: {service}",
            body=f"Canary at {percentage}% ({phase}) for version {version}",
            event_type=NotificationEventType.CANARY_PROMOTED,
            priority=NotificationPriority.LOW,
            service=service,
            version=version,
            environment=environment,
            metadata={"percentage": percentage, "phase": phase, **kwargs},
        )
        return await self.send(message)

    async def notify_health_degraded(
        self,
        service: str,
        environment: str,
        health_status: str,
        **kwargs,
    ) -> Dict[str, bool]:
        """Send health degraded notification."""
        message = NotificationMessage(
            title=f"⚠️ Health Degraded: {service}",
            body=f"Service health is {health_status} in {environment}",
            event_type=NotificationEventType.HEALTH_DEGRADED,
            priority=NotificationPriority.HIGH,
            service=service,
            environment=environment,
            metadata={"health_status": health_status, **kwargs},
        )
        return await self.send(message)


# Global client instance
_notification_client: Optional[NotificationClient] = None


def get_notification_client() -> NotificationClient:
    """Get global notification client."""
    global _notification_client
    if _notification_client is None:
        _notification_client = NotificationClient()
    return _notification_client
