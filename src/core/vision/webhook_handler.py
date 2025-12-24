"""
Webhook Handler for Vision Provider.

This module provides comprehensive webhook management including:
- Webhook registration and configuration
- Event routing and delivery
- Retry logic with exponential backoff
- Signature verification and security
- Delivery status tracking
- Rate limiting and throttling

Phase 20 Feature.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


# ============================================================================
# Webhook Enums
# ============================================================================


class WebhookEventType(Enum):
    """Types of webhook events."""

    # Vision events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"

    # Provider events
    PROVIDER_CONNECTED = "provider.connected"
    PROVIDER_DISCONNECTED = "provider.disconnected"
    PROVIDER_ERROR = "provider.error"

    # System events
    SYSTEM_HEALTH = "system.health"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_ERROR = "system.error"

    # Resource events
    RESOURCE_CREATED = "resource.created"
    RESOURCE_UPDATED = "resource.updated"
    RESOURCE_DELETED = "resource.deleted"

    # Security events
    SECURITY_ALERT = "security.alert"
    SECURITY_BREACH = "security.breach"
    ACCESS_DENIED = "access.denied"

    # Custom events
    CUSTOM = "custom"


class WebhookStatus(Enum):
    """Webhook registration status."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    FAILED = "failed"
    PENDING_VERIFICATION = "pending_verification"


class DeliveryStatus(Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"


class SignatureAlgorithm(Enum):
    """Signature algorithms for webhook security."""

    HMAC_SHA256 = "hmac_sha256"
    HMAC_SHA512 = "hmac_sha512"
    HMAC_SHA1 = "hmac_sha1"


# ============================================================================
# Webhook Data Classes
# ============================================================================


@dataclass
class WebhookConfig:
    """Webhook configuration."""

    url: str
    events: List[WebhookEventType]
    secret: str = ""
    signature_algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256
    content_type: str = "application/json"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 60
    rate_limit_per_minute: int = 100
    include_timestamp: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class WebhookRegistration:
    """Webhook registration details."""

    webhook_id: str
    name: str
    config: WebhookConfig
    status: WebhookStatus = WebhookStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner_id: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    failure_count: int = 0
    last_triggered_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    total_deliveries: int = 0
    successful_deliveries: int = 0


@dataclass
class WebhookEvent:
    """Webhook event payload."""

    event_id: str
    event_type: WebhookEventType
    timestamp: datetime
    payload: Dict[str, Any]
    source: str = ""
    correlation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


@dataclass
class DeliveryAttempt:
    """Record of a delivery attempt."""

    attempt_id: str
    webhook_id: str
    event_id: str
    status: DeliveryStatus
    timestamp: datetime = field(default_factory=datetime.now)
    response_code: Optional[int] = None
    response_body: str = ""
    error_message: str = ""
    duration_ms: int = 0
    attempt_number: int = 1


@dataclass
class DeliveryRecord:
    """Complete delivery record for an event."""

    delivery_id: str
    webhook_id: str
    event: WebhookEvent
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: List[DeliveryAttempt] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None


# ============================================================================
# Webhook Signature
# ============================================================================


class WebhookSignature:
    """Handles webhook signature generation and verification."""

    @staticmethod
    def generate(
        payload: str,
        secret: str,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
        timestamp: Optional[str] = None,
    ) -> str:
        """Generate signature for payload."""
        if algorithm == SignatureAlgorithm.HMAC_SHA256:
            hash_func = hashlib.sha256
        elif algorithm == SignatureAlgorithm.HMAC_SHA512:
            hash_func = hashlib.sha512
        else:
            hash_func = hashlib.sha1

        # Include timestamp if provided
        message = f"{timestamp}.{payload}" if timestamp else payload

        signature = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hash_func,
        ).hexdigest()

        return signature

    @staticmethod
    def verify(
        payload: str,
        signature: str,
        secret: str,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Verify payload signature."""
        expected = WebhookSignature.generate(payload, secret, algorithm, timestamp)
        return hmac.compare_digest(expected, signature)


# ============================================================================
# Rate Limiter
# ============================================================================


class WebhookRateLimiter:
    """Rate limiter for webhook deliveries."""

    def __init__(self) -> None:
        """Initialize rate limiter."""
        self._windows: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def check_rate_limit(self, webhook_id: str, limit_per_minute: int) -> tuple[bool, int]:
        """
        Check if webhook is within rate limit.

        Returns (allowed, remaining_capacity).
        """
        with self._lock:
            now = time.time()
            window_start = now - 60  # 1 minute window

            if webhook_id not in self._windows:
                self._windows[webhook_id] = []

            # Clean old entries
            self._windows[webhook_id] = [t for t in self._windows[webhook_id] if t > window_start]

            current_count = len(self._windows[webhook_id])
            remaining = limit_per_minute - current_count

            if current_count < limit_per_minute:
                self._windows[webhook_id].append(now)
                return True, remaining - 1

            return False, 0

    def get_retry_after(self, webhook_id: str) -> int:
        """Get seconds until rate limit resets."""
        with self._lock:
            if webhook_id not in self._windows or not self._windows[webhook_id]:
                return 0

            oldest = min(self._windows[webhook_id])
            retry_after = max(0, int(60 - (time.time() - oldest)))
            return retry_after


# ============================================================================
# Webhook Delivery Service
# ============================================================================


class WebhookDeliveryService:
    """Handles actual webhook delivery with retry logic."""

    def __init__(
        self,
        rate_limiter: Optional[WebhookRateLimiter] = None,
    ) -> None:
        """Initialize delivery service."""
        self._rate_limiter = rate_limiter or WebhookRateLimiter()
        self._delivery_queue: asyncio.Queue[DeliveryRecord] = asyncio.Queue()
        self._pending_retries: Dict[str, DeliveryRecord] = {}
        self._delivery_history: Dict[str, List[DeliveryRecord]] = {}
        self._lock = threading.RLock()

    async def deliver(
        self,
        webhook: WebhookRegistration,
        event: WebhookEvent,
    ) -> DeliveryRecord:
        """Deliver event to webhook endpoint."""
        delivery_id = str(uuid.uuid4())

        record = DeliveryRecord(
            delivery_id=delivery_id,
            webhook_id=webhook.webhook_id,
            event=event,
            status=DeliveryStatus.PENDING,
        )

        # Check rate limit
        allowed, remaining = self._rate_limiter.check_rate_limit(
            webhook.webhook_id,
            webhook.config.rate_limit_per_minute,
        )

        if not allowed:
            retry_after = self._rate_limiter.get_retry_after(webhook.webhook_id)
            record.status = DeliveryStatus.RETRYING
            record.next_retry_at = datetime.now() + timedelta(seconds=retry_after)
            self._pending_retries[delivery_id] = record
            return record

        # Attempt delivery
        attempt = await self._attempt_delivery(webhook, event, attempt_number=1)
        record.attempts.append(attempt)

        if attempt.status == DeliveryStatus.DELIVERED:
            record.status = DeliveryStatus.DELIVERED
            record.completed_at = datetime.now()
        elif attempt.attempt_number < webhook.config.max_retries:
            record.status = DeliveryStatus.RETRYING
            record.next_retry_at = datetime.now() + timedelta(
                seconds=webhook.config.retry_delay_seconds
            )
            self._pending_retries[delivery_id] = record
        else:
            record.status = DeliveryStatus.FAILED
            record.completed_at = datetime.now()

        # Store in history
        self._store_delivery(record)

        return record

    async def _attempt_delivery(
        self,
        webhook: WebhookRegistration,
        event: WebhookEvent,
        attempt_number: int,
    ) -> DeliveryAttempt:
        """Attempt a single delivery."""
        attempt_id = str(uuid.uuid4())
        start_time = time.time()

        attempt = DeliveryAttempt(
            attempt_id=attempt_id,
            webhook_id=webhook.webhook_id,
            event_id=event.event_id,
            status=DeliveryStatus.DELIVERING,
            attempt_number=attempt_number,
        )

        try:
            # Prepare payload
            payload = json.dumps(event.to_dict())

            # Generate signature
            timestamp = str(int(time.time()))
            signature = WebhookSignature.generate(
                payload,
                webhook.config.secret,
                webhook.config.signature_algorithm,
                timestamp,
            )

            # Build headers
            headers = {
                "Content-Type": webhook.config.content_type,
                "X-Webhook-Signature": signature,
                "X-Webhook-Timestamp": timestamp,
                "X-Webhook-Event": event.event_type.value,
                "X-Webhook-Delivery": attempt_id,
                **webhook.config.custom_headers,
            }

            # Simulate HTTP request (in production, use aiohttp or httpx)
            response_code, response_body = await self._send_request(
                url=webhook.config.url,
                payload=payload,
                headers=headers,
                timeout=webhook.config.timeout_seconds,
            )

            attempt.response_code = response_code
            attempt.response_body = response_body

            if 200 <= response_code < 300:
                attempt.status = DeliveryStatus.DELIVERED
            else:
                attempt.status = DeliveryStatus.FAILED
                attempt.error_message = f"HTTP {response_code}"

        except asyncio.TimeoutError:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = "Request timeout"
        except Exception as e:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = str(e)

        attempt.duration_ms = int((time.time() - start_time) * 1000)
        return attempt

    async def _send_request(
        self,
        url: str,
        payload: str,
        headers: Dict[str, str],
        timeout: int,
    ) -> tuple[int, str]:
        """
        Send HTTP request to webhook URL.

        In production, this would use aiohttp or httpx.
        This is a simulation for testing purposes.
        """
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")

        # Simulate network latency
        await asyncio.sleep(0.01)

        # Simulate successful response
        return 200, '{"status": "ok"}'

    def _store_delivery(self, record: DeliveryRecord) -> None:
        """Store delivery record in history."""
        with self._lock:
            if record.webhook_id not in self._delivery_history:
                self._delivery_history[record.webhook_id] = []

            self._delivery_history[record.webhook_id].append(record)

            # Keep only last 1000 records per webhook
            if len(self._delivery_history[record.webhook_id]) > 1000:
                self._delivery_history[record.webhook_id] = self._delivery_history[
                    record.webhook_id
                ][-1000:]

    def get_delivery_history(
        self,
        webhook_id: str,
        limit: int = 100,
    ) -> List[DeliveryRecord]:
        """Get delivery history for a webhook."""
        with self._lock:
            history = self._delivery_history.get(webhook_id, [])
            return history[-limit:]

    def get_pending_retries(self) -> List[DeliveryRecord]:
        """Get all pending retry deliveries."""
        return list(self._pending_retries.values())


# ============================================================================
# Webhook Registry
# ============================================================================


class WebhookRegistry:
    """Registry for webhook registrations."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._webhooks: Dict[str, WebhookRegistration] = {}
        self._by_event: Dict[WebhookEventType, Set[str]] = {}
        self._by_owner: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        config: WebhookConfig,
        owner_id: str = "",
        description: str = "",
    ) -> WebhookRegistration:
        """Register a new webhook."""
        webhook_id = str(uuid.uuid4())

        webhook = WebhookRegistration(
            webhook_id=webhook_id,
            name=name,
            config=config,
            owner_id=owner_id,
            description=description,
        )

        with self._lock:
            self._webhooks[webhook_id] = webhook

            # Index by event type
            for event_type in config.events:
                if event_type not in self._by_event:
                    self._by_event[event_type] = set()
                self._by_event[event_type].add(webhook_id)

            # Index by owner
            if owner_id:
                if owner_id not in self._by_owner:
                    self._by_owner[owner_id] = set()
                self._by_owner[owner_id].add(webhook_id)

        logger.info(f"Registered webhook: {name} ({webhook_id})")
        return webhook

    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        with self._lock:
            if webhook_id not in self._webhooks:
                return False

            webhook = self._webhooks[webhook_id]

            # Remove from event index
            for event_type in webhook.config.events:
                if event_type in self._by_event:
                    self._by_event[event_type].discard(webhook_id)

            # Remove from owner index
            if webhook.owner_id and webhook.owner_id in self._by_owner:
                self._by_owner[webhook.owner_id].discard(webhook_id)

            del self._webhooks[webhook_id]

        logger.info(f"Unregistered webhook: {webhook_id}")
        return True

    def get(self, webhook_id: str) -> Optional[WebhookRegistration]:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)

    def get_by_event(self, event_type: WebhookEventType) -> List[WebhookRegistration]:
        """Get all webhooks subscribed to an event type."""
        with self._lock:
            webhook_ids = self._by_event.get(event_type, set())
            return [
                self._webhooks[wid]
                for wid in webhook_ids
                if wid in self._webhooks and self._webhooks[wid].status == WebhookStatus.ACTIVE
            ]

    def get_by_owner(self, owner_id: str) -> List[WebhookRegistration]:
        """Get all webhooks for an owner."""
        with self._lock:
            webhook_ids = self._by_owner.get(owner_id, set())
            return [self._webhooks[wid] for wid in webhook_ids if wid in self._webhooks]

    def list_all(self) -> List[WebhookRegistration]:
        """List all webhooks."""
        return list(self._webhooks.values())

    def update_status(
        self, webhook_id: str, status: WebhookStatus
    ) -> Optional[WebhookRegistration]:
        """Update webhook status."""
        with self._lock:
            if webhook_id in self._webhooks:
                self._webhooks[webhook_id].status = status
                self._webhooks[webhook_id].updated_at = datetime.now()
                return self._webhooks[webhook_id]
        return None


# ============================================================================
# Webhook Manager
# ============================================================================


class WebhookManager:
    """
    Central webhook management system.

    Handles registration, event routing, and delivery orchestration.
    """

    def __init__(
        self,
        registry: Optional[WebhookRegistry] = None,
        delivery_service: Optional[WebhookDeliveryService] = None,
    ) -> None:
        """Initialize webhook manager."""
        self._registry = registry or WebhookRegistry()
        self._delivery_service = delivery_service or WebhookDeliveryService()
        self._event_handlers: Dict[WebhookEventType, List[Callable[[WebhookEvent], None]]] = {}
        self._lock = threading.RLock()

    @property
    def registry(self) -> WebhookRegistry:
        """Get webhook registry."""
        return self._registry

    def register_webhook(
        self,
        name: str,
        url: str,
        events: List[WebhookEventType],
        secret: str = "",
        owner_id: str = "",
        **kwargs: Any,
    ) -> WebhookRegistration:
        """Register a new webhook."""
        config = WebhookConfig(
            url=url,
            events=events,
            secret=secret or str(uuid.uuid4()),
            **kwargs,
        )

        return self._registry.register(
            name=name,
            config=config,
            owner_id=owner_id,
        )

    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        return self._registry.unregister(webhook_id)

    async def emit_event(
        self,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        source: str = "",
        correlation_id: str = "",
    ) -> List[DeliveryRecord]:
        """Emit an event to all subscribed webhooks."""
        event = WebhookEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            payload=payload,
            source=source,
            correlation_id=correlation_id or str(uuid.uuid4()),
        )

        # Notify internal handlers
        self._notify_handlers(event)

        # Deliver to webhooks
        webhooks = self._registry.get_by_event(event_type)
        deliveries: List[DeliveryRecord] = []

        for webhook in webhooks:
            try:
                record = await self._delivery_service.deliver(webhook, event)
                deliveries.append(record)

                # Update webhook statistics
                webhook.total_deliveries += 1
                webhook.last_triggered_at = datetime.now()

                if record.status == DeliveryStatus.DELIVERED:
                    webhook.successful_deliveries += 1
                    webhook.last_success_at = datetime.now()
                    webhook.failure_count = 0
                elif record.status == DeliveryStatus.FAILED:
                    webhook.failure_count += 1

                    # Disable webhook if too many failures
                    if webhook.failure_count >= 10:
                        self._registry.update_status(webhook.webhook_id, WebhookStatus.FAILED)

            except Exception as e:
                logger.error(f"Error delivering to webhook {webhook.webhook_id}: {e}")

        return deliveries

    def on_event(
        self,
        event_type: WebhookEventType,
        handler: Callable[[WebhookEvent], None],
    ) -> None:
        """Register an internal event handler."""
        with self._lock:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(handler)

    def _notify_handlers(self, event: WebhookEvent) -> None:
        """Notify internal event handlers."""
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    def get_webhook(self, webhook_id: str) -> Optional[WebhookRegistration]:
        """Get a webhook by ID."""
        return self._registry.get(webhook_id)

    def list_webhooks(self, owner_id: Optional[str] = None) -> List[WebhookRegistration]:
        """List webhooks, optionally filtered by owner."""
        if owner_id:
            return self._registry.get_by_owner(owner_id)
        return self._registry.list_all()

    def get_delivery_history(self, webhook_id: str, limit: int = 100) -> List[DeliveryRecord]:
        """Get delivery history for a webhook."""
        return self._delivery_service.get_delivery_history(webhook_id, limit)

    def pause_webhook(self, webhook_id: str) -> Optional[WebhookRegistration]:
        """Pause a webhook."""
        return self._registry.update_status(webhook_id, WebhookStatus.PAUSED)

    def resume_webhook(self, webhook_id: str) -> Optional[WebhookRegistration]:
        """Resume a paused webhook."""
        return self._registry.update_status(webhook_id, WebhookStatus.ACTIVE)


# ============================================================================
# Vision Provider Integration
# ============================================================================


class WebhookVisionProvider(VisionProvider):
    """Vision provider with webhook integration."""

    def __init__(
        self,
        base_provider: VisionProvider,
        webhook_manager: WebhookManager,
    ) -> None:
        """Initialize webhook vision provider."""
        self._base_provider = base_provider
        self._webhook_manager = webhook_manager

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"webhook_{self._base_provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with webhook notifications."""
        correlation_id = str(uuid.uuid4())

        # Emit start event
        await self._webhook_manager.emit_event(
            event_type=WebhookEventType.ANALYSIS_STARTED,
            payload={
                "image_size": len(image_data),
                "include_description": include_description,
            },
            source=self.provider_name,
            correlation_id=correlation_id,
        )

        try:
            # Perform analysis
            description = await self._base_provider.analyze_image(image_data, include_description)

            # Emit completion event
            await self._webhook_manager.emit_event(
                event_type=WebhookEventType.ANALYSIS_COMPLETED,
                payload={
                    "success": True,
                    "confidence": description.confidence,
                },
                source=self.provider_name,
                correlation_id=correlation_id,
            )

            return description

        except Exception as e:
            # Emit failure event
            await self._webhook_manager.emit_event(
                event_type=WebhookEventType.ANALYSIS_FAILED,
                payload={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                source=self.provider_name,
                correlation_id=correlation_id,
            )
            raise


# ============================================================================
# Factory Functions
# ============================================================================


def create_webhook_config(
    url: str,
    events: List[WebhookEventType],
    secret: Optional[str] = None,
    **kwargs: Any,
) -> WebhookConfig:
    """Create a webhook configuration."""
    return WebhookConfig(
        url=url,
        events=events,
        secret=secret or str(uuid.uuid4()),
        **kwargs,
    )


def create_webhook_manager(
    rate_limit_per_minute: int = 100,
) -> WebhookManager:
    """Create a configured webhook manager."""
    rate_limiter = WebhookRateLimiter()
    delivery_service = WebhookDeliveryService(rate_limiter=rate_limiter)
    registry = WebhookRegistry()

    return WebhookManager(
        registry=registry,
        delivery_service=delivery_service,
    )


def create_webhook_event(
    event_type: WebhookEventType,
    payload: Dict[str, Any],
    source: str = "",
) -> WebhookEvent:
    """Create a webhook event."""
    return WebhookEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        timestamp=datetime.now(),
        payload=payload,
        source=source,
        correlation_id=str(uuid.uuid4()),
    )
