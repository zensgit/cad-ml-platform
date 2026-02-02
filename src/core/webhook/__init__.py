"""Webhook Module.

Provides webhook infrastructure:
- Outbound webhook delivery
- Retry with backoff
- Signature verification
- Event filtering
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class WebhookEventType(Enum):
    """Standard webhook event types."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    CUSTOM = "custom"


@dataclass
class WebhookEvent:
    """A webhook event."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    source: str = ""
    subject: Optional[str] = None
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "subject": self.subject,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class WebhookEndpoint:
    """A webhook endpoint configuration."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    secret: Optional[str] = None
    events: List[str] = field(default_factory=list)  # Event types to subscribe
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Delivery settings
    timeout_seconds: float = 30.0
    max_retries: int = 5
    retry_delay: float = 60.0  # Initial delay in seconds

    def matches_event(self, event_type: str) -> bool:
        """Check if endpoint subscribes to event type."""
        if not self.events:  # Empty = all events
            return True
        return event_type in self.events or "*" in self.events


@dataclass
class WebhookDelivery:
    """A webhook delivery attempt."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    endpoint_id: str = ""
    event_id: str = ""
    status: WebhookStatus = WebhookStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    next_retry: Optional[datetime] = None
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "endpoint_id": self.endpoint_id,
            "event_id": self.event_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt.isoformat() if self.last_attempt else None,
            "response_status": self.response_status,
            "error": self.error,
        }


class WebhookSigner:
    """Signs webhook payloads for verification."""

    def __init__(self, algorithm: str = "sha256"):
        self._algorithm = algorithm

    def sign(self, payload: str, secret: str) -> str:
        """Create signature for payload."""
        signature = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            getattr(hashlib, self._algorithm),
        ).hexdigest()
        return f"{self._algorithm}={signature}"

    def verify(self, payload: str, secret: str, signature: str) -> bool:
        """Verify payload signature."""
        expected = self.sign(payload, secret)
        return hmac.compare_digest(expected, signature)


class WebhookDeliveryStore(ABC):
    """Abstract delivery storage."""

    @abstractmethod
    async def save(self, delivery: WebhookDelivery) -> bool:
        """Save delivery."""
        pass

    @abstractmethod
    async def get(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery by ID."""
        pass

    @abstractmethod
    async def get_pending(self) -> List[WebhookDelivery]:
        """Get pending deliveries ready for retry."""
        pass

    @abstractmethod
    async def get_by_event(self, event_id: str) -> List[WebhookDelivery]:
        """Get deliveries for an event."""
        pass


class InMemoryDeliveryStore(WebhookDeliveryStore):
    """In-memory delivery storage."""

    def __init__(self, max_entries: int = 10000):
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._max_entries = max_entries

    async def save(self, delivery: WebhookDelivery) -> bool:
        self._deliveries[delivery.id] = delivery

        # Cleanup old entries
        if len(self._deliveries) > self._max_entries:
            sorted_deliveries = sorted(
                self._deliveries.items(),
                key=lambda x: x[1].created_at,
            )
            to_remove = len(self._deliveries) - self._max_entries
            for key, _ in sorted_deliveries[:to_remove]:
                del self._deliveries[key]

        return True

    async def get(self, delivery_id: str) -> Optional[WebhookDelivery]:
        return self._deliveries.get(delivery_id)

    async def get_pending(self) -> List[WebhookDelivery]:
        now = datetime.utcnow()
        return [
            d for d in self._deliveries.values()
            if d.status in (WebhookStatus.PENDING, WebhookStatus.RETRYING)
            and (d.next_retry is None or d.next_retry <= now)
        ]

    async def get_by_event(self, event_id: str) -> List[WebhookDelivery]:
        return [d for d in self._deliveries.values() if d.event_id == event_id]


class WebhookClient(ABC):
    """Abstract HTTP client for webhook delivery."""

    @abstractmethod
    async def post(
        self,
        url: str,
        payload: str,
        headers: Dict[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        """Send POST request. Returns (status_code, body)."""
        pass


class SimpleWebhookClient(WebhookClient):
    """Simple webhook client using aiohttp if available."""

    async def post(
        self,
        url: str,
        payload: str,
        headers: Dict[str, str],
        timeout: float,
    ) -> tuple[int, str]:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    body = await response.text()
                    return response.status, body

        except ImportError:
            # Fallback: simulate success for testing
            logger.warning("aiohttp not available, simulating webhook delivery")
            return 200, '{"status": "ok"}'


class WebhookDispatcher:
    """Dispatches webhooks to endpoints."""

    def __init__(
        self,
        client: Optional[WebhookClient] = None,
        store: Optional[WebhookDeliveryStore] = None,
        signer: Optional[WebhookSigner] = None,
    ):
        self._client = client or SimpleWebhookClient()
        self._store = store or InMemoryDeliveryStore()
        self._signer = signer or WebhookSigner()
        self._endpoints: Dict[str, WebhookEndpoint] = {}

    def register_endpoint(self, endpoint: WebhookEndpoint) -> None:
        """Register a webhook endpoint."""
        self._endpoints[endpoint.id] = endpoint
        logger.info(f"Registered webhook endpoint: {endpoint.url}")

    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Unregister a webhook endpoint."""
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
            return True
        return False

    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get endpoint by ID."""
        return self._endpoints.get(endpoint_id)

    def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all endpoints."""
        return list(self._endpoints.values())

    async def dispatch(self, event: WebhookEvent) -> List[WebhookDelivery]:
        """Dispatch event to all matching endpoints."""
        deliveries = []

        for endpoint in self._endpoints.values():
            if not endpoint.enabled:
                continue

            if not endpoint.matches_event(event.type):
                continue

            delivery = WebhookDelivery(
                endpoint_id=endpoint.id,
                event_id=event.id,
            )

            await self._deliver(endpoint, event, delivery)
            await self._store.save(delivery)
            deliveries.append(delivery)

        return deliveries

    async def _deliver(
        self,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
        delivery: WebhookDelivery,
    ) -> None:
        """Attempt delivery to endpoint."""
        payload = event.to_json()

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-ID": delivery.id,
            "X-Event-ID": event.id,
            "X-Event-Type": event.type,
            "X-Timestamp": str(int(time.time())),
            **endpoint.headers,
        }

        # Add signature if secret is configured
        if endpoint.secret:
            signature = self._signer.sign(payload, endpoint.secret)
            headers["X-Signature"] = signature

        delivery.attempts += 1
        delivery.last_attempt = datetime.utcnow()

        try:
            status, body = await self._client.post(
                endpoint.url,
                payload,
                headers,
                endpoint.timeout_seconds,
            )

            delivery.response_status = status
            delivery.response_body = body[:1000] if body else None  # Truncate

            if 200 <= status < 300:
                delivery.status = WebhookStatus.DELIVERED
                delivery.delivered_at = datetime.utcnow()
                logger.info(f"Webhook delivered: {delivery.id} to {endpoint.url}")
            else:
                self._schedule_retry(endpoint, delivery)
                logger.warning(
                    f"Webhook delivery failed: {delivery.id}, status={status}"
                )

        except Exception as e:
            delivery.error = str(e)
            self._schedule_retry(endpoint, delivery)
            logger.error(f"Webhook delivery error: {delivery.id}, {e}")

    def _schedule_retry(
        self,
        endpoint: WebhookEndpoint,
        delivery: WebhookDelivery,
    ) -> None:
        """Schedule delivery retry with exponential backoff."""
        if delivery.attempts >= endpoint.max_retries:
            delivery.status = WebhookStatus.FAILED
            return

        delivery.status = WebhookStatus.RETRYING

        # Exponential backoff: delay * 2^(attempts-1)
        delay = endpoint.retry_delay * (2 ** (delivery.attempts - 1))
        delay = min(delay, 3600)  # Cap at 1 hour

        delivery.next_retry = datetime.utcnow() + timedelta(seconds=delay)

    async def retry_pending(self) -> int:
        """Retry all pending deliveries."""
        pending = await self._store.get_pending()
        count = 0

        for delivery in pending:
            endpoint = self._endpoints.get(delivery.endpoint_id)
            if not endpoint:
                delivery.status = WebhookStatus.FAILED
                delivery.error = "Endpoint not found"
                await self._store.save(delivery)
                continue

            # Reconstruct event (simplified - real impl would store events)
            event = WebhookEvent(id=delivery.event_id)

            await self._deliver(endpoint, event, delivery)
            await self._store.save(delivery)
            count += 1

        return count

    async def get_delivery_status(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery status."""
        return await self._store.get(delivery_id)


class WebhookReceiver:
    """Receives and verifies incoming webhooks."""

    def __init__(self, signer: Optional[WebhookSigner] = None):
        self._signer = signer or WebhookSigner()
        self._handlers: Dict[str, List[Callable[[WebhookEvent], Any]]] = {}

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[WebhookEvent], Any],
    ) -> None:
        """Register handler for event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def verify_signature(
        self,
        payload: str,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify webhook signature."""
        return self._signer.verify(payload, secret, signature)

    async def receive(
        self,
        payload: str,
        headers: Dict[str, str],
        secret: Optional[str] = None,
    ) -> WebhookEvent:
        """Receive and process webhook."""
        # Verify signature if secret provided
        if secret:
            signature = headers.get("X-Signature", "")
            if not self.verify_signature(payload, signature, secret):
                raise WebhookVerificationError("Invalid signature")

        # Parse event
        try:
            data = json.loads(payload)
            event = WebhookEvent(
                id=data.get("id", str(uuid.uuid4())),
                type=data.get("type", ""),
                source=data.get("source", ""),
                subject=data.get("subject"),
                data=data.get("data"),
                timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
                metadata=data.get("metadata", {}),
            )
        except json.JSONDecodeError as e:
            raise WebhookParseError(f"Invalid JSON: {e}")

        # Call handlers
        handlers = self._handlers.get(event.type, []) + self._handlers.get("*", [])

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Webhook handler error: {e}")

        return event


class WebhookError(Exception):
    """Base webhook error."""
    pass


class WebhookVerificationError(WebhookError):
    """Raised when signature verification fails."""
    pass


class WebhookParseError(WebhookError):
    """Raised when payload parsing fails."""
    pass


class WebhookRetryWorker:
    """Background worker for retrying failed webhooks."""

    def __init__(
        self,
        dispatcher: WebhookDispatcher,
        interval: float = 60.0,
    ):
        self._dispatcher = dispatcher
        self._interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the retry worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._retry_loop())
        logger.info("Webhook retry worker started")

    async def stop(self) -> None:
        """Stop the retry worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Webhook retry worker stopped")

    async def _retry_loop(self) -> None:
        """Retry loop."""
        while self._running:
            try:
                count = await self._dispatcher.retry_pending()
                if count > 0:
                    logger.info(f"Retried {count} webhook deliveries")
            except Exception as e:
                logger.error(f"Webhook retry error: {e}")

            await asyncio.sleep(self._interval)


__all__ = [
    "WebhookStatus",
    "WebhookEventType",
    "WebhookEvent",
    "WebhookEndpoint",
    "WebhookDelivery",
    "WebhookSigner",
    "WebhookDeliveryStore",
    "InMemoryDeliveryStore",
    "WebhookClient",
    "SimpleWebhookClient",
    "WebhookDispatcher",
    "WebhookReceiver",
    "WebhookError",
    "WebhookVerificationError",
    "WebhookParseError",
    "WebhookRetryWorker",
]
