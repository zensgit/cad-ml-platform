"""Webhook notifications for async vision processing.

Provides:
- Async completion callbacks
- Retry with exponential backoff
- Webhook signature verification
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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


class WebhookEventType(Enum):
    """Types of webhook events."""

    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"
    BATCH_STARTED = "batch.started"
    BATCH_PROGRESS = "batch.progress"
    BATCH_COMPLETED = "batch.completed"
    BUDGET_ALERT = "budget.alert"


class WebhookStatus(Enum):
    """Status of webhook delivery."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookEvent:
    """A webhook event to be delivered."""

    event_id: str
    event_type: WebhookEventType
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    status: WebhookStatus = WebhookStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""

    url: str
    secret: Optional[str] = None  # For signature verification
    events: List[WebhookEventType] = field(
        default_factory=lambda: list(WebhookEventType)
    )
    max_retries: int = 3
    retry_delay: float = 1.0  # Base delay in seconds
    timeout: float = 30.0
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def should_deliver(self, event_type: WebhookEventType) -> bool:
        """Check if this webhook should receive the event type."""
        return self.enabled and event_type in self.events


@dataclass
class DeliveryResult:
    """Result of webhook delivery attempt."""

    event_id: str
    webhook_url: str
    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    attempt: int = 1


class WebhookManager:
    """
    Manages webhook registrations and deliveries.

    Features:
    - Multiple webhook endpoints
    - Event filtering per endpoint
    - Retry with exponential backoff
    - Signature verification
    - Async delivery queue
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        delivery_concurrency: int = 5,
    ):
        """
        Initialize webhook manager.

        Args:
            max_queue_size: Maximum pending events in queue
            delivery_concurrency: Max concurrent deliveries
        """
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._queue: asyncio.Queue[WebhookEvent] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._delivery_semaphore = asyncio.Semaphore(delivery_concurrency)
        self._delivery_history: List[DeliveryResult] = []
        self._running = False
        self._worker_task: Optional[asyncio.Task[None]] = None

    def register_webhook(
        self,
        webhook_id: str,
        config: WebhookConfig,
    ) -> None:
        """
        Register a webhook endpoint.

        Args:
            webhook_id: Unique identifier for the webhook
            config: Webhook configuration
        """
        self._webhooks[webhook_id] = config
        logger.info(f"Registered webhook: {webhook_id} -> {config.url}")

    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Unregister a webhook endpoint.

        Args:
            webhook_id: Webhook identifier

        Returns:
            True if webhook was found and removed
        """
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            logger.info(f"Unregistered webhook: {webhook_id}")
            return True
        return False

    def list_webhooks(self) -> Dict[str, WebhookConfig]:
        """List all registered webhooks."""
        return dict(self._webhooks)

    async def emit(
        self,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
    ) -> str:
        """
        Emit a webhook event.

        Args:
            event_type: Type of event
            payload: Event payload

        Returns:
            Event ID
        """
        event = WebhookEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            payload=payload,
        )

        try:
            self._queue.put_nowait(event)
            logger.debug(f"Queued webhook event: {event.event_id}")
        except asyncio.QueueFull:
            logger.warning(f"Webhook queue full, dropping event: {event.event_id}")

        return event.event_id

    async def start(self) -> None:
        """Start the webhook delivery worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._delivery_worker())
        logger.info("Webhook delivery worker started")

    async def stop(self) -> None:
        """Stop the webhook delivery worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Webhook delivery worker stopped")

    async def _delivery_worker(self) -> None:
        """Background worker for delivering webhooks."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
                await self._deliver_event(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Webhook worker error: {e}")

    async def _deliver_event(self, event: WebhookEvent) -> None:
        """Deliver an event to all matching webhooks."""
        for webhook_id, config in self._webhooks.items():
            if config.should_deliver(event.event_type):
                asyncio.create_task(
                    self._deliver_to_webhook(event, webhook_id, config)
                )

    async def _deliver_to_webhook(
        self,
        event: WebhookEvent,
        webhook_id: str,
        config: WebhookConfig,
    ) -> DeliveryResult:
        """Deliver event to a specific webhook with retries."""
        async with self._delivery_semaphore:
            attempt = 0
            last_error = None

            while attempt < config.max_retries:
                attempt += 1

                try:
                    result = await self._send_webhook(event, config, attempt)

                    if result.success:
                        self._delivery_history.append(result)
                        logger.debug(
                            f"Webhook delivered: {event.event_id} -> {webhook_id}"
                        )
                        return result

                    last_error = result.error or f"HTTP {result.status_code}"

                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        f"Webhook delivery failed (attempt {attempt}): {e}"
                    )

                # Exponential backoff
                if attempt < config.max_retries:
                    delay = config.retry_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

            # All retries exhausted
            result = DeliveryResult(
                event_id=event.event_id,
                webhook_url=config.url,
                success=False,
                error=f"Max retries exceeded: {last_error}",
                attempt=attempt,
            )
            self._delivery_history.append(result)
            logger.error(
                f"Webhook delivery failed after {attempt} attempts: "
                f"{event.event_id} -> {webhook_id}"
            )
            return result

    async def _send_webhook(
        self,
        event: WebhookEvent,
        config: WebhookConfig,
        attempt: int,
    ) -> DeliveryResult:
        """Send a single webhook request."""
        start_time = time.time()
        payload = event.to_dict()
        body = json.dumps(payload)

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Event": event.event_type.value,
            "X-Webhook-Event-ID": event.event_id,
            "X-Webhook-Attempt": str(attempt),
            **config.headers,
        }

        # Add signature if secret is configured
        if config.secret:
            signature = self._sign_payload(body, config.secret)
            headers["X-Webhook-Signature"] = signature

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    config.url,
                    content=body,
                    headers=headers,
                    timeout=config.timeout,
                )
                duration_ms = (time.time() - start_time) * 1000

                success = 200 <= response.status_code < 300
                return DeliveryResult(
                    event_id=event.event_id,
                    webhook_url=config.url,
                    success=success,
                    status_code=response.status_code,
                    response_body=response.text[:500] if response.text else None,
                    duration_ms=duration_ms,
                    attempt=attempt,
                )

            except httpx.TimeoutException:
                return DeliveryResult(
                    event_id=event.event_id,
                    webhook_url=config.url,
                    success=False,
                    error="Request timeout",
                    duration_ms=(time.time() - start_time) * 1000,
                    attempt=attempt,
                )

            except httpx.RequestError as e:
                return DeliveryResult(
                    event_id=event.event_id,
                    webhook_url=config.url,
                    success=False,
                    error=str(e),
                    duration_ms=(time.time() - start_time) * 1000,
                    attempt=attempt,
                )

    @staticmethod
    def _sign_payload(payload: str, secret: str) -> str:
        """Generate HMAC signature for payload."""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: Request body
            signature: X-Webhook-Signature header value
            secret: Webhook secret

        Returns:
            True if signature is valid
        """
        expected = WebhookManager._sign_payload(payload, secret)
        return hmac.compare_digest(signature, expected)

    def get_delivery_history(
        self,
        limit: int = 100,
        event_type: Optional[WebhookEventType] = None,
    ) -> List[DeliveryResult]:
        """Get recent delivery history."""
        history = self._delivery_history[-limit:]
        if event_type:
            # Would need to track event_type in DeliveryResult
            pass
        return history


class WebhookVisionProvider:
    """
    Wrapper that adds webhook notifications to any VisionProvider.

    Emits events for analysis lifecycle.
    """

    def __init__(
        self,
        provider: VisionProvider,
        webhook_manager: WebhookManager,
        include_result_in_payload: bool = True,
    ):
        """
        Initialize webhook-enabled provider.

        Args:
            provider: The underlying vision provider
            webhook_manager: WebhookManager instance
            include_result_in_payload: Include full result in webhook payload
        """
        self._provider = provider
        self._webhook_manager = webhook_manager
        self._include_result = include_result_in_payload

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        request_id: Optional[str] = None,
    ) -> VisionDescription:
        """
        Analyze image with webhook notifications.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description
            request_id: Optional request ID for tracking

        Returns:
            VisionDescription with analysis results
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.time()

        # Emit started event
        await self._webhook_manager.emit(
            WebhookEventType.ANALYSIS_STARTED,
            {
                "request_id": request_id,
                "provider": self._provider.provider_name,
                "image_size": len(image_data),
                "timestamp": datetime.now().isoformat(),
            },
        )

        try:
            result = await self._provider.analyze_image(
                image_data, include_description
            )
            elapsed_ms = (time.time() - start_time) * 1000

            # Emit completed event
            payload: Dict[str, Any] = {
                "request_id": request_id,
                "provider": self._provider.provider_name,
                "elapsed_ms": elapsed_ms,
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

            if self._include_result:
                payload["result"] = {
                    "summary": result.summary,
                    "details": result.details,
                    "confidence": result.confidence,
                }

            await self._webhook_manager.emit(
                WebhookEventType.ANALYSIS_COMPLETED,
                payload,
            )

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            # Emit failed event
            await self._webhook_manager.emit(
                WebhookEventType.ANALYSIS_FAILED,
                {
                    "request_id": request_id,
                    "provider": self._provider.provider_name,
                    "elapsed_ms": elapsed_ms,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            raise

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def webhook_manager(self) -> WebhookManager:
        """Get the webhook manager."""
        return self._webhook_manager


# Global webhook manager instance
_global_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """
    Get the global webhook manager instance.

    Returns:
        WebhookManager singleton
    """
    global _global_webhook_manager
    if _global_webhook_manager is None:
        _global_webhook_manager = WebhookManager()
    return _global_webhook_manager


def create_webhook_provider(
    provider: VisionProvider,
    webhook_url: Optional[str] = None,
    webhook_secret: Optional[str] = None,
    webhook_manager: Optional[WebhookManager] = None,
) -> WebhookVisionProvider:
    """
    Factory to create a webhook-enabled provider wrapper.

    Args:
        provider: The underlying vision provider
        webhook_url: URL to send webhooks to
        webhook_secret: Secret for signature verification
        webhook_manager: Optional existing WebhookManager

    Returns:
        WebhookVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> webhook_provider = create_webhook_provider(
        ...     provider,
        ...     webhook_url="https://api.example.com/webhooks/vision",
        ...     webhook_secret="my-secret",
        ... )
        >>> # Start the webhook delivery worker
        >>> await webhook_provider.webhook_manager.start()
        >>> result = await webhook_provider.analyze_image(image_bytes)
    """
    manager = webhook_manager or get_webhook_manager()

    if webhook_url:
        config = WebhookConfig(
            url=webhook_url,
            secret=webhook_secret,
        )
        manager.register_webhook("default", config)

    return WebhookVisionProvider(provider=provider, webhook_manager=manager)
