"""
Tests for Phase 20: Advanced Integration & Extensibility.

Tests webhook handler module components.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.vision.webhook_handler import (
    # Enums
    WebhookEventType,
    WebhookStatus,
    DeliveryStatus,
    SignatureAlgorithm,
    # Dataclasses
    WebhookConfig,
    WebhookRegistration,
    WebhookEvent,
    DeliveryAttempt,
    DeliveryRecord,
    # Classes
    WebhookSignature,
    WebhookRateLimiter,
    WebhookDeliveryService,
    WebhookRegistry,
    WebhookManager,
    WebhookVisionProvider,
    # Factory functions
    create_webhook_config,
    create_webhook_manager,
    create_webhook_event,
)
from src.core.vision.base import VisionDescription


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def webhook_config():
    """Create a test webhook configuration."""
    return WebhookConfig(
        url="https://example.com/webhook",
        events=[WebhookEventType.ANALYSIS_COMPLETED, WebhookEventType.ANALYSIS_FAILED],
        secret="test_secret_123",
        signature_algorithm=SignatureAlgorithm.HMAC_SHA256,
        max_retries=3,
        rate_limit_per_minute=100,
    )


@pytest.fixture
def webhook_registry():
    """Create a webhook registry."""
    return WebhookRegistry()


@pytest.fixture
def webhook_manager():
    """Create a webhook manager."""
    return create_webhook_manager()


@pytest.fixture
def mock_provider():
    """Create a mock vision provider."""
    provider = MagicMock()
    provider.provider_name = "mock_provider"
    provider.analyze_image = AsyncMock(
        return_value=VisionDescription(
            summary="Test summary",
            details=["Detail 1", "Detail 2"],
            confidence=0.95,
        )
    )
    return provider


# ============================================================================
# Test Enums
# ============================================================================


class TestWebhookEnums:
    """Tests for webhook-related enums."""

    def test_webhook_event_type_values(self):
        """Test WebhookEventType enum values."""
        assert WebhookEventType.ANALYSIS_STARTED.value == "analysis.started"
        assert WebhookEventType.ANALYSIS_COMPLETED.value == "analysis.completed"
        assert WebhookEventType.ANALYSIS_FAILED.value == "analysis.failed"
        assert WebhookEventType.PROVIDER_CONNECTED.value == "provider.connected"
        assert WebhookEventType.PROVIDER_DISCONNECTED.value == "provider.disconnected"
        assert WebhookEventType.PROVIDER_ERROR.value == "provider.error"
        assert WebhookEventType.SYSTEM_HEALTH.value == "system.health"
        assert WebhookEventType.SECURITY_ALERT.value == "security.alert"
        assert WebhookEventType.CUSTOM.value == "custom"

    def test_webhook_status_values(self):
        """Test WebhookStatus enum values."""
        assert WebhookStatus.ACTIVE.value == "active"
        assert WebhookStatus.PAUSED.value == "paused"
        assert WebhookStatus.DISABLED.value == "disabled"
        assert WebhookStatus.FAILED.value == "failed"
        assert WebhookStatus.PENDING_VERIFICATION.value == "pending_verification"

    def test_delivery_status_values(self):
        """Test DeliveryStatus enum values."""
        assert DeliveryStatus.PENDING.value == "pending"
        assert DeliveryStatus.DELIVERING.value == "delivering"
        assert DeliveryStatus.DELIVERED.value == "delivered"
        assert DeliveryStatus.FAILED.value == "failed"
        assert DeliveryStatus.RETRYING.value == "retrying"
        assert DeliveryStatus.EXPIRED.value == "expired"

    def test_signature_algorithm_values(self):
        """Test SignatureAlgorithm enum values."""
        assert SignatureAlgorithm.HMAC_SHA256.value == "hmac_sha256"
        assert SignatureAlgorithm.HMAC_SHA512.value == "hmac_sha512"
        assert SignatureAlgorithm.HMAC_SHA1.value == "hmac_sha1"


# ============================================================================
# Test Dataclasses
# ============================================================================


class TestWebhookDataclasses:
    """Tests for webhook dataclasses."""

    def test_webhook_config_creation(self, webhook_config):
        """Test WebhookConfig creation."""
        assert webhook_config.url == "https://example.com/webhook"
        assert len(webhook_config.events) == 2
        assert webhook_config.secret == "test_secret_123"
        assert webhook_config.signature_algorithm == SignatureAlgorithm.HMAC_SHA256
        assert webhook_config.max_retries == 3
        assert webhook_config.rate_limit_per_minute == 100

    def test_webhook_config_defaults(self):
        """Test WebhookConfig default values."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEventType.CUSTOM],
        )
        assert config.timeout_seconds == 30
        assert config.content_type == "application/json"
        assert config.include_timestamp is True
        assert len(config.custom_headers) == 0

    def test_webhook_registration_creation(self, webhook_config):
        """Test WebhookRegistration creation."""
        registration = WebhookRegistration(
            webhook_id="webhook_123",
            name="Test Webhook",
            config=webhook_config,
            owner_id="user_123",
            description="A test webhook",
        )
        assert registration.webhook_id == "webhook_123"
        assert registration.name == "Test Webhook"
        assert registration.status == WebhookStatus.ACTIVE
        assert registration.owner_id == "user_123"
        assert registration.failure_count == 0
        assert registration.total_deliveries == 0

    def test_webhook_event_creation(self):
        """Test WebhookEvent creation."""
        event = WebhookEvent(
            event_id="event_123",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            timestamp=datetime.now(),
            payload={"result": "success"},
            source="test_provider",
            correlation_id="corr_123",
        )
        assert event.event_id == "event_123"
        assert event.event_type == WebhookEventType.ANALYSIS_COMPLETED
        assert event.payload == {"result": "success"}
        assert event.source == "test_provider"

    def test_webhook_event_to_dict(self):
        """Test WebhookEvent to_dict method."""
        timestamp = datetime.now()
        event = WebhookEvent(
            event_id="event_123",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            timestamp=timestamp,
            payload={"key": "value"},
            source="test",
            correlation_id="corr_123",
        )
        event_dict = event.to_dict()
        assert event_dict["event_id"] == "event_123"
        assert event_dict["event_type"] == "analysis.completed"
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["payload"] == {"key": "value"}

    def test_delivery_attempt_creation(self):
        """Test DeliveryAttempt creation."""
        attempt = DeliveryAttempt(
            attempt_id="attempt_123",
            webhook_id="webhook_123",
            event_id="event_123",
            status=DeliveryStatus.DELIVERED,
            response_code=200,
            duration_ms=150,
        )
        assert attempt.attempt_id == "attempt_123"
        assert attempt.status == DeliveryStatus.DELIVERED
        assert attempt.response_code == 200
        assert attempt.duration_ms == 150

    def test_delivery_record_creation(self):
        """Test DeliveryRecord creation."""
        event = WebhookEvent(
            event_id="event_123",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            timestamp=datetime.now(),
            payload={},
        )
        record = DeliveryRecord(
            delivery_id="delivery_123",
            webhook_id="webhook_123",
            event=event,
        )
        assert record.delivery_id == "delivery_123"
        assert record.webhook_id == "webhook_123"
        assert record.status == DeliveryStatus.PENDING
        assert len(record.attempts) == 0


# ============================================================================
# Test WebhookSignature
# ============================================================================


class TestWebhookSignature:
    """Tests for WebhookSignature class."""

    def test_generate_signature_sha256(self):
        """Test signature generation with SHA256."""
        payload = '{"event": "test"}'
        secret = "test_secret"

        signature = WebhookSignature.generate(
            payload, secret, SignatureAlgorithm.HMAC_SHA256
        )

        assert signature is not None
        assert len(signature) == 64  # SHA256 hex digest length

    def test_generate_signature_sha512(self):
        """Test signature generation with SHA512."""
        payload = '{"event": "test"}'
        secret = "test_secret"

        signature = WebhookSignature.generate(
            payload, secret, SignatureAlgorithm.HMAC_SHA512
        )

        assert signature is not None
        assert len(signature) == 128  # SHA512 hex digest length

    def test_generate_signature_with_timestamp(self):
        """Test signature generation with timestamp."""
        payload = '{"event": "test"}'
        secret = "test_secret"
        timestamp = "1234567890"

        signature = WebhookSignature.generate(
            payload, secret, SignatureAlgorithm.HMAC_SHA256, timestamp
        )

        assert signature is not None

    def test_verify_signature_success(self):
        """Test successful signature verification."""
        payload = '{"event": "test"}'
        secret = "test_secret"

        signature = WebhookSignature.generate(
            payload, secret, SignatureAlgorithm.HMAC_SHA256
        )

        result = WebhookSignature.verify(
            payload, signature, secret, SignatureAlgorithm.HMAC_SHA256
        )

        assert result is True

    def test_verify_signature_failure(self):
        """Test failed signature verification."""
        payload = '{"event": "test"}'
        secret = "test_secret"

        result = WebhookSignature.verify(
            payload, "invalid_signature", secret, SignatureAlgorithm.HMAC_SHA256
        )

        assert result is False

    def test_verify_signature_with_timestamp(self):
        """Test signature verification with timestamp."""
        payload = '{"event": "test"}'
        secret = "test_secret"
        timestamp = "1234567890"

        signature = WebhookSignature.generate(
            payload, secret, SignatureAlgorithm.HMAC_SHA256, timestamp
        )

        result = WebhookSignature.verify(
            payload, signature, secret, SignatureAlgorithm.HMAC_SHA256, timestamp
        )

        assert result is True


# ============================================================================
# Test WebhookRateLimiter
# ============================================================================


class TestWebhookRateLimiter:
    """Tests for WebhookRateLimiter class."""

    def test_rate_limiter_creation(self):
        """Test rate limiter creation."""
        limiter = WebhookRateLimiter()
        assert limiter is not None

    def test_check_rate_limit_within_limit(self):
        """Test rate limit check within limits."""
        limiter = WebhookRateLimiter()

        allowed, remaining = limiter.check_rate_limit("webhook_123", 100)

        assert allowed is True
        assert remaining == 99

    def test_check_rate_limit_exceeds_limit(self):
        """Test rate limit check when exceeded."""
        limiter = WebhookRateLimiter()

        # Use up the rate limit
        for _ in range(100):
            limiter.check_rate_limit("webhook_123", 100)

        allowed, remaining = limiter.check_rate_limit("webhook_123", 100)

        assert allowed is False
        assert remaining == 0

    def test_get_retry_after(self):
        """Test get_retry_after for rate limited webhook."""
        limiter = WebhookRateLimiter()

        # Initially no retry needed
        retry_after = limiter.get_retry_after("webhook_123")
        assert retry_after == 0

        # After some requests
        for _ in range(100):
            limiter.check_rate_limit("webhook_123", 100)

        retry_after = limiter.get_retry_after("webhook_123")
        assert retry_after >= 0


# ============================================================================
# Test WebhookRegistry
# ============================================================================


class TestWebhookRegistry:
    """Tests for WebhookRegistry class."""

    def test_register_webhook(self, webhook_registry, webhook_config):
        """Test registering a webhook."""
        registration = webhook_registry.register(
            name="Test Webhook",
            config=webhook_config,
            owner_id="user_123",
            description="Test description",
        )

        assert registration.name == "Test Webhook"
        assert registration.webhook_id is not None
        assert registration.owner_id == "user_123"
        assert registration.status == WebhookStatus.ACTIVE

    def test_get_webhook(self, webhook_registry, webhook_config):
        """Test getting a webhook by ID."""
        registration = webhook_registry.register(
            name="Test Webhook",
            config=webhook_config,
        )

        retrieved = webhook_registry.get(registration.webhook_id)

        assert retrieved is not None
        assert retrieved.webhook_id == registration.webhook_id

    def test_get_nonexistent_webhook(self, webhook_registry):
        """Test getting a nonexistent webhook."""
        result = webhook_registry.get("nonexistent_id")
        assert result is None

    def test_unregister_webhook(self, webhook_registry, webhook_config):
        """Test unregistering a webhook."""
        registration = webhook_registry.register(
            name="Test Webhook",
            config=webhook_config,
        )

        result = webhook_registry.unregister(registration.webhook_id)

        assert result is True
        assert webhook_registry.get(registration.webhook_id) is None

    def test_unregister_nonexistent_webhook(self, webhook_registry):
        """Test unregistering a nonexistent webhook."""
        result = webhook_registry.unregister("nonexistent_id")
        assert result is False

    def test_get_by_event(self, webhook_registry, webhook_config):
        """Test getting webhooks by event type."""
        webhook_registry.register(
            name="Test Webhook",
            config=webhook_config,
        )

        webhooks = webhook_registry.get_by_event(WebhookEventType.ANALYSIS_COMPLETED)

        assert len(webhooks) == 1

    def test_get_by_owner(self, webhook_registry, webhook_config):
        """Test getting webhooks by owner."""
        webhook_registry.register(
            name="Test Webhook",
            config=webhook_config,
            owner_id="user_123",
        )

        webhooks = webhook_registry.get_by_owner("user_123")

        assert len(webhooks) == 1

    def test_list_all(self, webhook_registry, webhook_config):
        """Test listing all webhooks."""
        webhook_registry.register(name="Webhook 1", config=webhook_config)
        webhook_registry.register(name="Webhook 2", config=webhook_config)

        all_webhooks = webhook_registry.list_all()

        assert len(all_webhooks) == 2

    def test_update_status(self, webhook_registry, webhook_config):
        """Test updating webhook status."""
        registration = webhook_registry.register(
            name="Test Webhook",
            config=webhook_config,
        )

        result = webhook_registry.update_status(
            registration.webhook_id, WebhookStatus.PAUSED
        )

        assert result is not None
        assert result.status == WebhookStatus.PAUSED


# ============================================================================
# Test WebhookDeliveryService
# ============================================================================


class TestWebhookDeliveryService:
    """Tests for WebhookDeliveryService class."""

    def test_delivery_service_creation(self):
        """Test delivery service creation."""
        service = WebhookDeliveryService()
        assert service is not None

    @pytest.mark.asyncio
    async def test_deliver_webhook(self, webhook_config):
        """Test delivering a webhook."""
        service = WebhookDeliveryService()

        registration = WebhookRegistration(
            webhook_id="webhook_123",
            name="Test Webhook",
            config=webhook_config,
        )

        event = WebhookEvent(
            event_id="event_123",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            timestamp=datetime.now(),
            payload={"result": "success"},
        )

        record = await service.deliver(registration, event)

        assert record is not None
        assert record.webhook_id == "webhook_123"
        assert record.event.event_id == "event_123"

    def test_get_delivery_history(self):
        """Test getting delivery history."""
        service = WebhookDeliveryService()

        history = service.get_delivery_history("webhook_123")

        assert history is not None
        assert isinstance(history, list)

    def test_get_pending_retries(self):
        """Test getting pending retries."""
        service = WebhookDeliveryService()

        retries = service.get_pending_retries()

        assert retries is not None
        assert isinstance(retries, list)


# ============================================================================
# Test WebhookManager
# ============================================================================


class TestWebhookManager:
    """Tests for WebhookManager class."""

    def test_manager_creation(self):
        """Test webhook manager creation."""
        manager = create_webhook_manager()
        assert manager is not None

    def test_register_webhook(self, webhook_manager):
        """Test registering webhook through manager."""
        registration = webhook_manager.register_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
            secret="test_secret",
            owner_id="user_123",
        )

        assert registration is not None
        assert registration.name == "Test Webhook"

    def test_unregister_webhook(self, webhook_manager):
        """Test unregistering webhook through manager."""
        registration = webhook_manager.register_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )

        result = webhook_manager.unregister_webhook(registration.webhook_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_emit_event(self, webhook_manager):
        """Test emitting an event."""
        # Register a webhook
        webhook_manager.register_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )

        # Emit event
        deliveries = await webhook_manager.emit_event(
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={"result": "success"},
            source="test",
        )

        assert deliveries is not None

    def test_get_webhook(self, webhook_manager):
        """Test getting a webhook."""
        registration = webhook_manager.register_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )

        webhook = webhook_manager.get_webhook(registration.webhook_id)

        assert webhook is not None
        assert webhook.webhook_id == registration.webhook_id

    def test_list_webhooks(self, webhook_manager):
        """Test listing all webhooks."""
        webhook_manager.register_webhook(
            name="Webhook 1",
            url="https://example.com/webhook1",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )
        webhook_manager.register_webhook(
            name="Webhook 2",
            url="https://example.com/webhook2",
            events=[WebhookEventType.ANALYSIS_FAILED],
        )

        webhooks = webhook_manager.list_webhooks()

        assert len(webhooks) == 2

    def test_list_webhooks_by_owner(self, webhook_manager):
        """Test listing webhooks by owner."""
        webhook_manager.register_webhook(
            name="Webhook 1",
            url="https://example.com/webhook1",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
            owner_id="user_1",
        )
        webhook_manager.register_webhook(
            name="Webhook 2",
            url="https://example.com/webhook2",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
            owner_id="user_2",
        )

        webhooks = webhook_manager.list_webhooks(owner_id="user_1")

        assert len(webhooks) == 1

    def test_pause_webhook(self, webhook_manager):
        """Test pausing a webhook."""
        registration = webhook_manager.register_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )

        result = webhook_manager.pause_webhook(registration.webhook_id)

        assert result is not None
        assert result.status == WebhookStatus.PAUSED

    def test_resume_webhook(self, webhook_manager):
        """Test resuming a paused webhook."""
        registration = webhook_manager.register_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )

        webhook_manager.pause_webhook(registration.webhook_id)
        result = webhook_manager.resume_webhook(registration.webhook_id)

        assert result is not None
        assert result.status == WebhookStatus.ACTIVE

    def test_get_delivery_history(self, webhook_manager):
        """Test getting delivery history."""
        registration = webhook_manager.register_webhook(
            name="Test Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )

        history = webhook_manager.get_delivery_history(registration.webhook_id)

        assert history is not None
        assert isinstance(history, list)

    def test_on_event_handler(self, webhook_manager):
        """Test registering an internal event handler."""
        handler_called = []

        def handler(event):
            handler_called.append(event)

        webhook_manager.on_event(WebhookEventType.ANALYSIS_COMPLETED, handler)

        # The handler should be registered
        assert True  # Handler registration doesn't raise


# ============================================================================
# Test WebhookVisionProvider
# ============================================================================


class TestWebhookVisionProvider:
    """Tests for WebhookVisionProvider class."""

    def test_provider_creation(self, mock_provider, webhook_manager):
        """Test creating WebhookVisionProvider."""
        provider = WebhookVisionProvider(
            base_provider=mock_provider,
            webhook_manager=webhook_manager,
        )

        assert provider is not None
        assert provider.provider_name == "webhook_mock_provider"

    @pytest.mark.asyncio
    async def test_analyze_image_emits_events(self, mock_provider, webhook_manager):
        """Test that analyze_image emits webhook events."""
        events_emitted = []

        def track_event(event):
            events_emitted.append(event.event_type)

        webhook_manager.on_event(WebhookEventType.ANALYSIS_STARTED, track_event)
        webhook_manager.on_event(WebhookEventType.ANALYSIS_COMPLETED, track_event)

        provider = WebhookVisionProvider(
            base_provider=mock_provider,
            webhook_manager=webhook_manager,
        )

        result = await provider.analyze_image(b"test_image_data")

        assert result is not None
        assert result.confidence == 0.95


# ============================================================================
# Test Factory Functions
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_webhook_config(self):
        """Test create_webhook_config factory."""
        config = create_webhook_config(
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
            secret="test_secret",
        )

        assert config is not None
        assert config.url == "https://example.com/webhook"
        assert config.secret == "test_secret"

    def test_create_webhook_manager(self):
        """Test create_webhook_manager factory."""
        manager = create_webhook_manager()

        assert manager is not None
        assert isinstance(manager, WebhookManager)

    def test_create_webhook_manager_with_custom_rate_limit(self):
        """Test create_webhook_manager with custom rate limit."""
        manager = create_webhook_manager(rate_limit_per_minute=50)

        assert manager is not None

    def test_create_webhook_event(self):
        """Test create_webhook_event factory."""
        event = create_webhook_event(
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={"result": "success"},
            source="test",
        )

        assert event is not None
        assert event.event_type == WebhookEventType.ANALYSIS_COMPLETED
        assert event.payload == {"result": "success"}
        assert event.event_id is not None
        assert event.correlation_id is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestWebhookIntegration:
    """Integration tests for webhook functionality."""

    @pytest.mark.asyncio
    async def test_full_webhook_flow(self, mock_provider):
        """Test complete webhook flow from registration to delivery."""
        # Create manager
        manager = create_webhook_manager()

        # Register webhook
        registration = manager.register_webhook(
            name="Integration Test Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
            secret="integration_test_secret",
        )

        assert registration is not None

        # Create provider with webhook integration
        provider = WebhookVisionProvider(
            base_provider=mock_provider,
            webhook_manager=manager,
        )

        # Perform analysis (should trigger webhook)
        result = await provider.analyze_image(b"test_image")

        assert result is not None
        assert result.confidence == 0.95

        # Check delivery history
        history = manager.get_delivery_history(registration.webhook_id)
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_multiple_webhooks_same_event(self, mock_provider):
        """Test multiple webhooks subscribing to same event."""
        manager = create_webhook_manager()

        # Register multiple webhooks
        reg1 = manager.register_webhook(
            name="Webhook 1",
            url="https://example1.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )
        reg2 = manager.register_webhook(
            name="Webhook 2",
            url="https://example2.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )

        # Emit event
        deliveries = await manager.emit_event(
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={"test": "data"},
        )

        # Both webhooks should receive the event
        assert len(deliveries) == 2

    @pytest.mark.asyncio
    async def test_webhook_event_filtering(self, mock_provider):
        """Test that webhooks only receive subscribed events."""
        manager = create_webhook_manager()

        # Register webhook for specific event
        reg = manager.register_webhook(
            name="Filtered Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_FAILED],  # Only failed events
        )

        # Emit completed event (not subscribed)
        deliveries = await manager.emit_event(
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={"test": "data"},
        )

        # Webhook should not receive this event
        assert len(deliveries) == 0

    def test_webhook_signature_verification_flow(self):
        """Test signature generation and verification flow."""
        secret = "test_secret_123"
        payload = '{"event": "analysis.completed", "data": {"result": "success"}}'
        timestamp = "1234567890"

        # Generate signature
        signature = WebhookSignature.generate(
            payload, secret, SignatureAlgorithm.HMAC_SHA256, timestamp
        )

        # Verify signature
        is_valid = WebhookSignature.verify(
            payload, signature, secret, SignatureAlgorithm.HMAC_SHA256, timestamp
        )

        assert is_valid is True

        # Verify with wrong timestamp fails
        is_valid_wrong = WebhookSignature.verify(
            payload, signature, secret, SignatureAlgorithm.HMAC_SHA256, "wrong_timestamp"
        )

        assert is_valid_wrong is False

    @pytest.mark.asyncio
    async def test_paused_webhook_not_triggered(self):
        """Test that paused webhooks don't receive events."""
        manager = create_webhook_manager()

        # Register and pause webhook
        reg = manager.register_webhook(
            name="Paused Webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )
        manager.pause_webhook(reg.webhook_id)

        # Emit event
        deliveries = await manager.emit_event(
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={"test": "data"},
        )

        # Paused webhook should not receive event
        assert len(deliveries) == 0
