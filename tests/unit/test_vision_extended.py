"""Unit tests for extended vision features.

Tests streaming, prompts, cost tracking, and webhooks.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision import VisionDescription, VisionProviderError
from src.core.vision.cost_tracker import (
    BudgetConfig,
    CostTrackedVisionProvider,
    CostTracker,
    PricingTier,
    create_cost_tracked_provider,
)
from src.core.vision.prompts import (
    BUILTIN_TEMPLATES,
    PromptConfig,
    PromptManager,
    PromptTemplate,
    PromptType,
    get_prompts,
    register_custom_template,
)
from src.core.vision.streaming import (
    BatchStreamProcessor,
    StreamConfig,
    StreamEvent,
    StreamEventType,
    StreamingVisionProvider,
    create_streaming_provider,
    stream_analysis,
)
from src.core.vision.webhooks import (
    WebhookConfig,
    WebhookEventType,
    WebhookManager,
    WebhookVisionProvider,
    create_webhook_provider,
)

# Sample image data
SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
)


class MockVisionProvider:
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        confidence: float = 0.9,
        latency: float = 0.01,
        fail: bool = False,
    ):
        self.name = name
        self.confidence = confidence
        self.latency = latency
        self.fail = fail
        self.call_count = 0

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        self.call_count += 1
        await asyncio.sleep(self.latency)

        if self.fail:
            raise VisionProviderError(self.name, "Simulated failure")

        return VisionDescription(
            summary=f"Summary from {self.name}",
            details=[f"Detail from {self.name}"],
            confidence=self.confidence,
        )

    @property
    def provider_name(self) -> str:
        return self.name


class TestStreamingVisionProvider:
    """Tests for StreamingVisionProvider."""

    @pytest.mark.asyncio
    async def test_streams_events(self):
        """Test streaming emits expected events."""
        mock = MockVisionProvider(latency=0.01)
        streaming = StreamingVisionProvider(mock)

        events = []
        async for event in streaming.analyze_image_stream(SAMPLE_PNG):
            events.append(event)

        # Should have start, progress, and complete events
        event_types = [e.event_type for e in events]
        assert StreamEventType.START in event_types
        assert StreamEventType.COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_complete_event_has_result(self):
        """Test complete event contains result."""
        mock = MockVisionProvider()
        streaming = StreamingVisionProvider(mock)

        complete_event = None
        async for event in streaming.analyze_image_stream(SAMPLE_PNG):
            if event.event_type == StreamEventType.COMPLETE:
                complete_event = event

        assert complete_event is not None
        assert "result" in complete_event.data
        assert complete_event.data["result"]["summary"] == "Summary from mock"

    @pytest.mark.asyncio
    async def test_error_event_on_failure(self):
        """Test error event emitted on failure."""
        mock = MockVisionProvider(fail=True)
        streaming = StreamingVisionProvider(mock)

        error_event = None
        async for event in streaming.analyze_image_stream(SAMPLE_PNG):
            if event.event_type == StreamEventType.ERROR:
                error_event = event

        assert error_event is not None
        assert "error" in error_event.data

    @pytest.mark.asyncio
    async def test_events_have_sequence_numbers(self):
        """Test events have sequential sequence numbers."""
        mock = MockVisionProvider(latency=0.01)
        streaming = StreamingVisionProvider(mock)

        events = []
        async for event in streaming.analyze_image_stream(SAMPLE_PNG):
            events.append(event)

        sequences = [e.sequence for e in events]
        assert sequences == sorted(sequences)
        assert sequences[0] == 1

    @pytest.mark.asyncio
    async def test_stream_event_to_sse(self):
        """Test StreamEvent to SSE conversion."""
        event = StreamEvent(
            event_type=StreamEventType.PROGRESS,
            data={"progress": 0.5},
            sequence=1,
        )

        sse = event.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        assert '"type": "progress"' in sse


class TestBatchStreamProcessor:
    """Tests for BatchStreamProcessor."""

    @pytest.mark.asyncio
    async def test_streams_batch_results(self):
        """Test batch processing streams results."""
        mock = MockVisionProvider(latency=0.01)
        processor = BatchStreamProcessor(mock, max_concurrency=2)

        images = [SAMPLE_PNG] * 3
        events = []
        async for event in processor.process_batch_stream(images):
            events.append(event)

        # Should have start, per-item events, and final complete
        start_events = [e for e in events if e.event_type == StreamEventType.START]
        complete_events = [e for e in events if e.event_type == StreamEventType.COMPLETE]

        assert len(start_events) >= 1
        assert len(complete_events) >= 1


class TestPromptManager:
    """Tests for PromptManager."""

    def test_builtin_templates_available(self):
        """Test built-in templates are available."""
        manager = PromptManager()

        assert "engineering_drawing" in manager.list_templates()
        assert "architectural" in manager.list_templates()
        assert "circuit_diagram" in manager.list_templates()

    def test_get_prompts(self):
        """Test getting prompts from template."""
        manager = PromptManager()

        system, user = manager.get_prompts("engineering_drawing")

        assert "engineering" in system.lower()
        assert len(user) > 0

    def test_register_custom_template(self):
        """Test registering custom template."""
        manager = PromptManager()

        template = PromptTemplate(
            name="custom_test",
            system_prompt="Custom system prompt",
            user_prompt_template="Analyze {subject}",
            variables=["subject"],
        )
        manager.register_template(template)

        assert "custom_test" in manager.list_templates()
        system, user = manager.get_prompts("custom_test", subject="PCB")
        assert "Custom system prompt" in system

    def test_register_duplicate_raises(self):
        """Test duplicate registration raises error."""
        manager = PromptManager()

        template = PromptTemplate(
            name="engineering_drawing",
            system_prompt="New prompt",
            user_prompt_template="New user prompt",
        )

        with pytest.raises(ValueError):
            manager.register_template(template, overwrite=False)

    def test_prompt_config_custom_type(self):
        """Test PromptConfig with custom type."""
        config = PromptConfig(
            template_type=PromptType.CUSTOM,
            custom_system_prompt="My custom prompt",
            custom_user_prompt="Analyze this",
        )

        system, user = config.get_prompts()
        assert system == "My custom prompt"
        assert user == "Analyze this"

    def test_prompt_config_additional_instructions(self):
        """Test PromptConfig with additional instructions."""
        config = PromptConfig(
            template_type=PromptType.GENERAL,
            additional_instructions="Focus on dimensions",
        )

        system, user = config.get_prompts()
        assert "Focus on dimensions" in system


class TestCostTracker:
    """Tests for CostTracker."""

    @pytest.mark.asyncio
    async def test_records_usage(self):
        """Test recording usage."""
        tracker = CostTracker()

        record = await tracker.record_usage(
            provider="openai",
            input_tokens=1000,
            output_tokens=500,
            images=1,
            success=True,
            latency_ms=1234.5,
        )

        assert record.provider == "openai"
        assert record.input_tokens == 1000
        assert record.cost_usd > 0

    @pytest.mark.asyncio
    async def test_daily_cost_tracking(self):
        """Test daily cost accumulation."""
        tracker = CostTracker()

        await tracker.record_usage(
            provider="openai",
            input_tokens=1000,
            output_tokens=500,
            images=1,
            success=True,
            latency_ms=100,
        )

        daily_cost = tracker.get_daily_cost()
        assert daily_cost > 0

    @pytest.mark.asyncio
    async def test_usage_summary(self):
        """Test getting usage summary."""
        tracker = CostTracker()

        for _ in range(3):
            await tracker.record_usage(
                provider="openai",
                input_tokens=1000,
                output_tokens=500,
                images=1,
                success=True,
                latency_ms=100,
            )

        summary = tracker.get_usage_summary(provider="openai")

        assert summary.total_requests == 3
        assert summary.successful_requests == 3
        assert summary.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_provider_breakdown(self):
        """Test cost breakdown by provider."""
        tracker = CostTracker()

        await tracker.record_usage(
            provider="openai",
            input_tokens=1000,
            output_tokens=500,
            images=1,
            success=True,
            latency_ms=100,
        )
        await tracker.record_usage(
            provider="anthropic",
            input_tokens=1000,
            output_tokens=500,
            images=1,
            success=True,
            latency_ms=100,
        )

        breakdown = tracker.get_provider_breakdown()

        assert "openai" in breakdown
        assert "anthropic" in breakdown

    @pytest.mark.asyncio
    async def test_budget_check(self):
        """Test budget checking."""
        config = BudgetConfig(
            daily_limit_usd=0.01,
            hard_limit=True,
        )
        tracker = CostTracker(budget_config=config)

        # Should be within budget initially
        assert await tracker.check_budget(0.005)

        # Should exceed budget
        assert not await tracker.check_budget(100.0)

    def test_pricing_tier_calculation(self):
        """Test pricing calculation."""
        pricing = PricingTier(
            provider="test",
            input_cost_per_1k_tokens=0.01,
            output_cost_per_1k_tokens=0.03,
            image_cost_per_image=0.001,
        )

        cost = pricing.calculate_cost(
            input_tokens=1000,
            output_tokens=1000,
            images=1,
        )

        assert cost == 0.01 + 0.03 + 0.001


class TestCostTrackedVisionProvider:
    """Tests for CostTrackedVisionProvider."""

    @pytest.mark.asyncio
    async def test_tracks_successful_request(self):
        """Test tracking successful request."""
        mock = MockVisionProvider()
        tracker = CostTracker()
        tracked = CostTrackedVisionProvider(mock, tracker)

        await tracked.analyze_image(SAMPLE_PNG)

        summary = tracker.get_usage_summary()
        assert summary.total_requests == 1
        assert summary.successful_requests == 1

    @pytest.mark.asyncio
    async def test_tracks_failed_request(self):
        """Test tracking failed request."""
        mock = MockVisionProvider(fail=True)
        tracker = CostTracker()
        tracked = CostTrackedVisionProvider(mock, tracker)

        with pytest.raises(VisionProviderError):
            await tracked.analyze_image(SAMPLE_PNG)

        summary = tracker.get_usage_summary()
        assert summary.total_requests == 1
        assert summary.failed_requests == 1


class TestWebhookManager:
    """Tests for WebhookManager."""

    def test_register_webhook(self):
        """Test webhook registration."""
        manager = WebhookManager()

        config = WebhookConfig(
            url="https://example.com/webhook",
            secret="test-secret",
        )
        manager.register_webhook("test", config)

        webhooks = manager.list_webhooks()
        assert "test" in webhooks

    def test_unregister_webhook(self):
        """Test webhook unregistration."""
        manager = WebhookManager()

        config = WebhookConfig(url="https://example.com/webhook")
        manager.register_webhook("test", config)

        result = manager.unregister_webhook("test")
        assert result is True

        webhooks = manager.list_webhooks()
        assert "test" not in webhooks

    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Test emitting event."""
        manager = WebhookManager()

        event_id = await manager.emit(
            WebhookEventType.ANALYSIS_COMPLETED,
            {"request_id": "123", "success": True},
        )

        assert event_id is not None
        assert len(event_id) > 0

    def test_signature_verification(self):
        """Test webhook signature verification."""
        payload = '{"test": "data"}'
        secret = "my-secret"

        signature = WebhookManager._sign_payload(payload, secret)
        assert WebhookManager.verify_signature(payload, signature, secret)

        # Wrong signature should fail
        assert not WebhookManager.verify_signature(payload, "wrong", secret)

    def test_webhook_config_event_filtering(self):
        """Test webhook event filtering."""
        config = WebhookConfig(
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED],
        )

        assert config.should_deliver(WebhookEventType.ANALYSIS_COMPLETED)
        assert not config.should_deliver(WebhookEventType.ANALYSIS_FAILED)


class TestWebhookVisionProvider:
    """Tests for WebhookVisionProvider."""

    @pytest.mark.asyncio
    async def test_emits_started_event(self):
        """Test emits started event."""
        mock = MockVisionProvider()
        manager = WebhookManager()
        webhook_provider = WebhookVisionProvider(mock, manager)

        emitted_events = []
        original_emit = manager.emit

        async def capture_emit(event_type, payload):
            emitted_events.append((event_type, payload))
            return await original_emit(event_type, payload)

        manager.emit = capture_emit

        await webhook_provider.analyze_image(SAMPLE_PNG)

        event_types = [e[0] for e in emitted_events]
        assert WebhookEventType.ANALYSIS_STARTED in event_types
        assert WebhookEventType.ANALYSIS_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_emits_failed_event_on_error(self):
        """Test emits failed event on error."""
        mock = MockVisionProvider(fail=True)
        manager = WebhookManager()
        webhook_provider = WebhookVisionProvider(mock, manager)

        emitted_events = []

        async def capture_emit(event_type, payload):
            emitted_events.append((event_type, payload))
            return "event-id"

        manager.emit = capture_emit

        with pytest.raises(VisionProviderError):
            await webhook_provider.analyze_image(SAMPLE_PNG)

        event_types = [e[0] for e in emitted_events]
        assert WebhookEventType.ANALYSIS_FAILED in event_types


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_streaming_provider(self):
        """Test creating streaming provider."""
        mock = MockVisionProvider()
        streaming = create_streaming_provider(mock)

        assert isinstance(streaming, StreamingVisionProvider)
        assert streaming.provider_name == "mock"

    def test_create_cost_tracked_provider(self):
        """Test creating cost tracked provider."""
        mock = MockVisionProvider()
        tracked = create_cost_tracked_provider(mock)

        assert isinstance(tracked, CostTrackedVisionProvider)
        assert tracked.provider_name == "mock"

    def test_create_webhook_provider(self):
        """Test creating webhook provider."""
        mock = MockVisionProvider()
        webhook = create_webhook_provider(mock)

        assert isinstance(webhook, WebhookVisionProvider)
        assert webhook.provider_name == "mock"
