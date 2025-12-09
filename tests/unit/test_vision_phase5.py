"""Tests for Phase 5 Vision Provider extensions.

Tests cover:
- A/B Testing (ab_testing.py)
- Response Validation (validation.py)
- Audit Logging (audit.py)
- Multi-Region Routing (multiregion.py)
- Request Prioritization (priority.py)
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.vision import VisionDescription
from src.core.vision.ab_testing import (
    ExperimentManager,
    ABTestingVisionProvider,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentResult,
    ExperimentAnalysis,
    Variant,
    VariantType,
    VariantStats,
    AllocationStrategy,
    StatisticalMethod,
    create_ab_testing_provider,
)
from src.core.vision.validation import (
    ResponseValidator,
    ValidatingVisionProvider,
    ValidationSchema,
    ValidationRule,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationRuleType,
    ValidationStats,
    ValidationError,
    STANDARD_SCHEMA,
    STRICT_SCHEMA,
    OCR_SCHEMA,
    create_validating_provider,
)
from src.core.vision.audit import (
    AuditLogger,
    AuditingVisionProvider,
    AuditEntry,
    AuditContext,
    AuditConfig,
    AuditEventType,
    AuditLevel,
    InMemoryAuditStorage,
    RetentionPolicy,
    create_auditing_provider,
)
from src.core.vision.multiregion import (
    RegionRouter,
    MultiRegionVisionProvider,
    Region,
    RegionConfig,
    RegionHealth,
    RegionStatus,
    RoutingStrategy,
    LatencyMeasurer,
    create_multiregion_provider,
)
from src.core.vision.priority import (
    RequestScheduler,
    PrioritizedVisionProvider,
    HeapPriorityQueue,
    MultiLevelQueue,
    Priority,
    QueueStrategy,
    RequestStatus,
    PrioritizedRequest,
    QueueStats,
    QueueFullError,
    create_prioritized_provider,
)


def create_test_description(
    summary: str = "Test summary",
    confidence: float = 0.95,
) -> VisionDescription:
    """Create a valid VisionDescription for testing."""
    return VisionDescription(
        summary=summary,
        confidence=confidence,
        details=["Detail 1", "Detail 2"],
    )


def create_mock_provider(name: str = "mock") -> MagicMock:
    """Create a mock vision provider."""
    provider = MagicMock()
    provider.provider_name = name
    provider.analyze_image = AsyncMock(return_value=create_test_description())
    return provider


# ============================================================================
# A/B Testing Tests
# ============================================================================


class TestVariantStats:
    """Tests for VariantStats class."""

    def test_initial_values(self):
        """Test initial stats values."""
        stats = VariantStats(variant_name="test")
        assert stats.total_requests == 0
        assert stats.success_rate == 0.0
        assert stats.average_latency_ms == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = VariantStats(
            variant_name="test",
            total_requests=10,
            successful_requests=8,
        )
        assert stats.success_rate == 0.8

    def test_average_latency(self):
        """Test average latency calculation."""
        stats = VariantStats(
            variant_name="test",
            total_requests=4,
            total_latency_ms=400.0,
        )
        assert stats.average_latency_ms == 100.0


class TestExperimentManager:
    """Tests for ExperimentManager class."""

    def test_create_experiment(self):
        """Test creating an experiment."""
        manager = ExperimentManager()
        control = create_mock_provider("control")
        treatment = create_mock_provider("treatment")

        config = manager.create_experiment(
            experiment_id="exp1",
            name="Test Experiment",
            control=control,
            treatments=[("treatment_a", treatment, 0.5)],
        )

        assert config.experiment_id == "exp1"
        assert config.status == ExperimentStatus.DRAFT
        assert len(config.variants) == 2

    def test_start_experiment(self):
        """Test starting an experiment."""
        manager = ExperimentManager()
        control = create_mock_provider()

        manager.create_experiment(
            experiment_id="exp1",
            name="Test",
            control=control,
            treatments=[],
        )
        manager.start_experiment("exp1")

        experiment = manager.get_experiment("exp1")
        assert experiment.status == ExperimentStatus.RUNNING

    def test_allocate_variant_hash_based(self):
        """Test hash-based variant allocation."""
        manager = ExperimentManager()
        control = create_mock_provider("control")
        treatment = create_mock_provider("treatment")

        manager.create_experiment(
            experiment_id="exp1",
            name="Test",
            control=control,
            treatments=[("treatment", treatment, 0.5)],
            allocation_strategy=AllocationStrategy.HASH_BASED,
        )
        manager.start_experiment("exp1")

        # Same request ID should always get same variant
        variant1 = manager.allocate_variant("exp1", "request_123")
        variant2 = manager.allocate_variant("exp1", "request_123")
        assert variant1.name == variant2.name

    def test_record_result(self):
        """Test recording experiment results."""
        manager = ExperimentManager()
        control = create_mock_provider()

        manager.create_experiment(
            experiment_id="exp1",
            name="Test",
            control=control,
            treatments=[],
        )
        manager.start_experiment("exp1")

        manager.record_result(
            experiment_id="exp1",
            variant_name="control",
            request_id="req1",
            latency_ms=100.0,
            success=True,
            confidence=0.95,
        )

        stats = manager.get_stats("exp1")
        assert stats["control"].total_requests == 1
        assert stats["control"].successful_requests == 1

    def test_analyze_experiment(self):
        """Test experiment analysis."""
        manager = ExperimentManager()
        control = create_mock_provider("control")
        treatment = create_mock_provider("treatment")

        manager.create_experiment(
            experiment_id="exp1",
            name="Test",
            control=control,
            treatments=[("treatment", treatment, 0.5)],
        )
        manager.start_experiment("exp1")

        # Record some results
        for i in range(50):
            manager.record_result("exp1", "control", f"req_c{i}", 100.0, True, 0.9)
            manager.record_result("exp1", "treatment", f"req_t{i}", 80.0, True, 0.95)

        analysis = manager.analyze_experiment("exp1")
        assert isinstance(analysis, ExperimentAnalysis)
        assert analysis.control_stats.total_requests == 50


class TestABTestingVisionProvider:
    """Tests for ABTestingVisionProvider class."""

    @pytest.mark.asyncio
    async def test_analyze_with_ab_testing(self):
        """Test analyzing with A/B testing."""
        control = create_mock_provider("control")
        treatment = create_mock_provider("treatment")

        provider, manager = create_ab_testing_provider(
            experiment_id="exp1",
            name="Test",
            control=control,
            treatments=[("treatment", treatment, 0.5)],
        )

        result = await provider.analyze_image(b"image_data")
        assert isinstance(result, VisionDescription)


# ============================================================================
# Validation Tests
# ============================================================================


class TestValidationRule:
    """Tests for ValidationRule class."""

    def test_rule_creation(self):
        """Test creating a validation rule."""
        rule = ValidationRule(
            name="test_rule",
            rule_type=ValidationRuleType.REQUIRED_FIELD,
            field="summary",
            severity=ValidationSeverity.ERROR,
        )
        assert rule.name == "test_rule"
        assert rule.enabled is True


class TestResponseValidator:
    """Tests for ResponseValidator class."""

    def test_validate_required_field(self):
        """Test required field validation."""
        schema = ValidationSchema(
            name="test",
            rules=[
                ValidationRule(
                    name="summary_required",
                    rule_type=ValidationRuleType.REQUIRED_FIELD,
                    field="summary",
                )
            ],
        )
        validator = ResponseValidator(schema)

        response = create_test_description(summary="Test summary")
        result = validator.validate(response)
        assert result.is_valid

    def test_validate_min_length(self):
        """Test minimum length validation."""
        schema = ValidationSchema(
            name="test",
            rules=[
                ValidationRule(
                    name="summary_length",
                    rule_type=ValidationRuleType.MIN_LENGTH,
                    field="summary",
                    params={"min_length": 100},
                    severity=ValidationSeverity.ERROR,
                )
            ],
        )
        validator = ResponseValidator(schema)

        response = create_test_description(summary="Short")
        result = validator.validate(response)
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_validate_confidence_threshold(self):
        """Test confidence threshold validation."""
        schema = ValidationSchema(
            name="test",
            rules=[
                ValidationRule(
                    name="confidence",
                    rule_type=ValidationRuleType.CONFIDENCE_THRESHOLD,
                    field="confidence",
                    params={"threshold": 0.9},
                    severity=ValidationSeverity.WARNING,
                )
            ],
        )
        validator = ResponseValidator(schema)

        response = create_test_description(confidence=0.8)
        result = validator.validate(response)
        assert result.is_valid  # Warning doesn't invalidate
        assert len(result.warnings) == 1

    def test_standard_schema(self):
        """Test using standard schema."""
        validator = ResponseValidator(STANDARD_SCHEMA)
        response = create_test_description()
        result = validator.validate(response)
        assert result.is_valid


class TestValidatingVisionProvider:
    """Tests for ValidatingVisionProvider class."""

    @pytest.mark.asyncio
    async def test_valid_response_passes(self):
        """Test that valid responses pass validation."""
        mock_provider = create_mock_provider()
        validator = ResponseValidator(STANDARD_SCHEMA)
        wrapper = ValidatingVisionProvider(mock_provider, validator)

        result = await wrapper.analyze_image(b"image")
        assert isinstance(result, VisionDescription)

    @pytest.mark.asyncio
    async def test_invalid_response_raises(self):
        """Test that invalid responses raise ValidationError."""
        mock_provider = create_mock_provider()
        # Return response with low confidence
        mock_provider.analyze_image.return_value = VisionDescription(
            summary="",  # Empty summary
            confidence=0.1,
        )

        validator = ResponseValidator(STRICT_SCHEMA)
        wrapper = ValidatingVisionProvider(mock_provider, validator, reject_invalid=True)

        with pytest.raises(ValidationError):
            await wrapper.analyze_image(b"image")


# ============================================================================
# Audit Tests
# ============================================================================


class TestAuditEntry:
    """Tests for AuditEntry class."""

    def test_entry_creation(self):
        """Test creating an audit entry."""
        entry = AuditEntry(
            entry_id="audit_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.REQUEST_COMPLETED,
            action="analyze",
            outcome="success",
        )
        assert entry.entry_id == "audit_001"
        assert entry.event_type == AuditEventType.REQUEST_COMPLETED

    def test_to_dict(self):
        """Test converting entry to dict."""
        entry = AuditEntry(
            entry_id="audit_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.REQUEST_STARTED,
        )
        data = entry.to_dict()
        assert "entry_id" in data
        assert "timestamp" in data
        assert data["event_type"] == "request_started"


class TestInMemoryAuditStorage:
    """Tests for InMemoryAuditStorage class."""

    def test_store_and_query(self):
        """Test storing and querying entries."""
        storage = InMemoryAuditStorage()
        entry = AuditEntry(
            entry_id="audit_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.REQUEST_COMPLETED,
        )
        storage.store(entry)

        results = storage.query()
        assert len(results) == 1
        assert results[0].entry_id == "audit_001"

    def test_query_by_event_type(self):
        """Test querying by event type."""
        storage = InMemoryAuditStorage()

        storage.store(AuditEntry(
            entry_id="1",
            timestamp=datetime.now(),
            event_type=AuditEventType.REQUEST_STARTED,
        ))
        storage.store(AuditEntry(
            entry_id="2",
            timestamp=datetime.now(),
            event_type=AuditEventType.REQUEST_COMPLETED,
        ))

        results = storage.query(event_types=[AuditEventType.REQUEST_STARTED])
        assert len(results) == 1
        assert results[0].entry_id == "1"


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_log_event(self):
        """Test logging an event."""
        logger = AuditLogger()
        entry = logger.log(
            event_type=AuditEventType.REQUEST_STARTED,
            action="analyze",
            resource="image",
        )
        assert entry is not None
        assert entry.action == "analyze"

    def test_query_logs(self):
        """Test querying audit logs."""
        logger = AuditLogger()
        logger.log(AuditEventType.REQUEST_STARTED, action="test1")
        logger.log(AuditEventType.REQUEST_COMPLETED, action="test2")

        results = logger.query()
        assert len(results) == 2

    def test_export_json(self):
        """Test exporting logs as JSON."""
        logger = AuditLogger()
        logger.log(AuditEventType.REQUEST_STARTED, action="test")

        export = logger.export(format="json")
        assert "test" in export


class TestAuditingVisionProvider:
    """Tests for AuditingVisionProvider class."""

    @pytest.mark.asyncio
    async def test_analyze_logs_events(self):
        """Test that analyze logs audit events."""
        mock_provider = create_mock_provider()
        provider, logger = create_auditing_provider(mock_provider)

        await provider.analyze_image(b"image")

        entries = logger.query()
        assert len(entries) >= 2  # Started and completed
        event_types = [e.event_type for e in entries]
        assert AuditEventType.REQUEST_STARTED in event_types
        assert AuditEventType.REQUEST_COMPLETED in event_types


# ============================================================================
# Multi-Region Tests
# ============================================================================


class TestRegionHealth:
    """Tests for RegionHealth class."""

    def test_initial_health(self):
        """Test initial health values."""
        health = RegionHealth(region=Region.US_EAST)
        assert health.status == RegionStatus.HEALTHY
        assert health.consecutive_failures == 0

    def test_update_success(self):
        """Test updating after success."""
        health = RegionHealth(region=Region.US_EAST)
        health.update_success(100.0)

        assert health.total_requests == 1
        assert health.total_successes == 1
        assert health.latency_ms == 100.0

    def test_update_failure_degrades_health(self):
        """Test that failures degrade health."""
        health = RegionHealth(region=Region.US_EAST)

        for _ in range(5):
            health.update_failure()

        assert health.status == RegionStatus.UNHEALTHY


class TestLatencyMeasurer:
    """Tests for LatencyMeasurer class."""

    def test_record_latency(self):
        """Test recording latency."""
        measurer = LatencyMeasurer()
        measurer.record_latency(Region.US_EAST, 100.0)

        latency = measurer.get_latency(Region.US_EAST)
        assert latency == 100.0

    def test_sorted_regions(self):
        """Test getting regions sorted by latency."""
        measurer = LatencyMeasurer()
        measurer.record_latency(Region.US_EAST, 100.0)
        measurer.record_latency(Region.EU_WEST, 50.0)
        measurer.record_latency(Region.ASIA_EAST, 200.0)

        sorted_regions = measurer.get_sorted_regions()
        assert sorted_regions[0][0] == Region.EU_WEST
        assert sorted_regions[-1][0] == Region.ASIA_EAST


class TestRegionRouter:
    """Tests for RegionRouter class."""

    def test_add_region(self):
        """Test adding a region."""
        router = RegionRouter()
        provider = create_mock_provider()

        router.add_region(RegionConfig(
            region=Region.US_EAST,
            provider=provider,
        ))

        available = router.get_available_regions()
        assert Region.US_EAST in available

    def test_select_region_latency_based(self):
        """Test latency-based region selection."""
        router = RegionRouter(strategy=RoutingStrategy.LATENCY)

        for region in [Region.US_EAST, Region.EU_WEST]:
            router.add_region(RegionConfig(
                region=region,
                provider=create_mock_provider(),
            ))

        # Record latencies
        router.record_result(Region.US_EAST, True, 100.0)
        router.record_result(Region.EU_WEST, True, 50.0)

        selected, fallbacks = router.select_region("req1")
        assert selected == Region.EU_WEST  # Lower latency

    def test_blocked_regions(self):
        """Test blocking regions."""
        router = RegionRouter(blocked_regions=[Region.US_EAST])

        router.add_region(RegionConfig(
            region=Region.US_EAST,
            provider=create_mock_provider(),
        ))

        available = router.get_available_regions()
        assert Region.US_EAST not in available


class TestMultiRegionVisionProvider:
    """Tests for MultiRegionVisionProvider class."""

    @pytest.mark.asyncio
    async def test_analyze_routes_to_region(self):
        """Test that analyze routes to a region."""
        provider1 = create_mock_provider("provider1")
        provider2 = create_mock_provider("provider2")

        multi_provider = create_multiregion_provider(
            regions=[
                (Region.US_EAST, provider1, 1.0),
                (Region.EU_WEST, provider2, 1.0),
            ],
            strategy=RoutingStrategy.ROUND_ROBIN,
        )

        result = await multi_provider.analyze_image(b"image")
        assert isinstance(result, VisionDescription)


# ============================================================================
# Priority Queue Tests
# ============================================================================


class TestPrioritizedRequest:
    """Tests for PrioritizedRequest class."""

    def test_request_creation(self):
        """Test creating a prioritized request."""
        request = PrioritizedRequest(
            priority_value=Priority.HIGH.value,
            sequence=1,
            request_id="req_001",
            priority=Priority.HIGH,
            image_data=b"image",
        )
        assert request.priority == Priority.HIGH
        assert request.status == RequestStatus.PENDING

    def test_is_expired(self):
        """Test expiration check."""
        request = PrioritizedRequest(
            priority_value=0,
            sequence=1,
            request_id="req_001",
            priority=Priority.NORMAL,
            image_data=b"image",
            deadline=datetime.now() - timedelta(hours=1),
        )
        assert request.is_expired


class TestHeapPriorityQueue:
    """Tests for HeapPriorityQueue class."""

    def test_enqueue_dequeue(self):
        """Test basic enqueue/dequeue."""
        queue = HeapPriorityQueue()

        request = PrioritizedRequest(
            priority_value=Priority.NORMAL.value,
            sequence=1,
            request_id="req1",
            priority=Priority.NORMAL,
            image_data=b"image",
        )
        queue.enqueue(request)

        assert queue.size() == 1
        dequeued = queue.dequeue()
        assert dequeued.request_id == "req1"
        assert queue.is_empty()

    def test_priority_ordering(self):
        """Test that higher priority requests are dequeued first."""
        queue = HeapPriorityQueue()

        low = PrioritizedRequest(
            priority_value=Priority.LOW.value,
            sequence=1,
            request_id="low",
            priority=Priority.LOW,
            image_data=b"image",
        )
        high = PrioritizedRequest(
            priority_value=Priority.HIGH.value,
            sequence=2,
            request_id="high",
            priority=Priority.HIGH,
            image_data=b"image",
        )

        queue.enqueue(low)
        queue.enqueue(high)

        first = queue.dequeue()
        assert first.request_id == "high"

    def test_max_size_limit(self):
        """Test queue max size limit."""
        queue = HeapPriorityQueue(max_size=2)

        for i in range(2):
            queue.enqueue(PrioritizedRequest(
                priority_value=Priority.NORMAL.value,
                sequence=i,
                request_id=f"req{i}",
                priority=Priority.NORMAL,
                image_data=b"image",
            ))

        with pytest.raises(QueueFullError):
            queue.enqueue(PrioritizedRequest(
                priority_value=Priority.NORMAL.value,
                sequence=3,
                request_id="req3",
                priority=Priority.NORMAL,
                image_data=b"image",
            ))


class TestMultiLevelQueue:
    """Tests for MultiLevelQueue class."""

    def test_strict_priority(self):
        """Test strict priority ordering."""
        queue = MultiLevelQueue(strategy=QueueStrategy.STRICT_PRIORITY)

        low = PrioritizedRequest(
            priority_value=Priority.LOW.value,
            sequence=1,
            request_id="low",
            priority=Priority.LOW,
            image_data=b"image",
        )
        critical = PrioritizedRequest(
            priority_value=Priority.CRITICAL.value,
            sequence=2,
            request_id="critical",
            priority=Priority.CRITICAL,
            image_data=b"image",
        )

        queue.enqueue(low)
        queue.enqueue(critical)

        first = queue.dequeue()
        assert first.request_id == "critical"

    def test_size_by_priority(self):
        """Test getting size by priority level."""
        queue = MultiLevelQueue()

        for _ in range(3):
            queue.enqueue(PrioritizedRequest(
                priority_value=Priority.HIGH.value,
                sequence=0,
                request_id="high",
                priority=Priority.HIGH,
                image_data=b"image",
            ))
        for _ in range(2):
            queue.enqueue(PrioritizedRequest(
                priority_value=Priority.LOW.value,
                sequence=0,
                request_id="low",
                priority=Priority.LOW,
                image_data=b"image",
            ))

        sizes = queue.size_by_priority()
        assert sizes[Priority.HIGH] == 3
        assert sizes[Priority.LOW] == 2


class TestRequestScheduler:
    """Tests for RequestScheduler class."""

    def test_submit_request(self):
        """Test submitting a request."""
        provider = create_mock_provider()
        scheduler = RequestScheduler(provider)

        request_id = scheduler.submit(
            image_data=b"image",
            priority=Priority.HIGH,
        )

        assert request_id is not None
        stats = scheduler.get_stats()
        assert stats.total_enqueued == 1

    @pytest.mark.asyncio
    async def test_process_next(self):
        """Test processing next request."""
        provider = create_mock_provider()
        scheduler = RequestScheduler(provider)

        scheduler.submit(b"image", Priority.NORMAL)
        result = await scheduler.process_next()

        assert result is not None
        request_id, description = result
        assert isinstance(description, VisionDescription)


class TestPrioritizedVisionProvider:
    """Tests for PrioritizedVisionProvider class."""

    @pytest.mark.asyncio
    async def test_analyze_with_priority(self):
        """Test analyzing with priority."""
        mock_provider = create_mock_provider()
        wrapper = create_prioritized_provider(mock_provider)

        result = await wrapper.analyze_image(b"image")
        assert isinstance(result, VisionDescription)

    @pytest.mark.asyncio
    async def test_critical_bypasses_queue(self):
        """Test that critical requests bypass queue."""
        mock_provider = create_mock_provider()
        wrapper = PrioritizedVisionProvider(mock_provider)

        result = await wrapper.analyze_with_priority(
            b"image", Priority.CRITICAL
        )

        assert isinstance(result, VisionDescription)
        # Critical should go directly, not through queue
        mock_provider.analyze_image.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase5Integration:
    """Integration tests for Phase 5 features."""

    @pytest.mark.asyncio
    async def test_validation_with_audit(self):
        """Test combining validation with audit logging."""
        mock_provider = create_mock_provider()

        # Add validation
        validator = ResponseValidator(STANDARD_SCHEMA)
        validated = ValidatingVisionProvider(mock_provider, validator)

        # Add audit logging
        audited, logger = create_auditing_provider(validated)

        result = await audited.analyze_image(b"image")
        assert isinstance(result, VisionDescription)

        # Check audit logs exist
        entries = logger.query()
        assert len(entries) >= 2

    @pytest.mark.asyncio
    async def test_priority_with_multiregion(self):
        """Test combining priority with multi-region."""
        provider1 = create_mock_provider("region1")
        provider2 = create_mock_provider("region2")

        multi = create_multiregion_provider(
            regions=[
                (Region.US_EAST, provider1, 1.0),
                (Region.EU_WEST, provider2, 1.0),
            ],
        )

        prioritized = create_prioritized_provider(multi)

        result = await prioritized.analyze_image(b"image")
        assert isinstance(result, VisionDescription)

    @pytest.mark.asyncio
    async def test_ab_testing_with_validation(self):
        """Test A/B testing with response validation."""
        control = create_mock_provider("control")
        treatment = create_mock_provider("treatment")

        ab_provider, manager = create_ab_testing_provider(
            experiment_id="exp1",
            name="Test",
            control=control,
            treatments=[("treatment", treatment, 0.5)],
        )

        # Add validation
        validator = ResponseValidator(STANDARD_SCHEMA)
        validated = ValidatingVisionProvider(ab_provider, validator)

        result = await validated.analyze_image(b"image")
        assert isinstance(result, VisionDescription)
