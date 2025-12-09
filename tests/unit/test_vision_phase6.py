"""Tests for Phase 6 Vision Provider features.

Phase 6 includes:
- Transformation pipelines
- Capability discovery
- Hot reload configuration
- Request deduplication
- Provider versioning
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from src.core.vision.base import VisionDescription, VisionProvider


# Helper functions
def create_test_description(
    summary: str = "Test summary",
    details: list = None,
    confidence: float = 0.85,
) -> VisionDescription:
    """Create a test VisionDescription."""
    return VisionDescription(
        summary=summary,
        details=details or ["Detail 1", "Detail 2"],
        confidence=confidence,
    )


def create_mock_provider(name: str = "mock") -> MagicMock:
    """Create a mock vision provider."""
    provider = MagicMock()
    provider.provider_name = name
    provider.analyze_image = AsyncMock(return_value=create_test_description())
    return provider


# ==================== Transformation Pipeline Tests ====================


class TestTransformationPipeline:
    """Tests for transformation pipeline module."""

    def test_transform_stage_enum(self):
        """Test TransformStage enum values."""
        from src.core.vision.transformation import TransformStage

        assert TransformStage.PRE_REQUEST.value == "pre_request"
        assert TransformStage.POST_RESPONSE.value == "post_response"
        assert TransformStage.ERROR.value == "error"

    def test_transform_priority_enum(self):
        """Test TransformPriority enum values."""
        from src.core.vision.transformation import TransformPriority

        assert TransformPriority.FIRST.value == 0
        assert TransformPriority.HIGH.value == 25
        assert TransformPriority.NORMAL.value == 50
        assert TransformPriority.LOW.value == 75
        assert TransformPriority.LAST.value == 100

    def test_transform_context_creation(self):
        """Test TransformContext creation."""
        from src.core.vision.transformation import TransformContext, TransformStage

        context = TransformContext(
            request_id="test-123",
            stage=TransformStage.PRE_REQUEST,
            provider_name="test_provider",
        )

        assert context.request_id == "test-123"
        assert context.stage == TransformStage.PRE_REQUEST
        assert context.provider_name == "test_provider"
        assert isinstance(context.metadata, dict)
        assert isinstance(context.errors, list)

    def test_transform_context_add_metadata(self):
        """Test adding metadata to context."""
        from src.core.vision.transformation import TransformContext, TransformStage

        context = TransformContext(
            request_id="test-123",
            stage=TransformStage.PRE_REQUEST,
            provider_name="test",
        )

        context.add_metadata("key1", "value1")
        context.add_metadata("key2", 123)

        assert context.metadata["key1"] == "value1"
        assert context.metadata["key2"] == 123

    def test_transform_context_add_error(self):
        """Test adding errors to context."""
        from src.core.vision.transformation import TransformContext, TransformStage

        context = TransformContext(
            request_id="test-123",
            stage=TransformStage.PRE_REQUEST,
            provider_name="test",
        )

        context.add_error("Error 1")
        context.add_error("Error 2")

        assert len(context.errors) == 2
        assert "Error 1" in context.errors

    def test_transformation_pipeline_creation(self):
        """Test TransformationPipeline creation."""
        from src.core.vision.transformation import TransformationPipeline

        pipeline = TransformationPipeline(name="test_pipeline")

        assert pipeline.name == "test_pipeline"
        assert pipeline.stats.total_transforms == 0

    def test_pipeline_stats(self):
        """Test PipelineStats tracking."""
        from src.core.vision.transformation import PipelineStats

        stats = PipelineStats()
        stats.record_transform("transform1", True, 10.5)
        stats.record_transform("transform1", True, 15.0)
        stats.record_transform("transform2", False, 5.0)

        assert stats.total_transforms == 3
        assert stats.successful_transforms == 2
        assert stats.failed_transforms == 1
        assert stats.total_duration_ms == 30.5
        assert stats.transforms_by_name["transform1"] == 2
        assert stats.errors_by_transform["transform2"] == 1

    def test_lambda_response_transformer(self):
        """Test LambdaResponseTransformer."""
        from src.core.vision.transformation import (
            LambdaResponseTransformer,
            TransformContext,
            TransformStage,
        )

        def boost_confidence(data: VisionDescription, ctx: TransformContext) -> VisionDescription:
            return VisionDescription(
                summary=data.summary,
                details=data.details,
                confidence=min(data.confidence * 1.2, 1.0),
            )

        transformer = LambdaResponseTransformer(
            name="boost",
            transform_fn=boost_confidence,
        )

        assert transformer.name == "boost"
        assert transformer.stage == TransformStage.POST_RESPONSE

        # Test transformation
        context = TransformContext(
            request_id="test",
            stage=TransformStage.POST_RESPONSE,
            provider_name="test",
        )
        input_data = create_test_description(confidence=0.7)
        output = transformer.transform(input_data, context)

        assert output.confidence == pytest.approx(0.84, rel=0.01)

    def test_confidence_boost_transformer(self):
        """Test ConfidenceBoostTransformer."""
        from src.core.vision.transformation import (
            ConfidenceBoostTransformer,
            TransformContext,
            TransformStage,
        )

        transformer = ConfidenceBoostTransformer(boost_factor=1.5, max_confidence=1.0)
        context = TransformContext(
            request_id="test",
            stage=TransformStage.POST_RESPONSE,
            provider_name="test",
        )

        # Test with value that would exceed 1.0
        input_data = create_test_description(confidence=0.8)
        output = transformer.transform(input_data, context)
        assert output.confidence == 1.0  # Capped at max

        # Test with value that stays under 1.0
        input_data = create_test_description(confidence=0.5)
        output = transformer.transform(input_data, context)
        assert output.confidence == pytest.approx(0.75, rel=0.01)

    def test_summary_prefix_transformer(self):
        """Test SummaryPrefixTransformer."""
        from src.core.vision.transformation import (
            SummaryPrefixTransformer,
            TransformContext,
            TransformStage,
        )

        transformer = SummaryPrefixTransformer(prefix="[ANALYZED] ")
        context = TransformContext(
            request_id="test",
            stage=TransformStage.POST_RESPONSE,
            provider_name="test",
        )

        input_data = create_test_description(summary="Original summary")
        output = transformer.transform(input_data, context)

        assert output.summary == "[ANALYZED] Original summary"

    def test_detail_filter_transformer(self):
        """Test DetailFilterTransformer."""
        from src.core.vision.transformation import (
            DetailFilterTransformer,
            TransformContext,
            TransformStage,
        )

        transformer = DetailFilterTransformer(
            min_length=5,
            keywords=["important"],
        )
        context = TransformContext(
            request_id="test",
            stage=TransformStage.POST_RESPONSE,
            provider_name="test",
        )

        input_data = VisionDescription(
            summary="Test",
            details=["ab", "important detail", "nothing here", "also important"],
            confidence=0.9,
        )
        output = transformer.transform(input_data, context)

        assert len(output.details) == 2
        assert "important detail" in output.details
        assert "also important" in output.details

    @pytest.mark.asyncio
    async def test_transforming_vision_provider(self):
        """Test TransformingVisionProvider."""
        from src.core.vision.transformation import (
            TransformingVisionProvider,
            TransformationPipeline,
            ConfidenceBoostTransformer,
        )

        mock_provider = create_mock_provider()
        pipeline = TransformationPipeline()
        pipeline.add_response_transformer(ConfidenceBoostTransformer(boost_factor=1.1))

        provider = TransformingVisionProvider(mock_provider, pipeline)

        result = await provider.analyze_image(b"test_image")

        assert result is not None
        mock_provider.analyze_image.assert_called_once()

    def test_create_transforming_provider(self):
        """Test create_transforming_provider factory."""
        from src.core.vision.transformation import create_transforming_provider

        mock_provider = create_mock_provider()
        provider = create_transforming_provider(mock_provider)

        assert provider.provider_name == "transforming_mock"


# ==================== Capability Discovery Tests ====================


class TestCapabilityDiscovery:
    """Tests for capability discovery module."""

    def test_capability_enum(self):
        """Test Capability enum values."""
        from src.core.vision.discovery import Capability

        assert Capability.IMAGE_ANALYSIS is not None
        assert Capability.OCR is not None
        assert Capability.FORMAT_JPEG is not None
        assert Capability.BATCH_PROCESSING is not None

    def test_capability_status_enum(self):
        """Test CapabilityStatus enum values."""
        from src.core.vision.discovery import CapabilityStatus

        assert CapabilityStatus.AVAILABLE.value == "available"
        assert CapabilityStatus.UNAVAILABLE.value == "unavailable"
        assert CapabilityStatus.DEGRADED.value == "degraded"
        assert CapabilityStatus.UNKNOWN.value == "unknown"

    def test_capability_info(self):
        """Test CapabilityInfo dataclass."""
        from src.core.vision.discovery import (
            CapabilityInfo,
            Capability,
            CapabilityStatus,
        )

        info = CapabilityInfo(
            capability=Capability.IMAGE_ANALYSIS,
            status=CapabilityStatus.AVAILABLE,
            version="1.0.0",
        )

        assert info.capability == Capability.IMAGE_ANALYSIS
        assert info.is_available() is True

        info.status = CapabilityStatus.UNAVAILABLE
        assert info.is_available() is False

    def test_provider_capabilities(self):
        """Test ProviderCapabilities class."""
        from src.core.vision.discovery import (
            ProviderCapabilities,
            Capability,
            CapabilityStatus,
        )

        caps = ProviderCapabilities(provider_name="test_provider")

        caps.add_capability(Capability.IMAGE_ANALYSIS, CapabilityStatus.AVAILABLE)
        caps.add_capability(Capability.OCR, CapabilityStatus.DEGRADED)

        assert caps.has_capability(Capability.IMAGE_ANALYSIS) is True
        assert caps.has_capability(Capability.OCR) is True
        assert caps.has_capability(Capability.FORMAT_PDF) is False

        available = caps.get_all_available()
        assert Capability.IMAGE_ANALYSIS in available
        assert Capability.OCR in available

    def test_capability_discovery_register(self):
        """Test CapabilityDiscovery registration."""
        from src.core.vision.discovery import (
            CapabilityDiscovery,
            Capability,
        )

        discovery = CapabilityDiscovery()

        caps = discovery.register_provider(
            "test_provider",
            [Capability.IMAGE_ANALYSIS, Capability.OCR],
        )

        assert caps.provider_name == "test_provider"
        assert caps.has_capability(Capability.IMAGE_ANALYSIS)
        assert caps.has_capability(Capability.OCR)

    def test_capability_requirement_matching(self):
        """Test CapabilityRequirement matching."""
        from src.core.vision.discovery import (
            CapabilityDiscovery,
            CapabilityRequirement,
            Capability,
        )

        discovery = CapabilityDiscovery()
        discovery.register_provider(
            "provider1",
            [Capability.IMAGE_ANALYSIS, Capability.OCR],
        )
        discovery.register_provider(
            "provider2",
            [Capability.IMAGE_ANALYSIS],
        )

        requirements = [
            CapabilityRequirement(capability=Capability.IMAGE_ANALYSIS, required=True),
            CapabilityRequirement(capability=Capability.OCR, required=True),
        ]

        result1 = discovery.match_requirements("provider1", requirements)
        result2 = discovery.match_requirements("provider2", requirements)

        assert result1.matches is True
        assert result1.score > result2.score
        assert result2.matches is False

    def test_find_matching_providers(self):
        """Test finding matching providers."""
        from src.core.vision.discovery import (
            CapabilityDiscovery,
            CapabilityRequirement,
            Capability,
        )

        discovery = CapabilityDiscovery()
        discovery.register_provider("p1", [Capability.IMAGE_ANALYSIS, Capability.OCR])
        discovery.register_provider("p2", [Capability.IMAGE_ANALYSIS])
        discovery.register_provider("p3", [Capability.OCR])

        requirements = [
            CapabilityRequirement(capability=Capability.IMAGE_ANALYSIS, required=True),
        ]

        matches = discovery.find_matching_providers(requirements)

        assert len(matches) >= 2
        assert matches[0].provider_name in ["p1", "p2"]

    @pytest.mark.asyncio
    async def test_capability_aware_provider(self):
        """Test CapabilityAwareVisionProvider."""
        from src.core.vision.discovery import (
            CapabilityAwareVisionProvider,
            CapabilityDiscovery,
            Capability,
        )

        mock_provider = create_mock_provider()
        discovery = CapabilityDiscovery()

        provider = CapabilityAwareVisionProvider(
            provider=mock_provider,
            discovery=discovery,
            capabilities=[Capability.IMAGE_ANALYSIS],
        )

        assert provider.has_capability(Capability.IMAGE_ANALYSIS) is True
        assert provider.has_capability(Capability.OCR) is False

        result = await provider.analyze_image(b"test")
        assert result is not None


# ==================== Hot Reload Tests ====================


class TestHotReload:
    """Tests for hot reload module."""

    def test_reload_trigger_enum(self):
        """Test ReloadTrigger enum values."""
        from src.core.vision.hot_reload import ReloadTrigger

        assert ReloadTrigger.FILE_CHANGE.value == "file_change"
        assert ReloadTrigger.MANUAL.value == "manual"
        assert ReloadTrigger.SCHEDULED.value == "scheduled"
        assert ReloadTrigger.API_CALL.value == "api_call"

    def test_reload_status_enum(self):
        """Test ReloadStatus enum values."""
        from src.core.vision.hot_reload import ReloadStatus

        assert ReloadStatus.PENDING.value == "pending"
        assert ReloadStatus.IN_PROGRESS.value == "in_progress"
        assert ReloadStatus.SUCCESS.value == "success"
        assert ReloadStatus.FAILED.value == "failed"

    def test_dict_config_source(self):
        """Test DictConfigSource."""
        from src.core.vision.hot_reload import DictConfigSource

        config = {"key1": "value1", "key2": 123}
        source = DictConfigSource(config, name="test")

        assert source.source_name == "dict:test"
        assert source.has_changed() is False

        loaded = source.load()
        assert loaded["key1"] == "value1"
        assert loaded["key2"] == 123

        source.update({"key1": "new_value"})
        assert source.has_changed() is True

    def test_hot_reload_config(self):
        """Test HotReloadConfig dataclass."""
        from src.core.vision.hot_reload import HotReloadConfig

        config = HotReloadConfig(
            enabled=True,
            poll_interval_seconds=10.0,
            max_retries=5,
        )

        assert config.enabled is True
        assert config.poll_interval_seconds == 10.0
        assert config.max_retries == 5
        assert config.rollback_on_error is True

    def test_config_validator(self):
        """Test ConfigValidator."""
        from src.core.vision.hot_reload import ConfigValidator

        validator = ConfigValidator()

        def check_required_key(config):
            if "required_key" not in config:
                return "Missing required_key"
            return None

        validator.add_rule(check_required_key)

        is_valid, errors = validator.validate({"required_key": "value"})
        assert is_valid is True
        assert len(errors) == 0

        is_valid, errors = validator.validate({})
        assert is_valid is False
        assert "Missing required_key" in errors

    def test_hot_reload_manager_load_initial(self):
        """Test HotReloadManager initial load."""
        from src.core.vision.hot_reload import HotReloadManager, DictConfigSource

        config = {"setting1": "value1"}
        source = DictConfigSource(config)
        manager = HotReloadManager(source)

        loaded = manager.load_initial()

        assert loaded["setting1"] == "value1"
        assert manager.version == 1
        assert manager.current_config is not None

    def test_hot_reload_manager_reload(self):
        """Test HotReloadManager reload."""
        from src.core.vision.hot_reload import (
            HotReloadManager,
            DictConfigSource,
            ReloadTrigger,
            ReloadStatus,
        )

        source = DictConfigSource({"v": 1})
        manager = HotReloadManager(source)
        manager.load_initial()

        source.update({"v": 2})
        event = manager.reload(ReloadTrigger.MANUAL)

        assert event.status == ReloadStatus.SUCCESS
        assert manager.version == 2
        assert manager.current_config["v"] == 2

    def test_hot_reload_manager_rollback(self):
        """Test HotReloadManager rollback."""
        from src.core.vision.hot_reload import HotReloadManager, DictConfigSource

        source = DictConfigSource({"v": 1})
        manager = HotReloadManager(source)
        manager.load_initial()

        source.update({"v": 2})
        manager.reload()

        assert manager.current_config["v"] == 2

        success = manager.rollback()
        assert success is True
        assert manager.current_config["v"] == 1

    @pytest.mark.asyncio
    async def test_hot_reloading_vision_provider(self):
        """Test HotReloadingVisionProvider."""
        from src.core.vision.hot_reload import (
            HotReloadingVisionProvider,
            HotReloadManager,
            DictConfigSource,
        )

        def factory(config):
            return create_mock_provider(config.get("name", "default"))

        source = DictConfigSource({"name": "test_provider"})
        manager = HotReloadManager(source)

        provider = HotReloadingVisionProvider(factory, manager)

        result = await provider.analyze_image(b"test")
        assert result is not None


# ==================== Deduplication Tests ====================


class TestDeduplication:
    """Tests for deduplication module."""

    def test_deduplication_strategy_enum(self):
        """Test DeduplicationStrategy enum values."""
        from src.core.vision.deduplication import DeduplicationStrategy

        assert DeduplicationStrategy.EXACT.value == "exact"
        assert DeduplicationStrategy.HASH.value == "hash"
        assert DeduplicationStrategy.SIMILARITY.value == "similarity"

    def test_hash_algorithm_enum(self):
        """Test HashAlgorithm enum values."""
        from src.core.vision.deduplication import HashAlgorithm

        assert HashAlgorithm.MD5.value == "md5"
        assert HashAlgorithm.SHA256.value == "sha256"

    def test_deduplication_config(self):
        """Test DeduplicationConfig dataclass."""
        from src.core.vision.deduplication import (
            DeduplicationConfig,
            DeduplicationStrategy,
        )

        config = DeduplicationConfig(
            strategy=DeduplicationStrategy.HASH,
            ttl_seconds=600.0,
            max_cache_size=500,
        )

        assert config.strategy == DeduplicationStrategy.HASH
        assert config.ttl_seconds == 600.0
        assert config.max_cache_size == 500

    def test_hash_key_generator(self):
        """Test HashKeyGenerator."""
        from src.core.vision.deduplication import HashKeyGenerator, HashAlgorithm

        generator = HashKeyGenerator(HashAlgorithm.SHA256)

        key1 = generator.generate_key(b"test_data", True)
        key2 = generator.generate_key(b"test_data", True)
        key3 = generator.generate_key(b"different_data", True)

        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key
        assert len(key1) == 64  # SHA256 hex length

    def test_deduplication_cache(self):
        """Test DeduplicationCache."""
        from src.core.vision.deduplication import DeduplicationCache

        cache = DeduplicationCache(max_size=100, ttl_seconds=300)

        description = create_test_description()
        cache.put("key1", description)

        result = cache.get("key1")
        assert result is not None
        assert result.summary == description.summary

        assert cache.get("nonexistent") is None

    def test_deduplication_cache_expiration(self):
        """Test cache expiration."""
        from src.core.vision.deduplication import DeduplicationCache, CachedResult
        from datetime import timedelta

        cache = DeduplicationCache(max_size=100, ttl_seconds=0.001)

        description = create_test_description()
        cache.put("key1", description)

        # Wait for expiration
        import time
        time.sleep(0.01)

        result = cache.get("key1")
        assert result is None  # Should be expired

    def test_deduplication_manager(self):
        """Test DeduplicationManager."""
        from src.core.vision.deduplication import DeduplicationManager

        manager = DeduplicationManager()

        # First request
        is_dup, result, key = manager.check_duplicate(b"test_image")
        assert is_dup is False
        assert result is None

        # Store result
        description = create_test_description()
        manager.store_result(key, description, len(b"test_image"))

        # Second request (should be duplicate)
        is_dup, result, _ = manager.check_duplicate(b"test_image")
        assert is_dup is True
        assert result is not None

    def test_deduplication_stats(self):
        """Test DeduplicationStats."""
        from src.core.vision.deduplication import DeduplicationStats

        stats = DeduplicationStats()
        stats.total_requests = 100
        stats.deduplicated_requests = 30
        stats.cache_hits = 30
        stats.cache_misses = 70

        assert stats.deduplication_rate == pytest.approx(0.3)
        assert stats.cache_hit_rate == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_deduplicating_vision_provider(self):
        """Test DeduplicatingVisionProvider."""
        from src.core.vision.deduplication import DeduplicatingVisionProvider

        mock_provider = create_mock_provider()
        provider = DeduplicatingVisionProvider(mock_provider)

        # First call
        result1 = await provider.analyze_image(b"test_image")
        assert mock_provider.analyze_image.call_count == 1

        # Second call (should be deduplicated)
        result2 = await provider.analyze_image(b"test_image")
        assert mock_provider.analyze_image.call_count == 1  # Not called again

        assert result1.summary == result2.summary


# ==================== Versioning Tests ====================


class TestVersioning:
    """Tests for versioning module."""

    def test_version_status_enum(self):
        """Test VersionStatus enum values."""
        from src.core.vision.versioning import VersionStatus

        assert VersionStatus.DEVELOPMENT.value == "development"
        assert VersionStatus.STABLE.value == "stable"
        assert VersionStatus.DEPRECATED.value == "deprecated"
        assert VersionStatus.RETIRED.value == "retired"

    def test_compatibility_level_enum(self):
        """Test CompatibilityLevel enum values."""
        from src.core.vision.versioning import CompatibilityLevel

        assert CompatibilityLevel.COMPATIBLE.value == "compatible"
        assert CompatibilityLevel.BREAKING.value == "breaking"

    def test_semantic_version_creation(self):
        """Test SemanticVersion creation."""
        from src.core.vision.versioning import SemanticVersion

        version = SemanticVersion(major=1, minor=2, patch=3)
        assert str(version) == "1.2.3"

        version_pre = SemanticVersion(major=2, minor=0, patch=0, prerelease="beta")
        assert str(version_pre) == "2.0.0-beta"

        version_build = SemanticVersion(
            major=1, minor=0, patch=0, prerelease="rc1", build="build123"
        )
        assert str(version_build) == "1.0.0-rc1+build123"

    def test_semantic_version_parse(self):
        """Test SemanticVersion parsing."""
        from src.core.vision.versioning import SemanticVersion

        v1 = SemanticVersion.parse("1.2.3")
        assert v1.major == 1
        assert v1.minor == 2
        assert v1.patch == 3

        v2 = SemanticVersion.parse("2.0.0-beta")
        assert v2.major == 2
        assert v2.prerelease == "beta"

        v3 = SemanticVersion.parse("1.0.0-rc1+build123")
        assert v3.prerelease == "rc1"
        assert v3.build == "build123"

    def test_semantic_version_comparison(self):
        """Test SemanticVersion comparison."""
        from src.core.vision.versioning import SemanticVersion

        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("1.1.0")
        v3 = SemanticVersion.parse("2.0.0")

        assert v1 < v2
        assert v2 < v3
        assert v1 == SemanticVersion.parse("1.0.0")

    def test_semantic_version_compatibility(self):
        """Test version compatibility checking."""
        from src.core.vision.versioning import SemanticVersion, CompatibilityLevel

        v1 = SemanticVersion.parse("1.0.0")
        v1_1 = SemanticVersion.parse("1.1.0")
        v2 = SemanticVersion.parse("2.0.0")

        assert v1.is_compatible_with(v1) == CompatibilityLevel.COMPATIBLE
        assert v1.is_compatible_with(v1_1) == CompatibilityLevel.BACKWARD_COMPATIBLE
        assert v1.is_compatible_with(v2) == CompatibilityLevel.BREAKING

    def test_provider_version(self):
        """Test ProviderVersion dataclass."""
        from src.core.vision.versioning import (
            ProviderVersion,
            SemanticVersion,
            VersionStatus,
        )

        mock_provider = create_mock_provider()
        pv = ProviderVersion(
            version=SemanticVersion.parse("1.0.0"),
            provider=mock_provider,
            status=VersionStatus.STABLE,
        )

        assert pv.is_available() is True
        assert pv.is_production_ready() is True

        pv.status = VersionStatus.RETIRED
        assert pv.is_available() is False

    def test_version_constraint(self):
        """Test VersionConstraint."""
        from src.core.vision.versioning import (
            VersionConstraint,
            ProviderVersion,
            SemanticVersion,
            VersionStatus,
        )

        constraint = VersionConstraint(
            min_version=SemanticVersion.parse("1.0.0"),
            max_version=SemanticVersion.parse("2.0.0"),
        )

        mock_provider = create_mock_provider()

        v1 = ProviderVersion(
            version=SemanticVersion.parse("1.5.0"),
            provider=mock_provider,
        )
        v2 = ProviderVersion(
            version=SemanticVersion.parse("0.9.0"),
            provider=mock_provider,
        )
        v3 = ProviderVersion(
            version=SemanticVersion.parse("2.1.0"),
            provider=mock_provider,
        )

        assert constraint.matches(v1) is True
        assert constraint.matches(v2) is False
        assert constraint.matches(v3) is False

    def test_version_registry(self):
        """Test VersionRegistry."""
        from src.core.vision.versioning import (
            VersionRegistry,
            SemanticVersion,
            VersionStatus,
        )

        registry = VersionRegistry()
        mock_provider = create_mock_provider()

        registry.register(
            "test_provider",
            SemanticVersion.parse("1.0.0"),
            mock_provider,
            VersionStatus.STABLE,
            is_default=True,
        )

        registry.register(
            "test_provider",
            SemanticVersion.parse("1.1.0"),
            mock_provider,
            VersionStatus.STABLE,
        )

        # Get default
        pv = registry.get_version("test_provider")
        assert pv is not None
        assert str(pv.version) == "1.0.0"

        # Get specific
        pv_specific = registry.get_version(
            "test_provider", SemanticVersion.parse("1.1.0")
        )
        assert pv_specific is not None
        assert str(pv_specific.version) == "1.1.0"

        # Get all versions
        all_versions = registry.get_all_versions("test_provider")
        assert len(all_versions) == 2

    def test_version_registry_deprecate(self):
        """Test version deprecation."""
        from src.core.vision.versioning import (
            VersionRegistry,
            SemanticVersion,
            VersionStatus,
        )

        registry = VersionRegistry()
        mock_provider = create_mock_provider()

        registry.register(
            "test_provider",
            SemanticVersion.parse("1.0.0"),
            mock_provider,
        )

        success = registry.deprecate("test_provider", SemanticVersion.parse("1.0.0"))
        assert success is True

        pv = registry.get_version("test_provider")
        assert pv.status == VersionStatus.DEPRECATED

    @pytest.mark.asyncio
    async def test_versioned_vision_provider(self):
        """Test VersionedVisionProvider."""
        from src.core.vision.versioning import (
            VersionedVisionProvider,
            VersionRegistry,
            SemanticVersion,
        )

        registry = VersionRegistry()
        mock_provider = create_mock_provider()

        registry.register(
            "test",
            SemanticVersion.parse("1.0.0"),
            mock_provider,
            is_default=True,
        )

        provider = VersionedVisionProvider("test", registry)

        result = await provider.analyze_image(b"test")
        assert result is not None
        assert "v1.0.0" in provider.provider_name


# ==================== Integration Tests ====================


class TestPhase6Integration:
    """Integration tests for Phase 6 features."""

    @pytest.mark.asyncio
    async def test_transformation_with_deduplication(self):
        """Test transformation pipeline with deduplication."""
        from src.core.vision.transformation import (
            TransformingVisionProvider,
            TransformationPipeline,
            ConfidenceBoostTransformer,
        )
        from src.core.vision.deduplication import DeduplicatingVisionProvider

        mock_provider = create_mock_provider()

        # Create transformation pipeline
        pipeline = TransformationPipeline()
        pipeline.add_response_transformer(ConfidenceBoostTransformer(boost_factor=1.1))

        # Wrap with transformation
        transforming = TransformingVisionProvider(mock_provider, pipeline)

        # Wrap with deduplication
        dedup = DeduplicatingVisionProvider(transforming)

        # First call
        result1 = await dedup.analyze_image(b"test_image")

        # Second call (deduplicated)
        result2 = await dedup.analyze_image(b"test_image")

        assert result1.summary == result2.summary
        assert mock_provider.analyze_image.call_count == 1

    @pytest.mark.asyncio
    async def test_capability_aware_with_versioning(self):
        """Test capability-aware provider with versioning."""
        from src.core.vision.discovery import (
            CapabilityAwareVisionProvider,
            CapabilityDiscovery,
            Capability,
        )
        from src.core.vision.versioning import (
            VersionRegistry,
            SemanticVersion,
        )

        mock_provider = create_mock_provider()

        # Setup versioning
        registry = VersionRegistry()
        registry.register(
            "test",
            SemanticVersion.parse("1.0.0"),
            mock_provider,
            is_default=True,
        )

        # Setup capability discovery
        discovery = CapabilityDiscovery()

        # Create capability-aware versioned provider
        pv = registry.get_version("test")
        capable = CapabilityAwareVisionProvider(
            provider=pv.provider,
            discovery=discovery,
            capabilities=[Capability.IMAGE_ANALYSIS, Capability.OCR],
        )

        assert capable.has_capability(Capability.IMAGE_ANALYSIS)
        result = await capable.analyze_image(b"test")
        assert result is not None
