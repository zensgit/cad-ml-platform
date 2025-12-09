"""
Tests for Vision Provider Phase 11 features.

Tests for:
- Encryption and Key Management
- Audit Logging and Compliance
- Data Pipeline (ETL)
- Stream Processing
- Data Validation
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.core.vision import VisionDescription, VisionProvider


# ============================================================================
# Mock Provider
# ============================================================================


class MockVisionProvider(VisionProvider):
    """Mock provider for testing."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def analyze_image(
        self, image_data: bytes, context: Optional[str] = None
    ) -> VisionDescription:
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate processing
        return VisionDescription(
            summary=f"Mock analysis from {self._name}",
            details=["Detailed mock description"],
            confidence=0.95,
        )


# ============================================================================
# Encryption Tests
# ============================================================================


class TestEncryption:
    """Tests for encryption module."""

    def test_encryption_algorithm_enum(self) -> None:
        """Test EncryptionAlgorithm enum."""
        from src.core.vision.encryption import EncryptionAlgorithm

        assert EncryptionAlgorithm.AES_256_GCM.value == "aes_256_gcm"
        assert EncryptionAlgorithm.RSA_4096.value == "rsa_4096"

    def test_key_metadata(self) -> None:
        """Test KeyMetadata."""
        from src.core.vision.encryption import (
            EncryptionAlgorithm,
            KeyMetadata,
            KeyStatus,
            KeyType,
        )

        metadata = KeyMetadata(
            key_id="key-1",
            key_type=KeyType.SYMMETRIC,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )

        assert metadata.key_id == "key-1"
        assert metadata.is_active()
        assert not metadata.is_expired()

        data = metadata.to_dict()
        assert data["key_id"] == "key-1"

    def test_key_metadata_expiration(self) -> None:
        """Test KeyMetadata expiration."""
        from src.core.vision.encryption import (
            EncryptionAlgorithm,
            KeyMetadata,
            KeyType,
        )

        metadata = KeyMetadata(
            key_id="key-1",
            key_type=KeyType.SYMMETRIC,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            expires_at=datetime.now() - timedelta(hours=1),
        )

        assert metadata.is_expired()
        assert not metadata.is_active()

    def test_in_memory_key_store(self) -> None:
        """Test InMemoryKeyStore."""
        from src.core.vision.encryption import (
            EncryptionAlgorithm,
            InMemoryKeyStore,
            KeyMetadata,
            KeyType,
        )

        store = InMemoryKeyStore()

        metadata = KeyMetadata(
            key_id="key-1",
            key_type=KeyType.SYMMETRIC,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )

        store.store_key("key-1", b"secret_key_data", metadata)

        result = store.get_key("key-1")
        assert result is not None
        key_data, stored_metadata = result
        assert key_data == b"secret_key_data"
        assert stored_metadata.key_id == "key-1"

    def test_key_manager_generate(self) -> None:
        """Test KeyManager key generation."""
        from src.core.vision.encryption import EncryptionAlgorithm, KeyManager

        manager = KeyManager()
        key_id = manager.generate_key(algorithm=EncryptionAlgorithm.AES_256_GCM)

        assert key_id is not None
        key = manager.get_key(key_id)
        assert key is not None
        assert len(key) == 32  # AES-256

    def test_key_manager_rotate(self) -> None:
        """Test KeyManager key rotation."""
        from src.core.vision.encryption import KeyManager

        manager = KeyManager()
        old_key_id = manager.generate_key()
        new_key_id = manager.rotate_key(old_key_id)

        assert new_key_id is not None
        assert new_key_id != old_key_id

    def test_simple_aes_encryptor(self) -> None:
        """Test SimpleAESEncryptor."""
        from src.core.vision.encryption import SimpleAESEncryptor

        encryptor = SimpleAESEncryptor()
        encryptor.set_key_id("test-key")

        key = b"0123456789abcdef0123456789abcdef"  # 32 bytes
        plaintext = b"Hello, World!"

        encrypted = encryptor.encrypt(plaintext, key)
        decrypted = encryptor.decrypt(encrypted, key)

        assert decrypted == plaintext

    def test_encryption_service(self) -> None:
        """Test EncryptionService."""
        from src.core.vision.encryption import EncryptionService

        service = EncryptionService()

        plaintext = b"Sensitive data"
        encrypted = service.encrypt(plaintext, purpose="test")
        decrypted = service.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encryption_service_string(self) -> None:
        """Test EncryptionService string encryption."""
        from src.core.vision.encryption import EncryptionService

        service = EncryptionService()

        plaintext = "Hello, encrypted world!"
        encrypted = service.encrypt_string(plaintext)
        decrypted = service.decrypt_string(encrypted)

        assert decrypted == plaintext

    def test_hasher(self) -> None:
        """Test Hasher."""
        from src.core.vision.encryption import HashAlgorithm, Hasher

        hasher = Hasher(HashAlgorithm.SHA256)

        data = b"test data"
        hash_result = hasher.hash(data)
        assert len(hash_result) == 32  # SHA256 produces 32 bytes

        hex_result = hasher.hash_hex(data)
        assert len(hex_result) == 64  # 32 bytes = 64 hex chars

    def test_hmac_authenticator(self) -> None:
        """Test HMACAuthenticator."""
        from src.core.vision.encryption import HMACAuthenticator

        key = b"secret_key"
        auth = HMACAuthenticator(key)

        data = b"message to sign"
        signature = auth.sign(data)

        assert auth.verify(data, signature)
        assert not auth.verify(b"wrong message", signature)

    def test_secure_storage(self) -> None:
        """Test SecureStorage."""
        from src.core.vision.encryption import SecureStorage

        storage = SecureStorage()

        storage.store("key1", b"secret value")
        value = storage.retrieve("key1")

        assert value == b"secret value"
        assert storage.exists("key1")
        assert not storage.exists("nonexistent")

    def test_secure_storage_expiration(self) -> None:
        """Test SecureStorage expiration."""
        from src.core.vision.encryption import SecureStorage

        storage = SecureStorage()
        storage.store("key1", b"expires soon", expires_in=timedelta(milliseconds=1))

        time.sleep(0.01)  # Wait for expiration
        assert storage.retrieve("key1") is None

    @pytest.mark.asyncio
    async def test_encrypted_vision_provider(self) -> None:
        """Test EncryptedVisionProvider."""
        from src.core.vision.encryption import create_encrypted_provider

        base_provider = MockVisionProvider()
        provider = create_encrypted_provider(base_provider)

        result = await provider.analyze_image(b"test image")

        assert result.summary == "Mock analysis from mock"
        assert "encrypted" in provider.provider_name


# ============================================================================
# Audit Logging Tests
# ============================================================================


class TestAuditLogging:
    """Tests for audit logging module."""

    def test_audit_event_type_enum(self) -> None:
        """Test AuditEventType enum."""
        from src.core.vision.audit_logging import AuditEventType

        assert AuditEventType.DATA_READ.value == "data_read"
        assert AuditEventType.IMAGE_ANALYZED.value == "image_analyzed"

    def test_audit_actor(self) -> None:
        """Test AuditActor."""
        from src.core.vision.audit_logging import AuditActor

        actor = AuditActor(
            actor_id="user-123",
            actor_type="user",
            ip_address="192.168.1.1",
        )

        assert actor.actor_id == "user-123"
        data = actor.to_dict()
        assert data["ip_address"] == "192.168.1.1"

    def test_audit_resource(self) -> None:
        """Test AuditResource."""
        from src.core.vision.audit_logging import AuditResource

        resource = AuditResource(
            resource_id="res-123",
            resource_type="image",
            resource_name="test.png",
        )

        assert resource.resource_id == "res-123"
        data = resource.to_dict()
        assert data["resource_type"] == "image"

    def test_audit_event(self) -> None:
        """Test AuditEvent."""
        from src.core.vision.audit_logging import AuditEvent, AuditEventType

        event = AuditEvent(
            event_id="evt-1",
            event_type=AuditEventType.DATA_READ,
            action="read_file",
            message="File was read",
        )

        assert event.event_id == "evt-1"
        assert event.get_hash() is not None

        data = event.to_dict()
        assert data["action"] == "read_file"

    def test_in_memory_audit_store(self) -> None:
        """Test InMemoryAuditStore."""
        from src.core.vision.audit_logging import (
            AuditEvent,
            AuditEventType,
            InMemoryAuditStore,
        )

        store = InMemoryAuditStore()

        event = AuditEvent(
            event_id="evt-1",
            event_type=AuditEventType.DATA_READ,
            action="test",
        )

        store.append(event)

        events = store.get_events()
        assert len(events) == 1
        assert events[0].event_id == "evt-1"

    def test_audit_logger(self) -> None:
        """Test AuditLogger."""
        from src.core.vision.audit_logging import AuditEventType, AuditLogger

        logger = AuditLogger()

        event = logger.log(
            event_type=AuditEventType.DATA_CREATED,
            action="create_record",
            message="Record created",
        )

        assert event.event_id is not None
        events = logger.get_events()
        assert len(events) == 1

    def test_audit_logger_correlation(self) -> None:
        """Test AuditLogger correlation ID."""
        from src.core.vision.audit_logging import AuditEventType, AuditLogger

        logger = AuditLogger()
        logger.set_correlation_id("corr-123")

        event = logger.log(
            event_type=AuditEventType.DATA_READ,
            action="test",
        )

        assert event.correlation_id == "corr-123"

    def test_audit_logger_access_event(self) -> None:
        """Test AuditLogger access logging."""
        from src.core.vision.audit_logging import (
            AuditActor,
            AuditLogger,
            AuditResource,
        )

        logger = AuditLogger()
        actor = AuditActor(actor_id="user-1", actor_type="user")
        resource = AuditResource(resource_id="res-1", resource_type="file")

        event = logger.log_access(actor, resource, granted=True)
        assert event.outcome == "success"

        event = logger.log_access(actor, resource, granted=False)
        assert event.outcome == "denied"

    def test_compliance_tracker(self) -> None:
        """Test ComplianceTracker."""
        from src.core.vision.audit_logging import (
            AuditLogger,
            ComplianceFramework,
            ComplianceRequirement,
            ComplianceTracker,
        )

        logger = AuditLogger()
        tracker = ComplianceTracker(logger)

        requirement = ComplianceRequirement(
            requirement_id="req-1",
            framework=ComplianceFramework.GDPR,
            name="Data Access Logging",
            description="Log all data access",
        )

        tracker.add_requirement(requirement)
        requirements = tracker.list_requirements()
        assert len(requirements) == 1

    def test_audit_trail(self) -> None:
        """Test AuditTrail."""
        from src.core.vision.audit_logging import AuditLogger, AuditTrail

        logger = AuditLogger()
        trail = AuditTrail(logger)

        trail.track_resource("res-1", "config", {"setting": "value1"})
        event = trail.update_resource("res-1", {"setting": "value2"})

        assert event is not None
        assert event.before_state == {"setting": "value1"}
        assert event.after_state == {"setting": "value2"}

    @pytest.mark.asyncio
    async def test_audited_vision_provider(self) -> None:
        """Test AuditedVisionProvider."""
        from src.core.vision.audit_logging import create_audited_provider

        base_provider = MockVisionProvider()
        provider = create_audited_provider(base_provider)

        result = await provider.analyze_image(b"test image")

        assert result.summary == "Mock analysis from mock"
        assert "audited" in provider.provider_name

        events = provider.get_audit_logger().get_events()
        assert len(events) >= 1


# ============================================================================
# Data Pipeline Tests
# ============================================================================


class TestDataPipeline:
    """Tests for data pipeline module."""

    def test_data_record(self) -> None:
        """Test DataRecord."""
        from src.core.vision.data_pipeline import DataFormat, DataRecord

        record = DataRecord(
            record_id="rec-1",
            data={"key": "value"},
            source="test",
        )

        assert record.record_id == "rec-1"
        assert record.format == DataFormat.DICT

        data = record.to_dict()
        assert data["source"] == "test"

    def test_in_memory_data_source(self) -> None:
        """Test InMemoryDataSource."""
        from src.core.vision.data_pipeline import DataRecord, InMemoryDataSource

        source = InMemoryDataSource()

        record1 = DataRecord(record_id="1", data={"a": 1})
        record2 = DataRecord(record_id="2", data={"b": 2})

        source.add_records([record1, record2])

        assert source.count() == 2

        batch = source.read_batch(1)
        assert len(batch) == 1

    def test_in_memory_data_sink(self) -> None:
        """Test InMemoryDataSink."""
        from src.core.vision.data_pipeline import DataRecord, InMemoryDataSink

        sink = InMemoryDataSink()

        record = DataRecord(record_id="1", data={"a": 1})
        sink.write(record)

        assert sink.count() == 1
        records = sink.get_records()
        assert records[0].record_id == "1"

    def test_map_transformer(self) -> None:
        """Test MapTransformer."""
        from src.core.vision.data_pipeline import DataRecord, MapTransformer

        transformer = MapTransformer(lambda x: x * 2)
        record = DataRecord(record_id="1", data=5)

        result = transformer.transform(record)

        assert result is not None
        assert result.data == 10

    def test_filter_transformer(self) -> None:
        """Test FilterTransformer."""
        from src.core.vision.data_pipeline import DataRecord, FilterTransformer

        transformer = FilterTransformer(lambda x: x > 5)

        record1 = DataRecord(record_id="1", data=3)
        record2 = DataRecord(record_id="2", data=10)

        assert transformer.transform(record1) is None
        assert transformer.transform(record2) is not None

    def test_field_extract_transformer(self) -> None:
        """Test FieldExtractTransformer."""
        from src.core.vision.data_pipeline import DataRecord, FieldExtractTransformer

        transformer = FieldExtractTransformer(["name", "age"])
        record = DataRecord(
            record_id="1",
            data={"name": "John", "age": 30, "email": "john@example.com"},
        )

        result = transformer.transform(record)

        assert result is not None
        assert result.data == {"name": "John", "age": 30}

    def test_enrich_transformer(self) -> None:
        """Test EnrichTransformer."""
        from src.core.vision.data_pipeline import DataRecord, EnrichTransformer

        transformer = EnrichTransformer({"source": "enriched"})
        record = DataRecord(record_id="1", data={"key": "value"})

        result = transformer.transform(record)

        assert result is not None
        assert result.data["source"] == "enriched"
        assert result.data["key"] == "value"

    def test_pipeline(self) -> None:
        """Test Pipeline."""
        from src.core.vision.data_pipeline import (
            DataRecord,
            InMemoryDataSink,
            InMemoryDataSource,
            MapTransformer,
            Pipeline,
            PipelineConfig,
            PipelineStage,
            PipelineStageType,
        )

        source = InMemoryDataSource([
            DataRecord(record_id="1", data=1),
            DataRecord(record_id="2", data=2),
            DataRecord(record_id="3", data=3),
        ])
        sink = InMemoryDataSink()

        config = PipelineConfig(pipeline_id="p1", name="test_pipeline")
        pipeline = Pipeline(config, source, sink)

        stage = PipelineStage(
            stage_id="s1",
            stage_type=PipelineStageType.TRANSFORM,
            name="double",
            transformer=MapTransformer(lambda x: x * 2),
        )
        pipeline.add_stage(stage)

        result = pipeline.run()

        assert result.total_records == 3
        assert result.processed_records == 3
        assert sink.count() == 3

    def test_pipeline_builder(self) -> None:
        """Test PipelineBuilder."""
        from src.core.vision.data_pipeline import (
            DataRecord,
            InMemoryDataSink,
            InMemoryDataSource,
            create_pipeline,
        )

        source = InMemoryDataSource([
            DataRecord(record_id="1", data=1),
            DataRecord(record_id="2", data=5),
            DataRecord(record_id="3", data=10),
        ])
        sink = InMemoryDataSink()

        pipeline = (
            create_pipeline("test")
            .from_source(source)
            .to_sink(sink)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 5)
            .build()
        )

        pipeline.set_source(source)
        pipeline.set_sink(sink)
        result = pipeline.run()

        assert result.processed_records <= 3

    def test_etl_pipeline(self) -> None:
        """Test ETLPipeline."""
        from src.core.vision.data_pipeline import create_etl_pipeline

        data = [1, 2, 3, 4, 5]
        results: List[int] = []

        pipeline = create_etl_pipeline(
            name="test_etl",
            extractor=lambda: iter(data),
            transformer=lambda x: x * 2,
            loader=lambda x: results.append(x) or True,
        )

        result = pipeline.run()

        assert result.total_records == 5
        assert results == [2, 4, 6, 8, 10]

    def test_batch_processor(self) -> None:
        """Test BatchProcessor."""
        from src.core.vision.data_pipeline import create_batch_processor

        processor = create_batch_processor(
            processor=lambda batch: [x * 2 for x in batch],
            batch_size=3,
        )

        results = processor.process_all([1, 2, 3, 4, 5])
        assert results == [2, 4, 6, 8, 10]


# ============================================================================
# Stream Processing Tests
# ============================================================================


class TestStreamProcessing:
    """Tests for stream processing module."""

    def test_stream_event(self) -> None:
        """Test StreamEvent."""
        from src.core.vision.stream_processing import StreamEvent

        event = StreamEvent(
            event_id="evt-1",
            data={"key": "value"},
            key="test-key",
        )

        assert event.event_id == "evt-1"
        data = event.to_dict()
        assert data["key"] == "test-key"

    def test_tumbling_window(self) -> None:
        """Test TumblingWindow."""
        from src.core.vision.stream_processing import StreamEvent, TumblingWindow

        window = TumblingWindow(size=timedelta(seconds=1))

        event = StreamEvent(event_id="1", data=1)
        window.add(event)

        events = window.get_events()
        assert len(events) == 1

    def test_sliding_window(self) -> None:
        """Test SlidingWindow."""
        from src.core.vision.stream_processing import SlidingWindow, StreamEvent

        window = SlidingWindow(
            size=timedelta(seconds=10),
            slide=timedelta(seconds=1),
        )

        event = StreamEvent(event_id="1", data=1)
        window.add(event)

        events = window.get_events()
        assert len(events) == 1

    def test_count_window(self) -> None:
        """Test CountWindow."""
        from src.core.vision.stream_processing import CountWindow, StreamEvent

        window = CountWindow(count=3)

        for i in range(5):
            window.add(StreamEvent(event_id=str(i), data=i))

        assert window.is_closed()
        events = window.get_events()
        assert len(events) == 3

    def test_map_operator(self) -> None:
        """Test MapOperator."""
        from src.core.vision.stream_processing import MapOperator, StreamEvent

        operator = MapOperator(lambda x: x * 2)
        event = StreamEvent(event_id="1", data=5)

        result = operator.process(event)

        assert result is not None
        assert result.data == 10

    def test_filter_operator(self) -> None:
        """Test FilterOperator."""
        from src.core.vision.stream_processing import FilterOperator, StreamEvent

        operator = FilterOperator(lambda x: x > 5)

        event1 = StreamEvent(event_id="1", data=3)
        event2 = StreamEvent(event_id="2", data=10)

        assert operator.process(event1) is None
        assert operator.process(event2) is not None

    def test_key_by_operator(self) -> None:
        """Test KeyByOperator."""
        from src.core.vision.stream_processing import KeyByOperator, StreamEvent

        operator = KeyByOperator(lambda x: str(x % 2))
        event = StreamEvent(event_id="1", data=5)

        result = operator.process(event)

        assert result is not None
        assert result.key == "1"

    def test_aggregators(self) -> None:
        """Test aggregators."""
        from src.core.vision.stream_processing import (
            AverageAggregator,
            CountAggregator,
            MaxAggregator,
            MinAggregator,
            SumAggregator,
        )

        # Sum
        sum_agg = SumAggregator()
        for i in [1, 2, 3]:
            sum_agg.add(i)
        assert sum_agg.get_result() == 6

        # Count
        count_agg = CountAggregator()
        for i in [1, 2, 3]:
            count_agg.add(i)
        assert count_agg.get_result() == 3

        # Average
        avg_agg = AverageAggregator()
        for i in [1, 2, 3]:
            avg_agg.add(i)
        assert avg_agg.get_result() == 2.0

        # Min
        min_agg = MinAggregator()
        for i in [3, 1, 2]:
            min_agg.add(i)
        assert min_agg.get_result() == 1

        # Max
        max_agg = MaxAggregator()
        for i in [1, 3, 2]:
            max_agg.add(i)
        assert max_agg.get_result() == 3

    def test_stream(self) -> None:
        """Test Stream."""
        from src.core.vision.stream_processing import Stream, StreamEvent

        stream: Stream[int] = Stream()
        results: List[int] = []

        stream.map(lambda x: x * 2)
        stream.filter(lambda x: x > 5)
        stream.add_sink(lambda e: results.append(e.data))

        # Process events directly
        for i in [1, 2, 3, 4, 5]:
            event = StreamEvent(event_id=str(i), data=i)
            stream.process_event(event)

        assert 10 in results  # 5 * 2 = 10
        assert 8 in results   # 4 * 2 = 8

    def test_stream_builder(self) -> None:
        """Test StreamBuilder."""
        from src.core.vision.stream_processing import create_stream

        stream = (
            create_stream()
            .map(lambda x: x * 2)
            .filter(lambda x: x > 0)
            .build()
        )

        assert stream is not None

    @pytest.mark.asyncio
    async def test_streaming_vision_provider(self) -> None:
        """Test StreamingVisionProvider."""
        from src.core.vision.stream_processing import create_streaming_provider

        base_provider = MockVisionProvider()
        provider = create_streaming_provider(base_provider)

        result = await provider.analyze_image(b"test image")

        assert result.summary == "Mock analysis from mock"
        assert "streaming" in provider.provider_name


# ============================================================================
# Data Validation Tests
# ============================================================================


class TestDataValidation:
    """Tests for data validation module."""

    def test_data_type_enum(self) -> None:
        """Test DataType enum."""
        from src.core.vision.data_validation import DataType

        assert DataType.STRING.value == "string"
        assert DataType.INTEGER.value == "integer"

    def test_validation_error(self) -> None:
        """Test ValidationError."""
        from src.core.vision.data_validation import ValidationError, ValidationSeverity

        error = ValidationError(
            error_id="err-1",
            field="name",
            message="Required field",
            severity=ValidationSeverity.ERROR,
        )

        assert error.field == "name"
        data = error.to_dict()
        assert data["severity"] == "error"

    def test_type_validator(self) -> None:
        """Test TypeValidator."""
        from src.core.vision.data_validation import DataType, TypeValidator

        validator = TypeValidator(DataType.STRING)

        result = validator.validate("hello", "field")
        assert result.valid

        result = validator.validate(123, "field")
        assert not result.valid

    def test_required_validator(self) -> None:
        """Test RequiredValidator."""
        from src.core.vision.data_validation import RequiredValidator

        validator = RequiredValidator()

        result = validator.validate("value", "field")
        assert result.valid

        result = validator.validate(None, "field")
        assert not result.valid

    def test_range_validator(self) -> None:
        """Test RangeValidator."""
        from src.core.vision.data_validation import RangeValidator

        validator = RangeValidator(min_value=0, max_value=100)

        assert validator.validate(50, "field").valid
        assert not validator.validate(-1, "field").valid
        assert not validator.validate(101, "field").valid

    def test_length_validator(self) -> None:
        """Test LengthValidator."""
        from src.core.vision.data_validation import LengthValidator

        validator = LengthValidator(min_length=2, max_length=10)

        assert validator.validate("hello", "field").valid
        assert not validator.validate("a", "field").valid
        assert not validator.validate("a" * 20, "field").valid

    def test_pattern_validator(self) -> None:
        """Test PatternValidator."""
        from src.core.vision.data_validation import PatternValidator

        validator = PatternValidator(r"^\d{3}-\d{4}$")

        assert validator.validate("123-4567", "field").valid
        assert not validator.validate("invalid", "field").valid

    def test_enum_validator(self) -> None:
        """Test EnumValidator."""
        from src.core.vision.data_validation import EnumValidator

        validator = EnumValidator(["a", "b", "c"])

        assert validator.validate("a", "field").valid
        assert not validator.validate("d", "field").valid

    def test_custom_validator(self) -> None:
        """Test CustomValidator."""
        from src.core.vision.data_validation import CustomValidator

        validator = CustomValidator(
            func=lambda x: isinstance(x, int) and x % 2 == 0,
            error_message="Must be even",
        )

        assert validator.validate(4, "field").valid
        assert not validator.validate(3, "field").valid

    def test_composite_validator(self) -> None:
        """Test CompositeValidator."""
        from src.core.vision.data_validation import (
            CompositeValidator,
            RangeValidator,
            TypeValidator,
            DataType,
        )

        validator = CompositeValidator([
            TypeValidator(DataType.INTEGER),
            RangeValidator(min_value=0, max_value=100),
        ])

        assert validator.validate(50, "field").valid
        assert not validator.validate("50", "field").valid
        assert not validator.validate(150, "field").valid

    def test_field_schema(self) -> None:
        """Test FieldSchema."""
        from src.core.vision.data_validation import DataType, FieldSchema

        schema = FieldSchema(
            name="age",
            data_type=DataType.INTEGER,
            required=True,
        )

        assert schema.validate(25).valid
        assert not schema.validate(None).valid

    def test_schema(self) -> None:
        """Test Schema."""
        from src.core.vision.data_validation import DataType, FieldSchema, Schema

        schema = Schema(
            name="user",
            fields=[
                FieldSchema(name="name", data_type=DataType.STRING, required=True),
                FieldSchema(name="age", data_type=DataType.INTEGER),
            ],
        )

        result = schema.validate({"name": "John", "age": 30})
        assert result.valid

        result = schema.validate({"age": 30})  # Missing name
        assert not result.valid

    def test_schema_builder(self) -> None:
        """Test SchemaBuilder."""
        from src.core.vision.data_validation import create_schema

        schema = (
            create_schema("person")
            .string_field("name", required=True, min_length=1)
            .integer_field("age", min_value=0, max_value=150)
            .enum_field("status", ["active", "inactive"])
            .build()
        )

        result = schema.validate({
            "name": "John",
            "age": 30,
            "status": "active",
        })
        assert result.valid

    def test_data_quality_checker(self) -> None:
        """Test DataQualityChecker."""
        from src.core.vision.data_validation import create_quality_checker

        checker = create_quality_checker()

        checker.add_check("not_empty", lambda x: len(x) > 0, "Data must not be empty")
        checker.add_check("has_key", lambda x: "id" in x, "Must have id")

        result = checker.check({"id": 1, "value": "test"})
        assert result["quality_score"] == 1.0

        result = checker.check({})
        assert result["quality_score"] < 1.0

    @pytest.mark.asyncio
    async def test_validated_vision_provider(self) -> None:
        """Test ValidatedVisionProvider."""
        from src.core.vision.data_validation import (
            create_schema,
            create_validated_provider,
        )

        # Allow extra fields since VisionDescription has summary, details, confidence
        output_schema = (
            create_schema("output")
            .string_field("summary", required=True)
            .float_field("confidence", min_value=0, max_value=1)
            .allow_extra(True)  # Allow details field
            .build()
        )

        base_provider = MockVisionProvider()
        provider = create_validated_provider(
            base_provider,
            output_schema=output_schema,
        )

        result = await provider.analyze_image(b"test image")

        assert result.summary == "Mock analysis from mock"
        assert "validated" in provider.provider_name


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase11Integration:
    """Integration tests for Phase 11 features."""

    @pytest.mark.asyncio
    async def test_encrypted_and_audited_provider(self) -> None:
        """Test encrypted and audited provider combination."""
        from src.core.vision.audit_logging import create_audited_provider
        from src.core.vision.encryption import create_encrypted_provider

        base_provider = MockVisionProvider()
        encrypted = create_encrypted_provider(base_provider)
        audited = create_audited_provider(encrypted)

        result = await audited.analyze_image(b"test image")

        assert result.summary == "Mock analysis from mock"
        events = audited.get_audit_logger().get_events()
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_validated_pipeline_provider(self) -> None:
        """Test validated provider with pipeline."""
        from src.core.vision.data_validation import (
            create_schema,
            create_validated_provider,
        )

        schema = create_schema("output").float_field("confidence").allow_extra(True).build()

        base_provider = MockVisionProvider()
        validated = create_validated_provider(base_provider, output_schema=schema)

        result = await validated.analyze_image(b"test image")
        assert result.confidence > 0

    def test_pipeline_with_validation(self) -> None:
        """Test pipeline with data validation."""
        from src.core.vision.data_pipeline import (
            DataRecord,
            FilterTransformer,
            InMemoryDataSink,
            InMemoryDataSource,
            MapTransformer,
            Pipeline,
            PipelineConfig,
            PipelineStage,
            PipelineStageType,
        )

        # Create pipeline that transforms and filters
        source = InMemoryDataSource([
            DataRecord(record_id="1", data={"value": 5}),
            DataRecord(record_id="2", data={"value": 15}),
            DataRecord(record_id="3", data={"value": 25}),
        ])
        sink = InMemoryDataSink()

        config = PipelineConfig(pipeline_id="p1", name="validated_pipeline")
        pipeline = Pipeline(config, source, sink)

        # Extract value and filter
        pipeline.add_stage(PipelineStage(
            stage_id="s1",
            stage_type=PipelineStageType.TRANSFORM,
            name="extract",
            transformer=MapTransformer(lambda x: x.get("value", 0)),
        ))
        pipeline.add_stage(PipelineStage(
            stage_id="s2",
            stage_type=PipelineStageType.FILTER,
            name="filter",
            transformer=FilterTransformer(lambda x: x > 10),
        ))

        result = pipeline.run()

        assert result.total_records == 3
        assert sink.count() == 2  # Only values > 10


# ============================================================================
# Import Tests
# ============================================================================


class TestPhase11Imports:
    """Tests for Phase 11 imports."""

    def test_encryption_imports(self) -> None:
        """Test encryption imports."""
        from src.core.vision import (
            EncryptionService,
            KeyManager,
            InMemoryKeyStore,
            Hasher,
            SecureStorage,
            EncryptionAlgorithm,
            KeyType,
            KeyStatus,
            create_encryption_service,
            create_encrypted_provider,
        )

        assert EncryptionService is not None
        assert KeyManager is not None
        assert create_encryption_service is not None

    def test_audit_logging_imports(self) -> None:
        """Test audit logging imports."""
        from src.core.vision import (
            AuditStore,
            AuditTrail,
            ComplianceFramework,
            AuditActor,
            AuditResource,
            create_audited_provider,
        )

        assert AuditStore is not None
        assert AuditTrail is not None
        assert create_audited_provider is not None

    def test_data_pipeline_imports(self) -> None:
        """Test data pipeline imports."""
        from src.core.vision import (
            Pipeline,
            PipelineBuilder,
            ETLPipeline,
            DataSource,
            DataSink,
            InMemoryDataSource,
            InMemoryDataSink,
            MapTransformer,
            FilterTransformer,
            PipelineStageType,
            PipelineStatus,
            DataFormat,
            DataRecord,
            create_pipeline,
            create_etl_pipeline,
        )

        assert Pipeline is not None
        assert PipelineBuilder is not None
        assert create_pipeline is not None

    def test_stream_processing_imports(self) -> None:
        """Test stream processing imports."""
        from src.core.vision import (
            Stream,
            StreamBuilder,
            Window,
            TumblingWindow,
            SlidingWindow,
            StreamOperator,
            MapOperator,
            FilterOperator,
            Aggregator,
            SumAggregator,
            CountAggregator,
            WindowType,
            StreamState,
            WindowResult,
            create_stream,
        )

        assert Stream is not None
        assert StreamBuilder is not None
        assert create_stream is not None

    def test_data_validation_imports(self) -> None:
        """Test data validation imports."""
        from src.core.vision import (
            Schema,
            SchemaBuilder,
            Validator,
            TypeValidator,
            RequiredValidator,
            RangeValidator,
            LengthValidator,
            PatternValidator,
            EnumValidator,
            CustomValidator,
            CompositeValidator,
            DataQualityChecker,
            ValidationStatus,
            DataType,
            FieldSchema,
            create_schema,
            create_validator,
            create_quality_checker,
            create_validated_provider,
        )

        assert Schema is not None
        assert SchemaBuilder is not None
        assert create_schema is not None
        assert create_validated_provider is not None
