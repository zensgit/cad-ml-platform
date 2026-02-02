"""Unit tests for Enterprise Features P44-P47.

P44: Feature Toggles Enhanced
P45: Secrets Management
P46: Audit Logging Enhanced
P47: Data Pipeline
"""

import asyncio
import json
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


# ============================================================================
# P44: Feature Toggles Enhanced Tests
# ============================================================================

class TestFeatureTogglesEnhanced:
    """Tests for Feature Toggles Enhanced module."""

    def test_toggle_state_enum(self):
        """Test ToggleState enum."""
        from src.core.feature_toggles_enhanced import ToggleState

        assert ToggleState.ON.value == "on"
        assert ToggleState.OFF.value == "off"

    def test_evaluation_context(self):
        """Test EvaluationContext."""
        from src.core.feature_toggles_enhanced import EvaluationContext

        context = EvaluationContext(
            user_id="user123",
            session_id="sess456",
            attributes={"plan": "premium"},
        )

        assert context.user_id == "user123"
        assert context.attributes["plan"] == "premium"

    def test_always_on_rule(self):
        """Test AlwaysOnRule."""
        from src.core.feature_toggles_enhanced import AlwaysOnRule, EvaluationContext

        rule = AlwaysOnRule()
        result = rule.evaluate(EvaluationContext())

        assert result.enabled is True

    def test_always_off_rule(self):
        """Test AlwaysOffRule."""
        from src.core.feature_toggles_enhanced import AlwaysOffRule, EvaluationContext

        rule = AlwaysOffRule()
        result = rule.evaluate(EvaluationContext())

        assert result.enabled is False

    def test_percentage_rule(self):
        """Test PercentageRule."""
        from src.core.feature_toggles_enhanced import PercentageRule, EvaluationContext

        # 100% should always be enabled
        rule = PercentageRule(percentage=100)
        for i in range(10):
            result = rule.evaluate(EvaluationContext(user_id=f"user{i}"))
            assert result.enabled is True

        # 0% should always be disabled
        rule = PercentageRule(percentage=0)
        for i in range(10):
            result = rule.evaluate(EvaluationContext(user_id=f"user{i}"))
            assert result.enabled is False

    def test_user_id_rule(self):
        """Test UserIdRule."""
        from src.core.feature_toggles_enhanced import UserIdRule, EvaluationContext

        rule = UserIdRule(user_ids={"user1", "user2"})

        # Allowed user
        result = rule.evaluate(EvaluationContext(user_id="user1"))
        assert result.enabled is True

        # Not allowed user
        result = rule.evaluate(EvaluationContext(user_id="user3"))
        assert result.enabled is False

    def test_attribute_rule(self):
        """Test AttributeRule."""
        from src.core.feature_toggles_enhanced import AttributeRule, EvaluationContext

        rule = AttributeRule(
            attribute="plan",
            values={"premium", "enterprise"},
        )

        # Matching attribute
        result = rule.evaluate(EvaluationContext(attributes={"plan": "premium"}))
        assert result.enabled is True

        # Non-matching attribute
        result = rule.evaluate(EvaluationContext(attributes={"plan": "free"}))
        assert result.enabled is False

    def test_feature_toggle(self):
        """Test FeatureToggle."""
        from src.core.feature_toggles_enhanced import (
            FeatureToggle, ToggleState, AlwaysOnRule, EvaluationContext
        )

        toggle = FeatureToggle(
            name="test_feature",
            description="A test feature",
            state=ToggleState.ON,
            rules=[AlwaysOnRule()],
        )

        assert toggle.name == "test_feature"
        assert toggle.is_enabled(EvaluationContext()) is True

    def test_composite_rule(self):
        """Test CompositeRule."""
        from src.core.feature_toggles_enhanced import (
            CompositeRule, UserIdRule, AttributeRule, EvaluationContext
        )

        # AND composition
        rule = CompositeRule(
            operator="and",
            rules=[
                UserIdRule(user_ids={"user1"}),
                AttributeRule(attribute="plan", values={"premium"}),
            ],
        )

        # Both conditions met
        result = rule.evaluate(EvaluationContext(
            user_id="user1",
            attributes={"plan": "premium"},
        ))
        assert result.enabled is True

        # Only one condition met
        result = rule.evaluate(EvaluationContext(
            user_id="user1",
            attributes={"plan": "free"},
        ))
        assert result.enabled is False

    @pytest.mark.asyncio
    async def test_in_memory_toggle_store(self):
        """Test InMemoryToggleStore."""
        from src.core.feature_toggles_enhanced import (
            InMemoryToggleStore, FeatureToggle, ToggleState
        )

        store = InMemoryToggleStore()
        toggle = FeatureToggle(name="test", state=ToggleState.ON)

        await store.save(toggle)
        loaded = await store.get("test")

        assert loaded is not None
        assert loaded.name == "test"

        # List toggles
        names = await store.list_toggles()
        assert "test" in names

        # Delete
        await store.delete("test")
        assert await store.get("test") is None

    @pytest.mark.asyncio
    async def test_toggle_manager(self):
        """Test ToggleManager."""
        from src.core.feature_toggles_enhanced import (
            ToggleManager, ToggleState, EvaluationContext
        )

        manager = ToggleManager()

        # Create toggle
        toggle = await manager.create_toggle(
            name="my_feature",
            description="My feature",
            default_state=ToggleState.ON,
        )
        assert toggle.name == "my_feature"

        # Check if enabled
        context = EvaluationContext(user_id="user1")
        enabled = await manager.is_enabled("my_feature", context)
        assert enabled is True

        # Disable toggle
        await manager.disable("my_feature")
        enabled = await manager.is_enabled("my_feature", context)
        assert enabled is False

    @pytest.mark.asyncio
    async def test_gradual_rollout(self):
        """Test gradual rollout."""
        from src.core.feature_toggles_enhanced import (
            ToggleManager, ToggleState, RolloutConfig
        )

        manager = ToggleManager()

        # Create toggle
        await manager.create_toggle(
            name="rollout_feature",
            default_state=ToggleState.OFF,
        )

        # Start rollout
        config = RolloutConfig(
            initial_percentage=10,
            target_percentage=50,
            increment=10,
            interval_seconds=0.1,
        )

        success = await manager.start_gradual_rollout("rollout_feature", config)
        assert success is True

        # Wait a bit for rollout
        await asyncio.sleep(0.3)

        # Should have rolled out some
        toggle = await manager.get_toggle("rollout_feature")
        assert toggle is not None


# ============================================================================
# P45: Secrets Management Tests
# ============================================================================

class TestSecretsManagement:
    """Tests for Secrets Management module."""

    def test_secret_type_enum(self):
        """Test SecretType enum."""
        from src.core.secrets import SecretType

        assert SecretType.PASSWORD.value == "password"
        assert SecretType.API_KEY.value == "api_key"
        assert SecretType.GENERIC.value == "generic"

    def test_secret_metadata(self):
        """Test SecretMetadata."""
        from src.core.secrets import SecretMetadata, SecretType

        metadata = SecretMetadata(
            name="test_secret",
            secret_type=SecretType.PASSWORD,
            description="A test secret",
            owner="test_user",
        )

        assert metadata.name == "test_secret"
        assert metadata.secret_type == SecretType.PASSWORD
        assert metadata.is_expired is False

    def test_secret_expiration(self):
        """Test secret expiration check."""
        from src.core.secrets import SecretMetadata

        # Not expired
        metadata = SecretMetadata(
            name="test",
            expires_at=datetime.utcnow() + timedelta(days=1),
        )
        assert metadata.is_expired is False

        # Expired
        metadata = SecretMetadata(
            name="test",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        assert metadata.is_expired is True

    def test_key_derivation(self):
        """Test KeyDerivation."""
        from src.core.secrets import KeyDerivation

        # Generate password
        password = KeyDerivation.generate_password(32)
        assert len(password) == 32

        # Generate API key
        api_key = KeyDerivation.generate_api_key("sk")
        assert api_key.startswith("sk_")

        # Derive key from password
        key, salt = KeyDerivation.derive_key("mypassword")
        assert len(key) == 32
        assert len(salt) == 16

        # Same password + salt = same key
        key2, _ = KeyDerivation.derive_key("mypassword", salt)
        assert key == key2

    def test_fernet_encryptor(self):
        """Test FernetEncryptor."""
        from src.core.secrets import FernetEncryptor

        encryptor = FernetEncryptor()

        # Encrypt
        plaintext = b"secret data"
        ciphertext, nonce, tag = encryptor.encrypt(plaintext)

        assert ciphertext != plaintext

        # Decrypt
        decrypted = encryptor.decrypt(ciphertext, nonce, tag)
        assert decrypted == plaintext

    def test_access_policy(self):
        """Test AccessPolicy."""
        from src.core.secrets import AccessPolicy

        policy = AccessPolicy(
            allowed_identities={"admin", "service1"},
            allowed_services={"api-server"},
        )

        # Allowed
        assert policy.is_allowed(identity="admin", service="api-server") is True

        # Not allowed identity
        assert policy.is_allowed(identity="user1", service="api-server") is False

    @pytest.mark.asyncio
    async def test_in_memory_secret_store(self):
        """Test InMemorySecretStore."""
        from src.core.secrets import (
            InMemorySecretStore, Secret, SecretMetadata, FernetEncryptor
        )

        encryptor = FernetEncryptor()
        store = InMemorySecretStore(encryptor)

        # Create and store secret
        encrypted, nonce, tag = encryptor.encrypt(b"my_secret")
        secret = Secret(
            name="test_secret",
            encrypted_value=encrypted,
            metadata=SecretMetadata(name="test_secret"),
            nonce=nonce,
            tag=tag,
        )

        await store.put(secret)

        # Retrieve
        loaded = await store.get("test_secret")
        assert loaded is not None
        assert loaded.name == "test_secret"

        # List
        names = await store.list_secrets()
        assert "test_secret" in names

        # Delete
        result = await store.delete("test_secret")
        assert result is True

    @pytest.mark.asyncio
    async def test_secrets_manager(self):
        """Test SecretsManager."""
        from src.core.secrets import SecretsManager, SecretType

        manager = SecretsManager()

        # Create secret
        secret = await manager.create_secret(
            name="db_password",
            value="super_secret_123",
            secret_type=SecretType.PASSWORD,
            description="Database password",
        )

        assert secret.name == "db_password"

        # Get secret
        value = await manager.get_secret("db_password")
        assert value == "super_secret_123"

        # Update secret
        new_secret = await manager.update_secret("db_password", "new_password_456")
        assert new_secret is not None
        assert new_secret.metadata.version == 2

        # Get updated value
        value = await manager.get_secret("db_password")
        assert value == "new_password_456"

        # Delete
        result = await manager.delete_secret("db_password")
        assert result is True

    @pytest.mark.asyncio
    async def test_secret_rotation(self):
        """Test secret rotation."""
        from src.core.secrets import SecretsManager, SecretType

        manager = SecretsManager()

        # Create secret
        await manager.create_secret(
            name="api_key",
            value="old_key",
            secret_type=SecretType.API_KEY,
        )

        # Rotate
        result = await manager.rotate_secret("api_key", new_value="new_key")
        assert result.success is True
        assert result.old_version == 1
        assert result.new_version == 2

        # Verify new value
        value = await manager.get_secret("api_key")
        assert value == "new_key"

    @pytest.mark.asyncio
    async def test_generate_and_store(self):
        """Test generate_and_store."""
        from src.core.secrets import SecretsManager, SecretType

        manager = SecretsManager()

        # Generate password
        secret, value = await manager.generate_and_store(
            name="generated_password",
            secret_type=SecretType.PASSWORD,
        )

        assert secret.name == "generated_password"
        assert len(value) == 32  # Default password length

        # Generate API key
        secret, value = await manager.generate_and_store(
            name="generated_api_key",
            secret_type=SecretType.API_KEY,
        )

        assert "sk_" in value or len(value) > 30


# ============================================================================
# P46: Audit Logging Enhanced Tests
# ============================================================================

class TestAuditLoggingEnhanced:
    """Tests for Audit Logging Enhanced module."""

    def test_audit_category_enum(self):
        """Test AuditCategory enum."""
        from src.core.audit_enhanced import AuditCategory

        assert AuditCategory.AUTHENTICATION.value == "authentication"
        assert AuditCategory.DATA_ACCESS.value == "data_access"

    def test_audit_context(self):
        """Test AuditContext."""
        from src.core.audit_enhanced import AuditContext

        context = AuditContext(
            user_id="user123",
            session_id="sess456",
            ip_address="192.168.1.1",
        )

        assert context.user_id == "user123"
        data = context.to_dict()
        assert data["user_id"] == "user123"

    def test_audit_record(self):
        """Test AuditRecord."""
        from src.core.audit_enhanced import (
            AuditRecord, AuditCategory, AuditSeverity, AuditOutcome
        )

        record = AuditRecord(
            record_id="rec123",
            timestamp=datetime.utcnow(),
            category=AuditCategory.AUTHENTICATION,
            action="login",
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
        )

        assert record.record_id == "rec123"
        assert record.action == "login"

        # To dict
        data = record.to_dict()
        assert data["category"] == "authentication"
        assert data["action"] == "login"

        # From dict
        restored = AuditRecord.from_dict(data)
        assert restored.record_id == record.record_id

    def test_record_integrity(self):
        """Test RecordIntegrity."""
        from src.core.audit_enhanced import (
            AuditRecord, AuditCategory, RecordIntegrity
        )

        integrity = RecordIntegrity()

        record = AuditRecord(
            record_id="rec123",
            timestamp=datetime.utcnow(),
            category=AuditCategory.AUTHENTICATION,
            action="login",
        )

        # Compute hash
        hash_value = integrity.compute_hash(record)
        assert len(hash_value) == 64  # SHA-256 hex

        # Sign record
        record.record_hash = hash_value
        signature = integrity.sign_record(record)
        record.signature = signature

        # Verify
        assert integrity.verify_hash(record) is True
        assert integrity.verify_signature(record) is True

    def test_record_builder(self):
        """Test RecordBuilder."""
        from src.core.audit_enhanced import (
            RecordBuilder, AuditCategory, AuditSeverity
        )

        builder = RecordBuilder()

        record = builder.create(
            category=AuditCategory.DATA_ACCESS,
            action="read",
            severity=AuditSeverity.INFO,
            resource_type="user",
            resource_id="123",
        )

        assert record.record_hash is not None
        assert record.signature is not None

    def test_audit_chain(self):
        """Test AuditChain."""
        from src.core.audit_enhanced import AuditChain, AuditCategory

        chain = AuditChain()

        # Add records
        r1 = chain.append(AuditCategory.AUTHENTICATION, "login")
        r2 = chain.append(AuditCategory.DATA_ACCESS, "read")
        r3 = chain.append(AuditCategory.DATA_ACCESS, "write")

        assert chain.metadata.record_count == 3
        assert r2.previous_hash == r1.record_hash
        assert r3.previous_hash == r2.record_hash

        # Verify chain
        valid, index = chain.verify()
        assert valid is True
        assert index is None

    @pytest.mark.asyncio
    async def test_in_memory_audit_storage(self):
        """Test InMemoryAuditStorage."""
        from src.core.audit_enhanced import (
            InMemoryAuditStorage, AuditRecord, AuditCategory
        )

        storage = InMemoryAuditStorage()

        record = AuditRecord(
            record_id="rec123",
            timestamp=datetime.utcnow(),
            category=AuditCategory.AUTHENTICATION,
            action="login",
        )

        # Store
        result = await storage.store(record)
        assert result is True

        # Get
        loaded = await storage.get("rec123")
        assert loaded is not None
        assert loaded.record_id == "rec123"

        # Query
        records = await storage.query(
            categories=[AuditCategory.AUTHENTICATION],
        )
        assert len(records) == 1

    def test_filter_builder(self):
        """Test FilterBuilder."""
        from src.core.audit_enhanced import (
            FilterBuilder, AuditRecord, AuditCategory, AuditSeverity
        )

        record = AuditRecord(
            record_id="rec123",
            timestamp=datetime.utcnow(),
            category=AuditCategory.AUTHENTICATION,
            action="login",
            severity=AuditSeverity.INFO,
        )

        # Build filter
        filter_group = (
            FilterBuilder()
            .where("action").equals("login")
            .where("severity").equals(AuditSeverity.INFO)
            .build()
        )

        # Evaluate
        assert filter_group.evaluate(record) is True

        # Non-matching filter
        filter_group = (
            FilterBuilder()
            .where("action").equals("logout")
            .build()
        )
        assert filter_group.evaluate(record) is False

    @pytest.mark.asyncio
    async def test_audit_manager(self):
        """Test AuditManager."""
        from src.core.audit_enhanced import (
            AuditManager, AuditCategory, AuditSeverity
        )

        manager = AuditManager()

        # Log event
        record = await manager.log(
            category=AuditCategory.AUTHENTICATION,
            action="login",
            description="User logged in",
        )

        assert record.record_id is not None
        assert record.category == AuditCategory.AUTHENTICATION

        # Log authentication event
        record = await manager.log_authentication(
            user_id="user123",
            action="login",
            success=True,
            ip_address="192.168.1.1",
        )

        assert record.context.user_id == "user123"

        # Log data access
        record = await manager.log_data_access(
            user_id="user123",
            resource_type="document",
            resource_id="doc456",
            action="read",
        )

        assert record.resource_type == "document"

    @pytest.mark.asyncio
    async def test_compliance_report(self):
        """Test compliance report generation."""
        from src.core.audit_enhanced import AuditManager, AuditCategory

        manager = AuditManager()

        # Log some events
        await manager.log(AuditCategory.AUTHENTICATION, "login")
        await manager.log(AuditCategory.DATA_ACCESS, "read")
        await manager.log(AuditCategory.SECURITY, "access_denied")

        # Generate report
        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow() + timedelta(hours=1)

        report = await manager.generate_compliance_report(start, end)

        assert report.total_records == 3
        assert "authentication" in report.records_by_category


# ============================================================================
# P47: Data Pipeline Tests
# ============================================================================

class TestDataPipeline:
    """Tests for Data Pipeline module."""

    def test_map_transformer(self):
        """Test MapTransformer."""
        from src.core.data_pipeline import MapTransformer

        transformer = MapTransformer(
            mappings={"old_name": "new_name", "old_value": "new_value"},
            include_unmapped=False,
        )

        data = {"old_name": "John", "old_value": 100, "extra": "ignored"}
        result = transformer.transform(data)

        assert result["new_name"] == "John"
        assert result["new_value"] == 100
        assert "extra" not in result

    def test_select_transformer(self):
        """Test SelectTransformer."""
        from src.core.data_pipeline import SelectTransformer

        transformer = SelectTransformer(fields=["name", "age"])

        data = {"name": "John", "age": 30, "city": "NYC"}
        result = transformer.transform(data)

        assert "name" in result
        assert "age" in result
        assert "city" not in result

    def test_type_convert_transformer(self):
        """Test TypeConvertTransformer."""
        from src.core.data_pipeline import TypeConvertTransformer

        transformer = TypeConvertTransformer(
            conversions={"age": "int", "price": "float", "active": "bool"},
        )

        data = {"age": "30", "price": "99.99", "active": "true"}
        result = transformer.transform(data)

        assert result["age"] == 30
        assert result["price"] == 99.99
        assert result["active"] is True

    def test_compute_transformer(self):
        """Test ComputeTransformer."""
        from src.core.data_pipeline import ComputeTransformer

        transformer = ComputeTransformer(
            computations={
                "full_name": lambda d: f"{d.get('first', '')} {d.get('last', '')}",
                "total": lambda d: d.get("price", 0) * d.get("quantity", 0),
            }
        )

        data = {"first": "John", "last": "Doe", "price": 10, "quantity": 5}
        result = transformer.transform(data)

        assert result["full_name"] == "John Doe"
        assert result["total"] == 50

    def test_filter_transformer(self):
        """Test FilterTransformer."""
        from src.core.data_pipeline import FilterTransformer

        transformer = FilterTransformer(
            predicate=lambda d: d.get("age", 0) >= 18
        )

        # Passes filter
        data = {"name": "John", "age": 25}
        result = transformer.transform(data)
        assert result is not None

        # Filtered out
        data = {"name": "Child", "age": 10}
        result = transformer.transform(data)
        assert result is None

    def test_flatten_transformer(self):
        """Test FlattenTransformer."""
        from src.core.data_pipeline import FlattenTransformer

        transformer = FlattenTransformer(separator=".")

        data = {
            "user": {
                "name": "John",
                "address": {
                    "city": "NYC",
                    "zip": "10001",
                }
            },
            "age": 30,
        }

        result = transformer.transform(data)

        assert result["user.name"] == "John"
        assert result["user.address.city"] == "NYC"
        assert result["age"] == 30

    def test_chain_transformer(self):
        """Test ChainTransformer."""
        from src.core.data_pipeline import (
            ChainTransformer, SelectTransformer, ComputeTransformer
        )

        transformer = ChainTransformer([
            SelectTransformer(fields=["first", "last"]),
            ComputeTransformer(
                computations={"full_name": lambda d: f"{d.get('first', '')} {d.get('last', '')}"}
            ),
        ])

        data = {"first": "John", "last": "Doe", "age": 30}
        result = transformer.transform(data)

        assert result["full_name"] == "John Doe"
        assert "age" not in result

    def test_group_by_aggregator(self):
        """Test GroupByAggregator."""
        from src.core.data_pipeline import GroupByAggregator, Aggregator, AggregatorType

        aggregator = GroupByAggregator(
            group_by=["category"],
            aggregators=[
                Aggregator("amount", AggregatorType.SUM, alias="total"),
                Aggregator("amount", AggregatorType.COUNT, alias="count"),
            ],
        )

        records = [
            {"category": "A", "amount": 100},
            {"category": "A", "amount": 200},
            {"category": "B", "amount": 50},
        ]

        results = aggregator.aggregate(records)

        # Find category A
        cat_a = next(r for r in results if r["category"] == "A")
        assert cat_a["total"] == 300
        assert cat_a["count"] == 2

    def test_stream_event(self):
        """Test StreamEvent."""
        from src.core.data_pipeline import StreamEvent

        event = StreamEvent(
            data={"key": "value"},
            key="my_key",
        )

        assert event.data["key"] == "value"
        assert event.key == "my_key"
        assert event.timestamp is not None

    @pytest.mark.asyncio
    async def test_in_memory_source_sink(self):
        """Test InMemorySource and InMemorySink."""
        from src.core.data_pipeline import InMemorySource, InMemorySink

        # Source
        source = InMemorySource([1, 2, 3, 4, 5])
        events = []
        async for event in source.read():
            events.append(event.data)

        assert events == [1, 2, 3, 4, 5]

        # Sink
        from src.core.data_pipeline import StreamEvent

        sink = InMemorySink()
        await sink.write(StreamEvent(data="test1"))
        await sink.write(StreamEvent(data="test2"))

        assert len(sink.events) == 2

    @pytest.mark.asyncio
    async def test_stream_pipeline(self):
        """Test StreamPipeline."""
        from src.core.data_pipeline import (
            StreamPipeline, InMemorySource, InMemorySink
        )

        source = InMemorySource([1, 2, 3, 4, 5])
        sink = InMemorySink()

        pipeline = (
            StreamPipeline(source, sink=sink)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 4)
        )

        await pipeline.run()

        results = [e.data for e in sink.events]
        assert results == [6, 8, 10]

    def test_job_status_enum(self):
        """Test JobStatus enum."""
        from src.core.data_pipeline import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.COMPLETED.value == "completed"

    def test_job_config(self):
        """Test JobConfig."""
        from src.core.data_pipeline import JobConfig, JobPriority

        config = JobConfig(
            max_retries=5,
            timeout_seconds=60,
            priority=JobPriority.HIGH,
        )

        assert config.max_retries == 5
        assert config.priority == JobPriority.HIGH

    @pytest.mark.asyncio
    async def test_function_job(self):
        """Test FunctionJob."""
        from src.core.data_pipeline import FunctionJob, JobStatus

        job = FunctionJob(func=lambda x: x * 2)
        result = await job.run(21)

        assert result.status == JobStatus.COMPLETED
        assert result.output == 42

    @pytest.mark.asyncio
    async def test_transform_job(self):
        """Test TransformJob."""
        from src.core.data_pipeline import (
            TransformJob, ComputeTransformer, JobStatus
        )

        transformer = ComputeTransformer(
            computations={"doubled": lambda d: d.get("value", 0) * 2}
        )

        job = TransformJob(transformer)
        data = [{"value": 1}, {"value": 2}, {"value": 3}]

        result = await job.run(data)

        assert result.status == JobStatus.COMPLETED
        assert len(result.output) == 3
        assert result.output[0]["doubled"] == 2

    @pytest.mark.asyncio
    async def test_parallel_batch_processor(self):
        """Test ParallelBatchProcessor."""
        from src.core.data_pipeline import ParallelBatchProcessor

        processor = ParallelBatchProcessor(
            processor=lambda x: x * 2,
            batch_size=2,
            max_workers=2,
        )

        results = await processor.process([1, 2, 3, 4, 5])

        assert sorted(results) == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_etl_pipeline(self):
        """Test ETLPipeline."""
        from src.core.data_pipeline import ETLPipeline, ComputeTransformer

        loaded_data = []

        pipeline = (
            ETLPipeline()
            .extract(lambda: [{"value": 1}, {"value": 2}, {"value": 3}])
            .transform(ComputeTransformer(
                computations={"doubled": lambda d: d.get("value", 0) * 2}
            ))
            .load(lambda batch: loaded_data.extend(batch))
            .batch_size(10)
        )

        stats = await pipeline.run()

        assert stats["extracted"] == 3
        assert stats["transformed"] == 3
        assert stats["loaded"] == 3
        assert len(loaded_data) == 3
        assert loaded_data[0]["doubled"] == 2


# ============================================================================
# Integration Tests
# ============================================================================

class TestP44P47Integration:
    """Integration tests for P44-P47 modules."""

    @pytest.mark.asyncio
    async def test_feature_toggle_with_audit(self):
        """Test feature toggles with audit logging."""
        from src.core.feature_toggles_enhanced import (
            ToggleManager, ToggleState, EvaluationContext
        )
        from src.core.audit_enhanced import AuditManager, AuditCategory

        toggle_manager = ToggleManager()
        audit_manager = AuditManager()

        # Create toggle
        await toggle_manager.create_toggle(
            name="new_feature",
            default_state=ToggleState.ON,
        )

        # Check toggle and log
        context = EvaluationContext(user_id="user123")
        enabled = await toggle_manager.is_enabled("new_feature", context)

        await audit_manager.log(
            category=AuditCategory.CONFIGURATION,
            action="toggle_check",
            description=f"Checked toggle 'new_feature': {enabled}",
        )

        assert enabled is True

    @pytest.mark.asyncio
    async def test_secrets_with_audit(self):
        """Test secrets management with audit logging."""
        from src.core.secrets import SecretsManager, SecretType
        from src.core.audit_enhanced import AuditManager, AuditCategory

        secrets_manager = SecretsManager()
        audit_manager = AuditManager()

        # Create secret
        await secrets_manager.create_secret(
            name="api_key",
            value="secret123",
            secret_type=SecretType.API_KEY,
        )

        await audit_manager.log(
            category=AuditCategory.SECURITY,
            action="secret_created",
            resource_type="secret",
            resource_id="api_key",
        )

        # Access secret
        value = await secrets_manager.get_secret("api_key")

        await audit_manager.log_data_access(
            user_id="service",
            resource_type="secret",
            resource_id="api_key",
            action="read",
        )

        assert value == "secret123"

    @pytest.mark.asyncio
    async def test_data_pipeline_with_transforms(self):
        """Test complete data pipeline."""
        from src.core.data_pipeline import (
            ETLPipeline, ChainTransformer, MapTransformer,
            ComputeTransformer, FilterTransformer
        )

        # Sample data
        source_data = [
            {"user_name": "John", "user_age": "25", "status": "active"},
            {"user_name": "Jane", "user_age": "17", "status": "active"},
            {"user_name": "Bob", "user_age": "30", "status": "inactive"},
        ]

        # Build transformer chain
        transformer = ChainTransformer([
            # Rename fields
            MapTransformer({"user_name": "name", "user_age": "age", "status": "status"}),
            # Compute derived fields
            ComputeTransformer({
                "age_int": lambda d: int(d.get("age", 0)),
            }),
            # Filter
            FilterTransformer(lambda d: d.get("age_int", 0) >= 18),
        ])

        loaded = []

        pipeline = (
            ETLPipeline()
            .extract(lambda: source_data)
            .transform(transformer)
            .load(lambda batch: loaded.extend(batch))
        )

        stats = await pipeline.run()

        # Should have filtered out Jane (age 17)
        assert stats["extracted"] == 3
        assert stats["transformed"] == 2
        assert len(loaded) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
