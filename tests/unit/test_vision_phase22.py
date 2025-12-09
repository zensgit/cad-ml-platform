"""
Tests for Phase 22: Advanced Security & Governance.

Tests security governance hub module components including policy management,
data classification, encryption, secrets, threat intelligence, and security posture.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.vision.security_governance import (
    # Enums
    DataClassification,
    PolicyType,
    PolicyStatus,
    PolicyAction,
    EncryptionAlgorithm,
    KeyStatus,
    SecretType,
    ThreatLevel,
    SecurityEventType,
    PostureStatus,
    # Dataclasses
    SecurityPolicy,
    PolicyEvaluation,
    DataClassificationRule,
    EncryptionKey,
    Secret,
    ThreatIndicator,
    SecurityEvent,
    SecurityPosture,
    GovernanceConfig,
    # Classes
    PolicyEngine,
    DataClassificationManager,
    KeyManager,
    SecretManager,
    ThreatIntelManager,
    SecurityEventCorrelator,
    SecurityPostureAssessor,
    SecurityGovernanceHub,
    SecureVisionProvider,
    # Factory functions
    create_governance_config,
    create_security_governance_hub,
    create_security_policy,
    create_classification_rule,
    create_threat_indicator,
    create_secure_provider,
)
from src.core.vision.base import VisionDescription


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def governance_config():
    """Create a test governance configuration."""
    return GovernanceConfig(
        key_rotation_days=30,
        secret_expiry_days=90,
        max_policy_priority=100,
        enable_threat_intel=True,
        enable_auto_classification=True,
        audit_retention_days=30,
    )


@pytest.fixture
def policy_engine(governance_config):
    """Create a policy engine."""
    return PolicyEngine(governance_config)


@pytest.fixture
def classification_manager(governance_config):
    """Create a data classification manager."""
    return DataClassificationManager(governance_config)


@pytest.fixture
def key_manager(governance_config):
    """Create a key manager."""
    return KeyManager(governance_config)


@pytest.fixture
def secret_manager(key_manager, governance_config):
    """Create a secret manager."""
    return SecretManager(key_manager, governance_config)


@pytest.fixture
def threat_intel_manager(governance_config):
    """Create a threat intelligence manager."""
    return ThreatIntelManager(governance_config)


@pytest.fixture
def event_correlator(governance_config):
    """Create a security event correlator."""
    return SecurityEventCorrelator(governance_config)


@pytest.fixture
def governance_hub(governance_config):
    """Create a security governance hub."""
    return SecurityGovernanceHub(governance_config)


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


class TestSecurityEnums:
    """Tests for security-related enums."""

    def test_data_classification_values(self):
        """Test DataClassification enum values."""
        assert DataClassification.PUBLIC.value == "public"
        assert DataClassification.INTERNAL.value == "internal"
        assert DataClassification.CONFIDENTIAL.value == "confidential"
        assert DataClassification.RESTRICTED.value == "restricted"
        assert DataClassification.TOP_SECRET.value == "top_secret"

    def test_policy_type_values(self):
        """Test PolicyType enum values."""
        assert PolicyType.ACCESS.value == "access"
        assert PolicyType.DATA.value == "data"
        assert PolicyType.ENCRYPTION.value == "encryption"
        assert PolicyType.RETENTION.value == "retention"
        assert PolicyType.AUDIT.value == "audit"

    def test_policy_status_values(self):
        """Test PolicyStatus enum values."""
        assert PolicyStatus.DRAFT.value == "draft"
        assert PolicyStatus.ACTIVE.value == "active"
        assert PolicyStatus.SUSPENDED.value == "suspended"
        assert PolicyStatus.DEPRECATED.value == "deprecated"
        assert PolicyStatus.ARCHIVED.value == "archived"

    def test_policy_action_values(self):
        """Test PolicyAction enum values."""
        assert PolicyAction.ALLOW.value == "allow"
        assert PolicyAction.DENY.value == "deny"
        assert PolicyAction.AUDIT.value == "audit"
        assert PolicyAction.ENCRYPT.value == "encrypt"
        assert PolicyAction.MASK.value == "mask"
        assert PolicyAction.QUARANTINE.value == "quarantine"

    def test_encryption_algorithm_values(self):
        """Test EncryptionAlgorithm enum values."""
        assert EncryptionAlgorithm.AES_256_GCM.value == "aes-256-gcm"
        assert EncryptionAlgorithm.AES_256_CBC.value == "aes-256-cbc"
        assert EncryptionAlgorithm.CHACHA20_POLY1305.value == "chacha20-poly1305"
        assert EncryptionAlgorithm.RSA_OAEP.value == "rsa-oaep"

    def test_key_status_values(self):
        """Test KeyStatus enum values."""
        assert KeyStatus.ACTIVE.value == "active"
        assert KeyStatus.INACTIVE.value == "inactive"
        assert KeyStatus.COMPROMISED.value == "compromised"
        assert KeyStatus.EXPIRED.value == "expired"
        assert KeyStatus.PENDING_ROTATION.value == "pending_rotation"

    def test_secret_type_values(self):
        """Test SecretType enum values."""
        assert SecretType.API_KEY.value == "api_key"
        assert SecretType.PASSWORD.value == "password"
        assert SecretType.CERTIFICATE.value == "certificate"
        assert SecretType.TOKEN.value == "token"
        assert SecretType.SSH_KEY.value == "ssh_key"
        assert SecretType.DATABASE_CREDENTIAL.value == "database_credential"

    def test_threat_level_values(self):
        """Test ThreatLevel enum values."""
        assert ThreatLevel.NONE.value == "none"
        assert ThreatLevel.LOW.value == "low"
        assert ThreatLevel.MEDIUM.value == "medium"
        assert ThreatLevel.HIGH.value == "high"
        assert ThreatLevel.CRITICAL.value == "critical"

    def test_security_event_type_values(self):
        """Test SecurityEventType enum values."""
        assert SecurityEventType.ACCESS_DENIED.value == "access_denied"
        assert SecurityEventType.ACCESS_GRANTED.value == "access_granted"
        assert SecurityEventType.POLICY_VIOLATION.value == "policy_violation"
        assert SecurityEventType.THREAT_DETECTED.value == "threat_detected"

    def test_posture_status_values(self):
        """Test PostureStatus enum values."""
        assert PostureStatus.EXCELLENT.value == "excellent"
        assert PostureStatus.GOOD.value == "good"
        assert PostureStatus.FAIR.value == "fair"
        assert PostureStatus.POOR.value == "poor"
        assert PostureStatus.CRITICAL.value == "critical"


# ============================================================================
# Test Dataclasses
# ============================================================================


class TestSecurityDataclasses:
    """Tests for security dataclasses."""

    def test_security_policy_creation(self):
        """Test SecurityPolicy creation."""
        policy = SecurityPolicy(
            policy_id="policy-1",
            name="Test Policy",
            policy_type=PolicyType.ACCESS,
            status=PolicyStatus.ACTIVE,
            action=PolicyAction.DENY,
            priority=10,
        )
        assert policy.policy_id == "policy-1"
        assert policy.name == "Test Policy"
        assert policy.policy_type == PolicyType.ACCESS
        assert policy.status == PolicyStatus.ACTIVE
        assert policy.priority == 10

    def test_policy_evaluation_creation(self):
        """Test PolicyEvaluation creation."""
        evaluation = PolicyEvaluation(
            policy_id="policy-1",
            decision=PolicyAction.ALLOW,
            matched_rules=["rule-1", "rule-2"],
            reason="Test reason",
        )
        assert evaluation.policy_id == "policy-1"
        assert evaluation.decision == PolicyAction.ALLOW
        assert len(evaluation.matched_rules) == 2

    def test_data_classification_rule_creation(self):
        """Test DataClassificationRule creation."""
        rule = DataClassificationRule(
            rule_id="rule-1",
            name="PII Detection",
            classification=DataClassification.CONFIDENTIAL,
            patterns=[r"\d{3}-\d{2}-\d{4}"],  # SSN pattern
            keywords=["ssn", "social security"],
        )
        assert rule.rule_id == "rule-1"
        assert rule.classification == DataClassification.CONFIDENTIAL
        assert len(rule.patterns) == 1
        assert len(rule.keywords) == 2

    def test_encryption_key_creation(self):
        """Test EncryptionKey creation."""
        key = EncryptionKey(
            key_id="key-1",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            status=KeyStatus.ACTIVE,
            version=1,
        )
        assert key.key_id == "key-1"
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert key.status == KeyStatus.ACTIVE
        assert key.version == 1

    def test_secret_creation(self):
        """Test Secret creation."""
        secret = Secret(
            secret_id="secret-1",
            name="api_key",
            secret_type=SecretType.API_KEY,
            encrypted_value="enc:abc123",
            version=1,
        )
        assert secret.secret_id == "secret-1"
        assert secret.name == "api_key"
        assert secret.secret_type == SecretType.API_KEY
        assert secret.version == 1

    def test_threat_indicator_creation(self):
        """Test ThreatIndicator creation."""
        indicator = ThreatIndicator(
            indicator_id="ioc-1",
            indicator_type="ip",
            value="192.168.1.100",
            threat_level=ThreatLevel.HIGH,
            source="threat_feed",
            confidence=0.95,
        )
        assert indicator.indicator_id == "ioc-1"
        assert indicator.indicator_type == "ip"
        assert indicator.threat_level == ThreatLevel.HIGH
        assert indicator.confidence == 0.95

    def test_security_event_creation(self):
        """Test SecurityEvent creation."""
        event = SecurityEvent(
            event_id="event-1",
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=ThreatLevel.MEDIUM,
            source="api_gateway",
            message="Access denied for resource",
            actor="user123",
            resource="/api/data",
        )
        assert event.event_id == "event-1"
        assert event.event_type == SecurityEventType.ACCESS_DENIED
        assert event.severity == ThreatLevel.MEDIUM
        assert event.actor == "user123"

    def test_security_posture_creation(self):
        """Test SecurityPosture creation."""
        posture = SecurityPosture(
            assessment_id="assess-1",
            status=PostureStatus.GOOD,
            score=85.0,
            findings=[{"severity": "low", "message": "Minor issue"}],
            recommendations=["Enable MFA"],
        )
        assert posture.assessment_id == "assess-1"
        assert posture.status == PostureStatus.GOOD
        assert posture.score == 85.0
        assert len(posture.findings) == 1

    def test_governance_config_defaults(self):
        """Test GovernanceConfig default values."""
        config = GovernanceConfig()
        assert config.key_rotation_days == 90
        assert config.secret_expiry_days == 365
        assert config.max_policy_priority == 1000
        assert config.enable_threat_intel is True
        assert config.enable_auto_classification is True


# ============================================================================
# Test PolicyEngine
# ============================================================================


class TestPolicyEngine:
    """Tests for PolicyEngine class."""

    def test_add_policy(self, policy_engine):
        """Test adding a policy."""
        policy = SecurityPolicy(
            policy_id="policy-1",
            name="Test Policy",
            policy_type=PolicyType.ACCESS,
        )
        policy_engine.add_policy(policy)
        assert policy_engine.get_policy("policy-1") is not None

    def test_remove_policy(self, policy_engine):
        """Test removing a policy."""
        policy = SecurityPolicy(
            policy_id="policy-remove",
            name="Remove Policy",
            policy_type=PolicyType.ACCESS,
        )
        policy_engine.add_policy(policy)
        assert policy_engine.remove_policy("policy-remove") is True
        assert policy_engine.get_policy("policy-remove") is None
        assert policy_engine.remove_policy("nonexistent") is False

    def test_list_policies(self, policy_engine):
        """Test listing policies."""
        for i in range(3):
            policy = SecurityPolicy(
                policy_id=f"policy-{i}",
                name=f"Policy {i}",
                policy_type=PolicyType.ACCESS if i < 2 else PolicyType.DATA,
                priority=i,
            )
            policy_engine.add_policy(policy)

        all_policies = policy_engine.list_policies()
        assert len(all_policies) == 3

        access_policies = policy_engine.list_policies(policy_type=PolicyType.ACCESS)
        assert len(access_policies) == 2

    def test_activate_policy(self, policy_engine):
        """Test activating a policy."""
        policy = SecurityPolicy(
            policy_id="policy-activate",
            name="Activate Policy",
            policy_type=PolicyType.ACCESS,
            status=PolicyStatus.DRAFT,
        )
        policy_engine.add_policy(policy)
        assert policy_engine.activate_policy("policy-activate") is True
        assert policy_engine.get_policy("policy-activate").status == PolicyStatus.ACTIVE

    def test_suspend_policy(self, policy_engine):
        """Test suspending a policy."""
        policy = SecurityPolicy(
            policy_id="policy-suspend",
            name="Suspend Policy",
            policy_type=PolicyType.ACCESS,
            status=PolicyStatus.ACTIVE,
        )
        policy_engine.add_policy(policy)
        assert policy_engine.suspend_policy("policy-suspend") is True
        assert policy_engine.get_policy("policy-suspend").status == PolicyStatus.SUSPENDED

    def test_evaluate_policy_allow(self, policy_engine):
        """Test policy evaluation with allow."""
        policy = SecurityPolicy(
            policy_id="policy-allow",
            name="Allow Policy",
            policy_type=PolicyType.ACCESS,
            status=PolicyStatus.ACTIVE,
            action=PolicyAction.ALLOW,
            conditions={"role": "admin"},
        )
        policy_engine.add_policy(policy)

        evaluation = policy_engine.evaluate(
            {"role": "admin"},
            policy_type=PolicyType.ACCESS,
        )
        assert evaluation.decision == PolicyAction.ALLOW

    def test_evaluate_policy_deny(self, policy_engine):
        """Test policy evaluation with deny."""
        policy = SecurityPolicy(
            policy_id="policy-deny",
            name="Deny Policy",
            policy_type=PolicyType.ACCESS,
            status=PolicyStatus.ACTIVE,
            action=PolicyAction.DENY,
            conditions={"role": "guest"},
        )
        policy_engine.add_policy(policy)

        evaluation = policy_engine.evaluate(
            {"role": "guest"},
            policy_type=PolicyType.ACCESS,
        )
        assert evaluation.decision == PolicyAction.DENY

    def test_evaluate_no_matching_policy(self, policy_engine):
        """Test evaluation when no policy matches."""
        evaluation = policy_engine.evaluate(
            {"role": "unknown"},
            policy_type=PolicyType.ACCESS,
        )
        assert evaluation.decision == PolicyAction.ALLOW  # Default


# ============================================================================
# Test DataClassificationManager
# ============================================================================


class TestDataClassificationManager:
    """Tests for DataClassificationManager class."""

    def test_add_rule(self, classification_manager):
        """Test adding a classification rule."""
        rule = DataClassificationRule(
            rule_id="rule-1",
            name="PII Rule",
            classification=DataClassification.CONFIDENTIAL,
            keywords=["ssn", "social security"],
        )
        classification_manager.add_rule(rule)
        assert len(classification_manager.list_rules()) == 1

    def test_remove_rule(self, classification_manager):
        """Test removing a classification rule."""
        rule = DataClassificationRule(
            rule_id="rule-remove",
            name="Remove Rule",
            classification=DataClassification.INTERNAL,
        )
        classification_manager.add_rule(rule)
        assert classification_manager.remove_rule("rule-remove") is True
        assert classification_manager.remove_rule("nonexistent") is False

    def test_classify_data_by_keyword(self, classification_manager):
        """Test data classification by keyword."""
        rule = DataClassificationRule(
            rule_id="rule-ssn",
            name="SSN Rule",
            classification=DataClassification.CONFIDENTIAL,
            keywords=["ssn", "social security number"],
        )
        classification_manager.add_rule(rule)

        classification = classification_manager.classify_data(
            "This contains SSN data",
            resource_id="doc-1",
        )
        assert classification == DataClassification.CONFIDENTIAL

    def test_classify_data_by_pattern(self, classification_manager):
        """Test data classification by regex pattern."""
        rule = DataClassificationRule(
            rule_id="rule-cc",
            name="Credit Card Rule",
            classification=DataClassification.RESTRICTED,
            patterns=[r"\d{4}-\d{4}-\d{4}-\d{4}"],
        )
        classification_manager.add_rule(rule)

        classification = classification_manager.classify_data(
            "Card: 1234-5678-9012-3456",
        )
        assert classification == DataClassification.RESTRICTED

    def test_classify_data_default_public(self, classification_manager):
        """Test default classification is public."""
        classification = classification_manager.classify_data("Normal text")
        assert classification == DataClassification.PUBLIC

    def test_get_classification(self, classification_manager):
        """Test getting stored classification."""
        classification_manager.set_classification(
            "resource-1",
            DataClassification.INTERNAL,
        )
        result = classification_manager.get_classification("resource-1")
        assert result == DataClassification.INTERNAL

    def test_classification_priority(self, classification_manager):
        """Test classification priority (most restrictive wins)."""
        # Add less restrictive rule
        rule1 = DataClassificationRule(
            rule_id="rule-internal",
            name="Internal Rule",
            classification=DataClassification.INTERNAL,
            keywords=["internal"],
        )
        # Add more restrictive rule
        rule2 = DataClassificationRule(
            rule_id="rule-secret",
            name="Secret Rule",
            classification=DataClassification.TOP_SECRET,
            keywords=["secret"],
        )
        classification_manager.add_rule(rule1)
        classification_manager.add_rule(rule2)

        # Should match most restrictive first
        classification = classification_manager.classify_data("secret internal data")
        assert classification == DataClassification.TOP_SECRET


# ============================================================================
# Test KeyManager
# ============================================================================


class TestKeyManager:
    """Tests for KeyManager class."""

    def test_generate_key(self, key_manager):
        """Test key generation."""
        key = key_manager.generate_key()
        assert key.key_id is not None
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert key.status == KeyStatus.ACTIVE
        assert key.version == 1

    def test_generate_key_with_algorithm(self, key_manager):
        """Test key generation with specific algorithm."""
        key = key_manager.generate_key(
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
        )
        assert key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305

    def test_get_key(self, key_manager):
        """Test getting a key."""
        key = key_manager.generate_key()
        retrieved = key_manager.get_key(key.key_id)
        assert retrieved is not None
        assert retrieved.key_id == key.key_id

    def test_get_active_key(self, key_manager):
        """Test getting active key."""
        key_manager.generate_key()
        active_key = key_manager.get_active_key()
        assert active_key is not None
        assert active_key.status == KeyStatus.ACTIVE

    def test_rotate_key(self, key_manager):
        """Test key rotation."""
        old_key = key_manager.generate_key()
        new_key = key_manager.rotate_key(old_key.key_id)

        assert new_key is not None
        assert new_key.rotated_from == old_key.key_id
        assert new_key.version == 2

        # Old key should be inactive
        old_key_updated = key_manager.get_key(old_key.key_id)
        assert old_key_updated.status == KeyStatus.INACTIVE

    def test_revoke_key(self, key_manager):
        """Test key revocation."""
        key = key_manager.generate_key()
        assert key_manager.revoke_key(key.key_id, "Compromised") is True

        revoked_key = key_manager.get_key(key.key_id)
        assert revoked_key.status == KeyStatus.COMPROMISED
        assert revoked_key.metadata.get("revocation_reason") == "Compromised"

    def test_list_keys(self, key_manager):
        """Test listing keys."""
        key_manager.generate_key()
        key_manager.generate_key()

        keys = key_manager.list_keys()
        assert len(keys) == 2

        active_keys = key_manager.list_keys(status=KeyStatus.ACTIVE)
        assert len(active_keys) == 2

    def test_get_keys_for_rotation(self, key_manager):
        """Test getting keys that need rotation."""
        # Generate key with short expiry
        key = key_manager.generate_key(expires_in_days=3)

        keys_for_rotation = key_manager.get_keys_for_rotation()
        assert len(keys_for_rotation) == 1
        assert keys_for_rotation[0].key_id == key.key_id


# ============================================================================
# Test SecretManager
# ============================================================================


class TestSecretManager:
    """Tests for SecretManager class."""

    def test_store_secret(self, secret_manager):
        """Test storing a secret."""
        secret = secret_manager.store_secret(
            name="api_key",
            value="secret_value_123",
            secret_type=SecretType.API_KEY,
        )
        assert secret.secret_id is not None
        assert secret.name == "api_key"
        assert secret.secret_type == SecretType.API_KEY
        assert secret.encrypted_value.startswith("enc:")

    def test_get_secret(self, secret_manager):
        """Test retrieving a secret."""
        secret = secret_manager.store_secret(
            name="password",
            value="my_password_123",
            secret_type=SecretType.PASSWORD,
        )

        retrieved_value = secret_manager.get_secret(secret.secret_id)
        assert retrieved_value == "my_password_123"

    def test_get_secret_by_name(self, secret_manager):
        """Test retrieving a secret by name."""
        secret_manager.store_secret(
            name="db_password",
            value="database_pass",
            secret_type=SecretType.DATABASE_CREDENTIAL,
        )

        retrieved = secret_manager.get_secret_by_name("db_password")
        assert retrieved == "database_pass"

    def test_rotate_secret(self, secret_manager):
        """Test secret rotation."""
        secret = secret_manager.store_secret(
            name="rotating_secret",
            value="old_value",
            secret_type=SecretType.API_KEY,
        )

        rotated = secret_manager.rotate_secret(
            secret.secret_id,
            "new_value",
        )
        assert rotated is not None
        assert rotated.version == 2

        # Verify new value
        new_value = secret_manager.get_secret(secret.secret_id)
        assert new_value == "new_value"

    def test_delete_secret(self, secret_manager):
        """Test deleting a secret."""
        secret = secret_manager.store_secret(
            name="delete_me",
            value="to_delete",
            secret_type=SecretType.TOKEN,
        )

        assert secret_manager.delete_secret(secret.secret_id) is True
        assert secret_manager.get_secret(secret.secret_id) is None

    def test_list_secrets(self, secret_manager):
        """Test listing secrets."""
        secret_manager.store_secret("key1", "val1", SecretType.API_KEY)
        secret_manager.store_secret("pass1", "val2", SecretType.PASSWORD)
        secret_manager.store_secret("key2", "val3", SecretType.API_KEY)

        all_secrets = secret_manager.list_secrets()
        assert len(all_secrets) == 3

        api_keys = secret_manager.list_secrets(secret_type=SecretType.API_KEY)
        assert len(api_keys) == 2

    def test_get_expiring_secrets(self, secret_manager):
        """Test getting expiring secrets."""
        secret_manager.store_secret(
            name="expiring",
            value="value",
            secret_type=SecretType.TOKEN,
            expires_in_days=7,
        )

        expiring = secret_manager.get_expiring_secrets(days=30)
        assert len(expiring) == 1


# ============================================================================
# Test ThreatIntelManager
# ============================================================================


class TestThreatIntelManager:
    """Tests for ThreatIntelManager class."""

    def test_add_indicator(self, threat_intel_manager):
        """Test adding a threat indicator."""
        indicator = ThreatIndicator(
            indicator_id="ioc-1",
            indicator_type="ip",
            value="192.168.1.100",
            threat_level=ThreatLevel.HIGH,
        )
        threat_intel_manager.add_indicator(indicator)
        assert len(threat_intel_manager.list_indicators()) == 1

    def test_remove_indicator(self, threat_intel_manager):
        """Test removing a threat indicator."""
        indicator = ThreatIndicator(
            indicator_id="ioc-remove",
            indicator_type="domain",
            value="malicious.com",
            threat_level=ThreatLevel.CRITICAL,
        )
        threat_intel_manager.add_indicator(indicator)
        assert threat_intel_manager.remove_indicator("ioc-remove") is True
        assert threat_intel_manager.remove_indicator("nonexistent") is False

    def test_check_threat(self, threat_intel_manager):
        """Test checking for threat match."""
        indicator = ThreatIndicator(
            indicator_id="ioc-check",
            indicator_type="ip",
            value="10.0.0.1",
            threat_level=ThreatLevel.MEDIUM,
        )
        threat_intel_manager.add_indicator(indicator)

        match = threat_intel_manager.check_threat("ip", "10.0.0.1")
        assert match is not None
        assert match.threat_level == ThreatLevel.MEDIUM

        no_match = threat_intel_manager.check_threat("ip", "10.0.0.2")
        assert no_match is None

    def test_get_indicators_by_level(self, threat_intel_manager):
        """Test getting indicators by threat level."""
        threat_intel_manager.add_indicator(ThreatIndicator(
            indicator_id="ioc-1",
            indicator_type="ip",
            value="1.1.1.1",
            threat_level=ThreatLevel.CRITICAL,
        ))
        threat_intel_manager.add_indicator(ThreatIndicator(
            indicator_id="ioc-2",
            indicator_type="ip",
            value="2.2.2.2",
            threat_level=ThreatLevel.LOW,
        ))

        critical = threat_intel_manager.get_indicators_by_level(ThreatLevel.CRITICAL)
        assert len(critical) == 1


# ============================================================================
# Test SecurityEventCorrelator
# ============================================================================


class TestSecurityEventCorrelator:
    """Tests for SecurityEventCorrelator class."""

    def test_record_event(self, event_correlator):
        """Test recording a security event."""
        event = SecurityEvent(
            event_id="event-1",
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=ThreatLevel.MEDIUM,
            source="api",
            message="Access denied",
        )
        event_correlator.record_event(event)

        events = event_correlator.get_events()
        assert len(events) == 1

    def test_get_events_filtered(self, event_correlator):
        """Test getting events with filters."""
        event_correlator.record_event(SecurityEvent(
            event_id="e1",
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=ThreatLevel.HIGH,
            source="api",
        ))
        event_correlator.record_event(SecurityEvent(
            event_id="e2",
            event_type=SecurityEventType.ACCESS_GRANTED,
            severity=ThreatLevel.LOW,
            source="api",
        ))

        denied_events = event_correlator.get_events(
            event_type=SecurityEventType.ACCESS_DENIED,
        )
        assert len(denied_events) == 1

        high_severity = event_correlator.get_events(severity=ThreatLevel.HIGH)
        assert len(high_severity) == 1

    def test_event_correlation(self, event_correlator):
        """Test event correlation."""
        # Record related events from same source
        event1 = SecurityEvent(
            event_id="e1",
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=ThreatLevel.MEDIUM,
            source="api",
        )
        event2 = SecurityEvent(
            event_id="e2",
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=ThreatLevel.MEDIUM,
            source="api",
        )

        event_correlator.record_event(event1)
        event_correlator.record_event(event2)

        # Second event should have correlation to first
        events = event_correlator.get_events()
        # Events from same source within time window are correlated
        assert len(events) == 2


# ============================================================================
# Test SecurityPostureAssessor
# ============================================================================


class TestSecurityPostureAssessor:
    """Tests for SecurityPostureAssessor class."""

    def test_assess_empty_state(self, governance_hub):
        """Test assessment with empty state."""
        assessment = governance_hub.posture.assess()
        assert assessment is not None
        assert assessment.assessment_id is not None
        # Empty state should have findings about missing policies/keys
        assert len(assessment.findings) > 0

    def test_assess_with_policies(self, governance_hub):
        """Test assessment with active policies."""
        # Add an active policy
        policy = create_security_policy(
            name="Test Policy",
            policy_type=PolicyType.ACCESS,
            action=PolicyAction.DENY,
        )
        policy.status = PolicyStatus.ACTIVE
        governance_hub.policies.add_policy(policy)

        assessment = governance_hub.posture.assess()
        assert assessment.metrics["active_policies"] == 1

    def test_assess_with_keys(self, governance_hub):
        """Test assessment with encryption keys."""
        governance_hub.keys.generate_key()

        assessment = governance_hub.posture.assess()
        assert assessment.metrics["active_keys"] == 1

    def test_get_latest_assessment(self, governance_hub):
        """Test getting latest assessment."""
        governance_hub.posture.assess()
        governance_hub.posture.assess()

        latest = governance_hub.posture.get_latest_assessment()
        assert latest is not None

    def test_assessment_history(self, governance_hub):
        """Test assessment history."""
        for _ in range(5):
            governance_hub.posture.assess()

        history = governance_hub.posture.get_assessment_history(limit=3)
        assert len(history) == 3


# ============================================================================
# Test SecurityGovernanceHub
# ============================================================================


class TestSecurityGovernanceHub:
    """Tests for SecurityGovernanceHub class."""

    def test_hub_initialization(self, governance_hub):
        """Test hub initialization."""
        assert governance_hub.policies is not None
        assert governance_hub.classification is not None
        assert governance_hub.keys is not None
        assert governance_hub.secrets is not None
        assert governance_hub.threat_intel is not None
        assert governance_hub.events is not None
        assert governance_hub.posture is not None

    def test_record_security_event(self, governance_hub):
        """Test recording a security event."""
        event = governance_hub.record_security_event(
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=ThreatLevel.MEDIUM,
            source="test",
            message="Test event",
        )
        assert event.event_id is not None
        assert event.event_type == SecurityEventType.ACCESS_DENIED

    def test_evaluate_access_allow(self, governance_hub):
        """Test access evaluation with allow."""
        policy = create_security_policy(
            name="Allow Admin",
            policy_type=PolicyType.ACCESS,
            action=PolicyAction.ALLOW,
            conditions={"actor": "admin"},
        )
        policy.status = PolicyStatus.ACTIVE
        governance_hub.policies.add_policy(policy)

        evaluation = governance_hub.evaluate_access(
            actor="admin",
            resource="/api/data",
            action="read",
        )
        assert evaluation.decision == PolicyAction.ALLOW

    def test_evaluate_access_deny(self, governance_hub):
        """Test access evaluation with deny."""
        policy = create_security_policy(
            name="Deny Guest",
            policy_type=PolicyType.ACCESS,
            action=PolicyAction.DENY,
            conditions={"actor": "guest"},
        )
        policy.status = PolicyStatus.ACTIVE
        governance_hub.policies.add_policy(policy)

        evaluation = governance_hub.evaluate_access(
            actor="guest",
            resource="/api/admin",
            action="write",
        )
        assert evaluation.decision == PolicyAction.DENY

    def test_get_governance_summary(self, governance_hub):
        """Test getting governance summary."""
        # Add some data
        governance_hub.keys.generate_key()
        governance_hub.secrets.store_secret("key", "value", SecretType.API_KEY)

        summary = governance_hub.get_governance_summary()

        assert "posture" in summary
        assert "policies" in summary
        assert "encryption" in summary
        assert "secrets" in summary
        assert "threat_intel" in summary
        assert "events" in summary
        assert "timestamp" in summary


# ============================================================================
# Test SecureVisionProvider
# ============================================================================


class TestSecureVisionProvider:
    """Tests for SecureVisionProvider class."""

    def test_provider_name(self, mock_provider, governance_hub):
        """Test provider name property."""
        provider = SecureVisionProvider(mock_provider, governance_hub)
        assert provider.provider_name == "secure_mock_provider"

    @pytest.mark.asyncio
    async def test_analyze_image_allowed(self, mock_provider, governance_hub):
        """Test image analysis when allowed."""
        # Add allow policy
        policy = create_security_policy(
            name="Allow Vision",
            policy_type=PolicyType.ACCESS,
            action=PolicyAction.ALLOW,
            conditions={"actor": "vision_client"},
        )
        policy.status = PolicyStatus.ACTIVE
        governance_hub.policies.add_policy(policy)

        provider = SecureVisionProvider(mock_provider, governance_hub)
        result = await provider.analyze_image(b"test_image_data")

        assert result.summary == "Test summary"
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_analyze_image_denied(self, mock_provider, governance_hub):
        """Test image analysis when denied."""
        # Add deny policy
        policy = create_security_policy(
            name="Deny Vision",
            policy_type=PolicyType.ACCESS,
            action=PolicyAction.DENY,
            conditions={"actor": "vision_client"},
        )
        policy.status = PolicyStatus.ACTIVE
        governance_hub.policies.add_policy(policy)

        provider = SecureVisionProvider(mock_provider, governance_hub)

        with pytest.raises(PermissionError):
            await provider.analyze_image(b"test_image_data")


# ============================================================================
# Test Factory Functions
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_governance_config(self):
        """Test create_governance_config factory."""
        config = create_governance_config(
            key_rotation_days=60,
            secret_expiry_days=180,
        )
        assert config.key_rotation_days == 60
        assert config.secret_expiry_days == 180

    def test_create_security_governance_hub(self):
        """Test create_security_governance_hub factory."""
        hub = create_security_governance_hub(
            key_rotation_days=45,
        )
        assert hub is not None
        assert hub._config.key_rotation_days == 45

    def test_create_security_policy(self):
        """Test create_security_policy factory."""
        policy = create_security_policy(
            name="Test Policy",
            policy_type=PolicyType.ACCESS,
            action=PolicyAction.ALLOW,
            priority=10,
        )
        assert policy.policy_id is not None
        assert policy.name == "Test Policy"
        assert policy.priority == 10

    def test_create_classification_rule(self):
        """Test create_classification_rule factory."""
        rule = create_classification_rule(
            name="PII Rule",
            classification=DataClassification.CONFIDENTIAL,
            keywords=["ssn"],
        )
        assert rule.rule_id is not None
        assert rule.name == "PII Rule"
        assert rule.classification == DataClassification.CONFIDENTIAL

    def test_create_threat_indicator(self):
        """Test create_threat_indicator factory."""
        indicator = create_threat_indicator(
            indicator_type="ip",
            value="10.0.0.1",
            threat_level=ThreatLevel.HIGH,
            source="threat_feed",
        )
        assert indicator.indicator_id is not None
        assert indicator.threat_level == ThreatLevel.HIGH

    def test_create_secure_provider(self, mock_provider):
        """Test create_secure_provider factory."""
        provider = create_secure_provider(mock_provider)
        assert provider is not None
        assert isinstance(provider, SecureVisionProvider)

    def test_create_secure_provider_with_hub(self, mock_provider, governance_hub):
        """Test create_secure_provider with existing hub."""
        provider = create_secure_provider(mock_provider, governance_hub)
        assert provider._hub is governance_hub


# ============================================================================
# Integration Tests
# ============================================================================


class TestSecurityIntegration:
    """Integration tests for security governance system."""

    def test_full_security_pipeline(self, mock_provider):
        """Test complete security pipeline."""
        # Create hub
        hub = create_security_governance_hub()

        # Add policies
        access_policy = create_security_policy(
            name="Allow All",
            policy_type=PolicyType.ACCESS,
            action=PolicyAction.ALLOW,
            conditions={"actor": "vision_client"},
        )
        access_policy.status = PolicyStatus.ACTIVE
        hub.policies.add_policy(access_policy)

        # Add classification rules
        rule = create_classification_rule(
            name="Sensitive Data",
            classification=DataClassification.CONFIDENTIAL,
            keywords=["password", "secret"],
        )
        hub.classification.add_rule(rule)

        # Generate encryption key
        hub.keys.generate_key()

        # Store secret
        hub.secrets.store_secret(
            "api_key",
            "secret_value",
            SecretType.API_KEY,
        )

        # Get summary
        summary = hub.get_governance_summary()
        assert summary["policies"]["active"] == 1
        assert summary["encryption"]["active_keys"] == 1
        assert summary["secrets"]["total"] == 1

    def test_threat_detection_pipeline(self):
        """Test threat detection and event correlation."""
        hub = create_security_governance_hub()

        # Add threat indicator
        indicator = create_threat_indicator(
            indicator_type="ip",
            value="192.168.1.100",
            threat_level=ThreatLevel.HIGH,
        )
        hub.threat_intel.add_indicator(indicator)

        # Check for threat
        match = hub.threat_intel.check_threat("ip", "192.168.1.100")
        assert match is not None
        assert match.threat_level == ThreatLevel.HIGH

        # Record event
        hub.record_security_event(
            event_type=SecurityEventType.THREAT_DETECTED,
            severity=ThreatLevel.HIGH,
            source="firewall",
            message="Known malicious IP detected",
        )

        # Verify event recorded
        events = hub.events.get_events(
            event_type=SecurityEventType.THREAT_DETECTED,
        )
        assert len(events) == 1

    def test_key_lifecycle(self):
        """Test complete key lifecycle."""
        hub = create_security_governance_hub()

        # Generate key
        key = hub.keys.generate_key(expires_in_days=30)
        assert key.status == KeyStatus.ACTIVE

        # Rotate key
        new_key = hub.keys.rotate_key(key.key_id)
        assert new_key.version == 2
        assert hub.keys.get_key(key.key_id).status == KeyStatus.INACTIVE

        # Revoke key
        hub.keys.revoke_key(new_key.key_id, "Test revocation")
        assert hub.keys.get_key(new_key.key_id).status == KeyStatus.COMPROMISED

    def test_posture_improvement(self):
        """Test security posture improvement."""
        hub = create_security_governance_hub()

        # Initial assessment (should have issues)
        initial = hub.posture.assess()
        initial_score = initial.score

        # Add active policy
        policy = create_security_policy(
            name="Security Policy",
            policy_type=PolicyType.ACCESS,
        )
        policy.status = PolicyStatus.ACTIVE
        hub.policies.add_policy(policy)

        # Add encryption key
        hub.keys.generate_key()

        # Re-assess
        improved = hub.posture.assess()

        # Score should improve
        assert improved.score >= initial_score
