"""Tests for Phase 14: Advanced Security & Privacy features.

This module tests:
- Access Control (RBAC, ABAC, policies)
- Data Masking (PII detection, masking strategies)
- Security Audit (threat detection, audit logging)
- Key Management (key rotation, secrets vault)
- Privacy Compliance (GDPR, consent management)
"""

from datetime import datetime, timedelta
from typing import Any

import pytest

from src.core.vision.base import VisionDescription, VisionProvider

# Access Control imports
from src.core.vision.access_control import (
    AccessControlManager,
    AccessControlVisionProvider,
    AccessDecision,
    AccessRequest,
    AccessResult,
    Permission,
    Policy,
    PolicyEffect,
    PolicyEngine,
    Resource,
    ResourceType,
    Role,
    RoleManager,
    User,
    create_access_manager,
    create_acl_provider,
    create_policy_engine,
    create_role_manager,
)

# Data Masking imports
from src.core.vision.data_masking import (
    DataMaskingVisionProvider,
    HeuristicPIIDetector,
    MaskingEngine,
    MaskingResult,
    MaskingStrategy,
    PIIMatch,
    PIIType,
    RegexPIIDetector,
    create_masking_engine,
    create_masking_provider,
    create_regex_detector,
    create_heuristic_detector,
)

# Security Audit imports
from src.core.vision.security_audit import (
    AlertManager,
    AnomalyDetector,
    AuditLogger,
    BruteForceDetector,
    IPReputationDetector,
    SecurityAuditManager,
    SecurityAuditVisionProvider,
    SecurityEvent,
    SecurityEventType,
    SeverityLevel,
    ThreatLevel,
    create_alert_manager,
    create_anomaly_detector,
    create_audit_logger,
    create_audit_provider,
    create_brute_force_detector,
    create_ip_reputation_detector,
    create_security_audit_manager,
)

# Key Management imports
from src.core.vision.key_management import (
    EncryptedKeyStore,
    InMemoryKeyStore,
    KeyAlgorithm,
    KeyGenerator,
    KeyManagedVisionProvider,
    KeyManager,
    KeyStatus,
    KeyType,
    SecretsVault,
    create_encrypted_key_store,
    create_in_memory_key_store,
    create_key_generator,
    create_key_managed_provider,
    create_key_manager,
    create_secrets_vault,
)

# Privacy Compliance imports
from src.core.vision.privacy_compliance import (
    Consent,
    ConsentManager,
    ConsentType,
    DataRetentionManager,
    DataRetentionPolicy,
    DataSubjectRequest,
    DataSubjectRequestHandler,
    DataSubjectRight,
    LegalBasis,
    PrivacyComplianceManager,
    PrivacyComplianceVisionProvider,
    PrivacyRegulation,
    ProcessingActivityLog,
    RequestStatus,
    create_activity_log,
    create_compliance_manager,
    create_compliance_provider,
    create_consent_manager,
    create_request_handler,
    create_retention_manager,
)


# Stub provider for testing
class SimpleStubProvider(VisionProvider):
    """Simple stub provider for testing."""

    @property
    def provider_name(self) -> str:
        return "simple_stub"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        return VisionDescription(
            summary="Test analysis",
            details=["Detail 1", "Detail 2"],
            confidence=0.95,
        )


# =============================================================================
# Access Control Tests
# =============================================================================


class TestRoleManager:
    """Tests for RoleManager."""

    def test_create_role(self) -> None:
        """Test role creation."""
        manager = create_role_manager()
        role = manager.create_role("admin", permissions={Permission.ADMIN})
        assert role.name == "admin"
        assert role.role_id is not None
        assert Permission.ADMIN in role.permissions

    def test_add_permission_and_assign_role(self) -> None:
        """Test assigning roles to users."""
        manager = create_role_manager()
        role = manager.create_role("editor", permissions={Permission.WRITE, Permission.READ})
        result = manager.assign_role("user1", role.role_id)
        assert result is True

    def test_check_permission(self) -> None:
        """Test permission checking."""
        manager = create_role_manager()
        role = manager.create_role("admin", permissions={Permission.ANALYZE})
        manager.assign_role("user1", role.role_id)

        has_perm = manager.check_permission("user1", Permission.ANALYZE)
        assert has_perm is True

        no_perm = manager.check_permission("user1", Permission.DELETE)
        assert no_perm is False

    def test_get_user_roles(self) -> None:
        """Test getting user roles."""
        manager = create_role_manager()
        role = manager.create_role("viewer", permissions={Permission.READ})
        manager.assign_role("user1", role.role_id)

        roles = manager.get_user_roles("user1")
        assert len(roles) == 1
        assert roles[0].name == "viewer"


class TestPolicyEngine:
    """Tests for PolicyEngine (ABAC)."""

    def test_create_policy(self) -> None:
        """Test policy creation."""
        engine = create_policy_engine()
        policy = Policy(
            policy_id="policy1",
            name="work_hours_only",
            effect=PolicyEffect.ALLOW,
        )
        engine.add_policy(policy)
        retrieved = engine.get_policy("policy1")
        assert retrieved is not None
        assert retrieved.name == "work_hours_only"

    def test_list_policies(self) -> None:
        """Test listing policies."""
        engine = create_policy_engine()
        policy1 = Policy(policy_id="p1", name="policy1", effect=PolicyEffect.ALLOW)
        policy2 = Policy(policy_id="p2", name="policy2", effect=PolicyEffect.DENY)
        engine.add_policy(policy1)
        engine.add_policy(policy2)

        policies = engine.list_policies()
        assert len(policies) == 2


class TestAccessControlManager:
    """Tests for AccessControlManager combining RBAC and ABAC."""

    def test_create_user(self) -> None:
        """Test user creation."""
        manager = create_access_manager()
        user = manager.create_user("testuser")
        assert user.username == "testuser"
        assert user.user_id is not None

    def test_check_access_with_rbac(self) -> None:
        """Test access check using RBAC."""
        manager = create_access_manager()

        # Create user
        user = manager.create_user("analyst")

        # Create and assign role
        role = manager.role_manager.create_role("analyst_role", permissions={Permission.ANALYZE})
        manager.role_manager.assign_role(user.user_id, role.role_id)

        # Register resource
        resource = manager.register_resource(ResourceType.IMAGE)

        # Check access
        result = manager.check_access(
            user.user_id,
            resource.resource_id,
            Permission.ANALYZE,
        )
        assert result.decision == AccessDecision.ALLOW


class TestAccessControlVisionProvider:
    """Tests for AccessControlVisionProvider."""

    @pytest.mark.asyncio
    async def test_analyze_image_with_access(self) -> None:
        """Test image analysis with proper access."""
        stub = SimpleStubProvider()
        manager = create_access_manager()

        # Create user and setup access
        user = manager.create_user("analyst")
        role = manager.role_manager.create_role("analyst", permissions={Permission.ANALYZE})
        manager.role_manager.assign_role(user.user_id, role.role_id)

        provider = create_acl_provider(stub, manager, user_id=user.user_id)

        # Register a default resource
        resource = manager.register_resource(ResourceType.IMAGE)

        result = await provider.analyze_image(b"test", resource_id=resource.resource_id)
        assert result.summary == "Test analysis"
        assert result.confidence == 0.95


# =============================================================================
# Data Masking Tests
# =============================================================================


class TestPIIDetection:
    """Tests for PII detection."""

    def test_regex_detector_email(self) -> None:
        """Test email detection."""
        detector = create_regex_detector()
        text = "Contact me at john@example.com"
        matches = detector.detect(text)
        assert len(matches) > 0
        assert any(m.pii_type == PIIType.EMAIL for m in matches)

    def test_regex_detector_phone(self) -> None:
        """Test phone number detection."""
        detector = create_regex_detector()
        text = "Call me at 555-123-4567"
        matches = detector.detect(text)
        assert len(matches) > 0
        assert any(m.pii_type == PIIType.PHONE for m in matches)

    def test_regex_detector_ssn(self) -> None:
        """Test SSN detection."""
        detector = create_regex_detector()
        text = "SSN: 123-45-6789"
        matches = detector.detect(text)
        assert len(matches) > 0
        assert any(m.pii_type == PIIType.SSN for m in matches)

    def test_regex_detector_credit_card(self) -> None:
        """Test credit card detection."""
        detector = create_regex_detector()
        text = "Card: 4111-1111-1111-1111"
        matches = detector.detect(text)
        assert len(matches) > 0
        assert any(m.pii_type == PIIType.CREDIT_CARD for m in matches)

    def test_heuristic_detector(self) -> None:
        """Test heuristic PII detection."""
        detector = create_heuristic_detector()
        text = "My email is john@test.com"
        matches = detector.detect(text)
        # Should detect at least the email
        assert len(matches) > 0


class TestMaskingEngine:
    """Tests for MaskingEngine."""

    def test_mask_text(self) -> None:
        """Test masking text with PII."""
        engine = create_masking_engine()
        text = "Contact john@example.com or call 555-123-4567"
        result = engine.mask_text(text)
        assert "john@example.com" not in result.masked_text
        # Original should be preserved
        assert result.original_text == text

    def test_detect_and_mask(self) -> None:
        """Test detection followed by masking."""
        engine = create_masking_engine()
        text = "SSN: 123-45-6789, Card: 4111-1111-1111-1111"
        result = engine.mask_text(text)
        assert "123-45-6789" not in result.masked_text
        assert "4111-1111-1111-1111" not in result.masked_text
        assert result.detections is not None


class TestDataMaskingVisionProvider:
    """Tests for DataMaskingVisionProvider."""

    @pytest.mark.asyncio
    async def test_analyze_with_masking(self) -> None:
        """Test image analysis with data masking."""
        stub = SimpleStubProvider()
        engine = create_masking_engine()
        provider = create_masking_provider(stub, engine)

        result = await provider.analyze_image(b"test")
        assert result.confidence == 0.95


# =============================================================================
# Security Audit Tests
# =============================================================================


class TestThreatDetectors:
    """Tests for threat detectors."""

    def test_brute_force_detector(self) -> None:
        """Test brute force attack detection."""
        detector = create_brute_force_detector(max_attempts=5, window_seconds=60)

        # Verify detector was created
        assert detector is not None

    def test_anomaly_detector(self) -> None:
        """Test anomaly detection."""
        detector = create_anomaly_detector()
        assert detector is not None

    def test_ip_reputation_detector(self) -> None:
        """Test IP reputation detection."""
        detector = create_ip_reputation_detector()
        assert detector is not None


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_create_logger(self) -> None:
        """Test logger creation."""
        logger = create_audit_logger()
        assert logger is not None

    def test_log_operations(self) -> None:
        """Test basic logging operations."""
        logger = create_audit_logger()
        # Logger should be created without errors
        assert logger is not None


class TestAlertManager:
    """Tests for AlertManager."""

    def test_create_alert_manager(self) -> None:
        """Test alert manager creation."""
        manager = create_alert_manager()
        assert manager is not None


class TestSecurityAuditManager:
    """Tests for SecurityAuditManager."""

    def test_create_manager(self) -> None:
        """Test security audit manager creation."""
        manager = create_security_audit_manager()
        assert manager is not None


class TestSecurityAuditVisionProvider:
    """Tests for SecurityAuditVisionProvider."""

    @pytest.mark.asyncio
    async def test_analyze_with_audit(self) -> None:
        """Test image analysis with security audit."""
        stub = SimpleStubProvider()
        manager = create_security_audit_manager()
        provider = create_audit_provider(stub, manager)

        result = await provider.analyze_image(b"test", user_id="user1")
        assert result.confidence == 0.95


# =============================================================================
# Key Management Tests
# =============================================================================


class TestKeyGenerator:
    """Tests for KeyGenerator."""

    def test_create_generator(self) -> None:
        """Test key generator creation."""
        generator = create_key_generator()
        assert generator is not None


class TestKeyStore:
    """Tests for KeyStore implementations."""

    def test_in_memory_store(self) -> None:
        """Test in-memory key storage."""
        store = create_in_memory_key_store()
        assert store is not None

    def test_encrypted_store(self) -> None:
        """Test encrypted key storage."""
        master_key = b"0" * 32  # 256-bit master key
        store = create_encrypted_key_store(master_key)
        assert store is not None


class TestSecretsVault:
    """Tests for SecretsVault."""

    def test_create_vault(self) -> None:
        """Test secrets vault creation."""
        vault = create_secrets_vault()
        assert vault is not None


class TestKeyManager:
    """Tests for KeyManager."""

    def test_create_key_manager(self) -> None:
        """Test key manager creation."""
        manager = create_key_manager()
        assert manager is not None


class TestKeyManagedVisionProvider:
    """Tests for KeyManagedVisionProvider."""

    @pytest.mark.asyncio
    async def test_analyze_with_key_management(self) -> None:
        """Test image analysis with key management."""
        stub = SimpleStubProvider()
        manager = create_key_manager()
        provider = create_key_managed_provider(stub, manager)

        result = await provider.analyze_image(b"test")
        assert result.confidence == 0.95


# =============================================================================
# Privacy Compliance Tests
# =============================================================================


class TestConsentManager:
    """Tests for ConsentManager."""

    def test_grant_consent(self) -> None:
        """Test granting consent."""
        manager = create_consent_manager()
        consent = manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.DATA_PROCESSING,
            purpose="Image analysis",
        )
        assert consent.granted is True
        assert consent.subject_id == "user1"

    def test_withdraw_consent(self) -> None:
        """Test withdrawing consent."""
        manager = create_consent_manager()
        consent = manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.MARKETING,
        )

        result = manager.withdraw_consent(consent.consent_id)
        assert result is True

        # Should no longer have valid consent
        has_consent = manager.has_valid_consent(
            "user1", ConsentType.MARKETING
        )
        assert has_consent is False

    def test_check_valid_consent(self) -> None:
        """Test checking valid consent."""
        manager = create_consent_manager()
        manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.DATA_STORAGE,
        )

        has_consent = manager.has_valid_consent(
            "user1", ConsentType.DATA_STORAGE
        )
        assert has_consent is True

        no_consent = manager.has_valid_consent(
            "user1", ConsentType.PROFILING
        )
        assert no_consent is False

    def test_consent_status(self) -> None:
        """Test getting consent status."""
        manager = create_consent_manager()
        manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.DATA_PROCESSING,
        )
        manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.ANALYTICS,
        )

        status = manager.get_consent_status("user1")
        assert status[ConsentType.DATA_PROCESSING] is True
        assert status[ConsentType.ANALYTICS] is True
        assert status[ConsentType.MARKETING] is False


class TestDataSubjectRequestHandler:
    """Tests for DataSubjectRequestHandler."""

    def test_submit_request(self) -> None:
        """Test submitting a data subject request."""
        handler = create_request_handler()
        request = handler.submit_request(
            subject_id="user1",
            right=DataSubjectRight.ACCESS,
        )
        assert request.subject_id == "user1"
        assert request.right == DataSubjectRight.ACCESS
        assert request.status == RequestStatus.PENDING

    def test_process_request(self) -> None:
        """Test processing a request."""
        handler = create_request_handler()

        # Register a handler for access requests
        def handle_access(req: DataSubjectRequest) -> str:
            return "Access data exported"

        handler.register_handler(DataSubjectRight.ACCESS, handle_access)

        request = handler.submit_request(
            subject_id="user1",
            right=DataSubjectRight.ACCESS,
        )
        processed = handler.process_request(request.request_id)

        assert processed is not None
        assert processed.status == RequestStatus.COMPLETED
        assert processed.response == "Access data exported"

    def test_get_pending_requests(self) -> None:
        """Test getting pending requests."""
        handler = create_request_handler()
        handler.submit_request("user1", DataSubjectRight.ERASURE)
        handler.submit_request("user2", DataSubjectRight.PORTABILITY)

        pending = handler.get_pending_requests()
        assert len(pending) == 2


class TestDataRetentionManager:
    """Tests for DataRetentionManager."""

    def test_add_policy(self) -> None:
        """Test adding retention policy."""
        manager = create_retention_manager()
        policy = DataRetentionPolicy(
            policy_id="policy1",
            data_category="user_images",
            retention_period=timedelta(days=90),
            legal_basis=LegalBasis.CONSENT,
            regulation=PrivacyRegulation.GDPR,
        )
        manager.add_policy(policy)

        retrieved = manager.get_policy("policy1")
        assert retrieved is not None
        assert retrieved.data_category == "user_images"

    def test_record_data_collection(self) -> None:
        """Test recording data collection."""
        manager = create_retention_manager()
        manager.record_data_collection(
            subject_id="user1",
            data_category="user_images",
        )
        # Just verify no exceptions


class TestProcessingActivityLog:
    """Tests for ProcessingActivityLog."""

    def test_log_activity(self) -> None:
        """Test logging processing activity."""
        log = create_activity_log()
        activity = log.log_activity(
            activity_type="image_analysis",
            subject_id="user1",
            data_categories=["image_data"],
            purpose="Analysis for quality check",
            legal_basis=LegalBasis.CONSENT,
        )
        assert activity.activity_type == "image_analysis"
        assert activity.subject_id == "user1"

    def test_get_activities_for_subject(self) -> None:
        """Test getting activities for a subject."""
        log = create_activity_log()
        log.log_activity(
            activity_type="analysis",
            subject_id="user1",
            data_categories=["images"],
            purpose="Test",
            legal_basis=LegalBasis.CONSENT,
        )
        log.log_activity(
            activity_type="storage",
            subject_id="user1",
            data_categories=["images"],
            purpose="Test",
            legal_basis=LegalBasis.CONTRACT,
        )
        log.log_activity(
            activity_type="analysis",
            subject_id="user2",
            data_categories=["images"],
            purpose="Test",
            legal_basis=LegalBasis.CONSENT,
        )

        user1_activities = log.get_activities_for_subject("user1")
        assert len(user1_activities) == 2

    def test_export_for_subject(self) -> None:
        """Test exporting activities for a subject."""
        log = create_activity_log()
        log.log_activity(
            activity_type="analysis",
            subject_id="user1",
            data_categories=["images"],
            purpose="Test",
            legal_basis=LegalBasis.CONSENT,
        )

        export = log.export_for_subject("user1")
        assert export["subject_id"] == "user1"
        assert len(export["activities"]) == 1


class TestPrivacyComplianceManager:
    """Tests for PrivacyComplianceManager."""

    def test_check_processing_allowed_with_consent(self) -> None:
        """Test checking if processing is allowed with consent."""
        manager = create_compliance_manager()
        manager.consent_manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.DATA_PROCESSING,
        )

        allowed, reason = manager.check_processing_allowed(
            subject_id="user1",
            processing_type=ConsentType.DATA_PROCESSING,
        )
        assert allowed is True

    def test_check_processing_denied_without_consent(self) -> None:
        """Test checking processing is denied without consent."""
        manager = create_compliance_manager()

        allowed, reason = manager.check_processing_allowed(
            subject_id="user1",
            processing_type=ConsentType.PROFILING,
        )
        assert allowed is False

    def test_handle_erasure_request(self) -> None:
        """Test handling erasure request."""
        manager = create_compliance_manager()
        manager.consent_manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.DATA_PROCESSING,
        )

        result = manager.handle_erasure_request("user1")
        assert result["subject_id"] == "user1"
        assert result["request_type"] == "erasure"

    def test_handle_access_request(self) -> None:
        """Test handling access request."""
        manager = create_compliance_manager()
        manager.consent_manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.DATA_STORAGE,
        )

        result = manager.handle_access_request("user1")
        assert result["subject_id"] == "user1"
        assert "data" in result

    def test_create_assessment(self) -> None:
        """Test creating privacy impact assessment."""
        manager = create_compliance_manager()
        assessment = manager.create_assessment(
            project_name="Image Analysis System",
            data_types=["images", "metadata"],
            processing_purposes=["analysis", "storage"],
            risk_level="medium",
            mitigations=["encryption", "access_control"],
        )
        assert assessment.project_name == "Image Analysis System"
        assert assessment.approved is False

    def test_approve_assessment(self) -> None:
        """Test approving privacy impact assessment."""
        manager = create_compliance_manager()
        assessment = manager.create_assessment(
            project_name="Test Project",
            data_types=["data"],
            processing_purposes=["analysis"],
            risk_level="low",
            mitigations=["encryption"],
        )

        result = manager.approve_assessment(
            assessment.assessment_id,
            approved_by="dpo@company.com",
        )
        assert result is True

        approved = manager.get_assessment(assessment.assessment_id)
        assert approved is not None
        assert approved.approved is True

    def test_compliance_status(self) -> None:
        """Test getting compliance status."""
        manager = create_compliance_manager()
        status = manager.get_compliance_status()

        assert "regulations" in status
        assert "pending_requests" in status
        assert "compliant" in status


class TestPrivacyComplianceVisionProvider:
    """Tests for PrivacyComplianceVisionProvider."""

    @pytest.mark.asyncio
    async def test_analyze_with_consent(self) -> None:
        """Test image analysis with valid consent."""
        stub = SimpleStubProvider()
        manager = create_compliance_manager()

        # Grant consent
        manager.consent_manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.DATA_PROCESSING,
        )

        provider = create_compliance_provider(stub, manager)
        result = await provider.analyze_image(
            b"test", subject_id="user1"
        )
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_analyze_without_consent(self) -> None:
        """Test image analysis without consent."""
        stub = SimpleStubProvider()
        manager = create_compliance_manager()
        provider = create_compliance_provider(
            stub, manager, require_consent=True
        )

        result = await provider.analyze_image(
            b"test", subject_id="user_no_consent"
        )
        # Should return a result indicating processing not allowed
        assert "not allowed" in result.summary.lower() or result.confidence == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase14Integration:
    """Integration tests for Phase 14 features."""

    @pytest.mark.asyncio
    async def test_full_security_pipeline(self) -> None:
        """Test full security pipeline with all Phase 14 features."""
        stub = SimpleStubProvider()

        # Set up access control
        access_manager = create_access_manager()
        user = access_manager.create_user("analyst")
        role = access_manager.role_manager.create_role("analyst", permissions={Permission.ANALYZE})
        access_manager.role_manager.assign_role(user.user_id, role.role_id)

        # Set up compliance
        compliance_manager = create_compliance_manager()
        compliance_manager.consent_manager.grant_consent(
            subject_id=user.user_id,
            consent_type=ConsentType.DATA_PROCESSING,
        )

        # Create compliance provider
        provider = create_compliance_provider(stub, compliance_manager)

        # Analyze with proper setup
        result = await provider.analyze_image(
            b"test", subject_id=user.user_id
        )
        assert result.confidence == 0.95

    def test_data_masking_operations(self) -> None:
        """Test data masking operations."""
        # Set up masking
        masking_engine = create_masking_engine()

        # Mask sensitive data
        text = "Contact john@example.com for SSN 123-45-6789"
        result = masking_engine.mask_text(text)
        assert "john@example.com" not in result.masked_text
        assert "123-45-6789" not in result.masked_text

    def test_key_management_operations(self) -> None:
        """Test key management operations."""
        # Create key manager
        key_manager = create_key_manager()
        assert key_manager is not None

        # Create secrets vault
        vault = create_secrets_vault()
        assert vault is not None

    def test_consent_with_retention(self) -> None:
        """Test consent management with data retention."""
        # Create compliance manager
        manager = create_compliance_manager()

        # Add retention policy
        policy = DataRetentionPolicy(
            policy_id="user_data_policy",
            data_category="user_images",
            retention_period=timedelta(days=30),
            legal_basis=LegalBasis.CONSENT,
            regulation=PrivacyRegulation.GDPR,
        )
        manager.retention_manager.add_policy(policy)

        # Grant consent
        manager.consent_manager.grant_consent(
            subject_id="user1",
            consent_type=ConsentType.DATA_STORAGE,
            expiry_days=30,
        )

        # Record data collection
        manager.retention_manager.record_data_collection(
            subject_id="user1",
            data_category="user_images",
        )

        # Check consent is valid
        has_consent = manager.consent_manager.has_valid_consent(
            "user1", ConsentType.DATA_STORAGE
        )
        assert has_consent is True
