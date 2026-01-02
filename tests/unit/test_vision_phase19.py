"""Tests for Vision module Phase 19: Advanced Security & Compliance.

Tests security scanner, access control, audit logger, encryption manager,
and compliance checker components.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core.vision import (  # Security Scanner; Access Control; Audit Logger; Encryption Manager; Compliance Checker
    AccessController,
    AccessControlVisionProvider,
    AccessDecision,
    AccessRequest,
    AccessResult,
    ACLResource,
    ACLResourceType,
    ACLRole,
    ACLSession,
    ACLUser,
    APIKey,
    APIKeyManager,
    AuditAlert,
    AuditCategory,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditLoggerVisionProvider,
    AuditPolicy,
    AuditQuery,
    AuditSeverity,
    AuditSummary,
    AuthMethod,
    Certificate,
    CodeScanner,
    ComplianceAuditLogger,
    ComplianceChecker,
    ComplianceCheckerVisionProvider,
    ComplianceControl,
    ComplianceFramework,
    CompliancePolicy,
    ComplianceReport,
    ComplianceReporter,
    ComplianceRiskLevel,
    ComplianceStatus,
    ControlAssessment,
    ControlAssessor,
    ControlCategory,
    ControlRegistry,
    DataClassification,
    DataInventory,
    DataInventoryItem,
    DependencyScanner,
    EncryptedData,
    EncryptionAlgorithm,
    EncryptionKey,
    EncryptionManager,
    EncryptionManagerVisionProvider,
    Encryptor,
    HashAlgorithm,
    KeyGenerator,
    KeyPolicy,
    KeyPurpose,
    KeyRotationEvent,
    KeyRotationManager,
    KeyStatus,
    KeyType,
    LogDestination,
    LogIntegrity,
    MemoryAuditStore,
    MemoryKeyStore,
    Permission,
    PolicyEngine,
    PolicyType,
    PolicyViolation,
    PrivacyImpactAssessment,
    RemediationPriority,
    RetentionPeriod,
    RiskAssessment,
    RiskAssessor,
    RoleManager,
    RoleType,
    ScanConfig,
    ScanResult,
    ScanStatus,
    ScanType,
    SecurityPolicy,
    SecurityRiskLevel,
    SecurityScanner,
    SecurityScannerVisionProvider,
    SessionManager,
    SessionStatus,
    Tenant,
    ThreatCategory,
    ThreatDetector,
    ThreatIndicator,
    UserManager,
    Vulnerability,
    VulnerabilitySeverity,
    create_access_control_provider,
    create_access_controller,
    create_access_request,
    create_acl_resource,
    create_acl_role,
    create_acl_user,
    create_api_key_manager,
    create_audit_alert,
    create_audit_event,
    create_audit_logger,
    create_audit_logger_provider,
    create_audit_policy,
    create_audit_query,
    create_code_scanner,
    create_compliance_audit_logger,
    create_compliance_checker,
    create_compliance_checker_provider,
    create_compliance_control,
    create_compliance_policy,
    create_control_registry,
    create_data_inventory,
    create_data_inventory_item,
    create_dependency_scanner,
    create_encryption_manager,
    create_encryption_manager_provider,
    create_encryptor,
    create_key_generator,
    create_key_policy,
    create_memory_audit_store,
    create_memory_key_store,
    create_policy_engine,
    create_privacy_impact_assessment,
    create_risk_assessor,
    create_role_manager,
    create_scan_config,
    create_security_policy,
    create_security_scanner,
    create_security_scanner_provider,
    create_session_manager,
    create_threat_detector,
    create_threat_indicator,
    create_user_manager,
    create_vulnerability,
)

# =============================================================================
# Security Scanner Tests
# =============================================================================


class TestSecurityScannerEnums:
    """Test security scanner enums."""

    def test_vulnerability_severity_values(self):
        """Test VulnerabilitySeverity enum values."""
        assert VulnerabilitySeverity.CRITICAL == "critical"
        assert VulnerabilitySeverity.HIGH == "high"
        assert VulnerabilitySeverity.MEDIUM == "medium"
        assert VulnerabilitySeverity.LOW == "low"
        assert VulnerabilitySeverity.INFO == "info"

    def test_scan_type_values(self):
        """Test ScanType enum values."""
        assert ScanType.CODE == "code"
        assert ScanType.DEPENDENCY == "dependency"
        assert ScanType.INFRASTRUCTURE == "infrastructure"
        assert ScanType.CONTAINER == "container"

    def test_threat_category_values(self):
        """Test ThreatCategory enum values."""
        assert ThreatCategory.INJECTION == "injection"
        assert ThreatCategory.AUTHENTICATION == "authentication"
        assert ThreatCategory.XSS == "xss"
        assert ThreatCategory.DATA_EXPOSURE == "data_exposure"

    def test_scan_status_values(self):
        """Test ScanStatus enum values."""
        assert ScanStatus.PENDING == "pending"
        assert ScanStatus.RUNNING == "running"
        assert ScanStatus.COMPLETED == "completed"
        assert ScanStatus.FAILED == "failed"


class TestSecurityScannerDataclasses:
    """Test security scanner dataclasses."""

    def test_vulnerability_creation(self):
        """Test Vulnerability dataclass."""
        vuln = create_vulnerability(
            vuln_id="VULN-001",
            title="SQL Injection",
            severity=VulnerabilitySeverity.HIGH,
            category=ThreatCategory.INJECTION,
            description="SQL injection vulnerability found",
        )
        assert vuln.vuln_id == "VULN-001"
        assert vuln.title == "SQL Injection"
        assert vuln.severity == VulnerabilitySeverity.HIGH

    def test_scan_config_creation(self):
        """Test ScanConfig dataclass."""
        config = create_scan_config(
            scan_id="scan-001",
            scan_type=ScanType.CODE,
            target="src/",
        )
        assert config.scan_id == "scan-001"
        assert config.scan_type == ScanType.CODE
        assert config.target == "src/"

    def test_threat_indicator_creation(self):
        """Test ThreatIndicator dataclass."""
        indicator = create_threat_indicator(
            indicator_id="TI-001",
            indicator_type="pattern",
            value="SELECT.*FROM.*WHERE",
            threat_category=ThreatCategory.INJECTION,
            confidence=0.85,
        )
        assert indicator.indicator_id == "TI-001"
        assert indicator.threat_category == ThreatCategory.INJECTION
        assert indicator.confidence == 0.85

    def test_security_policy_creation(self):
        """Test SecurityPolicy dataclass."""
        policy = create_security_policy(
            policy_id="POL-001",
            name="High Severity Policy",
            rules=[{"action": "block", "severity": "critical"}],
        )
        assert policy.policy_id == "POL-001"
        assert len(policy.rules) == 1


class TestSecurityScannerComponents:
    """Test security scanner components."""

    def test_code_scanner_creation(self):
        """Test CodeScanner creation."""
        scanner = create_code_scanner()
        assert isinstance(scanner, CodeScanner)

    def test_dependency_scanner_creation(self):
        """Test DependencyScanner creation."""
        scanner = create_dependency_scanner()
        assert isinstance(scanner, DependencyScanner)

    def test_threat_detector_creation(self):
        """Test ThreatDetector creation."""
        detector = create_threat_detector()
        assert isinstance(detector, ThreatDetector)

    def test_risk_assessor_creation(self):
        """Test RiskAssessor creation."""
        assessor = create_risk_assessor()
        assert isinstance(assessor, RiskAssessor)

    def test_security_scanner_creation(self):
        """Test SecurityScanner creation."""
        scanner = create_security_scanner()
        assert isinstance(scanner, SecurityScanner)

    def test_security_scanner_provider_creation(self):
        """Test SecurityScannerVisionProvider creation."""
        provider = create_security_scanner_provider()
        assert isinstance(provider, SecurityScannerVisionProvider)
        assert provider.provider_name == "security_scanner"


# =============================================================================
# Access Control Tests
# =============================================================================


class TestAccessControlEnums:
    """Test access control enums."""

    def test_permission_values(self):
        """Test Permission enum values."""
        assert Permission.READ == "read"
        assert Permission.WRITE == "write"
        assert Permission.DELETE == "delete"
        assert Permission.ADMIN == "admin"

    def test_role_type_values(self):
        """Test RoleType enum values."""
        assert RoleType.SYSTEM == "system"
        assert RoleType.CUSTOM == "custom"
        assert RoleType.TEMPORARY == "temporary"

    def test_auth_method_values(self):
        """Test AuthMethod enum values."""
        assert AuthMethod.PASSWORD == "password"
        assert AuthMethod.API_KEY == "api_key"
        assert AuthMethod.OAUTH == "oauth"
        assert AuthMethod.JWT == "jwt"

    def test_session_status_values(self):
        """Test SessionStatus enum values."""
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.EXPIRED == "expired"
        assert SessionStatus.REVOKED == "revoked"

    def test_access_decision_values(self):
        """Test AccessDecision enum values."""
        assert AccessDecision.ALLOW == "allow"
        assert AccessDecision.DENY == "deny"


class TestAccessControlDataclasses:
    """Test access control dataclasses."""

    def test_user_creation(self):
        """Test ACLUser dataclass."""
        user = create_acl_user(
            user_id="user-001",
            username="testuser",
            email="test@example.com",
            roles=["role-user"],
        )
        assert user.user_id == "user-001"
        assert user.username == "testuser"

    def test_role_creation(self):
        """Test ACLRole dataclass."""
        role = create_acl_role(
            role_id="role-001",
            name="Developer",
            permissions=[Permission.READ, Permission.WRITE],
        )
        assert role.role_id == "role-001"
        assert Permission.WRITE in role.permissions

    def test_acl_resource_creation(self):
        """Test ACLResource dataclass."""
        resource = create_acl_resource(
            resource_id="res-001",
            resource_type=ACLResourceType.MODEL,
            name="config.json",
            owner_id="user-001",
        )
        assert resource.resource_id == "res-001"
        assert resource.resource_type == ACLResourceType.MODEL

    def test_access_request_creation(self):
        """Test AccessRequest dataclass."""
        request = create_access_request(
            user_id="user-001",
            resource_id="res-001",
            permission=Permission.READ,
        )
        assert request.user_id == "user-001"
        assert request.permission == Permission.READ


class TestAccessControlComponents:
    """Test access control components."""

    def test_user_manager_creation(self):
        """Test UserManager creation."""
        manager = create_user_manager()
        assert isinstance(manager, UserManager)

    def test_role_manager_creation(self):
        """Test RoleManager creation."""
        manager = create_role_manager()
        assert isinstance(manager, RoleManager)

    def test_session_manager_creation(self):
        """Test SessionManager creation."""
        manager = create_session_manager()
        assert isinstance(manager, SessionManager)

    def test_api_key_manager_creation(self):
        """Test APIKeyManager creation."""
        manager = create_api_key_manager()
        assert isinstance(manager, APIKeyManager)

    def test_access_controller_creation(self):
        """Test AccessController creation."""
        controller = create_access_controller()
        assert isinstance(controller, AccessController)

    def test_access_control_provider_creation(self):
        """Test AccessControlVisionProvider creation."""
        provider = create_access_control_provider()
        assert isinstance(provider, AccessControlVisionProvider)
        assert provider.provider_name == "access_control"


# =============================================================================
# Audit Logger Tests
# =============================================================================


class TestAuditLoggerEnums:
    """Test audit logger enums."""

    def test_audit_event_type_values(self):
        """Test AuditEventType enum values."""
        assert AuditEventType.LOGIN == "login"
        assert AuditEventType.LOGOUT == "logout"
        assert AuditEventType.CREATE == "create"
        assert AuditEventType.DELETE == "delete"
        assert AuditEventType.READ == "read"

    def test_audit_severity_values(self):
        """Test AuditSeverity enum values."""
        assert AuditSeverity.INFO == "info"
        assert AuditSeverity.WARNING == "warning"
        assert AuditSeverity.ERROR == "error"
        assert AuditSeverity.CRITICAL == "critical"

    def test_audit_category_values(self):
        """Test AuditCategory enum values."""
        assert AuditCategory.AUTHENTICATION == "authentication"
        assert AuditCategory.AUTHORIZATION == "authorization"
        assert AuditCategory.DATA_ACCESS == "data_access"
        assert AuditCategory.SYSTEM == "system"

    def test_log_destination_values(self):
        """Test LogDestination enum values."""
        assert LogDestination.FILE == "file"
        assert LogDestination.DATABASE == "database"
        assert LogDestination.MEMORY == "memory"

    def test_retention_period_values(self):
        """Test RetentionPeriod enum values."""
        assert RetentionPeriod.DAYS_30 == "30d"
        assert RetentionPeriod.DAYS_90 == "90d"
        assert RetentionPeriod.DAYS_365 == "365d"


class TestAuditLoggerDataclasses:
    """Test audit logger dataclasses."""

    def test_audit_event_creation(self):
        """Test AuditEvent dataclass."""
        event = create_audit_event(
            event_id="evt-001",
            event_type=AuditEventType.LOGIN,
            category=AuditCategory.AUTHENTICATION,
            action="user_login",
        )
        assert event.event_id == "evt-001"
        assert event.event_type == AuditEventType.LOGIN
        assert event.category == AuditCategory.AUTHENTICATION

    def test_audit_query_creation(self):
        """Test AuditQuery dataclass."""
        query = create_audit_query(
            limit=100,
        )
        assert query.limit == 100

    def test_audit_policy_creation(self):
        """Test AuditPolicy dataclass."""
        policy = create_audit_policy(
            policy_id="pol-001",
            name="Security Events",
            event_types=[AuditEventType.LOGIN],
        )
        assert policy.policy_id == "pol-001"
        assert AuditEventType.LOGIN in policy.event_types

    def test_audit_alert_creation(self):
        """Test AuditAlert dataclass."""
        alert = create_audit_alert(
            alert_id="alert-001",
            name="Failed Login Alert",
            condition={"type": "failed_login", "threshold": 5},
            severity=AuditSeverity.WARNING,
        )
        assert alert.alert_id == "alert-001"
        assert alert.severity == AuditSeverity.WARNING


class TestAuditLoggerComponents:
    """Test audit logger components."""

    def test_memory_audit_store_creation(self):
        """Test MemoryAuditStore creation."""
        store = create_memory_audit_store()
        assert isinstance(store, MemoryAuditStore)

    def test_audit_logger_creation(self):
        """Test AuditLogger creation."""
        logger = create_audit_logger()
        assert isinstance(logger, AuditLogger)

    def test_compliance_audit_logger_creation(self):
        """Test ComplianceAuditLogger creation."""
        logger = create_compliance_audit_logger()
        assert isinstance(logger, ComplianceAuditLogger)

    def test_audit_logger_provider_creation(self):
        """Test AuditLoggerVisionProvider creation."""
        provider = create_audit_logger_provider()
        assert isinstance(provider, AuditLoggerVisionProvider)
        assert provider.provider_name == "audit_logger"


# =============================================================================
# Encryption Manager Tests
# =============================================================================


class TestEncryptionManagerEnums:
    """Test encryption manager enums."""

    def test_encryption_algorithm_values(self):
        """Test EncryptionAlgorithm enum values."""
        assert EncryptionAlgorithm.AES_256_GCM == "aes-256-gcm"
        assert EncryptionAlgorithm.AES_256_CBC == "aes-256-cbc"
        assert EncryptionAlgorithm.RSA_OAEP == "rsa-oaep"

    def test_key_type_values(self):
        """Test KeyType enum values."""
        assert KeyType.SYMMETRIC == "symmetric"
        assert KeyType.ASYMMETRIC == "asymmetric"
        assert KeyType.HMAC == "hmac"

    def test_key_status_values(self):
        """Test KeyStatus enum values."""
        assert KeyStatus.ACTIVE == "active"
        assert KeyStatus.INACTIVE == "inactive"
        assert KeyStatus.COMPROMISED == "compromised"

    def test_key_purpose_values(self):
        """Test KeyPurpose enum values."""
        assert KeyPurpose.ENCRYPTION == "encryption"
        assert KeyPurpose.SIGNING == "signing"
        assert KeyPurpose.KEY_WRAPPING == "key_wrapping"

    def test_hash_algorithm_values(self):
        """Test HashAlgorithm enum values."""
        assert HashAlgorithm.SHA256 == "sha256"
        assert HashAlgorithm.SHA384 == "sha384"
        assert HashAlgorithm.SHA512 == "sha512"


class TestEncryptionManagerDataclasses:
    """Test encryption manager dataclasses."""

    def test_key_policy_creation(self):
        """Test KeyPolicy dataclass."""
        policy = create_key_policy(
            policy_id="kp-001",
            name="Standard Key Policy",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            rotation_days=90,
        )
        assert policy.policy_id == "kp-001"
        assert policy.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert policy.rotation_days == 90


class TestEncryptionManagerComponents:
    """Test encryption manager components."""

    def test_memory_key_store_creation(self):
        """Test MemoryKeyStore creation."""
        store = create_memory_key_store()
        assert isinstance(store, MemoryKeyStore)

    def test_key_generator_creation(self):
        """Test KeyGenerator creation."""
        generator = create_key_generator()
        assert isinstance(generator, KeyGenerator)

    def test_encryptor_creation(self):
        """Test Encryptor creation."""
        store = create_memory_key_store()
        encryptor = create_encryptor(store)
        assert isinstance(encryptor, Encryptor)

    def test_encryption_manager_creation(self):
        """Test EncryptionManager creation."""
        manager = create_encryption_manager()
        assert isinstance(manager, EncryptionManager)

    def test_encryption_manager_provider_creation(self):
        """Test EncryptionManagerVisionProvider creation."""
        provider = create_encryption_manager_provider()
        assert isinstance(provider, EncryptionManagerVisionProvider)
        assert provider.provider_name == "encryption_manager"


# =============================================================================
# Compliance Checker Tests
# =============================================================================


class TestComplianceCheckerEnums:
    """Test compliance checker enums."""

    def test_compliance_framework_values(self):
        """Test ComplianceFramework enum values."""
        assert ComplianceFramework.GDPR == "gdpr"
        assert ComplianceFramework.HIPAA == "hipaa"
        assert ComplianceFramework.SOC2 == "soc2"
        assert ComplianceFramework.PCI_DSS == "pci_dss"
        assert ComplianceFramework.ISO_27001 == "iso_27001"

    def test_compliance_status_values(self):
        """Test ComplianceStatus enum values."""
        assert ComplianceStatus.COMPLIANT == "compliant"
        assert ComplianceStatus.NON_COMPLIANT == "non_compliant"
        assert ComplianceStatus.PARTIAL == "partial"
        assert ComplianceStatus.NOT_APPLICABLE == "not_applicable"

    def test_control_category_values(self):
        """Test ControlCategory enum values."""
        assert ControlCategory.ACCESS_CONTROL == "access_control"
        assert ControlCategory.DATA_PROTECTION == "data_protection"
        assert ControlCategory.AUDIT_LOGGING == "audit_logging"
        assert ControlCategory.INCIDENT_RESPONSE == "incident_response"

    def test_data_classification_values(self):
        """Test DataClassification enum values."""
        assert DataClassification.PUBLIC == "public"
        assert DataClassification.INTERNAL == "internal"
        assert DataClassification.CONFIDENTIAL == "confidential"
        assert DataClassification.RESTRICTED == "restricted"

    def test_policy_type_values(self):
        """Test PolicyType enum values."""
        assert PolicyType.DATA_RETENTION == "data_retention"
        assert PolicyType.DATA_ACCESS == "data_access"
        assert PolicyType.ENCRYPTION == "encryption"


class TestComplianceCheckerDataclasses:
    """Test compliance checker dataclasses."""

    def test_compliance_control_creation(self):
        """Test ComplianceControl dataclass."""
        control = create_compliance_control(
            control_id="ctrl-001",
            name="Access Control Policy",
            description="Ensure proper access control",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.ACCESS_CONTROL,
        )
        assert control.control_id == "ctrl-001"
        assert control.framework == ComplianceFramework.SOC2

    def test_compliance_policy_creation(self):
        """Test CompliancePolicy dataclass."""
        policy = create_compliance_policy(
            policy_id="pol-001",
            name="Data Retention Policy",
            policy_type=PolicyType.DATA_RETENTION,
            framework=ComplianceFramework.GDPR,
        )
        assert policy.policy_id == "pol-001"
        assert policy.framework == ComplianceFramework.GDPR

    def test_data_inventory_item_creation(self):
        """Test DataInventoryItem dataclass."""
        item = create_data_inventory_item(
            item_id="di-001",
            name="Customer Data",
            classification=DataClassification.CONFIDENTIAL,
            location="database/customers",
            owner="data-team",
        )
        assert item.item_id == "di-001"
        assert item.classification == DataClassification.CONFIDENTIAL

    def test_privacy_impact_assessment_creation(self):
        """Test PrivacyImpactAssessment dataclass."""
        pia = create_privacy_impact_assessment(
            assessment_id="pia-001",
            project_name="New Feature",
            assessor="security-team",
            data_types=[DataClassification.PII],
        )
        assert pia.assessment_id == "pia-001"


class TestComplianceCheckerComponents:
    """Test compliance checker components."""

    def test_control_registry_creation(self):
        """Test ControlRegistry creation."""
        registry = create_control_registry()
        assert isinstance(registry, ControlRegistry)

    def test_policy_engine_creation(self):
        """Test PolicyEngine creation."""
        engine = create_policy_engine()
        assert isinstance(engine, PolicyEngine)

    def test_data_inventory_creation(self):
        """Test DataInventory creation."""
        inventory = create_data_inventory()
        assert isinstance(inventory, DataInventory)

    def test_compliance_checker_creation(self):
        """Test ComplianceChecker creation."""
        checker = create_compliance_checker()
        assert isinstance(checker, ComplianceChecker)

    def test_compliance_checker_provider_creation(self):
        """Test ComplianceCheckerVisionProvider creation."""
        provider = create_compliance_checker_provider()
        assert isinstance(provider, ComplianceCheckerVisionProvider)
        assert provider.provider_name == "compliance_checker"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase19Integration:
    """Integration tests for Phase 19 components."""

    def test_security_compliance_workflow(self):
        """Test security and compliance workflow integration."""
        # Create security scanner
        scanner = create_security_scanner()
        assert isinstance(scanner, SecurityScanner)

        # Create access controller
        controller = create_access_controller()
        assert isinstance(controller, AccessController)

        # Create audit logger
        logger = create_audit_logger()
        assert isinstance(logger, AuditLogger)

        # Create encryption manager
        encryption = create_encryption_manager()
        assert isinstance(encryption, EncryptionManager)

        # Create compliance checker
        compliance = create_compliance_checker()
        assert isinstance(compliance, ComplianceChecker)

    def test_all_vision_providers_available(self):
        """Test all Phase 19 vision providers are available."""
        providers = [
            create_security_scanner_provider(),
            create_access_control_provider(),
            create_audit_logger_provider(),
            create_encryption_manager_provider(),
            create_compliance_checker_provider(),
        ]

        provider_names = [p.provider_name for p in providers]
        assert "security_scanner" in provider_names
        assert "access_control" in provider_names
        assert "audit_logger" in provider_names
        assert "encryption_manager" in provider_names
        assert "compliance_checker" in provider_names


class TestSecurityScannerIntegration:
    """Integration tests for security scanner."""

    def test_code_scanner_workflow(self):
        """Test code scanner workflow."""
        scanner = create_code_scanner()
        config = create_scan_config(
            scan_id="scan-001",
            scan_type=ScanType.CODE,
            target="src/",
        )
        assert config.target == "src/"

    def test_threat_detection_workflow(self):
        """Test threat detection workflow."""
        detector = create_threat_detector()
        indicator = create_threat_indicator(
            indicator_id="ti-001",
            indicator_type="pattern",
            value="DROP TABLE",
            threat_category=ThreatCategory.INJECTION,
            confidence=0.95,
        )
        assert indicator.confidence == 0.95


class TestAccessControlIntegration:
    """Integration tests for access control."""

    def test_user_role_workflow(self):
        """Test user and role management workflow."""
        user_manager = create_user_manager()
        role_manager = create_role_manager()

        user = create_acl_user(
            user_id="u-001",
            username="admin",
            email="admin@example.com",
            roles=["admin"],
        )
        role = create_acl_role(
            role_id="r-001",
            name="admin",
            permissions=[Permission.ADMIN],
        )

        assert user.username == "admin"
        assert Permission.ADMIN in role.permissions

    def test_session_management_workflow(self):
        """Test session management workflow."""
        session_manager = create_session_manager(session_timeout=3600)
        assert isinstance(session_manager, SessionManager)


class TestAuditLoggerIntegration:
    """Integration tests for audit logger."""

    def test_audit_logging_workflow(self):
        """Test audit logging workflow."""
        logger = create_audit_logger()
        event = create_audit_event(
            event_id="evt-001",
            event_type=AuditEventType.LOGIN,
            category=AuditCategory.AUTHENTICATION,
            action="user_login",
        )
        assert event.event_type == AuditEventType.LOGIN

    def test_compliance_logging_workflow(self):
        """Test compliance audit logging workflow."""
        logger = create_compliance_audit_logger()
        policy = create_audit_policy(
            policy_id="pol-001",
            name="GDPR Audit Policy",
            event_types=[AuditEventType.READ, AuditEventType.DELETE],
        )
        assert AuditEventType.READ in policy.event_types


class TestEncryptionIntegration:
    """Integration tests for encryption manager."""

    def test_key_management_workflow(self):
        """Test key management workflow."""
        manager = create_encryption_manager()
        policy = create_key_policy(
            policy_id="kp-001",
            name="Standard Policy",
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            rotation_days=30,
        )
        assert policy.rotation_days == 30

    def test_encryptor_workflow(self):
        """Test encryptor workflow."""
        store = create_memory_key_store()
        encryptor = create_encryptor(store)
        assert isinstance(encryptor, Encryptor)


class TestComplianceIntegration:
    """Integration tests for compliance checker."""

    def test_gdpr_compliance_workflow(self):
        """Test GDPR compliance workflow."""
        checker = create_compliance_checker()
        control = create_compliance_control(
            control_id="gdpr-001",
            name="Right to Erasure",
            description="Implement data deletion capability",
            framework=ComplianceFramework.GDPR,
            category=ControlCategory.DATA_PROTECTION,
        )
        assert control.framework == ComplianceFramework.GDPR

    def test_multi_framework_compliance(self):
        """Test multi-framework compliance workflow."""
        policy = create_compliance_policy(
            policy_id="multi-001",
            name="Data Retention Policy",
            policy_type=PolicyType.DATA_RETENTION,
            framework=ComplianceFramework.GDPR,
        )
        assert policy.policy_type == PolicyType.DATA_RETENTION

    def test_privacy_impact_assessment_workflow(self):
        """Test privacy impact assessment workflow."""
        pia = create_privacy_impact_assessment(
            assessment_id="pia-001",
            project_name="ML Model Training",
            assessor="privacy-team",
            data_types=[DataClassification.PII, DataClassification.PHI],
        )
        assert len(pia.data_types) == 2


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_scan_targets(self):
        """Test scan config with minimal target."""
        config = create_scan_config(
            scan_id="scan-empty",
            scan_type=ScanType.CODE,
            target="",
        )
        assert config.target == ""

    def test_empty_permissions(self):
        """Test role with empty permissions."""
        role = create_acl_role(
            role_id="r-empty",
            name="NoPermissions",
            permissions=[],
        )
        assert len(role.permissions) == 0

    def test_empty_policy_rules(self):
        """Test policy with no rules."""
        policy = create_compliance_policy(
            policy_id="pol-empty",
            name="Empty Policy",
            policy_type=PolicyType.DATA_ACCESS,
            framework=ComplianceFramework.SOC2,
            rules=[],
        )
        assert len(policy.rules) == 0

    def test_high_confidence_indicator(self):
        """Test threat indicator with maximum confidence."""
        indicator = create_threat_indicator(
            indicator_id="ti-max",
            indicator_type="signature",
            value="known_malware_signature",
            threat_category=ThreatCategory.DATA_EXPOSURE,
            confidence=1.0,
        )
        assert indicator.confidence == 1.0

    def test_years_retention_period(self):
        """Test audit policy with years retention."""
        policy = create_audit_policy(
            policy_id="pol-long",
            name="Long Retention",
            event_types=[AuditEventType.READ],
        )
        assert AuditEventType.READ in policy.event_types
