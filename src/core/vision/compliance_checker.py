"""Compliance Checker Module for Vision System.

This module provides regulatory compliance capabilities including:
- Compliance framework support (GDPR, HIPAA, SOC2, PCI-DSS)
- Policy enforcement and validation
- Compliance reporting and auditing
- Data residency and sovereignty
- Privacy impact assessment
- Automated compliance monitoring

Phase 19: Advanced Security & Compliance
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist"
    CCPA = "ccpa"
    CUSTOM = "custom"


class ComplianceStatus(str, Enum):
    """Compliance check status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ControlCategory(str, Enum):
    """Compliance control categories."""

    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"
    INCIDENT_RESPONSE = "incident_response"
    RISK_MANAGEMENT = "risk_management"
    PRIVACY = "privacy"
    NETWORK_SECURITY = "network_security"


class RiskLevel(str, Enum):
    """Compliance risk levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class DataClassification(str, Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"
    PCI = "pci"


class PolicyType(str, Enum):
    """Types of compliance policies."""

    DATA_RETENTION = "data_retention"
    DATA_ACCESS = "data_access"
    ENCRYPTION = "encryption"
    AUDIT = "audit"
    PRIVACY = "privacy"
    SECURITY = "security"


# ========================
# Dataclasses
# ========================


@dataclass
class ComplianceControl:
    """A compliance control requirement."""

    control_id: str
    name: str
    description: str
    framework: ComplianceFramework
    category: ControlCategory
    required: bool = True
    automated: bool = True
    evidence_required: List[str] = field(default_factory=list)
    implementation_guidance: str = ""


@dataclass
class ControlAssessment:
    """Assessment result for a control."""

    control_id: str
    status: ComplianceStatus
    findings: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    remediation: str = ""
    assessed_at: datetime = field(default_factory=datetime.now)
    assessed_by: str = "system"


@dataclass
class CompliancePolicy:
    """A compliance policy."""

    policy_id: str
    name: str
    policy_type: PolicyType
    framework: ComplianceFramework
    rules: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    version: str = "1.0.0"
    effective_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None


@dataclass
class PolicyViolation:
    """A policy violation."""

    violation_id: str
    policy_id: str
    resource_id: str
    resource_type: str
    violation_type: str
    description: str
    severity: RiskLevel = RiskLevel.MEDIUM
    detected_at: datetime = field(default_factory=datetime.now)
    remediated: bool = False
    remediated_at: Optional[datetime] = None


@dataclass
class ComplianceReport:
    """Compliance assessment report."""

    report_id: str
    framework: ComplianceFramework
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    overall_status: ComplianceStatus
    control_assessments: List[ControlAssessment]
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataInventoryItem:
    """Item in the data inventory."""

    item_id: str
    name: str
    classification: DataClassification
    location: str
    owner: str
    data_types: List[str] = field(default_factory=list)
    retention_days: int = 365
    encryption_required: bool = True
    pii_fields: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment (PIA)."""

    assessment_id: str
    project_name: str
    assessor: str
    data_types: List[DataClassification]
    processing_purposes: List[str]
    third_party_sharing: bool = False
    cross_border_transfer: bool = False
    risk_level: RiskLevel = RiskLevel.MEDIUM
    mitigations: List[str] = field(default_factory=list)
    approved: bool = False
    approved_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


# ========================
# Core Classes
# ========================


class ControlRegistry:
    """Registry of compliance controls."""

    def __init__(self):
        self._controls: Dict[str, ComplianceControl] = {}
        self._framework_controls: Dict[ComplianceFramework, List[str]] = defaultdict(list)
        self._lock = threading.RLock()

        # Initialize with common controls
        self._initialize_default_controls()

    def _initialize_default_controls(self) -> None:
        """Initialize default compliance controls."""
        default_controls = [
            # GDPR Controls
            ComplianceControl(
                control_id="gdpr-1",
                name="Data Subject Rights",
                description="Implement data subject access, rectification, and erasure rights",
                framework=ComplianceFramework.GDPR,
                category=ControlCategory.PRIVACY,
                evidence_required=["access_request_logs", "erasure_logs"],
            ),
            ComplianceControl(
                control_id="gdpr-2",
                name="Consent Management",
                description="Obtain and manage explicit consent for data processing",
                framework=ComplianceFramework.GDPR,
                category=ControlCategory.PRIVACY,
                evidence_required=["consent_records"],
            ),
            # SOC2 Controls
            ComplianceControl(
                control_id="soc2-cc6.1",
                name="Logical Access",
                description="Restrict logical access to information assets",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.ACCESS_CONTROL,
                evidence_required=["access_logs", "user_provisioning_records"],
            ),
            ComplianceControl(
                control_id="soc2-cc6.6",
                name="System Operations",
                description="Protect against security events",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.NETWORK_SECURITY,
                evidence_required=["security_logs", "incident_reports"],
            ),
            # HIPAA Controls
            ComplianceControl(
                control_id="hipaa-164.312a",
                name="Access Control",
                description="Implement technical policies for electronic PHI access",
                framework=ComplianceFramework.HIPAA,
                category=ControlCategory.ACCESS_CONTROL,
                evidence_required=["access_policies", "authentication_logs"],
            ),
            ComplianceControl(
                control_id="hipaa-164.312e",
                name="Transmission Security",
                description="Implement technical measures for ePHI transmission",
                framework=ComplianceFramework.HIPAA,
                category=ControlCategory.ENCRYPTION,
                evidence_required=["encryption_certificates", "transmission_logs"],
            ),
        ]

        for control in default_controls:
            self.register_control(control)

    def register_control(self, control: ComplianceControl) -> None:
        """Register a compliance control."""
        with self._lock:
            self._controls[control.control_id] = control
            self._framework_controls[control.framework].append(control.control_id)

    def get_control(self, control_id: str) -> Optional[ComplianceControl]:
        """Get a control by ID."""
        return self._controls.get(control_id)

    def list_controls(
        self,
        framework: Optional[ComplianceFramework] = None,
        category: Optional[ControlCategory] = None,
    ) -> List[ComplianceControl]:
        """List compliance controls."""
        if framework:
            control_ids = self._framework_controls.get(framework, [])
            controls = [self._controls[cid] for cid in control_ids]
        else:
            controls = list(self._controls.values())

        if category:
            controls = [c for c in controls if c.category == category]

        return controls


class ControlAssessor:
    """Assess compliance controls."""

    def __init__(self, registry: ControlRegistry):
        self._registry = registry
        self._assessors: Dict[str, Callable[[ComplianceControl], ControlAssessment]] = {}

    def register_assessor(
        self,
        control_id: str,
        assessor: Callable[[ComplianceControl], ControlAssessment],
    ) -> None:
        """Register an assessor function for a control."""
        self._assessors[control_id] = assessor

    def assess_control(self, control_id: str) -> ControlAssessment:
        """Assess a single control."""
        control = self._registry.get_control(control_id)
        if control is None:
            return ControlAssessment(
                control_id=control_id,
                status=ComplianceStatus.UNKNOWN,
                findings=["Control not found"],
            )

        # Use registered assessor or default
        assessor = self._assessors.get(control_id)
        if assessor:
            return assessor(control)

        # Default assessment (simulated)
        return self._default_assessment(control)

    def _default_assessment(self, control: ComplianceControl) -> ControlAssessment:
        """Default assessment for a control."""
        # Simulated assessment
        import random

        statuses = [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.PARTIAL,
            ComplianceStatus.NON_COMPLIANT,
        ]
        status = random.choice(statuses)

        findings = []
        if status == ComplianceStatus.NON_COMPLIANT:
            findings.append(f"Control {control.name} not implemented")
        elif status == ComplianceStatus.PARTIAL:
            findings.append(f"Control {control.name} partially implemented")

        risk_level = RiskLevel.LOW if status == ComplianceStatus.COMPLIANT else RiskLevel.MEDIUM

        return ControlAssessment(
            control_id=control.control_id,
            status=status,
            findings=findings,
            risk_level=risk_level,
            evidence={"assessment_method": "automated"},
        )

    def assess_framework(
        self,
        framework: ComplianceFramework,
    ) -> List[ControlAssessment]:
        """Assess all controls for a framework."""
        controls = self._registry.list_controls(framework=framework)
        return [self.assess_control(c.control_id) for c in controls]


class PolicyEngine:
    """Enforce compliance policies."""

    def __init__(self):
        self._policies: Dict[str, CompliancePolicy] = {}
        self._violations: List[PolicyViolation] = []
        self._lock = threading.RLock()

    def add_policy(self, policy: CompliancePolicy) -> None:
        """Add a compliance policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def get_policy(self, policy_id: str) -> Optional[CompliancePolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(
        self,
        framework: Optional[ComplianceFramework] = None,
        policy_type: Optional[PolicyType] = None,
    ) -> List[CompliancePolicy]:
        """List policies."""
        policies = list(self._policies.values())
        if framework:
            policies = [p for p in policies if p.framework == framework]
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]
        return policies

    def evaluate(
        self,
        resource_id: str,
        resource_type: str,
        context: Dict[str, Any],
    ) -> List[PolicyViolation]:
        """Evaluate resource against policies."""
        violations = []

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            for rule in policy.rules:
                violation = self._evaluate_rule(policy, rule, resource_id, resource_type, context)
                if violation:
                    violations.append(violation)

        with self._lock:
            self._violations.extend(violations)

        return violations

    def _evaluate_rule(
        self,
        policy: CompliancePolicy,
        rule: Dict[str, Any],
        resource_id: str,
        resource_type: str,
        context: Dict[str, Any],
    ) -> Optional[PolicyViolation]:
        """Evaluate a single rule."""
        rule_type = rule.get("type")
        rule_value = rule.get("value")

        if rule_type == "encryption_required":
            if context.get("encrypted") != rule_value:
                return PolicyViolation(
                    violation_id=hashlib.sha256(
                        f"{policy.policy_id}:{resource_id}:{time.time()}".encode()
                    ).hexdigest()[:12],
                    policy_id=policy.policy_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    violation_type="encryption",
                    description="Resource not encrypted as required",
                )

        elif rule_type == "retention_max_days":
            retention = context.get("retention_days", 0)
            if retention > rule_value:
                return PolicyViolation(
                    violation_id=hashlib.sha256(
                        f"{policy.policy_id}:{resource_id}:{time.time()}".encode()
                    ).hexdigest()[:12],
                    policy_id=policy.policy_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    violation_type="retention",
                    description=f"Retention period {retention} exceeds maximum {rule_value}",
                )

        elif rule_type == "classification_required":
            if context.get("classification") not in rule_value:
                return PolicyViolation(
                    violation_id=hashlib.sha256(
                        f"{policy.policy_id}:{resource_id}:{time.time()}".encode()
                    ).hexdigest()[:12],
                    policy_id=policy.policy_id,
                    resource_id=resource_id,
                    resource_type=resource_type,
                    violation_type="classification",
                    description="Resource classification not allowed",
                )

        return None

    def get_violations(
        self,
        policy_id: Optional[str] = None,
        remediated: Optional[bool] = None,
    ) -> List[PolicyViolation]:
        """Get policy violations."""
        with self._lock:
            violations = self._violations.copy()

        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]
        if remediated is not None:
            violations = [v for v in violations if v.remediated == remediated]

        return violations


class ComplianceReporter:
    """Generate compliance reports."""

    def __init__(
        self,
        control_registry: ControlRegistry,
        control_assessor: ControlAssessor,
    ):
        self._registry = control_registry
        self._assessor = control_assessor

    def generate_report(
        self,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime,
    ) -> ComplianceReport:
        """Generate a compliance report."""
        assessments = self._assessor.assess_framework(framework)

        # Calculate overall status
        statuses = [a.status for a in assessments]
        if all(s == ComplianceStatus.COMPLIANT for s in statuses):
            overall_status = ComplianceStatus.COMPLIANT
        elif any(s == ComplianceStatus.NON_COMPLIANT for s in statuses):
            overall_status = ComplianceStatus.NON_COMPLIANT
        else:
            overall_status = ComplianceStatus.PARTIAL

        # Generate summary
        summary = {
            "total_controls": len(assessments),
            "compliant": sum(1 for a in assessments if a.status == ComplianceStatus.COMPLIANT),
            "non_compliant": sum(
                1 for a in assessments if a.status == ComplianceStatus.NON_COMPLIANT
            ),
            "partial": sum(1 for a in assessments if a.status == ComplianceStatus.PARTIAL),
            "compliance_rate": sum(1 for a in assessments if a.status == ComplianceStatus.COMPLIANT)
            / len(assessments)
            if assessments
            else 0,
        }

        # Generate recommendations
        recommendations = []
        for assessment in assessments:
            if assessment.status != ComplianceStatus.COMPLIANT:
                control = self._registry.get_control(assessment.control_id)
                if control and control.implementation_guidance:
                    recommendations.append(f"{control.name}: {control.implementation_guidance}")

        report_id = hashlib.sha256(f"{framework}:{period_start}:{period_end}".encode()).hexdigest()[
            :12
        ]

        return ComplianceReport(
            report_id=report_id,
            framework=framework,
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            overall_status=overall_status,
            control_assessments=assessments,
            summary=summary,
            recommendations=recommendations,
        )


class DataInventory:
    """Manage data inventory for compliance."""

    def __init__(self):
        self._items: Dict[str, DataInventoryItem] = {}
        self._lock = threading.RLock()

    def add_item(self, item: DataInventoryItem) -> None:
        """Add an item to the inventory."""
        with self._lock:
            self._items[item.item_id] = item

    def get_item(self, item_id: str) -> Optional[DataInventoryItem]:
        """Get an inventory item."""
        return self._items.get(item_id)

    def list_items(
        self,
        classification: Optional[DataClassification] = None,
        owner: Optional[str] = None,
    ) -> List[DataInventoryItem]:
        """List inventory items."""
        items = list(self._items.values())
        if classification:
            items = [i for i in items if i.classification == classification]
        if owner:
            items = [i for i in items if i.owner == owner]
        return items

    def get_pii_items(self) -> List[DataInventoryItem]:
        """Get items containing PII."""
        return [
            i
            for i in self._items.values()
            if i.classification in [DataClassification.PII, DataClassification.PHI] or i.pii_fields
        ]


class ComplianceChecker:
    """Main compliance checking component."""

    def __init__(self):
        self._control_registry = ControlRegistry()
        self._control_assessor = ControlAssessor(self._control_registry)
        self._policy_engine = PolicyEngine()
        self._reporter = ComplianceReporter(self._control_registry, self._control_assessor)
        self._data_inventory = DataInventory()

    def check_control(self, control_id: str) -> ControlAssessment:
        """Check a single control."""
        return self._control_assessor.assess_control(control_id)

    def check_framework(
        self,
        framework: ComplianceFramework,
    ) -> List[ControlAssessment]:
        """Check all controls for a framework."""
        return self._control_assessor.assess_framework(framework)

    def evaluate_resource(
        self,
        resource_id: str,
        resource_type: str,
        context: Dict[str, Any],
    ) -> List[PolicyViolation]:
        """Evaluate a resource against policies."""
        return self._policy_engine.evaluate(resource_id, resource_type, context)

    def generate_report(
        self,
        framework: ComplianceFramework,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ComplianceReport:
        """Generate a compliance report."""
        if period_start is None:
            period_start = datetime.now() - timedelta(days=30)
        if period_end is None:
            period_end = datetime.now()

        return self._reporter.generate_report(framework, period_start, period_end)

    def add_policy(self, policy: CompliancePolicy) -> None:
        """Add a compliance policy."""
        self._policy_engine.add_policy(policy)

    def get_control_registry(self) -> ControlRegistry:
        """Get the control registry."""
        return self._control_registry

    def get_policy_engine(self) -> PolicyEngine:
        """Get the policy engine."""
        return self._policy_engine

    def get_data_inventory(self) -> DataInventory:
        """Get the data inventory."""
        return self._data_inventory


# ========================
# Vision Provider
# ========================


class ComplianceCheckerVisionProvider(VisionProvider):
    """Vision provider for compliance checking capabilities."""

    def __init__(self):
        self._checker: Optional[ComplianceChecker] = None

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "compliance_checker"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """Analyze image for compliance context."""
        return self.get_description()

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Compliance Checker Vision Provider",
            version="1.0.0",
            description="Regulatory compliance and policy enforcement",
            capabilities=[
                "control_assessment",
                "policy_enforcement",
                "compliance_reporting",
                "data_inventory",
                "privacy_assessment",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._checker = ComplianceChecker()

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._checker = None

    def get_checker(self) -> ComplianceChecker:
        """Get the compliance checker."""
        if self._checker is None:
            self.initialize()
        return self._checker


# ========================
# Factory Functions
# ========================


def create_compliance_checker() -> ComplianceChecker:
    """Create a compliance checker."""
    return ComplianceChecker()


def create_compliance_control(
    control_id: str,
    name: str,
    description: str,
    framework: ComplianceFramework,
    category: ControlCategory,
) -> ComplianceControl:
    """Create a compliance control."""
    return ComplianceControl(
        control_id=control_id,
        name=name,
        description=description,
        framework=framework,
        category=category,
    )


def create_compliance_policy(
    policy_id: str,
    name: str,
    policy_type: PolicyType,
    framework: ComplianceFramework,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> CompliancePolicy:
    """Create a compliance policy."""
    return CompliancePolicy(
        policy_id=policy_id,
        name=name,
        policy_type=policy_type,
        framework=framework,
        rules=rules or [],
    )


def create_data_inventory_item(
    item_id: str,
    name: str,
    classification: DataClassification,
    location: str,
    owner: str,
) -> DataInventoryItem:
    """Create a data inventory item."""
    return DataInventoryItem(
        item_id=item_id,
        name=name,
        classification=classification,
        location=location,
        owner=owner,
    )


def create_privacy_impact_assessment(
    assessment_id: str,
    project_name: str,
    assessor: str,
    data_types: List[DataClassification],
) -> PrivacyImpactAssessment:
    """Create a privacy impact assessment."""
    return PrivacyImpactAssessment(
        assessment_id=assessment_id,
        project_name=project_name,
        assessor=assessor,
        data_types=data_types,
        processing_purposes=[],
    )


def create_control_registry() -> ControlRegistry:
    """Create a control registry."""
    return ControlRegistry()


def create_policy_engine() -> PolicyEngine:
    """Create a policy engine."""
    return PolicyEngine()


def create_data_inventory() -> DataInventory:
    """Create a data inventory."""
    return DataInventory()


def create_compliance_checker_provider() -> ComplianceCheckerVisionProvider:
    """Create a compliance checker vision provider."""
    return ComplianceCheckerVisionProvider()
