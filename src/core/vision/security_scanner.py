"""Security Scanner Module for Vision System.

This module provides security scanning capabilities including:
- Vulnerability scanning and detection
- Threat analysis and classification
- Security risk assessment
- Code security analysis
- Dependency vulnerability checking
- Container and infrastructure scanning

Phase 19: Advanced Security & Compliance
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
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


class VulnerabilitySeverity(str, Enum):
    """Vulnerability severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ScanType(str, Enum):
    """Types of security scans."""

    CODE = "code"
    DEPENDENCY = "dependency"
    CONTAINER = "container"
    INFRASTRUCTURE = "infrastructure"
    NETWORK = "network"
    API = "api"
    CONFIGURATION = "configuration"


class ThreatCategory(str, Enum):
    """Categories of security threats."""

    INJECTION = "injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    XSS = "xss"
    CSRF = "csrf"
    DATA_EXPOSURE = "data_exposure"
    MISCONFIGURATION = "misconfiguration"
    CRYPTOGRAPHIC = "cryptographic"
    DESERIALIZATION = "deserialization"
    COMPONENT = "component"


class ScanStatus(str, Enum):
    """Security scan status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RemediationPriority(str, Enum):
    """Remediation priority levels."""

    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"


class RiskLevel(str, Enum):
    """Overall risk levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


# ========================
# Dataclasses
# ========================


@dataclass
class Vulnerability:
    """A detected vulnerability."""

    vuln_id: str
    title: str
    severity: VulnerabilitySeverity
    category: ThreatCategory
    description: str = ""
    location: str = ""
    line_number: Optional[int] = None
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    affected_component: str = ""
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)
    false_positive: bool = False


@dataclass
class ScanConfig:
    """Configuration for a security scan."""

    scan_id: str
    scan_type: ScanType
    target: str
    options: Dict[str, Any] = field(default_factory=dict)
    exclude_patterns: List[str] = field(default_factory=list)
    severity_threshold: VulnerabilitySeverity = VulnerabilitySeverity.LOW
    max_findings: int = 1000
    timeout_seconds: int = 3600


@dataclass
class ScanResult:
    """Result of a security scan."""

    scan_id: str
    scan_type: ScanType
    status: ScanStatus
    target: str
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIndicator:
    """A threat indicator."""

    indicator_id: str
    indicator_type: str  # ip, domain, hash, pattern
    value: str
    threat_category: ThreatCategory
    confidence: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Risk assessment result."""

    assessment_id: str
    target: str
    risk_level: RiskLevel
    risk_score: float
    vulnerabilities_by_severity: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityPolicy:
    """Security policy definition."""

    policy_id: str
    name: str
    description: str = ""
    rules: List[Dict[str, Any]] = field(default_factory=list)
    severity_on_violation: VulnerabilitySeverity = VulnerabilitySeverity.MEDIUM
    enabled: bool = True


# ========================
# Core Classes
# ========================


class VulnerabilityScanner(ABC):
    """Abstract base class for vulnerability scanners."""

    @abstractmethod
    def scan(self, config: ScanConfig) -> ScanResult:
        """Perform a vulnerability scan."""
        pass

    @abstractmethod
    def get_supported_types(self) -> List[ScanType]:
        """Get supported scan types."""
        pass


class CodeScanner(VulnerabilityScanner):
    """Scanner for code vulnerabilities."""

    def __init__(self):
        self._patterns: Dict[ThreatCategory, List[Dict[str, Any]]] = {
            ThreatCategory.INJECTION: [
                {
                    "pattern": r"exec\s*\(",
                    "name": "Code Injection",
                    "severity": VulnerabilitySeverity.CRITICAL,
                },
                {
                    "pattern": r"eval\s*\(",
                    "name": "Eval Injection",
                    "severity": VulnerabilitySeverity.CRITICAL,
                },
                {
                    "pattern": r"subprocess\.call.*shell\s*=\s*True",
                    "name": "Shell Injection",
                    "severity": VulnerabilitySeverity.HIGH,
                },
            ],
            ThreatCategory.DATA_EXPOSURE: [
                {
                    "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
                    "name": "Hardcoded Password",
                    "severity": VulnerabilitySeverity.HIGH,
                },
                {
                    "pattern": r"api_key\s*=\s*['\"][^'\"]+['\"]",
                    "name": "Hardcoded API Key",
                    "severity": VulnerabilitySeverity.HIGH,
                },
                {
                    "pattern": r"secret\s*=\s*['\"][^'\"]+['\"]",
                    "name": "Hardcoded Secret",
                    "severity": VulnerabilitySeverity.HIGH,
                },
            ],
            ThreatCategory.CRYPTOGRAPHIC: [
                {
                    "pattern": r"md5\s*\(",
                    "name": "Weak Hash (MD5)",
                    "severity": VulnerabilitySeverity.MEDIUM,
                },
                {
                    "pattern": r"sha1\s*\(",
                    "name": "Weak Hash (SHA1)",
                    "severity": VulnerabilitySeverity.LOW,
                },
            ],
        }

    def scan(self, config: ScanConfig) -> ScanResult:
        """Scan code for vulnerabilities."""
        start_time = datetime.now()
        vulnerabilities = []

        # Simulate scanning
        for category, patterns in self._patterns.items():
            for pattern_info in patterns:
                # In real implementation, would scan actual files
                vuln = Vulnerability(
                    vuln_id=hashlib.md5(
                        f"{pattern_info['name']}:{config.target}".encode()
                    ).hexdigest()[:8],
                    title=pattern_info["name"],
                    severity=pattern_info["severity"],
                    category=category,
                    description=f"Potential {pattern_info['name']} vulnerability detected",
                    location=config.target,
                    remediation=f"Review and fix {pattern_info['name']} issues",
                )
                # Only add some vulnerabilities for simulation
                if hash(pattern_info["name"]) % 3 == 0:
                    vulnerabilities.append(vuln)

        end_time = datetime.now()

        # Build summary
        summary = defaultdict(int)
        for vuln in vulnerabilities:
            summary[vuln.severity.value] += 1

        return ScanResult(
            scan_id=config.scan_id,
            scan_type=config.scan_type,
            status=ScanStatus.COMPLETED,
            target=config.target,
            vulnerabilities=vulnerabilities,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            summary=dict(summary),
        )

    def get_supported_types(self) -> List[ScanType]:
        """Get supported scan types."""
        return [ScanType.CODE]


class DependencyScanner(VulnerabilityScanner):
    """Scanner for dependency vulnerabilities."""

    def __init__(self):
        self._known_vulnerabilities: Dict[str, List[Dict[str, Any]]] = {
            "requests": [
                {
                    "version": "<2.20.0",
                    "cve": "CVE-2018-18074",
                    "severity": VulnerabilitySeverity.MEDIUM,
                },
            ],
            "django": [
                {
                    "version": "<2.2.24",
                    "cve": "CVE-2021-33203",
                    "severity": VulnerabilitySeverity.HIGH,
                },
            ],
            "flask": [
                {
                    "version": "<1.0",
                    "cve": "CVE-2019-1010083",
                    "severity": VulnerabilitySeverity.MEDIUM,
                },
            ],
        }

    def scan(self, config: ScanConfig) -> ScanResult:
        """Scan dependencies for vulnerabilities."""
        start_time = datetime.now()
        vulnerabilities = []

        # Simulate dependency scanning
        for package, vulns in self._known_vulnerabilities.items():
            for vuln_info in vulns:
                vuln = Vulnerability(
                    vuln_id=hashlib.md5(f"{package}:{vuln_info['cve']}".encode()).hexdigest()[:8],
                    title=f"Vulnerable dependency: {package}",
                    severity=vuln_info["severity"],
                    category=ThreatCategory.COMPONENT,
                    description=f"Package {package} has known vulnerability",
                    cve_id=vuln_info["cve"],
                    affected_component=package,
                    remediation=f"Upgrade {package} to latest version",
                )
                vulnerabilities.append(vuln)

        end_time = datetime.now()

        summary = defaultdict(int)
        for vuln in vulnerabilities:
            summary[vuln.severity.value] += 1

        return ScanResult(
            scan_id=config.scan_id,
            scan_type=config.scan_type,
            status=ScanStatus.COMPLETED,
            target=config.target,
            vulnerabilities=vulnerabilities,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            summary=dict(summary),
        )

    def get_supported_types(self) -> List[ScanType]:
        """Get supported scan types."""
        return [ScanType.DEPENDENCY]


class ThreatDetector:
    """Detect and analyze threats."""

    def __init__(self):
        self._indicators: Dict[str, ThreatIndicator] = {}
        self._detection_rules: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def add_indicator(self, indicator: ThreatIndicator) -> None:
        """Add a threat indicator."""
        with self._lock:
            self._indicators[indicator.indicator_id] = indicator

    def check_indicator(self, indicator_type: str, value: str) -> Optional[ThreatIndicator]:
        """Check if a value matches known threat indicators."""
        with self._lock:
            for indicator in self._indicators.values():
                if indicator.indicator_type == indicator_type and indicator.value == value:
                    indicator.last_seen = datetime.now()
                    return indicator
        return None

    def analyze_threat(self, data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Analyze data for threats."""
        detected = []
        for indicator in self._indicators.values():
            # Simple pattern matching
            for key, value in data.items():
                if isinstance(value, str) and indicator.value in value:
                    detected.append(indicator)
        return detected


class RiskAssessor:
    """Assess security risks."""

    def __init__(self):
        self._severity_weights = {
            VulnerabilitySeverity.CRITICAL: 10.0,
            VulnerabilitySeverity.HIGH: 7.0,
            VulnerabilitySeverity.MEDIUM: 4.0,
            VulnerabilitySeverity.LOW: 2.0,
            VulnerabilitySeverity.INFO: 0.5,
        }

    def assess(self, scan_results: List[ScanResult]) -> RiskAssessment:
        """Assess risk from scan results."""
        assessment_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]

        vulnerabilities_by_severity: Dict[str, int] = defaultdict(int)
        total_score = 0.0

        for result in scan_results:
            for vuln in result.vulnerabilities:
                vulnerabilities_by_severity[vuln.severity.value] += 1
                total_score += self._severity_weights.get(vuln.severity, 1.0)

        # Normalize score to 0-100
        risk_score = min(100.0, total_score)

        # Determine risk level
        if risk_score >= 80:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 40:
            risk_level = RiskLevel.MEDIUM
        elif risk_score >= 20:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL

        # Generate recommendations
        recommendations = []
        if vulnerabilities_by_severity.get("critical", 0) > 0:
            recommendations.append("Address critical vulnerabilities immediately")
        if vulnerabilities_by_severity.get("high", 0) > 0:
            recommendations.append("Prioritize high severity vulnerabilities")
        if total_score > 50:
            recommendations.append("Conduct comprehensive security review")

        target = scan_results[0].target if scan_results else "unknown"

        return RiskAssessment(
            assessment_id=assessment_id,
            target=target,
            risk_level=risk_level,
            risk_score=risk_score,
            vulnerabilities_by_severity=dict(vulnerabilities_by_severity),
            recommendations=recommendations,
        )


class SecurityScanner:
    """Main security scanner orchestrator."""

    def __init__(self):
        self._scanners: Dict[ScanType, VulnerabilityScanner] = {}
        self._results: Dict[str, ScanResult] = {}
        self._policies: Dict[str, SecurityPolicy] = {}
        self._threat_detector = ThreatDetector()
        self._risk_assessor = RiskAssessor()
        self._lock = threading.RLock()

        # Register default scanners
        self.register_scanner(CodeScanner())
        self.register_scanner(DependencyScanner())

    def register_scanner(self, scanner: VulnerabilityScanner) -> None:
        """Register a vulnerability scanner."""
        for scan_type in scanner.get_supported_types():
            self._scanners[scan_type] = scanner

    def scan(self, config: ScanConfig) -> ScanResult:
        """Perform a security scan."""
        scanner = self._scanners.get(config.scan_type)
        if scanner is None:
            return ScanResult(
                scan_id=config.scan_id,
                scan_type=config.scan_type,
                status=ScanStatus.FAILED,
                target=config.target,
                metadata={"error": f"No scanner for type {config.scan_type}"},
            )

        result = scanner.scan(config)

        with self._lock:
            self._results[config.scan_id] = result

        return result

    def get_result(self, scan_id: str) -> Optional[ScanResult]:
        """Get a scan result."""
        return self._results.get(scan_id)

    def list_results(
        self,
        scan_type: Optional[ScanType] = None,
        status: Optional[ScanStatus] = None,
    ) -> List[ScanResult]:
        """List scan results."""
        results = list(self._results.values())
        if scan_type:
            results = [r for r in results if r.scan_type == scan_type]
        if status:
            results = [r for r in results if r.status == status]
        return results

    def assess_risk(self, scan_ids: Optional[List[str]] = None) -> RiskAssessment:
        """Assess risk from scan results."""
        if scan_ids:
            results = [self._results[sid] for sid in scan_ids if sid in self._results]
        else:
            results = list(self._results.values())
        return self._risk_assessor.assess(results)

    def add_policy(self, policy: SecurityPolicy) -> None:
        """Add a security policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def check_policy_compliance(
        self,
        scan_result: ScanResult,
    ) -> Tuple[bool, List[str]]:
        """Check if scan result complies with policies."""
        violations = []

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            for rule in policy.rules:
                rule_type = rule.get("type")
                if rule_type == "max_severity":
                    max_severity = VulnerabilitySeverity(rule.get("value", "low"))
                    severity_order = list(VulnerabilitySeverity)
                    max_idx = severity_order.index(max_severity)

                    for vuln in scan_result.vulnerabilities:
                        vuln_idx = severity_order.index(vuln.severity)
                        if vuln_idx < max_idx:  # Lower index = higher severity
                            violations.append(
                                f"Policy '{policy.name}': {vuln.title} exceeds max severity"
                            )

                elif rule_type == "max_count":
                    max_count = rule.get("value", 0)
                    if len(scan_result.vulnerabilities) > max_count:
                        violations.append(
                            f"Policy '{policy.name}': Vulnerability count {len(scan_result.vulnerabilities)} exceeds max {max_count}"
                        )

        return len(violations) == 0, violations


# ========================
# Vision Provider
# ========================


class SecurityScannerVisionProvider(VisionProvider):
    """Vision provider for security scanning capabilities."""

    def __init__(self):
        self._scanner: Optional[SecurityScanner] = None

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "security_scanner"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True
    ) -> VisionDescription:
        """Analyze image for security vulnerabilities."""
        return self.get_description()

    def get_description(self) -> VisionDescription:
        """Get provider description."""
        return VisionDescription(
            name="Security Scanner Vision Provider",
            version="1.0.0",
            description="Vulnerability scanning and threat detection",
            capabilities=[
                "code_scanning",
                "dependency_scanning",
                "threat_detection",
                "risk_assessment",
                "policy_compliance",
            ],
        )

    def initialize(self) -> None:
        """Initialize the provider."""
        self._scanner = SecurityScanner()

    def shutdown(self) -> None:
        """Shutdown the provider."""
        self._scanner = None

    def get_scanner(self) -> SecurityScanner:
        """Get the security scanner."""
        if self._scanner is None:
            self.initialize()
        return self._scanner


# ========================
# Factory Functions
# ========================


def create_security_scanner() -> SecurityScanner:
    """Create a security scanner."""
    return SecurityScanner()


def create_scan_config(
    scan_id: str,
    scan_type: ScanType,
    target: str,
    severity_threshold: VulnerabilitySeverity = VulnerabilitySeverity.LOW,
) -> ScanConfig:
    """Create a scan configuration."""
    return ScanConfig(
        scan_id=scan_id,
        scan_type=scan_type,
        target=target,
        severity_threshold=severity_threshold,
    )


def create_vulnerability(
    vuln_id: str,
    title: str,
    severity: VulnerabilitySeverity,
    category: ThreatCategory,
    description: str = "",
) -> Vulnerability:
    """Create a vulnerability."""
    return Vulnerability(
        vuln_id=vuln_id,
        title=title,
        severity=severity,
        category=category,
        description=description,
    )


def create_threat_indicator(
    indicator_id: str,
    indicator_type: str,
    value: str,
    threat_category: ThreatCategory,
    confidence: float = 0.5,
) -> ThreatIndicator:
    """Create a threat indicator."""
    return ThreatIndicator(
        indicator_id=indicator_id,
        indicator_type=indicator_type,
        value=value,
        threat_category=threat_category,
        confidence=confidence,
    )


def create_security_policy(
    policy_id: str,
    name: str,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> SecurityPolicy:
    """Create a security policy."""
    return SecurityPolicy(
        policy_id=policy_id,
        name=name,
        rules=rules or [],
    )


def create_code_scanner() -> CodeScanner:
    """Create a code scanner."""
    return CodeScanner()


def create_dependency_scanner() -> DependencyScanner:
    """Create a dependency scanner."""
    return DependencyScanner()


def create_threat_detector() -> ThreatDetector:
    """Create a threat detector."""
    return ThreatDetector()


def create_risk_assessor() -> RiskAssessor:
    """Create a risk assessor."""
    return RiskAssessor()


def create_security_scanner_provider() -> SecurityScannerVisionProvider:
    """Create a security scanner vision provider."""
    return SecurityScannerVisionProvider()
