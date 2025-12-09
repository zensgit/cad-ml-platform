"""
Security Governance Hub for Vision Provider.

This module provides a centralized security governance system including:
- Security policy management and enforcement
- Data classification and protection
- Encryption key management
- Secret management
- Security event correlation
- Threat intelligence integration
- Security posture assessment
- Governance reporting

Phase 22 Feature.
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


# ============================================================================
# Security Governance Enums
# ============================================================================


class DataClassification(Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class PolicyType(Enum):
    """Types of security policies."""

    ACCESS = "access"
    DATA = "data"
    ENCRYPTION = "encryption"
    RETENTION = "retention"
    NETWORK = "network"
    AUDIT = "audit"
    INCIDENT = "incident"


class PolicyStatus(Enum):
    """Policy status states."""

    DRAFT = "draft"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class PolicyAction(Enum):
    """Policy enforcement actions."""

    ALLOW = "allow"
    DENY = "deny"
    AUDIT = "audit"
    ENCRYPT = "encrypt"
    MASK = "mask"
    QUARANTINE = "quarantine"


class EncryptionAlgorithm(Enum):
    """Encryption algorithms."""

    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    RSA_OAEP = "rsa-oaep"


class KeyStatus(Enum):
    """Encryption key status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPROMISED = "compromised"
    EXPIRED = "expired"
    PENDING_ROTATION = "pending_rotation"


class SecretType(Enum):
    """Types of secrets."""

    API_KEY = "api_key"
    PASSWORD = "password"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    SSH_KEY = "ssh_key"
    DATABASE_CREDENTIAL = "database_credential"


class ThreatLevel(Enum):
    """Threat intelligence levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Security event types."""

    ACCESS_DENIED = "access_denied"
    ACCESS_GRANTED = "access_granted"
    POLICY_VIOLATION = "policy_violation"
    ANOMALY_DETECTED = "anomaly_detected"
    THREAT_DETECTED = "threat_detected"
    KEY_ROTATION = "key_rotation"
    SECRET_ACCESS = "secret_access"
    CONFIGURATION_CHANGE = "configuration_change"


class PostureStatus(Enum):
    """Security posture status."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


# ============================================================================
# Security Governance Data Classes
# ============================================================================


@dataclass
class SecurityPolicy:
    """Security policy definition."""

    policy_id: str
    name: str
    policy_type: PolicyType
    status: PolicyStatus = PolicyStatus.DRAFT
    description: str = ""
    rules: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    action: PolicyAction = PolicyAction.DENY
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEvaluation:
    """Result of policy evaluation."""

    policy_id: str
    decision: PolicyAction
    matched_rules: List[str] = field(default_factory=list)
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataClassificationRule:
    """Data classification rule."""

    rule_id: str
    name: str
    classification: DataClassification
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    auto_encrypt: bool = False
    auto_mask: bool = False


@dataclass
class EncryptionKey:
    """Encryption key metadata."""

    key_id: str
    algorithm: EncryptionAlgorithm
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    rotated_from: Optional[str] = None
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Secret:
    """Secret entry."""

    secret_id: str
    name: str
    secret_type: SecretType
    encrypted_value: str = ""
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatIndicator:
    """Threat intelligence indicator."""

    indicator_id: str
    indicator_type: str  # ip, domain, hash, url
    value: str
    threat_level: ThreatLevel
    source: str = ""
    description: str = ""
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class SecurityEvent:
    """Security event record."""

    event_id: str
    event_type: SecurityEventType
    severity: ThreatLevel
    source: str
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    actor: str = ""
    resource: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    correlated_events: List[str] = field(default_factory=list)


@dataclass
class SecurityPosture:
    """Security posture assessment."""

    assessment_id: str
    status: PostureStatus
    score: float  # 0-100
    timestamp: datetime = field(default_factory=datetime.now)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GovernanceConfig:
    """Security governance configuration."""

    key_rotation_days: int = 90
    secret_expiry_days: int = 365
    max_policy_priority: int = 1000
    enable_threat_intel: bool = True
    enable_auto_classification: bool = True
    audit_retention_days: int = 90


# ============================================================================
# Policy Engine
# ============================================================================


class PolicyEngine:
    """Manages and evaluates security policies."""

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        """Initialize policy engine."""
        self._config = config or GovernanceConfig()
        self._policies: Dict[str, SecurityPolicy] = {}
        self._lock = threading.RLock()

    def add_policy(self, policy: SecurityPolicy) -> None:
        """Add a security policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a security policy."""
        with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                return True
            return False

    def get_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get a policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def list_policies(
        self,
        policy_type: Optional[PolicyType] = None,
        status: Optional[PolicyStatus] = None,
    ) -> List[SecurityPolicy]:
        """List policies with optional filtering."""
        with self._lock:
            policies = list(self._policies.values())
            if policy_type:
                policies = [p for p in policies if p.policy_type == policy_type]
            if status:
                policies = [p for p in policies if p.status == status]
            return sorted(policies, key=lambda p: p.priority, reverse=True)

    def activate_policy(self, policy_id: str) -> bool:
        """Activate a policy."""
        with self._lock:
            policy = self._policies.get(policy_id)
            if policy:
                policy.status = PolicyStatus.ACTIVE
                policy.updated_at = datetime.now()
                return True
            return False

    def suspend_policy(self, policy_id: str) -> bool:
        """Suspend a policy."""
        with self._lock:
            policy = self._policies.get(policy_id)
            if policy:
                policy.status = PolicyStatus.SUSPENDED
                policy.updated_at = datetime.now()
                return True
            return False

    def evaluate(
        self,
        context: Dict[str, Any],
        policy_type: Optional[PolicyType] = None,
    ) -> PolicyEvaluation:
        """Evaluate policies against context."""
        with self._lock:
            active_policies = [
                p for p in self._policies.values()
                if p.status == PolicyStatus.ACTIVE
                and (policy_type is None or p.policy_type == policy_type)
            ]

        # Sort by priority (highest first)
        active_policies.sort(key=lambda p: p.priority, reverse=True)

        matched_rules = []
        decision = PolicyAction.ALLOW
        reason = "No matching policy"

        for policy in active_policies:
            if self._matches_conditions(policy, context):
                matched_rules.append(policy.policy_id)
                decision = policy.action
                reason = f"Matched policy: {policy.name}"
                break  # First matching policy wins

        return PolicyEvaluation(
            policy_id=matched_rules[0] if matched_rules else "",
            decision=decision,
            matched_rules=matched_rules,
            reason=reason,
            context=context,
        )

    def _matches_conditions(
        self,
        policy: SecurityPolicy,
        context: Dict[str, Any],
    ) -> bool:
        """Check if context matches policy conditions."""
        for key, expected in policy.conditions.items():
            actual = context.get(key)
            if actual != expected:
                return False
        return True


# ============================================================================
# Data Classification Manager
# ============================================================================


class DataClassificationManager:
    """Manages data classification and protection."""

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        """Initialize data classification manager."""
        self._config = config or GovernanceConfig()
        self._rules: Dict[str, DataClassificationRule] = {}
        self._classifications: Dict[str, DataClassification] = {}
        self._lock = threading.RLock()

    def add_rule(self, rule: DataClassificationRule) -> None:
        """Add a classification rule."""
        with self._lock:
            self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a classification rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
            return False

    def classify_data(self, data: str, resource_id: str = "") -> DataClassification:
        """Classify data based on rules."""
        with self._lock:
            rules = list(self._rules.values())

        # Sort by classification severity (most restrictive first)
        classification_order = [
            DataClassification.TOP_SECRET,
            DataClassification.RESTRICTED,
            DataClassification.CONFIDENTIAL,
            DataClassification.INTERNAL,
            DataClassification.PUBLIC,
        ]

        rules.sort(
            key=lambda r: classification_order.index(r.classification)
        )

        for rule in rules:
            if self._matches_rule(rule, data):
                if resource_id:
                    with self._lock:
                        self._classifications[resource_id] = rule.classification
                return rule.classification

        return DataClassification.PUBLIC

    def get_classification(self, resource_id: str) -> Optional[DataClassification]:
        """Get classification for a resource."""
        with self._lock:
            return self._classifications.get(resource_id)

    def set_classification(
        self,
        resource_id: str,
        classification: DataClassification,
    ) -> None:
        """Manually set classification for a resource."""
        with self._lock:
            self._classifications[resource_id] = classification

    def list_rules(self) -> List[DataClassificationRule]:
        """List all classification rules."""
        with self._lock:
            return list(self._rules.values())

    def _matches_rule(self, rule: DataClassificationRule, data: str) -> bool:
        """Check if data matches a classification rule."""
        import re

        # Check patterns
        for pattern in rule.patterns:
            if re.search(pattern, data, re.IGNORECASE):
                return True

        # Check keywords
        data_lower = data.lower()
        for keyword in rule.keywords:
            if keyword.lower() in data_lower:
                return True

        return False


# ============================================================================
# Key Manager
# ============================================================================


class KeyManager:
    """Manages encryption keys."""

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        """Initialize key manager."""
        self._config = config or GovernanceConfig()
        self._keys: Dict[str, EncryptionKey] = {}
        self._key_data: Dict[str, bytes] = {}  # Simulated key storage
        self._lock = threading.RLock()

    def generate_key(
        self,
        algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
        expires_in_days: Optional[int] = None,
    ) -> EncryptionKey:
        """Generate a new encryption key."""
        key_id = str(uuid.uuid4())
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        elif self._config.key_rotation_days:
            expires_at = datetime.now() + timedelta(days=self._config.key_rotation_days)

        key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            expires_at=expires_at,
        )

        # Generate actual key data (in production, use proper crypto)
        key_bytes = secrets.token_bytes(32)  # 256-bit key

        with self._lock:
            self._keys[key_id] = key
            self._key_data[key_id] = key_bytes

        return key

    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get key metadata."""
        with self._lock:
            return self._keys.get(key_id)

    def get_active_key(
        self,
        algorithm: Optional[EncryptionAlgorithm] = None,
    ) -> Optional[EncryptionKey]:
        """Get the active key for encryption."""
        with self._lock:
            for key in self._keys.values():
                if key.status == KeyStatus.ACTIVE:
                    if algorithm is None or key.algorithm == algorithm:
                        return key
            return None

    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Rotate an encryption key."""
        with self._lock:
            old_key = self._keys.get(key_id)
            if not old_key:
                return None

            # Mark old key for rotation
            old_key.status = KeyStatus.PENDING_ROTATION

        # Generate new key
        new_key = self.generate_key(old_key.algorithm)
        new_key.rotated_from = key_id
        new_key.version = old_key.version + 1

        with self._lock:
            old_key.status = KeyStatus.INACTIVE

        return new_key

    def revoke_key(self, key_id: str, reason: str = "") -> bool:
        """Revoke a key (mark as compromised)."""
        with self._lock:
            key = self._keys.get(key_id)
            if key:
                key.status = KeyStatus.COMPROMISED
                key.metadata["revocation_reason"] = reason
                key.metadata["revoked_at"] = datetime.now().isoformat()
                return True
            return False

    def list_keys(
        self,
        status: Optional[KeyStatus] = None,
    ) -> List[EncryptionKey]:
        """List keys with optional status filter."""
        with self._lock:
            keys = list(self._keys.values())
            if status:
                keys = [k for k in keys if k.status == status]
            return keys

    def get_keys_for_rotation(self) -> List[EncryptionKey]:
        """Get keys that need rotation."""
        now = datetime.now()
        with self._lock:
            return [
                k for k in self._keys.values()
                if k.status == KeyStatus.ACTIVE
                and k.expires_at
                and k.expires_at <= now + timedelta(days=7)
            ]


# ============================================================================
# Secret Manager
# ============================================================================


class SecretManager:
    """Manages secrets and credentials."""

    def __init__(
        self,
        key_manager: KeyManager,
        config: Optional[GovernanceConfig] = None,
    ) -> None:
        """Initialize secret manager."""
        self._config = config or GovernanceConfig()
        self._key_manager = key_manager
        self._secrets: Dict[str, Secret] = {}
        self._lock = threading.RLock()

    def store_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Secret:
        """Store a secret."""
        secret_id = str(uuid.uuid4())
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        elif self._config.secret_expiry_days:
            expires_at = datetime.now() + timedelta(days=self._config.secret_expiry_days)

        # Encrypt the value (simplified - in production use actual encryption)
        encrypted_value = self._encrypt(value)

        secret = Secret(
            secret_id=secret_id,
            name=name,
            secret_type=secret_type,
            encrypted_value=encrypted_value,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        with self._lock:
            self._secrets[secret_id] = secret

        return secret

    def get_secret(self, secret_id: str) -> Optional[str]:
        """Get and decrypt a secret value."""
        with self._lock:
            secret = self._secrets.get(secret_id)
            if not secret:
                return None

            # Check expiration
            if secret.expires_at and secret.expires_at < datetime.now():
                return None

            # Update access info
            secret.last_accessed = datetime.now()
            secret.access_count += 1

        # Decrypt and return
        return self._decrypt(secret.encrypted_value)

    def get_secret_by_name(self, name: str) -> Optional[str]:
        """Get secret by name."""
        with self._lock:
            for secret in self._secrets.values():
                if secret.name == name:
                    return self.get_secret(secret.secret_id)
        return None

    def rotate_secret(
        self,
        secret_id: str,
        new_value: str,
    ) -> Optional[Secret]:
        """Rotate a secret value."""
        with self._lock:
            old_secret = self._secrets.get(secret_id)
            if not old_secret:
                return None

            # Create new version
            encrypted_value = self._encrypt(new_value)
            old_secret.encrypted_value = encrypted_value
            old_secret.version += 1
            old_secret.created_at = datetime.now()

            return old_secret

    def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret."""
        with self._lock:
            if secret_id in self._secrets:
                del self._secrets[secret_id]
                return True
            return False

    def list_secrets(
        self,
        secret_type: Optional[SecretType] = None,
    ) -> List[Secret]:
        """List secrets (metadata only, not values)."""
        with self._lock:
            secrets = list(self._secrets.values())
            if secret_type:
                secrets = [s for s in secrets if s.secret_type == secret_type]
            return secrets

    def get_expiring_secrets(self, days: int = 30) -> List[Secret]:
        """Get secrets expiring within specified days."""
        cutoff = datetime.now() + timedelta(days=days)
        with self._lock:
            return [
                s for s in self._secrets.values()
                if s.expires_at and s.expires_at <= cutoff
            ]

    def _encrypt(self, value: str) -> str:
        """Encrypt a value (simplified implementation)."""
        # In production, use actual encryption with KeyManager
        encoded = base64.b64encode(value.encode()).decode()
        return f"enc:{encoded}"

    def _decrypt(self, encrypted: str) -> str:
        """Decrypt a value (simplified implementation)."""
        if encrypted.startswith("enc:"):
            encoded = encrypted[4:]
            return base64.b64decode(encoded).decode()
        return encrypted


# ============================================================================
# Threat Intelligence Manager
# ============================================================================


class ThreatIntelManager:
    """Manages threat intelligence."""

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        """Initialize threat intelligence manager."""
        self._config = config or GovernanceConfig()
        self._indicators: Dict[str, ThreatIndicator] = {}
        self._lock = threading.RLock()

    def add_indicator(self, indicator: ThreatIndicator) -> None:
        """Add a threat indicator."""
        with self._lock:
            self._indicators[indicator.indicator_id] = indicator

    def remove_indicator(self, indicator_id: str) -> bool:
        """Remove a threat indicator."""
        with self._lock:
            if indicator_id in self._indicators:
                del self._indicators[indicator_id]
                return True
            return False

    def check_threat(
        self,
        indicator_type: str,
        value: str,
    ) -> Optional[ThreatIndicator]:
        """Check if a value matches any threat indicator."""
        with self._lock:
            for indicator in self._indicators.values():
                if (indicator.indicator_type == indicator_type
                        and indicator.value == value):
                    indicator.last_seen = datetime.now()
                    return indicator
            return None

    def get_indicators_by_level(
        self,
        threat_level: ThreatLevel,
    ) -> List[ThreatIndicator]:
        """Get indicators by threat level."""
        with self._lock:
            return [
                i for i in self._indicators.values()
                if i.threat_level == threat_level
            ]

    def list_indicators(
        self,
        indicator_type: Optional[str] = None,
    ) -> List[ThreatIndicator]:
        """List all threat indicators."""
        with self._lock:
            indicators = list(self._indicators.values())
            if indicator_type:
                indicators = [
                    i for i in indicators
                    if i.indicator_type == indicator_type
                ]
            return indicators


# ============================================================================
# Security Event Correlator
# ============================================================================


class SecurityEventCorrelator:
    """Correlates security events."""

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        """Initialize security event correlator."""
        self._config = config or GovernanceConfig()
        self._events: List[SecurityEvent] = []
        self._correlation_rules: List[Callable[[List[SecurityEvent]], List[str]]] = []
        self._lock = threading.RLock()

    def record_event(self, event: SecurityEvent) -> None:
        """Record a security event."""
        with self._lock:
            self._events.append(event)

            # Run correlation
            event.correlated_events = self._correlate(event)

    def add_correlation_rule(
        self,
        rule: Callable[[List[SecurityEvent]], List[str]],
    ) -> None:
        """Add a correlation rule."""
        with self._lock:
            self._correlation_rules.append(rule)

    def get_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[ThreatLevel] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SecurityEvent]:
        """Get security events with filtering."""
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]
        if since:
            events = [e for e in events if e.timestamp >= since]

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_correlated_events(self, event_id: str) -> List[SecurityEvent]:
        """Get events correlated to a specific event."""
        with self._lock:
            for event in self._events:
                if event.event_id == event_id:
                    correlated_ids = event.correlated_events
                    return [
                        e for e in self._events
                        if e.event_id in correlated_ids
                    ]
            return []

    def _correlate(self, event: SecurityEvent) -> List[str]:
        """Find correlated events."""
        correlated = []

        # Time-based correlation (events within 5 minutes)
        window = timedelta(minutes=5)
        with self._lock:
            for existing in self._events:
                if existing.event_id == event.event_id:
                    continue
                time_diff = abs((event.timestamp - existing.timestamp).total_seconds())
                if time_diff <= window.total_seconds():
                    # Same source or same resource
                    if (existing.source == event.source
                            or existing.resource == event.resource):
                        correlated.append(existing.event_id)

        return correlated


# ============================================================================
# Security Posture Assessor
# ============================================================================


class SecurityPostureAssessor:
    """Assesses overall security posture."""

    def __init__(
        self,
        policy_engine: PolicyEngine,
        key_manager: KeyManager,
        secret_manager: SecretManager,
        config: Optional[GovernanceConfig] = None,
    ) -> None:
        """Initialize security posture assessor."""
        self._config = config or GovernanceConfig()
        self._policy_engine = policy_engine
        self._key_manager = key_manager
        self._secret_manager = secret_manager
        self._assessments: List[SecurityPosture] = []
        self._lock = threading.RLock()

    def assess(self) -> SecurityPosture:
        """Perform security posture assessment."""
        findings = []
        recommendations = []
        metrics = {}

        # Check policies
        active_policies = self._policy_engine.list_policies(status=PolicyStatus.ACTIVE)
        total_policies = len(self._policy_engine.list_policies())
        metrics["active_policies"] = len(active_policies)
        metrics["total_policies"] = total_policies

        if total_policies == 0:
            findings.append({
                "category": "policy",
                "severity": "high",
                "message": "No security policies defined",
            })
            recommendations.append("Define and activate security policies")

        # Check keys
        keys_for_rotation = self._key_manager.get_keys_for_rotation()
        active_keys = self._key_manager.list_keys(status=KeyStatus.ACTIVE)
        metrics["active_keys"] = len(active_keys)
        metrics["keys_pending_rotation"] = len(keys_for_rotation)

        if keys_for_rotation:
            findings.append({
                "category": "encryption",
                "severity": "medium",
                "message": f"{len(keys_for_rotation)} keys pending rotation",
            })
            recommendations.append("Rotate encryption keys approaching expiration")

        if not active_keys:
            findings.append({
                "category": "encryption",
                "severity": "high",
                "message": "No active encryption keys",
            })
            recommendations.append("Generate at least one active encryption key")

        # Check secrets
        expiring_secrets = self._secret_manager.get_expiring_secrets(30)
        total_secrets = len(self._secret_manager.list_secrets())
        metrics["total_secrets"] = total_secrets
        metrics["expiring_secrets"] = len(expiring_secrets)

        if expiring_secrets:
            findings.append({
                "category": "secrets",
                "severity": "medium",
                "message": f"{len(expiring_secrets)} secrets expiring within 30 days",
            })
            recommendations.append("Rotate expiring secrets")

        # Calculate score
        score = self._calculate_score(findings, metrics)
        status = self._determine_status(score)

        assessment = SecurityPosture(
            assessment_id=str(uuid.uuid4()),
            status=status,
            score=score,
            findings=findings,
            recommendations=recommendations,
            metrics=metrics,
        )

        with self._lock:
            self._assessments.append(assessment)

        return assessment

    def get_latest_assessment(self) -> Optional[SecurityPosture]:
        """Get the most recent assessment."""
        with self._lock:
            return self._assessments[-1] if self._assessments else None

    def get_assessment_history(self, limit: int = 10) -> List[SecurityPosture]:
        """Get assessment history."""
        with self._lock:
            return self._assessments[-limit:]

    def _calculate_score(
        self,
        findings: List[Dict[str, Any]],
        metrics: Dict[str, float],
    ) -> float:
        """Calculate security score (0-100)."""
        score = 100.0

        # Deduct for findings
        for finding in findings:
            severity = finding.get("severity", "low")
            if severity == "critical":
                score -= 25
            elif severity == "high":
                score -= 15
            elif severity == "medium":
                score -= 10
            else:
                score -= 5

        # Bonus for good practices
        if metrics.get("active_policies", 0) > 0:
            score += 5
        if metrics.get("active_keys", 0) > 0:
            score += 5

        return max(0, min(100, score))

    def _determine_status(self, score: float) -> PostureStatus:
        """Determine posture status from score."""
        if score >= 90:
            return PostureStatus.EXCELLENT
        elif score >= 75:
            return PostureStatus.GOOD
        elif score >= 60:
            return PostureStatus.FAIR
        elif score >= 40:
            return PostureStatus.POOR
        else:
            return PostureStatus.CRITICAL


# ============================================================================
# Security Governance Hub
# ============================================================================


class SecurityGovernanceHub:
    """
    Central security governance orchestration hub.

    Integrates policy management, data classification, encryption,
    secrets, threat intelligence, and security posture assessment.
    """

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        """Initialize security governance hub."""
        self._config = config or GovernanceConfig()
        self._policy_engine = PolicyEngine(self._config)
        self._classification = DataClassificationManager(self._config)
        self._key_manager = KeyManager(self._config)
        self._secret_manager = SecretManager(self._key_manager, self._config)
        self._threat_intel = ThreatIntelManager(self._config)
        self._event_correlator = SecurityEventCorrelator(self._config)
        self._posture_assessor = SecurityPostureAssessor(
            self._policy_engine,
            self._key_manager,
            self._secret_manager,
            self._config,
        )
        self._lock = threading.RLock()

    @property
    def policies(self) -> PolicyEngine:
        """Get policy engine."""
        return self._policy_engine

    @property
    def classification(self) -> DataClassificationManager:
        """Get data classification manager."""
        return self._classification

    @property
    def keys(self) -> KeyManager:
        """Get key manager."""
        return self._key_manager

    @property
    def secrets(self) -> SecretManager:
        """Get secret manager."""
        return self._secret_manager

    @property
    def threat_intel(self) -> ThreatIntelManager:
        """Get threat intelligence manager."""
        return self._threat_intel

    @property
    def events(self) -> SecurityEventCorrelator:
        """Get security event correlator."""
        return self._event_correlator

    @property
    def posture(self) -> SecurityPostureAssessor:
        """Get security posture assessor."""
        return self._posture_assessor

    def record_security_event(
        self,
        event_type: SecurityEventType,
        severity: ThreatLevel,
        source: str,
        message: str = "",
        **kwargs: Any,
    ) -> SecurityEvent:
        """Record a security event."""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            source=source,
            message=message,
            **kwargs,
        )
        self._event_correlator.record_event(event)
        return event

    def evaluate_access(
        self,
        actor: str,
        resource: str,
        action: str,
    ) -> PolicyEvaluation:
        """Evaluate access request against policies."""
        context = {
            "actor": actor,
            "resource": resource,
            "action": action,
        }

        evaluation = self._policy_engine.evaluate(
            context,
            policy_type=PolicyType.ACCESS,
        )

        # Record event
        event_type = (
            SecurityEventType.ACCESS_GRANTED
            if evaluation.decision == PolicyAction.ALLOW
            else SecurityEventType.ACCESS_DENIED
        )
        self.record_security_event(
            event_type=event_type,
            severity=ThreatLevel.LOW,
            source="policy_engine",
            message=evaluation.reason,
            actor=actor,
            resource=resource,
        )

        return evaluation

    def get_governance_summary(self) -> Dict[str, Any]:
        """Get summary of security governance state."""
        assessment = self._posture_assessor.assess()

        return {
            "posture": {
                "status": assessment.status.value,
                "score": assessment.score,
            },
            "policies": {
                "total": len(self._policy_engine.list_policies()),
                "active": len(self._policy_engine.list_policies(
                    status=PolicyStatus.ACTIVE
                )),
            },
            "encryption": {
                "active_keys": len(self._key_manager.list_keys(
                    status=KeyStatus.ACTIVE
                )),
                "pending_rotation": len(self._key_manager.get_keys_for_rotation()),
            },
            "secrets": {
                "total": len(self._secret_manager.list_secrets()),
                "expiring_soon": len(self._secret_manager.get_expiring_secrets(30)),
            },
            "threat_intel": {
                "total_indicators": len(self._threat_intel.list_indicators()),
                "critical": len(self._threat_intel.get_indicators_by_level(
                    ThreatLevel.CRITICAL
                )),
            },
            "events": {
                "recent": len(self._event_correlator.get_events(limit=100)),
            },
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# Vision Provider Integration
# ============================================================================


class SecureVisionProvider(VisionProvider):
    """Vision provider with security governance integration."""

    def __init__(
        self,
        base_provider: VisionProvider,
        hub: SecurityGovernanceHub,
    ) -> None:
        """Initialize secure vision provider."""
        self._base_provider = base_provider
        self._hub = hub

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"secure_{self._base_provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with security governance."""
        # Record access attempt
        evaluation = self._hub.evaluate_access(
            actor="vision_client",
            resource=f"image:{len(image_data)}",
            action="analyze",
        )

        if evaluation.decision == PolicyAction.DENY:
            raise PermissionError(f"Access denied: {evaluation.reason}")

        # Classify image data
        classification = self._hub.classification.classify_data(
            str(image_data[:100]),  # Sample for classification
            resource_id=f"image_{id(image_data)}",
        )

        # Record event
        self._hub.record_security_event(
            event_type=SecurityEventType.ACCESS_GRANTED,
            severity=ThreatLevel.NONE,
            source=self.provider_name,
            message=f"Image analysis with classification: {classification.value}",
            resource=f"image:{len(image_data)}",
        )

        # Perform analysis
        result = await self._base_provider.analyze_image(
            image_data,
            include_description,
        )

        return result


# ============================================================================
# Factory Functions
# ============================================================================


def create_governance_config(
    key_rotation_days: int = 90,
    secret_expiry_days: int = 365,
    **kwargs: Any,
) -> GovernanceConfig:
    """Create a governance configuration."""
    return GovernanceConfig(
        key_rotation_days=key_rotation_days,
        secret_expiry_days=secret_expiry_days,
        **kwargs,
    )


def create_security_governance_hub(
    key_rotation_days: int = 90,
    secret_expiry_days: int = 365,
    **kwargs: Any,
) -> SecurityGovernanceHub:
    """Create a security governance hub."""
    config = GovernanceConfig(
        key_rotation_days=key_rotation_days,
        secret_expiry_days=secret_expiry_days,
        **kwargs,
    )
    return SecurityGovernanceHub(config)


def create_security_policy(
    name: str,
    policy_type: PolicyType,
    action: PolicyAction = PolicyAction.DENY,
    conditions: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    **kwargs: Any,
) -> SecurityPolicy:
    """Create a security policy."""
    return SecurityPolicy(
        policy_id=str(uuid.uuid4()),
        name=name,
        policy_type=policy_type,
        action=action,
        conditions=conditions or {},
        priority=priority,
        **kwargs,
    )


def create_classification_rule(
    name: str,
    classification: DataClassification,
    patterns: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    **kwargs: Any,
) -> DataClassificationRule:
    """Create a data classification rule."""
    return DataClassificationRule(
        rule_id=str(uuid.uuid4()),
        name=name,
        classification=classification,
        patterns=patterns or [],
        keywords=keywords or [],
        **kwargs,
    )


def create_threat_indicator(
    indicator_type: str,
    value: str,
    threat_level: ThreatLevel,
    source: str = "",
    **kwargs: Any,
) -> ThreatIndicator:
    """Create a threat indicator."""
    return ThreatIndicator(
        indicator_id=str(uuid.uuid4()),
        indicator_type=indicator_type,
        value=value,
        threat_level=threat_level,
        source=source,
        **kwargs,
    )


def create_secure_provider(
    base_provider: VisionProvider,
    hub: Optional[SecurityGovernanceHub] = None,
) -> SecureVisionProvider:
    """Create a secure vision provider."""
    if hub is None:
        hub = create_security_governance_hub()
    return SecureVisionProvider(base_provider, hub)
