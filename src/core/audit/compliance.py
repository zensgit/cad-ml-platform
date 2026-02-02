"""Compliance Tracking and Data Access Logging.

Provides compliance framework support for GDPR, SOC2, HIPAA.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"  # EU General Data Protection Regulation
    SOC2 = "soc2"  # Service Organization Control 2
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO_27001 = "iso_27001"  # Information Security Management
    CCPA = "ccpa"  # California Consumer Privacy Act


class DataCategory(str, Enum):
    """Categories of data for compliance."""

    PERSONAL = "personal"  # PII
    SENSITIVE = "sensitive"  # Special category data
    FINANCIAL = "financial"  # Financial information
    HEALTH = "health"  # Protected health information
    AUTHENTICATION = "authentication"  # Credentials, tokens
    TECHNICAL = "technical"  # System/technical data
    PUBLIC = "public"  # Non-sensitive public data


class AccessPurpose(str, Enum):
    """Purpose of data access."""

    USER_REQUEST = "user_request"
    SERVICE_OPERATION = "service_operation"
    ANALYTICS = "analytics"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    DEBUGGING = "debugging"
    BACKUP = "backup"
    DELETION = "deletion"


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""

    name: str
    category: DataCategory
    retention_days: int
    frameworks: Set[ComplianceFramework] = field(default_factory=set)
    auto_delete: bool = True
    archive_before_delete: bool = True
    description: str = ""

    def is_expired(self, created_at: datetime) -> bool:
        """Check if data has exceeded retention period."""
        return datetime.utcnow() > created_at + timedelta(days=self.retention_days)


# Default retention policies
DEFAULT_RETENTION_POLICIES: Dict[DataCategory, RetentionPolicy] = {
    DataCategory.PERSONAL: RetentionPolicy(
        name="personal_data_retention",
        category=DataCategory.PERSONAL,
        retention_days=365 * 3,  # 3 years
        frameworks={ComplianceFramework.GDPR, ComplianceFramework.CCPA},
        description="Personal data retained for 3 years",
    ),
    DataCategory.SENSITIVE: RetentionPolicy(
        name="sensitive_data_retention",
        category=DataCategory.SENSITIVE,
        retention_days=365,  # 1 year
        frameworks={ComplianceFramework.GDPR, ComplianceFramework.HIPAA},
        description="Sensitive data retained for 1 year",
    ),
    DataCategory.FINANCIAL: RetentionPolicy(
        name="financial_data_retention",
        category=DataCategory.FINANCIAL,
        retention_days=365 * 7,  # 7 years
        frameworks={ComplianceFramework.SOC2, ComplianceFramework.PCI_DSS},
        description="Financial data retained for 7 years",
    ),
    DataCategory.HEALTH: RetentionPolicy(
        name="health_data_retention",
        category=DataCategory.HEALTH,
        retention_days=365 * 6,  # 6 years
        frameworks={ComplianceFramework.HIPAA},
        description="Health data retained for 6 years",
    ),
    DataCategory.AUTHENTICATION: RetentionPolicy(
        name="auth_data_retention",
        category=DataCategory.AUTHENTICATION,
        retention_days=90,  # 90 days
        frameworks={ComplianceFramework.SOC2, ComplianceFramework.ISO_27001},
        description="Authentication logs retained for 90 days",
    ),
    DataCategory.TECHNICAL: RetentionPolicy(
        name="technical_data_retention",
        category=DataCategory.TECHNICAL,
        retention_days=365,  # 1 year
        frameworks={ComplianceFramework.SOC2},
        description="Technical logs retained for 1 year",
    ),
}


@dataclass
class DataAccessLog:
    """Log entry for data access."""

    access_id: str
    timestamp: datetime
    user_id: str
    tenant_id: Optional[str]
    data_category: DataCategory
    resource_type: str
    resource_id: str
    access_type: str  # read, write, delete
    purpose: AccessPurpose
    fields_accessed: List[str] = field(default_factory=list)
    legal_basis: Optional[str] = None  # For GDPR
    consent_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "access_id": self.access_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "data_category": self.data_category.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "access_type": self.access_type,
            "purpose": self.purpose.value,
            "fields_accessed": self.fields_accessed,
            "legal_basis": self.legal_basis,
            "consent_id": self.consent_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
        }


@dataclass
class ConsentRecord:
    """Record of user consent."""

    consent_id: str
    user_id: str
    tenant_id: Optional[str]
    purpose: str
    scope: Set[str]  # What data/operations consented to
    granted_at: datetime
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    version: str = "1.0"
    source: str = "web"  # web, api, mobile
    ip_address: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.revoked_at:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


@dataclass
class DataSubjectRequest:
    """Data subject request (GDPR Article 15-22)."""

    request_id: str
    user_id: str
    tenant_id: Optional[str]
    request_type: str  # access, rectification, erasure, portability, restriction
    submitted_at: datetime
    deadline: datetime  # Usually 30 days for GDPR
    status: str = "pending"  # pending, processing, completed, rejected
    completed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    notes: str = ""

    @property
    def is_overdue(self) -> bool:
        """Check if request is past deadline."""
        if self.status == "completed":
            return False
        return datetime.utcnow() > self.deadline


class ComplianceTracker:
    """Tracks compliance requirements and data access."""

    def __init__(
        self,
        enabled_frameworks: Optional[Set[ComplianceFramework]] = None,
        retention_policies: Optional[Dict[DataCategory, RetentionPolicy]] = None,
    ):
        self.enabled_frameworks = enabled_frameworks or {
            ComplianceFramework.GDPR,
            ComplianceFramework.SOC2,
        }
        self.retention_policies = retention_policies or DEFAULT_RETENTION_POLICIES

        self._access_logs: List[DataAccessLog] = []
        self._consent_records: Dict[str, ConsentRecord] = {}  # consent_id -> record
        self._dsr_requests: Dict[str, DataSubjectRequest] = {}  # request_id -> request
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def log_data_access(
        self,
        user_id: str,
        data_category: DataCategory,
        resource_type: str,
        resource_id: str,
        access_type: str,
        purpose: AccessPurpose,
        tenant_id: Optional[str] = None,
        fields_accessed: Optional[List[str]] = None,
        legal_basis: Optional[str] = None,
        consent_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataAccessLog:
        """Log a data access event.

        Args:
            user_id: User performing access
            data_category: Category of data accessed
            resource_type: Type of resource
            resource_id: ID of resource
            access_type: Type of access (read/write/delete)
            purpose: Purpose of access
            tenant_id: Tenant identifier
            fields_accessed: Specific fields accessed
            legal_basis: Legal basis for processing (GDPR)
            consent_id: Reference to consent record
            ip_address: Client IP
            user_agent: Client user agent
            metadata: Additional metadata

        Returns:
            Created DataAccessLog
        """
        import uuid

        log_entry = DataAccessLog(
            access_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=user_id,
            tenant_id=tenant_id,
            data_category=data_category,
            resource_type=resource_type,
            resource_id=resource_id,
            access_type=access_type,
            purpose=purpose,
            fields_accessed=fields_accessed or [],
            legal_basis=legal_basis,
            consent_id=consent_id,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )

        async with self._get_lock():
            self._access_logs.append(log_entry)

            # Trim old logs (keep last 100k)
            if len(self._access_logs) > 100000:
                self._access_logs = self._access_logs[-100000:]

        logger.debug(f"Logged data access: {log_entry.access_id}")
        return log_entry

    async def record_consent(
        self,
        user_id: str,
        purpose: str,
        scope: Set[str],
        tenant_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        version: str = "1.0",
        source: str = "web",
        ip_address: Optional[str] = None,
    ) -> ConsentRecord:
        """Record user consent.

        Args:
            user_id: User giving consent
            purpose: Purpose of consent
            scope: Scope of consent
            tenant_id: Tenant identifier
            expires_in_days: Days until consent expires
            version: Consent version
            source: Source of consent
            ip_address: Client IP

        Returns:
            Created ConsentRecord
        """
        import uuid

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        consent = ConsentRecord(
            consent_id=str(uuid.uuid4()),
            user_id=user_id,
            tenant_id=tenant_id,
            purpose=purpose,
            scope=scope,
            granted_at=datetime.utcnow(),
            expires_at=expires_at,
            version=version,
            source=source,
            ip_address=ip_address,
        )

        async with self._get_lock():
            self._consent_records[consent.consent_id] = consent

        logger.info(f"Recorded consent: {consent.consent_id} for user {user_id}")
        return consent

    async def revoke_consent(self, consent_id: str) -> bool:
        """Revoke user consent.

        Args:
            consent_id: Consent to revoke

        Returns:
            True if revoked successfully
        """
        async with self._get_lock():
            consent = self._consent_records.get(consent_id)
            if not consent:
                return False

            consent.revoked_at = datetime.utcnow()
            logger.info(f"Revoked consent: {consent_id}")
            return True

    async def check_consent(
        self,
        user_id: str,
        purpose: str,
        scope_item: Optional[str] = None,
    ) -> Optional[ConsentRecord]:
        """Check if user has valid consent.

        Args:
            user_id: User to check
            purpose: Required purpose
            scope_item: Specific scope item required

        Returns:
            Valid ConsentRecord if found, None otherwise
        """
        async with self._get_lock():
            for consent in self._consent_records.values():
                if consent.user_id != user_id:
                    continue
                if consent.purpose != purpose:
                    continue
                if not consent.is_valid:
                    continue
                if scope_item and scope_item not in consent.scope:
                    continue
                return consent

        return None

    async def submit_dsr(
        self,
        user_id: str,
        request_type: str,
        tenant_id: Optional[str] = None,
        deadline_days: int = 30,
    ) -> DataSubjectRequest:
        """Submit a Data Subject Request.

        Args:
            user_id: User making request
            request_type: Type of request
            tenant_id: Tenant identifier
            deadline_days: Days to respond

        Returns:
            Created DataSubjectRequest
        """
        import uuid

        request = DataSubjectRequest(
            request_id=str(uuid.uuid4()),
            user_id=user_id,
            tenant_id=tenant_id,
            request_type=request_type,
            submitted_at=datetime.utcnow(),
            deadline=datetime.utcnow() + timedelta(days=deadline_days),
        )

        async with self._get_lock():
            self._dsr_requests[request.request_id] = request

        logger.info(f"DSR submitted: {request.request_id} ({request_type}) for user {user_id}")
        return request

    async def complete_dsr(
        self,
        request_id: str,
        response_data: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> bool:
        """Complete a Data Subject Request.

        Args:
            request_id: Request to complete
            response_data: Response data
            notes: Additional notes

        Returns:
            True if completed successfully
        """
        async with self._get_lock():
            request = self._dsr_requests.get(request_id)
            if not request:
                return False

            request.status = "completed"
            request.completed_at = datetime.utcnow()
            request.response_data = response_data
            request.notes = notes

            logger.info(f"DSR completed: {request_id}")
            return True

    async def get_user_data_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of data held about a user (for GDPR Article 15).

        Args:
            user_id: User to get data for

        Returns:
            Summary of user data
        """
        async with self._get_lock():
            # Get access logs
            access_logs = [
                log.to_dict() for log in self._access_logs
                if log.user_id == user_id
            ][-100]  # Last 100

            # Get consent records
            consents = [
                {
                    "consent_id": c.consent_id,
                    "purpose": c.purpose,
                    "scope": list(c.scope),
                    "granted_at": c.granted_at.isoformat(),
                    "is_valid": c.is_valid,
                }
                for c in self._consent_records.values()
                if c.user_id == user_id
            ]

            # Get DSR history
            dsr_history = [
                {
                    "request_id": r.request_id,
                    "type": r.request_type,
                    "status": r.status,
                    "submitted_at": r.submitted_at.isoformat(),
                }
                for r in self._dsr_requests.values()
                if r.user_id == user_id
            ]

        return {
            "user_id": user_id,
            "generated_at": datetime.utcnow().isoformat(),
            "data_categories_accessed": list(set(log["data_category"] for log in access_logs)),
            "recent_access_logs": access_logs[:20],
            "consent_records": consents,
            "dsr_history": dsr_history,
            "retention_policies": {
                cat.value: {
                    "retention_days": policy.retention_days,
                    "description": policy.description,
                }
                for cat, policy in self.retention_policies.items()
            },
        }

    async def get_overdue_dsr(self) -> List[DataSubjectRequest]:
        """Get all overdue DSR requests."""
        async with self._get_lock():
            return [r for r in self._dsr_requests.values() if r.is_overdue]

    async def get_expiring_consents(self, days: int = 30) -> List[ConsentRecord]:
        """Get consents expiring within given days."""
        cutoff = datetime.utcnow() + timedelta(days=days)
        async with self._get_lock():
            return [
                c for c in self._consent_records.values()
                if c.is_valid and c.expires_at and c.expires_at <= cutoff
            ]

    def get_retention_policy(self, category: DataCategory) -> RetentionPolicy:
        """Get retention policy for a data category."""
        return self.retention_policies.get(category, DEFAULT_RETENTION_POLICIES[DataCategory.TECHNICAL])

    async def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance status report."""
        async with self._get_lock():
            total_access_logs = len(self._access_logs)
            total_consents = len(self._consent_records)
            valid_consents = sum(1 for c in self._consent_records.values() if c.is_valid)
            total_dsr = len(self._dsr_requests)
            pending_dsr = sum(1 for r in self._dsr_requests.values() if r.status == "pending")
            overdue_dsr = sum(1 for r in self._dsr_requests.values() if r.is_overdue)

            # Data category breakdown
            category_counts: Dict[str, int] = {}
            for log in self._access_logs:
                cat = log.data_category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "enabled_frameworks": [f.value for f in self.enabled_frameworks],
            "metrics": {
                "total_access_logs": total_access_logs,
                "total_consents": total_consents,
                "valid_consents": valid_consents,
                "total_dsr_requests": total_dsr,
                "pending_dsr_requests": pending_dsr,
                "overdue_dsr_requests": overdue_dsr,
            },
            "data_category_access": category_counts,
            "retention_policies": {
                cat.value: {
                    "retention_days": policy.retention_days,
                    "auto_delete": policy.auto_delete,
                }
                for cat, policy in self.retention_policies.items()
            },
            "compliance_status": {
                "dsr_compliance": overdue_dsr == 0,
                "consent_tracking": valid_consents > 0 or total_consents == 0,
                "data_access_logging": total_access_logs > 0,
            },
        }


# Global compliance tracker
_compliance_tracker: Optional[ComplianceTracker] = None


def get_compliance_tracker() -> ComplianceTracker:
    """Get global compliance tracker."""
    global _compliance_tracker
    if _compliance_tracker is None:
        _compliance_tracker = ComplianceTracker()
    return _compliance_tracker
