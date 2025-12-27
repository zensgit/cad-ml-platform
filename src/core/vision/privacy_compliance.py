"""Privacy compliance module for Vision Provider system.

This module implements GDPR, CCPA, and other privacy regulation compliance
features including consent management, data retention, and subject rights.
"""

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set

from .base import VisionDescription, VisionProvider


class PrivacyRegulation(Enum):
    """Supported privacy regulations."""

    GDPR = "gdpr"  # EU General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    LGPD = "lgpd"  # Brazilian General Data Protection Law
    PIPEDA = "pipeda"  # Canadian Personal Information Protection
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    CUSTOM = "custom"


class ConsentType(Enum):
    """Types of consent that can be requested."""

    DATA_PROCESSING = "data_processing"
    DATA_STORAGE = "data_storage"
    DATA_SHARING = "data_sharing"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    PROFILING = "profiling"
    AUTOMATED_DECISIONS = "automated_decisions"
    CROSS_BORDER_TRANSFER = "cross_border_transfer"


class LegalBasis(Enum):
    """Legal basis for data processing under GDPR."""

    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRight(Enum):
    """Data subject rights under privacy regulations."""

    ACCESS = "access"  # Right to access data
    RECTIFICATION = "rectification"  # Right to correct data
    ERASURE = "erasure"  # Right to be forgotten
    RESTRICTION = "restriction"  # Right to restrict processing
    PORTABILITY = "portability"  # Right to data portability
    OBJECTION = "objection"  # Right to object to processing
    AUTOMATED_DECISIONS = "automated_decisions"  # Rights related to automated decisions


class RequestStatus(Enum):
    """Status of data subject requests."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Consent:
    """Represents a user's consent."""

    consent_id: str
    subject_id: str
    consent_type: ConsentType
    granted: bool
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    purpose: str = ""
    legal_basis: LegalBasis = LegalBasis.CONSENT
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if not self.granted:
            return False
        if self.withdrawn_at is not None:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


@dataclass
class DataSubjectRequest:
    """Represents a data subject rights request."""

    request_id: str
    subject_id: str
    right: DataSubjectRight
    status: RequestStatus
    created_at: datetime
    deadline: datetime
    completed_at: Optional[datetime] = None
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRetentionPolicy:
    """Data retention policy configuration."""

    policy_id: str
    data_category: str
    retention_period: timedelta
    legal_basis: LegalBasis
    regulation: PrivacyRegulation
    auto_delete: bool = True
    archive_before_delete: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingActivity:
    """Records a data processing activity."""

    activity_id: str
    activity_type: str
    subject_id: str
    data_categories: List[str]
    purpose: str
    legal_basis: LegalBasis
    timestamp: datetime
    processor: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment (PIA/DPIA) record."""

    assessment_id: str
    project_name: str
    data_types: List[str]
    processing_purposes: List[str]
    risk_level: str  # low, medium, high, critical
    mitigations: List[str]
    approved: bool
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    review_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsentStore(ABC):
    """Abstract base class for consent storage."""

    @abstractmethod
    def store_consent(self, consent: Consent) -> None:
        """Store a consent record."""
        pass

    @abstractmethod
    def get_consent(self, consent_id: str) -> Optional[Consent]:
        """Retrieve a consent record."""
        pass

    @abstractmethod
    def get_consents_for_subject(self, subject_id: str) -> List[Consent]:
        """Get all consents for a data subject."""
        pass

    @abstractmethod
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw a consent."""
        pass

    @abstractmethod
    def delete_consents(self, subject_id: str) -> int:
        """Delete all consents for a subject."""
        pass


class InMemoryConsentStore(ConsentStore):
    """In-memory consent storage implementation."""

    def __init__(self) -> None:
        self._consents: Dict[str, Consent] = {}
        self._subject_consents: Dict[str, Set[str]] = {}
        self._lock = Lock()

    def store_consent(self, consent: Consent) -> None:
        """Store a consent record."""
        with self._lock:
            self._consents[consent.consent_id] = consent
            if consent.subject_id not in self._subject_consents:
                self._subject_consents[consent.subject_id] = set()
            self._subject_consents[consent.subject_id].add(consent.consent_id)

    def get_consent(self, consent_id: str) -> Optional[Consent]:
        """Retrieve a consent record."""
        return self._consents.get(consent_id)

    def get_consents_for_subject(self, subject_id: str) -> List[Consent]:
        """Get all consents for a data subject."""
        consent_ids = self._subject_consents.get(subject_id, set())
        return [self._consents[cid] for cid in consent_ids if cid in self._consents]

    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw a consent."""
        with self._lock:
            if consent_id in self._consents:
                consent = self._consents[consent_id]
                consent.withdrawn_at = datetime.now()
                return True
            return False

    def delete_consents(self, subject_id: str) -> int:
        """Delete all consents for a subject."""
        with self._lock:
            consent_ids = self._subject_consents.pop(subject_id, set())
            count = 0
            for cid in consent_ids:
                if cid in self._consents:
                    del self._consents[cid]
                    count += 1
            return count


class ConsentManager:
    """Manages user consents and consent verification."""

    def __init__(
        self,
        store: Optional[ConsentStore] = None,
        default_expiry_days: int = 365,
    ) -> None:
        self._store = store or InMemoryConsentStore()
        self._default_expiry = timedelta(days=default_expiry_days)
        self._consent_templates: Dict[ConsentType, Dict[str, Any]] = {}

    def register_consent_template(
        self,
        consent_type: ConsentType,
        purpose: str,
        legal_basis: LegalBasis,
        expiry_days: Optional[int] = None,
    ) -> None:
        """Register a consent template for a specific type."""
        self._consent_templates[consent_type] = {
            "purpose": purpose,
            "legal_basis": legal_basis,
            "expiry_days": expiry_days or self._default_expiry.days,
        }

    def grant_consent(
        self,
        subject_id: str,
        consent_type: ConsentType,
        purpose: Optional[str] = None,
        expiry_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Consent:
        """Grant consent for a specific type of processing."""
        template = self._consent_templates.get(consent_type, {})

        now = datetime.now()
        days = expiry_days or template.get("expiry_days", self._default_expiry.days)

        consent = Consent(
            consent_id=str(uuid.uuid4()),
            subject_id=subject_id,
            consent_type=consent_type,
            granted=True,
            granted_at=now,
            expires_at=now + timedelta(days=days),
            purpose=purpose or template.get("purpose", ""),
            legal_basis=template.get("legal_basis", LegalBasis.CONSENT),
            metadata=metadata or {},
        )

        self._store.store_consent(consent)
        return consent

    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw a specific consent."""
        return self._store.withdraw_consent(consent_id)

    def has_valid_consent(
        self,
        subject_id: str,
        consent_type: ConsentType,
    ) -> bool:
        """Check if subject has valid consent for a specific type."""
        consents = self._store.get_consents_for_subject(subject_id)
        return any(c.consent_type == consent_type and c.is_valid() for c in consents)

    def get_consent_status(self, subject_id: str) -> Dict[ConsentType, bool]:
        """Get consent status for all types for a subject."""
        consents = self._store.get_consents_for_subject(subject_id)
        status: Dict[ConsentType, bool] = {}
        for consent_type in ConsentType:
            status[consent_type] = any(
                c.consent_type == consent_type and c.is_valid() for c in consents
            )
        return status

    def get_all_consents(self, subject_id: str) -> List[Consent]:
        """Get all consent records for a subject."""
        return self._store.get_consents_for_subject(subject_id)

    def delete_all_consents(self, subject_id: str) -> int:
        """Delete all consents for a subject (for erasure requests)."""
        return self._store.delete_consents(subject_id)


class DataSubjectRequestHandler:
    """Handles data subject rights requests."""

    def __init__(
        self,
        deadline_days: int = 30,  # GDPR requires response within 30 days
    ) -> None:
        self._requests: Dict[str, DataSubjectRequest] = {}
        self._subject_requests: Dict[str, List[str]] = {}
        self._handlers: Dict[DataSubjectRight, Callable[[DataSubjectRequest], str]] = {}
        self._deadline_days = deadline_days
        self._lock = Lock()

    def register_handler(
        self,
        right: DataSubjectRight,
        handler: Callable[[DataSubjectRequest], str],
    ) -> None:
        """Register a handler for a specific right."""
        self._handlers[right] = handler

    def submit_request(
        self,
        subject_id: str,
        right: DataSubjectRight,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataSubjectRequest:
        """Submit a data subject request."""
        now = datetime.now()
        request = DataSubjectRequest(
            request_id=str(uuid.uuid4()),
            subject_id=subject_id,
            right=right,
            status=RequestStatus.PENDING,
            created_at=now,
            deadline=now + timedelta(days=self._deadline_days),
            metadata=metadata or {},
        )

        with self._lock:
            self._requests[request.request_id] = request
            if subject_id not in self._subject_requests:
                self._subject_requests[subject_id] = []
            self._subject_requests[subject_id].append(request.request_id)

        return request

    def process_request(self, request_id: str) -> Optional[DataSubjectRequest]:
        """Process a pending request."""
        with self._lock:
            if request_id not in self._requests:
                return None

            request = self._requests[request_id]
            if request.status != RequestStatus.PENDING:
                return request

            request.status = RequestStatus.IN_PROGRESS

        # Execute handler if registered
        if request.right in self._handlers:
            try:
                response = self._handlers[request.right](request)
                with self._lock:
                    request.response = response
                    request.status = RequestStatus.COMPLETED
                    request.completed_at = datetime.now()
            except Exception as e:
                with self._lock:
                    request.response = f"Error: {str(e)}"
                    request.status = RequestStatus.REJECTED
        else:
            with self._lock:
                request.response = "No handler registered for this right"
                request.status = RequestStatus.REJECTED

        return request

    def get_request(self, request_id: str) -> Optional[DataSubjectRequest]:
        """Get a specific request."""
        return self._requests.get(request_id)

    def get_requests_for_subject(self, subject_id: str) -> List[DataSubjectRequest]:
        """Get all requests for a subject."""
        request_ids = self._subject_requests.get(subject_id, [])
        return [self._requests[rid] for rid in request_ids if rid in self._requests]

    def get_pending_requests(self) -> List[DataSubjectRequest]:
        """Get all pending requests."""
        return [r for r in self._requests.values() if r.status == RequestStatus.PENDING]

    def get_overdue_requests(self) -> List[DataSubjectRequest]:
        """Get requests that have passed their deadline."""
        now = datetime.now()
        return [
            r
            for r in self._requests.values()
            if r.status in (RequestStatus.PENDING, RequestStatus.IN_PROGRESS) and now > r.deadline
        ]


class DataRetentionManager:
    """Manages data retention policies and enforcement."""

    def __init__(self) -> None:
        self._policies: Dict[str, DataRetentionPolicy] = {}
        self._data_records: Dict[
            str, Dict[str, datetime]
        ] = {}  # subject -> {category -> timestamp}
        self._lock = Lock()

    def add_policy(self, policy: DataRetentionPolicy) -> None:
        """Add a retention policy."""
        with self._lock:
            self._policies[policy.policy_id] = policy

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a retention policy."""
        with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                return True
            return False

    def get_policy(self, policy_id: str) -> Optional[DataRetentionPolicy]:
        """Get a specific policy."""
        return self._policies.get(policy_id)

    def get_policies_for_category(self, data_category: str) -> List[DataRetentionPolicy]:
        """Get all policies for a data category."""
        return [p for p in self._policies.values() if p.data_category == data_category]

    def record_data_collection(
        self,
        subject_id: str,
        data_category: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record when data was collected for retention tracking."""
        with self._lock:
            if subject_id not in self._data_records:
                self._data_records[subject_id] = {}
            self._data_records[subject_id][data_category] = timestamp or datetime.now()

    def get_data_for_deletion(self) -> List[tuple]:
        """Get data that should be deleted based on retention policies."""
        now = datetime.now()
        to_delete = []

        with self._lock:
            for subject_id, categories in self._data_records.items():
                for category, collected_at in categories.items():
                    policies = self.get_policies_for_category(category)
                    for policy in policies:
                        if policy.auto_delete:
                            expiry = collected_at + policy.retention_period
                            if now > expiry:
                                to_delete.append((subject_id, category, policy.policy_id))

        return to_delete

    def mark_data_deleted(self, subject_id: str, data_category: str) -> bool:
        """Mark data as deleted after retention enforcement."""
        with self._lock:
            if subject_id in self._data_records:
                if data_category in self._data_records[subject_id]:
                    del self._data_records[subject_id][data_category]
                    return True
        return False


class ProcessingActivityLog:
    """Maintains a log of processing activities for compliance."""

    def __init__(self, max_entries: int = 100000) -> None:
        self._activities: List[ProcessingActivity] = []
        self._max_entries = max_entries
        self._lock = Lock()

    def log_activity(
        self,
        activity_type: str,
        subject_id: str,
        data_categories: List[str],
        purpose: str,
        legal_basis: LegalBasis,
        processor: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessingActivity:
        """Log a processing activity."""
        activity = ProcessingActivity(
            activity_id=str(uuid.uuid4()),
            activity_type=activity_type,
            subject_id=subject_id,
            data_categories=data_categories,
            purpose=purpose,
            legal_basis=legal_basis,
            timestamp=datetime.now(),
            processor=processor,
            metadata=metadata or {},
        )

        with self._lock:
            self._activities.append(activity)
            # Enforce max entries
            if len(self._activities) > self._max_entries:
                self._activities = self._activities[-self._max_entries :]

        return activity

    def get_activities_for_subject(self, subject_id: str) -> List[ProcessingActivity]:
        """Get all processing activities for a subject."""
        return [a for a in self._activities if a.subject_id == subject_id]

    def get_activities_by_type(self, activity_type: str) -> List[ProcessingActivity]:
        """Get activities by type."""
        return [a for a in self._activities if a.activity_type == activity_type]

    def get_activities_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[ProcessingActivity]:
        """Get activities within a time range."""
        return [a for a in self._activities if start <= a.timestamp <= end]

    def export_for_subject(self, subject_id: str) -> Dict[str, Any]:
        """Export processing activities for a subject (for access requests)."""
        activities = self.get_activities_for_subject(subject_id)
        return {
            "subject_id": subject_id,
            "export_date": datetime.now().isoformat(),
            "activities": [
                {
                    "activity_id": a.activity_id,
                    "type": a.activity_type,
                    "data_categories": a.data_categories,
                    "purpose": a.purpose,
                    "legal_basis": a.legal_basis.value,
                    "timestamp": a.timestamp.isoformat(),
                    "processor": a.processor,
                }
                for a in activities
            ],
        }


class PrivacyComplianceManager:
    """Central manager for privacy compliance."""

    def __init__(
        self,
        consent_manager: Optional[ConsentManager] = None,
        request_handler: Optional[DataSubjectRequestHandler] = None,
        retention_manager: Optional[DataRetentionManager] = None,
        activity_log: Optional[ProcessingActivityLog] = None,
        regulations: Optional[List[PrivacyRegulation]] = None,
    ) -> None:
        self._consent_manager = consent_manager or ConsentManager()
        self._request_handler = request_handler or DataSubjectRequestHandler()
        self._retention_manager = retention_manager or DataRetentionManager()
        self._activity_log = activity_log or ProcessingActivityLog()
        self._regulations = set(regulations or [PrivacyRegulation.GDPR])
        self._assessments: Dict[str, PrivacyImpactAssessment] = {}

    @property
    def consent_manager(self) -> ConsentManager:
        """Get the consent manager."""
        return self._consent_manager

    @property
    def request_handler(self) -> DataSubjectRequestHandler:
        """Get the request handler."""
        return self._request_handler

    @property
    def retention_manager(self) -> DataRetentionManager:
        """Get the retention manager."""
        return self._retention_manager

    @property
    def activity_log(self) -> ProcessingActivityLog:
        """Get the activity log."""
        return self._activity_log

    def add_regulation(self, regulation: PrivacyRegulation) -> None:
        """Add a regulation to comply with."""
        self._regulations.add(regulation)

    def get_regulations(self) -> Set[PrivacyRegulation]:
        """Get active regulations."""
        return self._regulations.copy()

    def check_processing_allowed(
        self,
        subject_id: str,
        processing_type: ConsentType,
        legal_basis: Optional[LegalBasis] = None,
    ) -> tuple:
        """Check if processing is allowed and return reason."""
        # Check for valid consent
        if self._consent_manager.has_valid_consent(subject_id, processing_type):
            return True, "Valid consent exists"

        # Check for other legal bases
        if legal_basis and legal_basis != LegalBasis.CONSENT:
            # For non-consent legal bases, processing may be allowed
            if legal_basis in (
                LegalBasis.CONTRACT,
                LegalBasis.LEGAL_OBLIGATION,
                LegalBasis.VITAL_INTERESTS,
            ):
                return True, f"Legal basis: {legal_basis.value}"

        return False, "No valid consent or legal basis"

    def record_processing(
        self,
        subject_id: str,
        activity_type: str,
        data_categories: List[str],
        purpose: str,
        legal_basis: LegalBasis,
        processor: str = "",
    ) -> ProcessingActivity:
        """Record a processing activity."""
        return self._activity_log.log_activity(
            activity_type=activity_type,
            subject_id=subject_id,
            data_categories=data_categories,
            purpose=purpose,
            legal_basis=legal_basis,
            processor=processor,
        )

    def handle_erasure_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle a right to erasure (right to be forgotten) request."""
        results = {
            "subject_id": subject_id,
            "request_type": "erasure",
            "timestamp": datetime.now().isoformat(),
            "actions": [],
        }

        # Delete consents
        deleted_consents = self._consent_manager.delete_all_consents(subject_id)
        results["actions"].append({"type": "consent_deletion", "count": deleted_consents})

        # Note: In a real implementation, this would trigger data deletion
        # across all systems. Here we just log the request.
        results["actions"].append({"type": "data_deletion_initiated", "status": "pending"})

        return results

    def handle_access_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle a data access request."""
        results = {
            "subject_id": subject_id,
            "request_type": "access",
            "timestamp": datetime.now().isoformat(),
            "data": {},
        }

        # Get consent records
        consents = self._consent_manager.get_all_consents(subject_id)
        results["data"]["consents"] = [
            {
                "type": c.consent_type.value,
                "granted": c.granted,
                "granted_at": c.granted_at.isoformat() if c.granted_at else None,
                "purpose": c.purpose,
            }
            for c in consents
        ]

        # Get processing activities
        results["data"]["processing_activities"] = self._activity_log.export_for_subject(subject_id)

        return results

    def create_assessment(
        self,
        project_name: str,
        data_types: List[str],
        processing_purposes: List[str],
        risk_level: str,
        mitigations: List[str],
    ) -> PrivacyImpactAssessment:
        """Create a Privacy Impact Assessment."""
        assessment = PrivacyImpactAssessment(
            assessment_id=str(uuid.uuid4()),
            project_name=project_name,
            data_types=data_types,
            processing_purposes=processing_purposes,
            risk_level=risk_level,
            mitigations=mitigations,
            approved=False,
        )
        self._assessments[assessment.assessment_id] = assessment
        return assessment

    def approve_assessment(
        self,
        assessment_id: str,
        approved_by: str,
        review_days: int = 365,
    ) -> bool:
        """Approve a Privacy Impact Assessment."""
        if assessment_id not in self._assessments:
            return False

        assessment = self._assessments[assessment_id]
        assessment.approved = True
        assessment.approved_by = approved_by
        assessment.approved_at = datetime.now()
        assessment.review_date = datetime.now() + timedelta(days=review_days)
        return True

    def get_assessment(self, assessment_id: str) -> Optional[PrivacyImpactAssessment]:
        """Get a Privacy Impact Assessment."""
        return self._assessments.get(assessment_id)

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status."""
        pending_requests = self._request_handler.get_pending_requests()
        overdue_requests = self._request_handler.get_overdue_requests()
        expired_data = self._retention_manager.get_data_for_deletion()

        return {
            "regulations": [r.value for r in self._regulations],
            "pending_requests": len(pending_requests),
            "overdue_requests": len(overdue_requests),
            "data_pending_deletion": len(expired_data),
            "assessments_pending_review": sum(
                1
                for a in self._assessments.values()
                if a.review_date and datetime.now() > a.review_date
            ),
            "compliant": len(overdue_requests) == 0,
        }


class PrivacyComplianceVisionProvider(VisionProvider):
    """Vision provider wrapper that enforces privacy compliance."""

    def __init__(
        self,
        provider: VisionProvider,
        compliance_manager: PrivacyComplianceManager,
        require_consent: bool = True,
        consent_type: ConsentType = ConsentType.DATA_PROCESSING,
    ) -> None:
        self._provider = provider
        self._compliance = compliance_manager
        self._require_consent = require_consent
        self._consent_type = consent_type

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return f"privacy_compliance_{self._provider.provider_name}"

    def _get_subject_id(self, image_data: bytes, **kwargs: Any) -> str:
        """Get subject ID from kwargs or generate from image hash."""
        return kwargs.get("subject_id", hashlib.sha256(image_data).hexdigest()[:16])

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image with privacy compliance checks."""
        subject_id = self._get_subject_id(image_data, **kwargs)
        legal_basis = kwargs.get("legal_basis", LegalBasis.CONSENT)

        # Check if processing is allowed
        if self._require_consent:
            allowed, reason = self._compliance.check_processing_allowed(
                subject_id, self._consent_type, legal_basis
            )
            if not allowed:
                return VisionDescription(
                    summary=f"Processing not allowed: {reason}",
                    details=[
                        "Consent required for image analysis",
                        f"Subject ID: {subject_id}",
                        f"Required consent type: {self._consent_type.value}",
                    ],
                    confidence=0.0,
                )

        # Record processing activity
        self._compliance.record_processing(
            subject_id=subject_id,
            activity_type="image_analysis",
            data_categories=["image_data", "biometric_data"],
            purpose=kwargs.get("purpose", "Image analysis"),
            legal_basis=legal_basis,
            processor=self.provider_name,
        )

        # Perform actual analysis
        return await self._provider.analyze_image(image_data, include_description, **kwargs)


# Factory functions


def create_consent_manager(
    default_expiry_days: int = 365,
) -> ConsentManager:
    """Create a consent manager instance."""
    return ConsentManager(default_expiry_days=default_expiry_days)


def create_consent_store() -> ConsentStore:
    """Create an in-memory consent store."""
    return InMemoryConsentStore()


def create_request_handler(deadline_days: int = 30) -> DataSubjectRequestHandler:
    """Create a data subject request handler."""
    return DataSubjectRequestHandler(deadline_days=deadline_days)


def create_retention_manager() -> DataRetentionManager:
    """Create a data retention manager."""
    return DataRetentionManager()


def create_activity_log(max_entries: int = 100000) -> ProcessingActivityLog:
    """Create a processing activity log."""
    return ProcessingActivityLog(max_entries=max_entries)


def create_compliance_manager(
    regulations: Optional[List[PrivacyRegulation]] = None,
) -> PrivacyComplianceManager:
    """Create a privacy compliance manager."""
    return PrivacyComplianceManager(regulations=regulations)


def create_compliance_provider(
    provider: VisionProvider,
    compliance_manager: Optional[PrivacyComplianceManager] = None,
    require_consent: bool = True,
    consent_type: ConsentType = ConsentType.DATA_PROCESSING,
) -> PrivacyComplianceVisionProvider:
    """Create a privacy-compliant vision provider."""
    manager = compliance_manager or create_compliance_manager()
    return PrivacyComplianceVisionProvider(
        provider=provider,
        compliance_manager=manager,
        require_consent=require_consent,
        consent_type=consent_type,
    )
