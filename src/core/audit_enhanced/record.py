"""Tamper-Proof Audit Records.

Provides immutable audit records with:
- Cryptographic integrity verification
- Hash chains for tamper detection
- Digital signatures
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AuditCategory(Enum):
    """Categories of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    SYSTEM = "system"
    USER_ACTION = "user_action"
    API_CALL = "api_call"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditOutcome(Enum):
    """Outcome of audited operation."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    DENIED = "denied"
    ERROR = "error"


@dataclass
class AuditContext:
    """Context information for audit records."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    service_name: Optional[str] = None
    environment: Optional[str] = None
    correlation_id: Optional[str] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.user_id:
            result["user_id"] = self.user_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.request_id:
            result["request_id"] = self.request_id
        if self.ip_address:
            result["ip_address"] = self.ip_address
        if self.user_agent:
            result["user_agent"] = self.user_agent
        if self.service_name:
            result["service_name"] = self.service_name
        if self.environment:
            result["environment"] = self.environment
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.custom_attributes:
            result["custom"] = self.custom_attributes
        return result


@dataclass
class AuditRecord:
    """Immutable audit record with integrity verification."""

    # Required fields
    record_id: str
    timestamp: datetime
    category: AuditCategory
    action: str

    # Optional fields
    severity: AuditSeverity = AuditSeverity.INFO
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    context: Optional[AuditContext] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    description: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Integrity fields
    previous_hash: Optional[str] = None
    record_hash: Optional[str] = None
    signature: Optional[str] = None

    # Metadata
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.context is None:
            self.context = AuditContext()

    def to_dict(self, include_integrity: bool = True) -> Dict[str, Any]:
        """Convert record to dictionary."""
        result = {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "action": self.action,
            "severity": self.severity.value,
            "outcome": self.outcome.value,
            "context": self.context.to_dict() if self.context else {},
            "schema_version": self.schema_version,
        }

        if self.resource_type:
            result["resource_type"] = self.resource_type
        if self.resource_id:
            result["resource_id"] = self.resource_id
        if self.description:
            result["description"] = self.description
        if self.details:
            result["details"] = self.details
        if self.tags:
            result["tags"] = self.tags

        if include_integrity:
            if self.previous_hash:
                result["previous_hash"] = self.previous_hash
            if self.record_hash:
                result["record_hash"] = self.record_hash
            if self.signature:
                result["signature"] = self.signature

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        """Create record from dictionary."""
        context_data = data.get("context", {})
        context = AuditContext(
            user_id=context_data.get("user_id"),
            session_id=context_data.get("session_id"),
            request_id=context_data.get("request_id"),
            ip_address=context_data.get("ip_address"),
            user_agent=context_data.get("user_agent"),
            service_name=context_data.get("service_name"),
            environment=context_data.get("environment"),
            correlation_id=context_data.get("correlation_id"),
            custom_attributes=context_data.get("custom", {}),
        )

        return cls(
            record_id=data["record_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            category=AuditCategory(data["category"]),
            action=data["action"],
            severity=AuditSeverity(data.get("severity", "info")),
            outcome=AuditOutcome(data.get("outcome", "success")),
            context=context,
            resource_type=data.get("resource_type"),
            resource_id=data.get("resource_id"),
            description=data.get("description"),
            details=data.get("details", {}),
            tags=data.get("tags", []),
            previous_hash=data.get("previous_hash"),
            record_hash=data.get("record_hash"),
            signature=data.get("signature"),
            schema_version=data.get("schema_version", "1.0"),
        )


class RecordIntegrity:
    """Handles cryptographic integrity for audit records."""

    def __init__(self, secret_key: Optional[bytes] = None):
        self._secret_key = secret_key or secrets.token_bytes(32)

    def compute_hash(self, record: AuditRecord) -> str:
        """Compute SHA-256 hash of record content."""
        content = record.to_dict(include_integrity=False)
        if record.previous_hash:
            content["previous_hash"] = record.previous_hash

        canonical = json.dumps(content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def sign_record(self, record: AuditRecord) -> str:
        """Create HMAC signature for record."""
        record_hash = record.record_hash or self.compute_hash(record)
        signature = hmac.new(
            self._secret_key,
            record_hash.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def verify_hash(self, record: AuditRecord) -> bool:
        """Verify record hash matches content."""
        if not record.record_hash:
            return False
        expected_hash = self.compute_hash(record)
        return hmac.compare_digest(record.record_hash, expected_hash)

    def verify_signature(self, record: AuditRecord) -> bool:
        """Verify record signature."""
        if not record.signature or not record.record_hash:
            return False
        expected_sig = self.sign_record(record)
        return hmac.compare_digest(record.signature, expected_sig)

    def verify_chain(self, records: List[AuditRecord]) -> tuple[bool, Optional[int]]:
        """Verify hash chain integrity.

        Returns:
            Tuple of (is_valid, first_invalid_index)
        """
        if not records:
            return True, None

        for i, record in enumerate(records):
            # Verify individual record hash
            if not self.verify_hash(record):
                return False, i

            # Verify chain link
            if i > 0:
                expected_prev = records[i - 1].record_hash
                if record.previous_hash != expected_prev:
                    return False, i

        return True, None


class RecordBuilder:
    """Builder for creating audit records."""

    def __init__(self, integrity: Optional[RecordIntegrity] = None):
        self._integrity = integrity or RecordIntegrity()
        self._last_hash: Optional[str] = None

    def create(
        self,
        category: AuditCategory,
        action: str,
        **kwargs,
    ) -> AuditRecord:
        """Create a new audit record with integrity."""
        record_id = kwargs.pop("record_id", None) or self._generate_id()
        timestamp = kwargs.pop("timestamp", None) or datetime.utcnow()

        record = AuditRecord(
            record_id=record_id,
            timestamp=timestamp,
            category=category,
            action=action,
            previous_hash=self._last_hash,
            **kwargs,
        )

        # Compute integrity
        record.record_hash = self._integrity.compute_hash(record)
        record.signature = self._integrity.sign_record(record)

        # Update chain
        self._last_hash = record.record_hash

        return record

    def set_last_hash(self, hash_value: str) -> None:
        """Set the last hash for chain continuation."""
        self._last_hash = hash_value

    def _generate_id(self) -> str:
        """Generate unique record ID."""
        timestamp_part = hex(int(time.time() * 1000))[2:]
        random_part = secrets.token_hex(8)
        return f"audit_{timestamp_part}_{random_part}"


@dataclass
class ChainMetadata:
    """Metadata for an audit chain."""
    chain_id: str
    created_at: datetime
    last_record_id: Optional[str] = None
    last_hash: Optional[str] = None
    record_count: int = 0
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "created_at": self.created_at.isoformat(),
            "last_record_id": self.last_record_id,
            "last_hash": self.last_hash,
            "record_count": self.record_count,
            "first_timestamp": self.first_timestamp.isoformat() if self.first_timestamp else None,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
        }


class AuditChain:
    """Manages a chain of audit records."""

    def __init__(
        self,
        chain_id: Optional[str] = None,
        integrity: Optional[RecordIntegrity] = None,
    ):
        self.chain_id = chain_id or f"chain_{secrets.token_hex(8)}"
        self._integrity = integrity or RecordIntegrity()
        self._builder = RecordBuilder(self._integrity)
        self._records: List[AuditRecord] = []
        self._metadata = ChainMetadata(
            chain_id=self.chain_id,
            created_at=datetime.utcnow(),
        )

    @property
    def metadata(self) -> ChainMetadata:
        return self._metadata

    @property
    def records(self) -> List[AuditRecord]:
        return self._records.copy()

    def append(
        self,
        category: AuditCategory,
        action: str,
        **kwargs,
    ) -> AuditRecord:
        """Append a new record to the chain."""
        record = self._builder.create(category, action, **kwargs)
        self._records.append(record)

        # Update metadata
        self._metadata.last_record_id = record.record_id
        self._metadata.last_hash = record.record_hash
        self._metadata.record_count += 1
        self._metadata.last_timestamp = record.timestamp
        if self._metadata.first_timestamp is None:
            self._metadata.first_timestamp = record.timestamp

        return record

    def verify(self) -> tuple[bool, Optional[int]]:
        """Verify the entire chain integrity."""
        return self._integrity.verify_chain(self._records)

    def get_record(self, record_id: str) -> Optional[AuditRecord]:
        """Get record by ID."""
        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def export(self) -> Dict[str, Any]:
        """Export chain with metadata."""
        return {
            "metadata": self._metadata.to_dict(),
            "records": [r.to_dict() for r in self._records],
        }

    @classmethod
    def import_chain(
        cls,
        data: Dict[str, Any],
        integrity: Optional[RecordIntegrity] = None,
    ) -> "AuditChain":
        """Import chain from exported data."""
        metadata = data["metadata"]
        chain = cls(chain_id=metadata["chain_id"], integrity=integrity)

        for record_data in data.get("records", []):
            record = AuditRecord.from_dict(record_data)
            chain._records.append(record)
            chain._builder.set_last_hash(record.record_hash)

        # Restore metadata
        chain._metadata.record_count = len(chain._records)
        if chain._records:
            chain._metadata.first_timestamp = chain._records[0].timestamp
            chain._metadata.last_timestamp = chain._records[-1].timestamp
            chain._metadata.last_record_id = chain._records[-1].record_id
            chain._metadata.last_hash = chain._records[-1].record_hash

        return chain
