"""Enhanced Audit Logging Module.

Provides enterprise-grade audit logging:
- Tamper-proof records with cryptographic integrity
- Hash chains for verification
- Advanced query interface
- Compliance reporting
- Multiple storage backends
"""

from src.core.audit_enhanced.record import (
    AuditCategory,
    AuditSeverity,
    AuditOutcome,
    AuditContext,
    AuditRecord,
    RecordIntegrity,
    RecordBuilder,
    ChainMetadata,
    AuditChain,
)
from src.core.audit_enhanced.storage import (
    AuditStorage,
    InMemoryAuditStorage,
    FileAuditStorage,
    RotatingFileStorage,
)
from src.core.audit_enhanced.query import (
    ComparisonOperator,
    LogicalOperator,
    FilterCondition,
    FilterGroup,
    FilterBuilder,
    AggregationType,
    AggregationResult,
    TimeSeriesPoint,
    QueryResult,
    AuditQuery,
    AuditExporter,
)
from src.core.audit_enhanced.manager import (
    RetentionPolicy,
    ComplianceReport,
    AuditManager,
    audit_logged,
)

__all__ = [
    # Record
    "AuditCategory",
    "AuditSeverity",
    "AuditOutcome",
    "AuditContext",
    "AuditRecord",
    "RecordIntegrity",
    "RecordBuilder",
    "ChainMetadata",
    "AuditChain",
    # Storage
    "AuditStorage",
    "InMemoryAuditStorage",
    "FileAuditStorage",
    "RotatingFileStorage",
    # Query
    "ComparisonOperator",
    "LogicalOperator",
    "FilterCondition",
    "FilterGroup",
    "FilterBuilder",
    "AggregationType",
    "AggregationResult",
    "TimeSeriesPoint",
    "QueryResult",
    "AuditQuery",
    "AuditExporter",
    # Manager
    "RetentionPolicy",
    "ComplianceReport",
    "AuditManager",
    "audit_logged",
]
