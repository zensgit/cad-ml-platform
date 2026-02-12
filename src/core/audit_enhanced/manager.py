"""Audit Manager.

High-level interface for audit logging:
- Event creation and storage
- Chain management
- Compliance reporting
- Retention policies
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.audit_enhanced.record import (
    AuditCategory,
    AuditChain,
    AuditContext,
    AuditOutcome,
    AuditRecord,
    AuditSeverity,
    RecordBuilder,
    RecordIntegrity,
)
from src.core.audit_enhanced.storage import AuditStorage, InMemoryAuditStorage
from src.core.audit_enhanced.query import (
    AuditExporter,
    AuditQuery,
    FilterBuilder,
    QueryResult,
)

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Policy for audit record retention."""
    name: str
    categories: Set[AuditCategory]
    retention_days: int
    archive_before_delete: bool = True
    compress: bool = True


@dataclass
class ComplianceReport:
    """Compliance report for audit data."""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_records: int
    records_by_category: Dict[str, int]
    records_by_severity: Dict[str, int]
    records_by_outcome: Dict[str, int]
    chain_integrity: bool
    integrity_violations: List[str]
    retention_compliance: bool
    summary: str


class AuditManager:
    """High-level audit management interface."""

    def __init__(
        self,
        storage: Optional[AuditStorage] = None,
        integrity: Optional[RecordIntegrity] = None,
        default_context: Optional[AuditContext] = None,
    ):
        self._storage = storage or InMemoryAuditStorage()
        self._integrity = integrity or RecordIntegrity()
        self._default_context = default_context
        self._builder = RecordBuilder(self._integrity)
        self._query = AuditQuery(self._storage)
        self._exporter = AuditExporter(self._storage)
        self._retention_policies: Dict[str, RetentionPolicy] = {}
        self._listeners: List[Callable[[AuditRecord], None]] = []

    # Event logging methods

    async def log(
        self,
        category: AuditCategory,
        action: str,
        context: Optional[AuditContext] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        description: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> AuditRecord:
        """Log an audit event."""
        merged_context = self._merge_context(context)

        record = self._builder.create(
            category=category,
            action=action,
            context=merged_context,
            severity=severity,
            outcome=outcome,
            resource_type=resource_type,
            resource_id=resource_id,
            description=description,
            details=details or {},
            tags=tags or [],
        )

        await self._storage.store(record)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(record)
            except Exception as e:
                logger.error(f"Listener error: {e}")

        return record

    async def log_authentication(
        self,
        user_id: str,
        action: str,  # login, logout, failed_login, password_change
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Log authentication event."""
        context = AuditContext(
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return await self.log(
            category=AuditCategory.AUTHENTICATION,
            action=action,
            context=context,
            severity=AuditSeverity.WARNING if not success else AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            details=details,
        )

    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,  # read, write, delete
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Log data access event."""
        context = AuditContext(user_id=user_id)
        return await self.log(
            category=AuditCategory.DATA_ACCESS,
            action=action,
            context=context,
            resource_type=resource_type,
            resource_id=resource_id,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            details=details,
        )

    async def log_configuration_change(
        self,
        user_id: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Log configuration change."""
        context = AuditContext(user_id=user_id)
        change_details = {
            "config_key": config_key,
            "old_value": str(old_value),
            "new_value": str(new_value),
            **(details or {}),
        }
        return await self.log(
            category=AuditCategory.CONFIGURATION,
            action="config_change",
            context=context,
            resource_type="configuration",
            resource_id=config_key,
            severity=AuditSeverity.WARNING,
            outcome=AuditOutcome.SUCCESS,
            details=change_details,
        )

    async def log_security_event(
        self,
        event_type: str,  # threat_detected, access_denied, policy_violation
        severity: AuditSeverity = AuditSeverity.WARNING,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        description: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Log security event."""
        context = AuditContext(
            user_id=user_id,
            ip_address=ip_address,
        )
        return await self.log(
            category=AuditCategory.SECURITY,
            action=event_type,
            context=context,
            severity=severity,
            outcome=AuditOutcome.DENIED,
            description=description,
            details=details,
            tags=["security"],
        )

    async def log_api_call(
        self,
        user_id: Optional[str],
        endpoint: str,
        method: str,
        status_code: int,
        request_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Log API call."""
        context = AuditContext(
            user_id=user_id,
            request_id=request_id,
        )
        api_details = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            **({"duration_ms": duration_ms} if duration_ms else {}),
            **(details or {}),
        }
        outcome = AuditOutcome.SUCCESS if status_code < 400 else AuditOutcome.FAILURE
        return await self.log(
            category=AuditCategory.API_CALL,
            action=f"{method} {endpoint}",
            context=context,
            severity=AuditSeverity.INFO,
            outcome=outcome,
            details=api_details,
        )

    # Query methods

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[List[AuditCategory]] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> QueryResult:
        """Query audit records."""
        builder = FilterBuilder()

        if categories:
            builder.where("category").is_in(categories)
        if user_id:
            builder.where("context.user_id").equals(user_id)

        return await self._query.execute(
            filter_group=builder.build(),
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )

    async def get_record(self, record_id: str) -> Optional[AuditRecord]:
        """Get single record by ID."""
        return await self._storage.get(record_id)

    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> List[AuditRecord]:
        """Get user's recent activity."""
        start_time = datetime.utcnow() - timedelta(days=days)
        result = await self.query(
            start_time=start_time,
            user_id=user_id,
            limit=1000,
        )
        return result.records

    # Integrity verification

    async def verify_chain_integrity(
        self,
        start_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> tuple[bool, List[str]]:
        """Verify integrity of audit chain."""
        records = await self._storage.get_chain_records(limit=limit)

        if not records:
            return True, []

        violations = []
        for i, record in enumerate(records):
            # Verify hash
            if not self._integrity.verify_hash(record):
                violations.append(f"Hash mismatch at record {record.record_id}")

            # Verify signature
            if record.signature and not self._integrity.verify_signature(record):
                violations.append(f"Signature invalid at record {record.record_id}")

            # Verify chain link
            if i > 0:
                expected_prev = records[i - 1].record_hash
                if record.previous_hash != expected_prev:
                    violations.append(
                        f"Chain broken at record {record.record_id}: "
                        f"expected {expected_prev}, got {record.previous_hash}"
                    )

        return len(violations) == 0, violations

    # Retention management

    def add_retention_policy(self, policy: RetentionPolicy) -> None:
        """Add retention policy."""
        self._retention_policies[policy.name] = policy
        logger.info(f"Added retention policy: {policy.name}")

    async def apply_retention_policies(self) -> Dict[str, int]:
        """Apply all retention policies."""
        results = {}
        now = datetime.utcnow()

        for name, policy in self._retention_policies.items():
            cutoff = now - timedelta(days=policy.retention_days)

            # Get records to delete
            records = await self._storage.query(
                end_time=cutoff,
                categories=list(policy.categories),
                limit=100000,
            )

            if policy.archive_before_delete:
                # Archive records (simplified - would write to archive storage)
                logger.info(f"Archiving {len(records)} records for policy {name}")

            # Delete old records
            deleted = await self._storage.delete_before(cutoff)
            results[name] = deleted
            logger.info(f"Policy {name}: deleted {deleted} records")

        return results

    # Compliance reporting

    async def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> ComplianceReport:
        """Generate compliance report for period."""
        import secrets as sec

        # Get all records in period
        records = await self._storage.query(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        # Count by category
        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_outcome: Dict[str, int] = {}

        for record in records:
            cat = record.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            sev = record.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            out = record.outcome.value
            by_outcome[out] = by_outcome.get(out, 0) + 1

        # Verify integrity
        integrity_ok, violations = await self.verify_chain_integrity()

        # Check retention compliance
        retention_ok = True
        for policy in self._retention_policies.values():
            cutoff = datetime.utcnow() - timedelta(days=policy.retention_days)
            old_count = await self._storage.count(
                end_time=cutoff,
                categories=list(policy.categories),
            )
            if old_count > 0:
                retention_ok = False
                break

        return ComplianceReport(
            report_id=f"report_{sec.token_hex(8)}",
            generated_at=datetime.utcnow(),
            period_start=start_time,
            period_end=end_time,
            total_records=len(records),
            records_by_category=by_category,
            records_by_severity=by_severity,
            records_by_outcome=by_outcome,
            chain_integrity=integrity_ok,
            integrity_violations=violations,
            retention_compliance=retention_ok,
            summary=self._generate_summary(len(records), by_severity, integrity_ok),
        )

    def _generate_summary(
        self,
        total: int,
        by_severity: Dict[str, int],
        integrity_ok: bool,
    ) -> str:
        """Generate human-readable summary."""
        critical = by_severity.get("critical", 0)
        errors = by_severity.get("error", 0)
        warnings = by_severity.get("warning", 0)

        lines = [
            f"Total audit records: {total}",
            f"Critical events: {critical}",
            f"Errors: {errors}",
            f"Warnings: {warnings}",
            f"Chain integrity: {'OK' if integrity_ok else 'COMPROMISED'}",
        ]
        return "; ".join(lines)

    # Export

    async def export_records(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json",
    ) -> str:
        """Export records in specified format."""
        records = await self._storage.query(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        if format == "json":
            return await self._exporter.to_json(records)
        elif format == "csv":
            return await self._exporter.to_csv(records)
        elif format == "jsonl":
            return await self._exporter.to_jsonl(records)
        else:
            raise ValueError(f"Unsupported format: {format}")

    # Listener management

    def add_listener(self, listener: Callable[[AuditRecord], None]) -> None:
        """Add event listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[AuditRecord], None]) -> None:
        """Remove event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    # Context management

    def _merge_context(self, context: Optional[AuditContext]) -> AuditContext:
        """Merge provided context with default context."""
        if not context and not self._default_context:
            return AuditContext()

        if not self._default_context:
            return context or AuditContext()

        if not context:
            return self._default_context

        # Merge: provided context takes precedence
        return AuditContext(
            user_id=context.user_id or self._default_context.user_id,
            session_id=context.session_id or self._default_context.session_id,
            request_id=context.request_id or self._default_context.request_id,
            ip_address=context.ip_address or self._default_context.ip_address,
            user_agent=context.user_agent or self._default_context.user_agent,
            service_name=context.service_name or self._default_context.service_name,
            environment=context.environment or self._default_context.environment,
            correlation_id=context.correlation_id or self._default_context.correlation_id,
            custom_attributes={
                **self._default_context.custom_attributes,
                **context.custom_attributes,
            },
        )


def audit_logged(
    manager: AuditManager,
    category: AuditCategory,
    action: Optional[str] = None,
):
    """Decorator to automatically log function calls."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_action = action or func.__name__

            try:
                result = await func(*args, **kwargs)
                await manager.log(
                    category=category,
                    action=func_action,
                    outcome=AuditOutcome.SUCCESS,
                    details={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
                )
                return result
            except Exception as e:
                await manager.log(
                    category=category,
                    action=func_action,
                    outcome=AuditOutcome.ERROR,
                    severity=AuditSeverity.ERROR,
                    details={"error": str(e)},
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_action = action or func.__name__

            try:
                result = func(*args, **kwargs)
                asyncio.create_task(manager.log(
                    category=category,
                    action=func_action,
                    outcome=AuditOutcome.SUCCESS,
                ))
                return result
            except Exception as e:
                asyncio.create_task(manager.log(
                    category=category,
                    action=func_action,
                    outcome=AuditOutcome.ERROR,
                    severity=AuditSeverity.ERROR,
                    details={"error": str(e)},
                ))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
