"""SLA Monitor Module for Vision System.

This module provides SLA monitoring capabilities including:
- SLA definition and tracking
- Uptime monitoring and availability
- Performance SLA compliance
- SLA reporting and analytics
- Alert integration for SLA breaches
- Historical SLA data analysis

Phase 17: Advanced Observability & Monitoring
"""

from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider


# ========================
# Enums
# ========================


class SLAType(str, Enum):
    """Types of SLAs."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


class SLAStatus(str, Enum):
    """SLA compliance status."""

    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class UptimeStatus(str, Enum):
    """Service uptime status."""

    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class ReportPeriod(str, Enum):
    """Report time periods."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class CheckType(str, Enum):
    """Health check types."""

    HTTP = "http"
    TCP = "tcp"
    PING = "ping"
    CUSTOM = "custom"


# ========================
# Data Classes
# ========================


@dataclass
class SLADefinition:
    """Definition of an SLA."""

    sla_id: str
    name: str
    sla_type: SLAType
    target_value: float
    warning_threshold: float  # Threshold for "at risk" status
    measurement_window: timedelta = field(default_factory=lambda: timedelta(hours=1))
    description: str = ""
    service: str = ""
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class SLAMeasurement:
    """A single SLA measurement."""

    sla_id: str
    timestamp: datetime
    value: float
    status: SLAStatus
    sample_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLAComplianceReport:
    """SLA compliance report."""

    sla_id: str
    name: str
    period_start: datetime
    period_end: datetime
    target_value: float
    actual_value: float
    compliance_percentage: float
    status: SLAStatus
    breach_count: int = 0
    total_measurements: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UptimeCheck:
    """Configuration for uptime check."""

    check_id: str
    name: str
    check_type: CheckType
    target: str
    interval: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    expected_status: int = 200  # For HTTP checks
    enabled: bool = True


@dataclass
class UptimeRecord:
    """A single uptime record."""

    check_id: str
    timestamp: datetime
    status: UptimeStatus
    response_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class UptimeSummary:
    """Summary of uptime for a service."""

    check_id: str
    name: str
    period_start: datetime
    period_end: datetime
    uptime_percentage: float
    downtime_minutes: float
    total_checks: int
    successful_checks: int
    failed_checks: int
    avg_response_time_ms: float
    incidents: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IncidentRecord:
    """Record of a service incident."""

    incident_id: str
    check_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: UptimeStatus = UptimeStatus.DOWN
    duration_minutes: float = 0.0
    error_message: str = ""
    resolution_notes: str = ""


@dataclass
class SLAAlert:
    """Alert for SLA breach or risk."""

    alert_id: str
    sla_id: str
    timestamp: datetime
    status: SLAStatus
    current_value: float
    target_value: float
    message: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


# ========================
# Uptime Tracker
# ========================


class UptimeTracker:
    """Tracks service uptime."""

    def __init__(self):
        self._checks: Dict[str, UptimeCheck] = {}
        self._records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._incidents: Dict[str, List[IncidentRecord]] = defaultdict(list)
        self._current_status: Dict[str, UptimeStatus] = {}
        self._lock = threading.Lock()

    def register_check(self, check: UptimeCheck) -> bool:
        """Register an uptime check."""
        with self._lock:
            if check.check_id in self._checks:
                return False
            self._checks[check.check_id] = check
            self._current_status[check.check_id] = UptimeStatus.UP
            return True

    def record_check(
        self,
        check_id: str,
        status: UptimeStatus,
        response_time_ms: float = 0.0,
        error: Optional[str] = None
    ) -> UptimeRecord:
        """Record an uptime check result."""
        record = UptimeRecord(
            check_id=check_id,
            timestamp=datetime.now(),
            status=status,
            response_time_ms=response_time_ms,
            error=error
        )

        with self._lock:
            self._records[check_id].append(record)
            previous_status = self._current_status.get(check_id, UptimeStatus.UP)
            self._current_status[check_id] = status

            # Handle incident tracking
            if status in (UptimeStatus.DOWN, UptimeStatus.DEGRADED):
                if previous_status == UptimeStatus.UP:
                    # New incident
                    incident = IncidentRecord(
                        incident_id=f"inc_{int(time.time() * 1000)}",
                        check_id=check_id,
                        start_time=datetime.now(),
                        status=status,
                        error_message=error or ""
                    )
                    self._incidents[check_id].append(incident)

            elif status == UptimeStatus.UP and previous_status != UptimeStatus.UP:
                # Incident resolved
                incidents = self._incidents.get(check_id, [])
                if incidents and not incidents[-1].end_time:
                    incident = incidents[-1]
                    incident.end_time = datetime.now()
                    incident.duration_minutes = (
                        incident.end_time - incident.start_time
                    ).total_seconds() / 60.0

        return record

    def get_uptime_summary(
        self,
        check_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[UptimeSummary]:
        """Get uptime summary for a check."""
        with self._lock:
            check = self._checks.get(check_id)
            if not check:
                return None

            records = list(self._records.get(check_id, []))

        if not records:
            return None

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        if not records:
            return None

        successful = sum(1 for r in records if r.status == UptimeStatus.UP)
        failed = len(records) - successful
        response_times = [r.response_time_ms for r in records if r.response_time_ms > 0]

        # Calculate downtime
        downtime_minutes = 0.0
        with self._lock:
            for incident in self._incidents.get(check_id, []):
                if start_time and incident.start_time < start_time:
                    continue
                if incident.duration_minutes:
                    downtime_minutes += incident.duration_minutes

        period_start = start_time or records[0].timestamp
        period_end = end_time or records[-1].timestamp

        return UptimeSummary(
            check_id=check_id,
            name=check.name,
            period_start=period_start,
            period_end=period_end,
            uptime_percentage=successful / len(records) * 100 if records else 100.0,
            downtime_minutes=downtime_minutes,
            total_checks=len(records),
            successful_checks=successful,
            failed_checks=failed,
            avg_response_time_ms=statistics.mean(response_times) if response_times else 0.0
        )

    def get_current_status(self, check_id: str) -> UptimeStatus:
        """Get current status for a check."""
        with self._lock:
            return self._current_status.get(check_id, UptimeStatus.UNKNOWN)

    def get_incidents(
        self,
        check_id: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[IncidentRecord]:
        """Get incident records."""
        with self._lock:
            if check_id:
                incidents = self._incidents.get(check_id, [])[:]
            else:
                incidents = []
                for inc_list in self._incidents.values():
                    incidents.extend(inc_list)

        if start_time:
            incidents = [i for i in incidents if i.start_time >= start_time]

        return sorted(incidents, key=lambda x: x.start_time, reverse=True)


# ========================
# SLA Tracker
# ========================


class SLATracker:
    """Tracks SLA compliance."""

    def __init__(self):
        self._definitions: Dict[str, SLADefinition] = {}
        self._measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._alerts: List[SLAAlert] = []
        self._lock = threading.Lock()

    def register_sla(self, definition: SLADefinition) -> bool:
        """Register an SLA definition."""
        with self._lock:
            if definition.sla_id in self._definitions:
                return False
            self._definitions[definition.sla_id] = definition
            return True

    def update_sla(self, definition: SLADefinition) -> bool:
        """Update an SLA definition."""
        with self._lock:
            if definition.sla_id not in self._definitions:
                return False
            self._definitions[definition.sla_id] = definition
            return True

    def delete_sla(self, sla_id: str) -> bool:
        """Delete an SLA definition."""
        with self._lock:
            if sla_id not in self._definitions:
                return False
            del self._definitions[sla_id]
            return True

    def record_measurement(
        self,
        sla_id: str,
        value: float,
        sample_count: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SLAMeasurement]:
        """Record an SLA measurement."""
        with self._lock:
            definition = self._definitions.get(sla_id)
            if not definition:
                return None

        status = self._evaluate_status(value, definition)

        measurement = SLAMeasurement(
            sla_id=sla_id,
            timestamp=datetime.now(),
            value=value,
            status=status,
            sample_count=sample_count,
            metadata=metadata or {}
        )

        with self._lock:
            self._measurements[sla_id].append(measurement)

        # Generate alert if needed
        if status in (SLAStatus.AT_RISK, SLAStatus.BREACHED):
            self._create_alert(measurement, definition)

        return measurement

    def _evaluate_status(
        self,
        value: float,
        definition: SLADefinition
    ) -> SLAStatus:
        """Evaluate SLA status based on value."""
        # For availability/throughput: higher is better
        # For latency/error_rate: lower is better

        if definition.sla_type in (SLAType.AVAILABILITY, SLAType.THROUGHPUT):
            if value >= definition.target_value:
                return SLAStatus.COMPLIANT
            elif value >= definition.warning_threshold:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.BREACHED
        else:  # LATENCY, ERROR_RATE
            if value <= definition.target_value:
                return SLAStatus.COMPLIANT
            elif value <= definition.warning_threshold:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.BREACHED

    def _create_alert(
        self,
        measurement: SLAMeasurement,
        definition: SLADefinition
    ) -> SLAAlert:
        """Create an SLA alert."""
        alert = SLAAlert(
            alert_id=f"sla_alert_{int(time.time() * 1000)}",
            sla_id=measurement.sla_id,
            timestamp=measurement.timestamp,
            status=measurement.status,
            current_value=measurement.value,
            target_value=definition.target_value,
            message=f"SLA '{definition.name}' is {measurement.status.value}: "
                    f"current={measurement.value:.2f}, target={definition.target_value:.2f}"
        )

        with self._lock:
            self._alerts.append(alert)

        return alert

    def get_compliance_report(
        self,
        sla_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> Optional[SLAComplianceReport]:
        """Get SLA compliance report."""
        with self._lock:
            definition = self._definitions.get(sla_id)
            if not definition:
                return None

            measurements = list(self._measurements.get(sla_id, []))

        if not measurements:
            return None

        if period_start:
            measurements = [m for m in measurements if m.timestamp >= period_start]
        if period_end:
            measurements = [m for m in measurements if m.timestamp <= period_end]

        if not measurements:
            return None

        values = [m.value for m in measurements]
        avg_value = statistics.mean(values)
        breach_count = sum(1 for m in measurements if m.status == SLAStatus.BREACHED)
        compliant_count = sum(1 for m in measurements if m.status == SLAStatus.COMPLIANT)

        compliance_percentage = compliant_count / len(measurements) * 100 if measurements else 0.0

        # Determine overall status
        if compliance_percentage >= 99.0:
            overall_status = SLAStatus.COMPLIANT
        elif compliance_percentage >= 95.0:
            overall_status = SLAStatus.AT_RISK
        else:
            overall_status = SLAStatus.BREACHED

        return SLAComplianceReport(
            sla_id=sla_id,
            name=definition.name,
            period_start=period_start or measurements[0].timestamp,
            period_end=period_end or measurements[-1].timestamp,
            target_value=definition.target_value,
            actual_value=avg_value,
            compliance_percentage=compliance_percentage,
            status=overall_status,
            breach_count=breach_count,
            total_measurements=len(measurements)
        )

    def get_current_status(self, sla_id: str) -> SLAStatus:
        """Get current SLA status."""
        with self._lock:
            measurements = list(self._measurements.get(sla_id, []))

        if not measurements:
            return SLAStatus.UNKNOWN

        return measurements[-1].status

    def get_alerts(
        self,
        sla_id: Optional[str] = None,
        acknowledged: Optional[bool] = None
    ) -> List[SLAAlert]:
        """Get SLA alerts."""
        with self._lock:
            alerts = self._alerts[:]

        if sla_id:
            alerts = [a for a in alerts if a.sla_id == sla_id]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an SLA alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    alert.acknowledged_at = datetime.now()
                    return True
            return False

    def get_all_slas(self) -> List[SLADefinition]:
        """Get all SLA definitions."""
        with self._lock:
            return list(self._definitions.values())


# ========================
# SLA Reporter
# ========================


class SLAReporter:
    """Generates SLA reports."""

    def __init__(self, sla_tracker: SLATracker, uptime_tracker: UptimeTracker):
        self._sla_tracker = sla_tracker
        self._uptime_tracker = uptime_tracker

    def generate_summary_report(
        self,
        period: ReportPeriod = ReportPeriod.DAILY
    ) -> Dict[str, Any]:
        """Generate a summary report."""
        now = datetime.now()

        if period == ReportPeriod.HOURLY:
            start_time = now - timedelta(hours=1)
        elif period == ReportPeriod.DAILY:
            start_time = now - timedelta(days=1)
        elif period == ReportPeriod.WEEKLY:
            start_time = now - timedelta(weeks=1)
        elif period == ReportPeriod.MONTHLY:
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=90)

        sla_reports = []
        for sla in self._sla_tracker.get_all_slas():
            report = self._sla_tracker.get_compliance_report(
                sla.sla_id, start_time, now
            )
            if report:
                sla_reports.append({
                    "sla_id": report.sla_id,
                    "name": report.name,
                    "status": report.status.value,
                    "compliance_percentage": report.compliance_percentage,
                    "target_value": report.target_value,
                    "actual_value": report.actual_value,
                    "breach_count": report.breach_count
                })

        return {
            "period": period.value,
            "period_start": start_time.isoformat(),
            "period_end": now.isoformat(),
            "generated_at": now.isoformat(),
            "sla_summary": sla_reports,
            "total_slas": len(sla_reports),
            "compliant_slas": sum(1 for r in sla_reports if r["status"] == "compliant"),
            "at_risk_slas": sum(1 for r in sla_reports if r["status"] == "at_risk"),
            "breached_slas": sum(1 for r in sla_reports if r["status"] == "breached")
        }

    def export_report(
        self,
        report: Dict[str, Any],
        format: str = "json"
    ) -> str:
        """Export report to specified format."""
        if format == "json":
            return json.dumps(report, indent=2, default=str)
        elif format == "markdown":
            return self._to_markdown(report)
        return json.dumps(report)

    def _to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to markdown format."""
        lines = [
            f"# SLA Report - {report['period'].capitalize()}",
            f"",
            f"**Period:** {report['period_start']} to {report['period_end']}",
            f"**Generated:** {report['generated_at']}",
            f"",
            f"## Summary",
            f"",
            f"- Total SLAs: {report['total_slas']}",
            f"- Compliant: {report['compliant_slas']}",
            f"- At Risk: {report['at_risk_slas']}",
            f"- Breached: {report['breached_slas']}",
            f"",
            f"## SLA Details",
            f"",
            f"| SLA | Status | Compliance | Target | Actual |",
            f"|-----|--------|------------|--------|--------|"
        ]

        for sla in report.get("sla_summary", []):
            lines.append(
                f"| {sla['name']} | {sla['status']} | "
                f"{sla['compliance_percentage']:.1f}% | "
                f"{sla['target_value']:.2f} | {sla['actual_value']:.2f} |"
            )

        return "\n".join(lines)


# ========================
# SLA Monitor
# ========================


class SLAMonitor:
    """Main SLA monitor coordinating all SLA operations."""

    def __init__(self):
        self._sla_tracker = SLATracker()
        self._uptime_tracker = UptimeTracker()
        self._reporter = SLAReporter(self._sla_tracker, self._uptime_tracker)
        self._lock = threading.Lock()

    # SLA Management

    def define_sla(
        self,
        sla_id: str,
        name: str,
        sla_type: SLAType,
        target_value: float,
        warning_threshold: float,
        **kwargs: Any
    ) -> SLADefinition:
        """Define a new SLA."""
        definition = SLADefinition(
            sla_id=sla_id,
            name=name,
            sla_type=sla_type,
            target_value=target_value,
            warning_threshold=warning_threshold,
            **kwargs
        )
        self._sla_tracker.register_sla(definition)
        return definition

    def record_sla_metric(
        self,
        sla_id: str,
        value: float,
        **metadata: Any
    ) -> Optional[SLAMeasurement]:
        """Record an SLA metric."""
        return self._sla_tracker.record_measurement(sla_id, value, metadata=metadata)

    def get_sla_status(self, sla_id: str) -> SLAStatus:
        """Get current SLA status."""
        return self._sla_tracker.get_current_status(sla_id)

    def get_sla_report(
        self,
        sla_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[SLAComplianceReport]:
        """Get SLA compliance report."""
        return self._sla_tracker.get_compliance_report(sla_id, start_time, end_time)

    # Uptime Management

    def register_uptime_check(
        self,
        check_id: str,
        name: str,
        target: str,
        check_type: CheckType = CheckType.HTTP,
        interval_seconds: int = 60
    ) -> UptimeCheck:
        """Register an uptime check."""
        check = UptimeCheck(
            check_id=check_id,
            name=name,
            check_type=check_type,
            target=target,
            interval=timedelta(seconds=interval_seconds)
        )
        self._uptime_tracker.register_check(check)
        return check

    def record_uptime_check(
        self,
        check_id: str,
        status: UptimeStatus,
        response_time_ms: float = 0.0,
        error: Optional[str] = None
    ) -> UptimeRecord:
        """Record an uptime check result."""
        return self._uptime_tracker.record_check(
            check_id, status, response_time_ms, error
        )

    def get_uptime_summary(
        self,
        check_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[UptimeSummary]:
        """Get uptime summary."""
        return self._uptime_tracker.get_uptime_summary(check_id, start_time, end_time)

    def get_incidents(
        self,
        check_id: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[IncidentRecord]:
        """Get incident records."""
        return self._uptime_tracker.get_incidents(check_id, start_time)

    # Reporting

    def generate_report(
        self,
        period: ReportPeriod = ReportPeriod.DAILY
    ) -> Dict[str, Any]:
        """Generate SLA report."""
        return self._reporter.generate_summary_report(period)

    def export_report(
        self,
        period: ReportPeriod = ReportPeriod.DAILY,
        format: str = "json"
    ) -> str:
        """Export SLA report."""
        report = self.generate_report(period)
        return self._reporter.export_report(report, format)

    # Alerts

    def get_active_alerts(self) -> List[SLAAlert]:
        """Get active (unacknowledged) SLA alerts."""
        return self._sla_tracker.get_alerts(acknowledged=False)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an SLA alert."""
        return self._sla_tracker.acknowledge_alert(alert_id, acknowledged_by)


# ========================
# SLA Monitor Provider
# ========================


class SLAMonitorVisionProvider(VisionProvider):
    """Vision provider with SLA monitoring integration."""

    def __init__(
        self,
        base_provider: VisionProvider,
        sla_monitor: Optional[SLAMonitor] = None
    ):
        self._base_provider = base_provider
        self._sla_monitor = sla_monitor or SLAMonitor()
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        self._setup_default_slas()

    def _setup_default_slas(self) -> None:
        """Set up default SLAs."""
        # Availability SLA: 99.9% uptime
        self._sla_monitor.define_sla(
            sla_id="vision_availability",
            name="Vision Service Availability",
            sla_type=SLAType.AVAILABILITY,
            target_value=99.9,
            warning_threshold=99.5,
            service=self.provider_name
        )

        # Latency SLA: P95 < 2000ms
        self._sla_monitor.define_sla(
            sla_id="vision_latency",
            name="Vision Service Latency",
            sla_type=SLAType.LATENCY,
            target_value=2000.0,
            warning_threshold=3000.0,
            service=self.provider_name
        )

        # Error rate SLA: < 1%
        self._sla_monitor.define_sla(
            sla_id="vision_error_rate",
            name="Vision Service Error Rate",
            sla_type=SLAType.ERROR_RATE,
            target_value=1.0,
            warning_threshold=2.0,
            service=self.provider_name
        )

        # Register uptime check
        self._sla_monitor.register_uptime_check(
            check_id="vision_health",
            name="Vision Health Check",
            target=f"{self._base_provider.provider_name}/health"
        )

    @property
    def provider_name(self) -> str:
        return f"sla_monitor_{self._base_provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        context: Optional[Dict[str, Any]] = None
    ) -> VisionDescription:
        """Analyze image with SLA tracking."""
        self._request_count += 1
        start_time = time.time()

        try:
            result = await self._base_provider.analyze_image(image_data, context)

            latency_ms = (time.time() - start_time) * 1000
            self._total_latency += latency_ms

            # Record metrics
            self._sla_monitor.record_sla_metric("vision_latency", latency_ms)

            # Calculate availability
            success_rate = (self._request_count - self._error_count) / self._request_count * 100
            self._sla_monitor.record_sla_metric("vision_availability", success_rate)

            # Calculate error rate
            error_rate = self._error_count / self._request_count * 100
            self._sla_monitor.record_sla_metric("vision_error_rate", error_rate)

            # Record uptime check
            self._sla_monitor.record_uptime_check(
                "vision_health",
                UptimeStatus.UP,
                latency_ms
            )

            return result

        except Exception as e:
            self._error_count += 1
            latency_ms = (time.time() - start_time) * 1000

            # Record failure
            self._sla_monitor.record_uptime_check(
                "vision_health",
                UptimeStatus.DOWN,
                latency_ms,
                str(e)
            )

            raise

    def get_sla_monitor(self) -> SLAMonitor:
        """Get the SLA monitor."""
        return self._sla_monitor


# ========================
# Factory Functions
# ========================


def create_sla_monitor() -> SLAMonitor:
    """Create a new SLA monitor."""
    return SLAMonitor()


def create_sla_definition(
    sla_id: str,
    name: str,
    sla_type: SLAType,
    target_value: float,
    warning_threshold: float,
    **kwargs: Any
) -> SLADefinition:
    """Create an SLA definition."""
    return SLADefinition(
        sla_id=sla_id,
        name=name,
        sla_type=sla_type,
        target_value=target_value,
        warning_threshold=warning_threshold,
        **kwargs
    )


def create_uptime_check(
    check_id: str,
    name: str,
    target: str,
    check_type: CheckType = CheckType.HTTP
) -> UptimeCheck:
    """Create an uptime check configuration."""
    return UptimeCheck(
        check_id=check_id,
        name=name,
        check_type=check_type,
        target=target
    )


def create_sla_monitor_provider(
    base_provider: VisionProvider,
    sla_monitor: Optional[SLAMonitor] = None
) -> SLAMonitorVisionProvider:
    """Create an SLA monitor vision provider."""
    return SLAMonitorVisionProvider(base_provider, sla_monitor)
