"""
Reporting Module - Phase 12.

Provides report generation capabilities including dashboard data,
export functionality, visualization support, and scheduled reports.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
import json

from .base import VisionProvider, VisionDescription


# ============================================================================
# Enums
# ============================================================================


class ReportType(Enum):
    """Types of reports."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    COMPARISON = "comparison"
    TREND = "trend"
    AUDIT = "audit"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"


class ChartType(Enum):
    """Types of charts for visualization."""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TABLE = "table"


class ReportStatus(Enum):
    """Report generation status."""

    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ScheduleFrequency(Enum):
    """Report schedule frequency."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DataSeries:
    """A series of data for charts."""

    name: str
    values: List[float]
    labels: List[str] = field(default_factory=list)
    color: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartConfig:
    """Configuration for a chart."""

    chart_id: str
    chart_type: ChartType
    title: str
    series: List[DataSeries] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    width: int = 800
    height: int = 400
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSection:
    """A section of a report."""

    section_id: str
    title: str
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    charts: List[ChartConfig] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    order: int = 0


@dataclass
class ReportConfig:
    """Configuration for a report."""

    report_id: str
    name: str
    report_type: ReportType = ReportType.SUMMARY
    format: ReportFormat = ReportFormat.JSON
    sections: List[ReportSection] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """A generated report."""

    report_id: str
    name: str
    report_type: ReportType
    status: ReportStatus = ReportStatus.PENDING
    content: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardWidget:
    """A widget on a dashboard."""

    widget_id: str
    title: str
    widget_type: str  # metric, chart, table, etc.
    data: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)
    refresh_interval_seconds: int = 60


@dataclass
class Dashboard:
    """A dashboard configuration."""

    dashboard_id: str
    name: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScheduledReport:
    """A scheduled report configuration."""

    schedule_id: str
    report_config: ReportConfig
    frequency: ScheduleFrequency
    recipients: List[str] = field(default_factory=list)
    next_run: datetime = field(default_factory=datetime.utcnow)
    last_run: Optional[datetime] = None
    enabled: bool = True


@dataclass
class ExportResult:
    """Result of exporting a report."""

    report_id: str
    format: ReportFormat
    content: str
    file_name: str
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# Report Builder
# ============================================================================


class ReportBuilder:
    """Builder for creating reports."""

    def __init__(self, report_id: str, name: str) -> None:
        self._report_id = report_id
        self._name = name
        self._report_type = ReportType.SUMMARY
        self._format = ReportFormat.JSON
        self._sections: List[ReportSection] = []
        self._filters: Dict[str, Any] = {}
        self._parameters: Dict[str, Any] = {}

    def report_type(self, report_type: ReportType) -> "ReportBuilder":
        """Set report type."""
        self._report_type = report_type
        return self

    def format(self, fmt: ReportFormat) -> "ReportBuilder":
        """Set report format."""
        self._format = fmt
        return self

    def add_section(
        self,
        section_id: str,
        title: str,
        content: str = "",
        data: Optional[Dict[str, Any]] = None,
    ) -> "ReportBuilder":
        """Add a section."""
        section = ReportSection(
            section_id=section_id,
            title=title,
            content=content,
            data=data or {},
            order=len(self._sections),
        )
        self._sections.append(section)
        return self

    def add_chart(
        self,
        section_id: str,
        chart_config: ChartConfig,
    ) -> "ReportBuilder":
        """Add a chart to a section."""
        for section in self._sections:
            if section.section_id == section_id:
                section.charts.append(chart_config)
                break
        return self

    def add_table(
        self,
        section_id: str,
        table_data: Dict[str, Any],
    ) -> "ReportBuilder":
        """Add a table to a section."""
        for section in self._sections:
            if section.section_id == section_id:
                section.tables.append(table_data)
                break
        return self

    def filter(self, key: str, value: Any) -> "ReportBuilder":
        """Add a filter."""
        self._filters[key] = value
        return self

    def parameter(self, key: str, value: Any) -> "ReportBuilder":
        """Add a parameter."""
        self._parameters[key] = value
        return self

    def build(self) -> ReportConfig:
        """Build the report configuration."""
        return ReportConfig(
            report_id=self._report_id,
            name=self._name,
            report_type=self._report_type,
            format=self._format,
            sections=self._sections,
            filters=self._filters,
            parameters=self._parameters,
        )


# ============================================================================
# Report Formatter
# ============================================================================


class ReportFormatter(ABC):
    """Abstract report formatter."""

    @abstractmethod
    def format(self, report: Report) -> str:
        """Format a report."""
        pass


class JSONFormatter(ReportFormatter):
    """JSON report formatter."""

    def format(self, report: Report) -> str:
        """Format report as JSON."""
        data = {
            "report_id": report.report_id,
            "name": report.name,
            "type": report.report_type.value,
            "status": report.status.value,
            "created_at": report.created_at.isoformat(),
            "sections": [
                {
                    "id": s.section_id,
                    "title": s.title,
                    "content": s.content,
                    "data": s.data,
                    "charts": [
                        {
                            "id": c.chart_id,
                            "type": c.chart_type.value,
                            "title": c.title,
                            "series": [
                                {"name": ser.name, "values": ser.values}
                                for ser in c.series
                            ],
                        }
                        for c in s.charts
                    ],
                    "tables": s.tables,
                }
                for s in report.sections
            ],
            "metadata": report.metadata,
        }
        return json.dumps(data, indent=2)


class MarkdownFormatter(ReportFormatter):
    """Markdown report formatter."""

    def format(self, report: Report) -> str:
        """Format report as Markdown."""
        lines = [
            f"# {report.name}",
            "",
            f"**Report ID:** {report.report_id}",
            f"**Type:** {report.report_type.value}",
            f"**Generated:** {report.created_at.isoformat()}",
            "",
        ]

        for section in sorted(report.sections, key=lambda s: s.order):
            lines.append(f"## {section.title}")
            lines.append("")

            if section.content:
                lines.append(section.content)
                lines.append("")

            if section.data:
                lines.append("### Data")
                for key, value in section.data.items():
                    lines.append(f"- **{key}:** {value}")
                lines.append("")

            for table in section.tables:
                if "headers" in table and "rows" in table:
                    headers = table["headers"]
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in table["rows"]:
                        lines.append("| " + " | ".join(str(c) for c in row) + " |")
                    lines.append("")

        return "\n".join(lines)


class CSVFormatter(ReportFormatter):
    """CSV report formatter."""

    def format(self, report: Report) -> str:
        """Format report as CSV."""
        lines = []

        for section in report.sections:
            for table in section.tables:
                if "headers" in table and "rows" in table:
                    lines.append(",".join(table["headers"]))
                    for row in table["rows"]:
                        lines.append(",".join(str(c) for c in row))
                    lines.append("")

        return "\n".join(lines)


class HTMLFormatter(ReportFormatter):
    """HTML report formatter."""

    def format(self, report: Report) -> str:
        """Format report as HTML."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{report.name}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f4f4f4; }",
            "h1 { color: #333; }",
            "h2 { color: #666; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{report.name}</h1>",
            f"<p><strong>Report ID:</strong> {report.report_id}</p>",
            f"<p><strong>Generated:</strong> {report.created_at.isoformat()}</p>",
        ]

        for section in sorted(report.sections, key=lambda s: s.order):
            html.append(f"<h2>{section.title}</h2>")

            if section.content:
                html.append(f"<p>{section.content}</p>")

            if section.data:
                html.append("<ul>")
                for key, value in section.data.items():
                    html.append(f"<li><strong>{key}:</strong> {value}</li>")
                html.append("</ul>")

            for table in section.tables:
                if "headers" in table and "rows" in table:
                    html.append("<table>")
                    html.append("<tr>")
                    for h in table["headers"]:
                        html.append(f"<th>{h}</th>")
                    html.append("</tr>")
                    for row in table["rows"]:
                        html.append("<tr>")
                        for cell in row:
                            html.append(f"<td>{cell}</td>")
                        html.append("</tr>")
                    html.append("</table>")

        html.extend(["</body>", "</html>"])
        return "\n".join(html)


# ============================================================================
# Report Generator
# ============================================================================


class ReportGenerator:
    """Generates reports from configurations."""

    def __init__(self) -> None:
        self._formatters: Dict[ReportFormat, ReportFormatter] = {
            ReportFormat.JSON: JSONFormatter(),
            ReportFormat.MARKDOWN: MarkdownFormatter(),
            ReportFormat.CSV: CSVFormatter(),
            ReportFormat.HTML: HTMLFormatter(),
        }
        self._data_providers: Dict[str, Callable[[], Dict[str, Any]]] = {}

    def register_formatter(
        self,
        fmt: ReportFormat,
        formatter: ReportFormatter,
    ) -> None:
        """Register a custom formatter."""
        self._formatters[fmt] = formatter

    def register_data_provider(
        self,
        name: str,
        provider: Callable[[], Dict[str, Any]],
    ) -> None:
        """Register a data provider."""
        self._data_providers[name] = provider

    def generate(self, config: ReportConfig) -> Report:
        """Generate a report from configuration."""
        report = Report(
            report_id=config.report_id,
            name=config.name,
            report_type=config.report_type,
            status=ReportStatus.GENERATING,
            sections=config.sections.copy(),
        )

        # Enrich sections with data providers
        for section in report.sections:
            for provider_name, provider in self._data_providers.items():
                if provider_name in config.parameters:
                    section.data.update(provider())

        # Format content
        formatter = self._formatters.get(config.format)
        if formatter:
            report.content = formatter.format(report)

        report.status = ReportStatus.COMPLETED
        report.completed_at = datetime.utcnow()

        return report

    def export(self, report: Report, fmt: ReportFormat) -> ExportResult:
        """Export a report in specified format."""
        formatter = self._formatters.get(fmt)
        if not formatter:
            raise ValueError(f"No formatter for format: {fmt}")

        content = formatter.format(report)
        extensions = {
            ReportFormat.JSON: "json",
            ReportFormat.CSV: "csv",
            ReportFormat.HTML: "html",
            ReportFormat.MARKDOWN: "md",
            ReportFormat.PDF: "pdf",
        }

        return ExportResult(
            report_id=report.report_id,
            format=fmt,
            content=content,
            file_name=f"{report.report_id}.{extensions.get(fmt, 'txt')}",
            size_bytes=len(content.encode()),
        )


# ============================================================================
# Dashboard Manager
# ============================================================================


class DashboardManager:
    """Manages dashboards."""

    def __init__(self) -> None:
        self._dashboards: Dict[str, Dashboard] = {}
        self._widget_providers: Dict[str, Callable[[], Dict[str, Any]]] = {}

    def create_dashboard(
        self,
        dashboard_id: str,
        name: str,
    ) -> Dashboard:
        """Create a dashboard."""
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
        )
        self._dashboards[dashboard_id] = dashboard
        return dashboard

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get a dashboard."""
        return self._dashboards.get(dashboard_id)

    def add_widget(
        self,
        dashboard_id: str,
        widget: DashboardWidget,
    ) -> bool:
        """Add a widget to a dashboard."""
        dashboard = self._dashboards.get(dashboard_id)
        if dashboard:
            dashboard.widgets.append(widget)
            dashboard.updated_at = datetime.utcnow()
            return True
        return False

    def remove_widget(
        self,
        dashboard_id: str,
        widget_id: str,
    ) -> bool:
        """Remove a widget from a dashboard."""
        dashboard = self._dashboards.get(dashboard_id)
        if dashboard:
            dashboard.widgets = [
                w for w in dashboard.widgets if w.widget_id != widget_id
            ]
            dashboard.updated_at = datetime.utcnow()
            return True
        return False

    def register_widget_provider(
        self,
        widget_type: str,
        provider: Callable[[], Dict[str, Any]],
    ) -> None:
        """Register a widget data provider."""
        self._widget_providers[widget_type] = provider

    def refresh_widget(self, widget: DashboardWidget) -> DashboardWidget:
        """Refresh widget data."""
        provider = self._widget_providers.get(widget.widget_type)
        if provider:
            widget.data = provider()
        return widget

    def list_dashboards(self) -> List[Dashboard]:
        """List all dashboards."""
        return list(self._dashboards.values())


# ============================================================================
# Report Scheduler
# ============================================================================


class ReportScheduler:
    """Schedules and manages report generation."""

    def __init__(self, generator: ReportGenerator) -> None:
        self._generator = generator
        self._schedules: Dict[str, ScheduledReport] = {}

    def schedule(
        self,
        schedule_id: str,
        config: ReportConfig,
        frequency: ScheduleFrequency,
        recipients: Optional[List[str]] = None,
    ) -> ScheduledReport:
        """Schedule a report."""
        scheduled = ScheduledReport(
            schedule_id=schedule_id,
            report_config=config,
            frequency=frequency,
            recipients=recipients or [],
            next_run=self._calculate_next_run(frequency),
        )
        self._schedules[schedule_id] = scheduled
        return scheduled

    def _calculate_next_run(self, frequency: ScheduleFrequency) -> datetime:
        """Calculate next run time."""
        now = datetime.utcnow()
        if frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif frequency == ScheduleFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == ScheduleFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == ScheduleFrequency.MONTHLY:
            return now + timedelta(days=30)
        return now

    def get_schedule(self, schedule_id: str) -> Optional[ScheduledReport]:
        """Get a schedule."""
        return self._schedules.get(schedule_id)

    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule."""
        if schedule_id in self._schedules:
            self._schedules[schedule_id].enabled = False
            return True
        return False

    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule."""
        if schedule_id in self._schedules:
            self._schedules[schedule_id].enabled = True
            return True
        return False

    def run_due_reports(self) -> List[Report]:
        """Run all due reports."""
        now = datetime.utcnow()
        reports: List[Report] = []

        for scheduled in self._schedules.values():
            if scheduled.enabled and scheduled.next_run <= now:
                report = self._generator.generate(scheduled.report_config)
                reports.append(report)

                scheduled.last_run = now
                scheduled.next_run = self._calculate_next_run(scheduled.frequency)

        return reports

    def list_schedules(self) -> List[ScheduledReport]:
        """List all schedules."""
        return list(self._schedules.values())


# ============================================================================
# Reporting Vision Provider
# ============================================================================


class ReportingVisionProvider(VisionProvider):
    """Vision provider with reporting capabilities."""

    def __init__(
        self,
        provider: VisionProvider,
        generator: ReportGenerator,
    ) -> None:
        self._provider = provider
        self._generator = generator
        self._analysis_history: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"reporting_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image and record for reporting."""
        result = await self._provider.analyze_image(image_data, prompt, **kwargs)

        # Record analysis
        self._analysis_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": result.confidence,
            "summary_length": len(result.summary),
            "detail_count": len(result.details),
        })

        return result

    def generate_report(self, report_type: ReportType = ReportType.SUMMARY) -> Report:
        """Generate a report from analysis history."""
        builder = ReportBuilder("vision_report", "Vision Analysis Report")
        builder.report_type(report_type)

        # Add summary section
        total = len(self._analysis_history)
        avg_confidence = (
            sum(h["confidence"] for h in self._analysis_history) / total
            if total > 0
            else 0
        )

        builder.add_section(
            "summary",
            "Summary",
            f"Total analyses: {total}",
            data={
                "total_analyses": total,
                "average_confidence": round(avg_confidence, 4),
            },
        )

        # Add history table
        if self._analysis_history:
            builder.add_table(
                "summary",
                {
                    "headers": ["Timestamp", "Confidence", "Summary Length"],
                    "rows": [
                        [h["timestamp"], h["confidence"], h["summary_length"]]
                        for h in self._analysis_history[-10:]  # Last 10
                    ],
                },
            )

        config = builder.build()
        return self._generator.generate(config)

    def get_generator(self) -> ReportGenerator:
        """Get the report generator."""
        return self._generator


# ============================================================================
# Factory Functions
# ============================================================================


def create_report_generator() -> ReportGenerator:
    """Create a report generator."""
    return ReportGenerator()


def create_report_builder(report_id: str, name: str) -> ReportBuilder:
    """Create a report builder."""
    return ReportBuilder(report_id, name)


def create_dashboard_manager() -> DashboardManager:
    """Create a dashboard manager."""
    return DashboardManager()


def create_report_scheduler(generator: ReportGenerator) -> ReportScheduler:
    """Create a report scheduler."""
    return ReportScheduler(generator)


def create_reporting_provider(
    provider: VisionProvider,
    generator: Optional[ReportGenerator] = None,
) -> ReportingVisionProvider:
    """Create a reporting vision provider."""
    return ReportingVisionProvider(
        provider=provider,
        generator=generator or create_report_generator(),
    )


def create_chart_config(
    chart_id: str,
    chart_type: ChartType,
    title: str,
    series: Optional[List[DataSeries]] = None,
) -> ChartConfig:
    """Create a chart configuration."""
    return ChartConfig(
        chart_id=chart_id,
        chart_type=chart_type,
        title=title,
        series=series or [],
    )


def create_data_series(
    name: str,
    values: List[float],
    labels: Optional[List[str]] = None,
) -> DataSeries:
    """Create a data series."""
    return DataSeries(
        name=name,
        values=values,
        labels=labels or [],
    )


def create_dashboard_widget(
    widget_id: str,
    title: str,
    widget_type: str,
    data: Optional[Dict[str, Any]] = None,
) -> DashboardWidget:
    """Create a dashboard widget."""
    return DashboardWidget(
        widget_id=widget_id,
        title=title,
        widget_type=widget_type,
        data=data or {},
    )
