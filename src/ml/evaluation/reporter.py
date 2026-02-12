"""
Evaluation report generation.

Provides:
- Markdown report generation
- HTML report generation
- JSON export
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ml.evaluation.metrics import MetricsReport
from src.ml.evaluation.confusion import ConfusionAnalysis
from src.ml.evaluation.error_analysis import ErrorCaseCollection

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Report output format."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Model Evaluation Report"
    include_confusion_matrix: bool = True
    include_per_class_metrics: bool = True
    include_error_analysis: bool = True
    include_error_cases: bool = False  # Include individual error cases
    max_error_cases: int = 50
    top_confusion_pairs: int = 10
    show_timestamps: bool = True


class EvaluationReporter:
    """
    Generator for evaluation reports.

    Supports:
    - Markdown reports
    - HTML reports
    - JSON export
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.

        Args:
            config: Report configuration
        """
        self._config = config or ReportConfig()

    def generate(
        self,
        metrics: MetricsReport,
        confusion: Optional[ConfusionAnalysis] = None,
        errors: Optional[ErrorCaseCollection] = None,
        format: ReportFormat = ReportFormat.MARKDOWN,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate evaluation report.

        Args:
            metrics: Metrics report
            confusion: Confusion analysis
            errors: Error case collection
            format: Output format
            model_name: Name of evaluated model
            dataset_name: Name of evaluation dataset
            extra_info: Additional information to include

        Returns:
            Report string
        """
        if format == ReportFormat.MARKDOWN:
            return self._generate_markdown(
                metrics, confusion, errors, model_name, dataset_name, extra_info
            )
        elif format == ReportFormat.HTML:
            return self._generate_html(
                metrics, confusion, errors, model_name, dataset_name, extra_info
            )
        elif format == ReportFormat.JSON:
            return self._generate_json(
                metrics, confusion, errors, model_name, dataset_name, extra_info
            )
        else:
            raise ValueError(f"Unknown format: {format}")

    def save(
        self,
        report: str,
        path: Path,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> None:
        """
        Save report to file.

        Args:
            report: Report content
            path: Output path
            format: Report format
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Add appropriate extension
        if format == ReportFormat.MARKDOWN and not path.suffix:
            path = path.with_suffix(".md")
        elif format == ReportFormat.HTML and not path.suffix:
            path = path.with_suffix(".html")
        elif format == ReportFormat.JSON and not path.suffix:
            path = path.with_suffix(".json")

        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Saved evaluation report to {path}")

    def _generate_markdown(
        self,
        metrics: MetricsReport,
        confusion: Optional[ConfusionAnalysis],
        errors: Optional[ErrorCaseCollection],
        model_name: Optional[str],
        dataset_name: Optional[str],
        extra_info: Optional[Dict[str, Any]],
    ) -> str:
        """Generate Markdown report."""
        lines = []
        config = self._config

        # Title
        lines.append(f"# {config.title}")
        lines.append("")

        if config.show_timestamps:
            lines.append(f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
            lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append("")
        if model_name:
            lines.append(f"- **Model**: {model_name}")
        if dataset_name:
            lines.append(f"- **Dataset**: {dataset_name}")
        lines.append(f"- **Total Samples**: {metrics.total_samples:,}")
        lines.append(f"- **Number of Classes**: {metrics.num_classes}")
        lines.append("")

        # Key Metrics
        lines.append("## Key Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| **Accuracy** | {metrics.accuracy:.4f} |")
        lines.append(f"| **F1 (Macro)** | {metrics.f1_macro:.4f} |")
        lines.append(f"| **F1 (Weighted)** | {metrics.f1_weighted:.4f} |")
        lines.append(f"| **Precision (Macro)** | {metrics.precision_macro:.4f} |")
        lines.append(f"| **Recall (Macro)** | {metrics.recall_macro:.4f} |")
        lines.append("")

        # Per-Class Metrics
        if config.include_per_class_metrics:
            lines.append("## Per-Class Metrics")
            lines.append("")
            lines.append("| Class | Precision | Recall | F1 | Support |")
            lines.append("|-------|-----------|--------|----|---------:|")

            for label in sorted(metrics.per_class.keys()):
                cm = metrics.per_class[label]
                lines.append(
                    f"| {label} | {cm.precision:.4f} | {cm.recall:.4f} | "
                    f"{cm.f1:.4f} | {cm.support:,} |"
                )
            lines.append("")

            # Worst performing classes
            worst = metrics.get_worst_classes("f1", 5)
            if worst:
                lines.append("### Worst Performing Classes (by F1)")
                lines.append("")
                for label, f1 in worst:
                    lines.append(f"- **{label}**: F1 = {f1:.4f}")
                lines.append("")

        # Confusion Matrix
        if config.include_confusion_matrix and confusion:
            lines.append("## Confusion Matrix")
            lines.append("")
            lines.append(confusion.to_markdown_table())
            lines.append("")

            # Top confusion pairs
            if confusion.top_confusion_pairs:
                lines.append("### Top Confusion Pairs")
                lines.append("")
                lines.append("| Class A | Class B | A→B | B→A | Total |")
                lines.append("|---------|---------|-----|-----|------:|")

                for cp in confusion.top_confusion_pairs[:config.top_confusion_pairs]:
                    lines.append(
                        f"| {cp.class_a} | {cp.class_b} | {cp.a_as_b} | "
                        f"{cp.b_as_a} | {cp.total_confusion} |"
                    )
                lines.append("")

        # Error Analysis
        if config.include_error_analysis and errors:
            lines.append("## Error Analysis")
            lines.append("")
            lines.append(f"- **Total Errors**: {errors.error_count:,}")
            lines.append(f"- **Error Rate**: {errors.error_rate:.2%}")
            lines.append(f"- **High-Confidence Errors**: {len(errors.high_confidence_errors):,}")
            lines.append(f"- **Boundary Cases**: {len(errors.boundary_cases):,}")
            lines.append("")

            # Error patterns
            if errors.patterns:
                lines.append("### Detected Error Patterns")
                lines.append("")

                for pattern in errors.patterns:
                    severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(pattern.severity, "⚪")
                    lines.append(f"#### {severity_emoji} {pattern.pattern_type.replace('_', ' ').title()}")
                    lines.append("")
                    lines.append(f"**Description**: {pattern.description}")
                    lines.append("")
                    lines.append(f"- Affected Classes: {', '.join(pattern.affected_classes)}")
                    lines.append(f"- Error Count: {pattern.error_count}")
                    if pattern.suggested_action:
                        lines.append(f"- **Suggested Action**: {pattern.suggested_action}")
                    lines.append("")

            # Individual error cases
            if config.include_error_cases and errors.errors:
                lines.append("### Sample Error Cases")
                lines.append("")
                lines.append("| Sample ID | True | Predicted | Confidence |")
                lines.append("|-----------|------|-----------|------------|")

                for error in errors.errors[:config.max_error_cases]:
                    lines.append(
                        f"| {error.sample_id} | {error.true_label} | "
                        f"{error.pred_label} | {error.confidence:.4f} |"
                    )
                lines.append("")

        # Extra Info
        if extra_info:
            lines.append("## Additional Information")
            lines.append("")
            for key, value in extra_info.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        return "\n".join(lines)

    def _generate_html(
        self,
        metrics: MetricsReport,
        confusion: Optional[ConfusionAnalysis],
        errors: Optional[ErrorCaseCollection],
        model_name: Optional[str],
        dataset_name: Optional[str],
        extra_info: Optional[Dict[str, Any]],
    ) -> str:
        """Generate HTML report."""
        # Generate markdown first, then wrap in HTML
        md_content = self._generate_markdown(
            metrics, confusion, errors, model_name, dataset_name, extra_info
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self._config.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f4f4f4;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .metrics-box {{
            background: #f0f7ff;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <pre style="white-space: pre-wrap; font-family: inherit;">
{md_content}
    </pre>
</body>
</html>"""
        return html

    def _generate_json(
        self,
        metrics: MetricsReport,
        confusion: Optional[ConfusionAnalysis],
        errors: Optional[ErrorCaseCollection],
        model_name: Optional[str],
        dataset_name: Optional[str],
        extra_info: Optional[Dict[str, Any]],
    ) -> str:
        """Generate JSON report."""
        report_data = {
            "title": self._config.title,
            "generated_at": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "metrics": metrics.to_dict(),
        }

        if confusion:
            report_data["confusion_analysis"] = confusion.to_dict()

        if errors:
            if self._config.include_error_cases:
                report_data["error_analysis"] = errors.to_dict()
            else:
                report_data["error_analysis"] = errors.to_summary_dict()

        if extra_info:
            report_data["extra_info"] = extra_info

        return json.dumps(report_data, indent=2, ensure_ascii=False)
