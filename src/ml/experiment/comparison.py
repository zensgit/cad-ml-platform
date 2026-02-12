"""
Experiment comparison utilities.

Provides:
- Multi-run comparison
- Metric visualization data
- Comparison reports
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RunComparison:
    """Comparison data for a single run."""
    run_id: str
    experiment_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]  # Final/best metrics
    tags: Dict[str, str]
    duration_seconds: Optional[float]
    status: str


@dataclass
class ComparisonReport:
    """Report comparing multiple runs."""
    runs: List[RunComparison]
    param_diff: Dict[str, List[Any]]  # Params that differ across runs
    metric_summary: Dict[str, Dict[str, float]]  # metric -> {min, max, mean}
    best_runs: Dict[str, str]  # metric -> best run_id
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "runs": [
                {
                    "run_id": r.run_id,
                    "experiment_name": r.experiment_name,
                    "params": r.params,
                    "metrics": r.metrics,
                    "tags": r.tags,
                    "duration_seconds": r.duration_seconds,
                    "status": r.status,
                }
                for r in self.runs
            ],
            "param_diff": self.param_diff,
            "metric_summary": self.metric_summary,
            "best_runs": self.best_runs,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Experiment Comparison Report",
            "",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total runs compared: {len(self.runs)}",
            "",
            "## Best Runs by Metric",
            "",
        ]

        for metric, run_id in self.best_runs.items():
            summary = self.metric_summary.get(metric, {})
            best_value = summary.get("max", "N/A")
            lines.append(f"- **{metric}**: {run_id} ({best_value:.4f})")

        lines.extend([
            "",
            "## Metric Summary",
            "",
            "| Metric | Min | Max | Mean |",
            "|--------|-----|-----|------|",
        ])

        for metric, stats in self.metric_summary.items():
            lines.append(
                f"| {metric} | {stats.get('min', 'N/A'):.4f} | "
                f"{stats.get('max', 'N/A'):.4f} | {stats.get('mean', 'N/A'):.4f} |"
            )

        if self.param_diff:
            lines.extend([
                "",
                "## Parameter Differences",
                "",
                "| Parameter | Values |",
                "|-----------|--------|",
            ])

            for param, values in self.param_diff.items():
                unique_values = list(set(str(v) for v in values))
                lines.append(f"| {param} | {', '.join(unique_values)} |")

        lines.extend([
            "",
            "## Run Details",
            "",
            "| Run ID | Experiment | Duration (s) | Status |",
            "|--------|------------|--------------|--------|",
        ])

        for run in self.runs:
            duration = f"{run.duration_seconds:.1f}" if run.duration_seconds else "N/A"
            lines.append(f"| {run.run_id} | {run.experiment_name} | {duration} | {run.status} |")

        return "\n".join(lines)

    def save(self, path: Path, format: str = "json") -> None:
        """
        Save report to file.

        Args:
            path: Output path
            format: "json" or "markdown"
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format == "markdown":
            with open(path, "w") as f:
                f.write(self.to_markdown())
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved comparison report to {path}")


class ExperimentComparison:
    """
    Utility for comparing experiment runs.

    Supports:
    - Multi-run metric comparison
    - Parameter difference analysis
    - Best run identification
    - Report generation
    """

    def __init__(self):
        """Initialize comparison utility."""
        self._runs: List[RunComparison] = []

    def add_run(
        self,
        run_id: str,
        experiment_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        duration_seconds: Optional[float] = None,
        status: str = "completed",
    ) -> None:
        """
        Add a run for comparison.

        Args:
            run_id: Run identifier
            experiment_name: Experiment name
            params: Run parameters
            metrics: Final metrics
            tags: Run tags
            duration_seconds: Run duration
            status: Run status
        """
        self._runs.append(RunComparison(
            run_id=run_id,
            experiment_name=experiment_name,
            params=params,
            metrics=metrics,
            tags=tags or {},
            duration_seconds=duration_seconds,
            status=status,
        ))

    def add_run_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Add a run from dictionary.

        Args:
            data: Run data dictionary
        """
        # Extract final metrics (last value for each metric)
        final_metrics = {}
        for name, history in data.get("metrics", {}).items():
            if history:
                final_metrics[name] = history[-1].get("value", 0)

        self.add_run(
            run_id=data["run_id"],
            experiment_name=data.get("experiment_name", ""),
            params=data.get("params", {}),
            metrics=final_metrics,
            tags=data.get("tags", {}),
            duration_seconds=data.get("duration_seconds"),
            status=data.get("status", "completed"),
        )

    def clear(self) -> None:
        """Clear all runs."""
        self._runs.clear()

    def _find_param_differences(self) -> Dict[str, List[Any]]:
        """Find parameters that differ across runs."""
        if not self._runs:
            return {}

        all_params = set()
        for run in self._runs:
            all_params.update(run.params.keys())

        diff_params = {}
        for param in all_params:
            values = [run.params.get(param) for run in self._runs]
            unique_values = set(str(v) for v in values)
            if len(unique_values) > 1:
                diff_params[param] = values

        return diff_params

    def _compute_metric_summary(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for each metric."""
        if not self._runs:
            return {}

        all_metrics = set()
        for run in self._runs:
            all_metrics.update(run.metrics.keys())

        summaries = {}
        for metric in all_metrics:
            values = [run.metrics.get(metric) for run in self._runs if run.metrics.get(metric) is not None]
            if values:
                summaries[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }

        return summaries

    def _find_best_runs(self, mode: str = "max") -> Dict[str, str]:
        """
        Find best run for each metric.

        Args:
            mode: "max" or "min" for optimization direction

        Returns:
            Dict mapping metric name to best run_id
        """
        if not self._runs:
            return {}

        all_metrics = set()
        for run in self._runs:
            all_metrics.update(run.metrics.keys())

        best_runs = {}
        for metric in all_metrics:
            best_run = None
            best_value = None

            for run in self._runs:
                value = run.metrics.get(metric)
                if value is None:
                    continue

                if best_value is None:
                    best_run = run.run_id
                    best_value = value
                elif mode == "max" and value > best_value:
                    best_run = run.run_id
                    best_value = value
                elif mode == "min" and value < best_value:
                    best_run = run.run_id
                    best_value = value

            if best_run:
                best_runs[metric] = best_run

        return best_runs

    def compare(self, mode: str = "max") -> ComparisonReport:
        """
        Generate comparison report.

        Args:
            mode: "max" or "min" for determining best runs

        Returns:
            ComparisonReport
        """
        return ComparisonReport(
            runs=self._runs.copy(),
            param_diff=self._find_param_differences(),
            metric_summary=self._compute_metric_summary(),
            best_runs=self._find_best_runs(mode),
        )

    def get_best_run(
        self,
        metric: str,
        mode: str = "max",
    ) -> Optional[RunComparison]:
        """
        Get best run for a specific metric.

        Args:
            metric: Metric name
            mode: "max" or "min"

        Returns:
            Best RunComparison or None
        """
        best_runs = self._find_best_runs(mode)
        best_run_id = best_runs.get(metric)

        if best_run_id is None:
            return None

        for run in self._runs:
            if run.run_id == best_run_id:
                return run

        return None

    def rank_runs(
        self,
        metric: str,
        mode: str = "max",
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank runs by metric.

        Args:
            metric: Metric to rank by
            mode: "max" or "min"
            top_k: Return only top K

        Returns:
            List of (run_id, metric_value) tuples
        """
        ranked = []
        for run in self._runs:
            value = run.metrics.get(metric)
            if value is not None:
                ranked.append((run.run_id, value))

        reverse = mode == "max"
        ranked.sort(key=lambda x: x[1], reverse=reverse)

        if top_k:
            ranked = ranked[:top_k]

        return ranked

    def __len__(self) -> int:
        return len(self._runs)

    def __repr__(self) -> str:
        return f"ExperimentComparison(runs={len(self._runs)})"
