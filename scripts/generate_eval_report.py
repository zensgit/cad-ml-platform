#!/usr/bin/env python3
"""
Generate Static HTML Evaluation Report

Creates a standalone HTML report from evaluation history JSON files.
Report includes:
- Latest Combined Score summary
- Historical evaluation table
- Embedded trend charts (base64 inline)
- Health check links

Usage:
    python3 scripts/generate_eval_report.py
    python3 scripts/generate_eval_report.py --out reports/eval_history/report
    python3 scripts/generate_eval_report.py --redact-commit  # Hide commit hashes
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from scripts.eval_report_data_helpers import (
        load_plot_base64_assets,
    )
    from scripts.eval_signal_reporting_helpers import (
        build_eval_signal_report_context,
        eval_signal_report_rows,
        load_eval_signal_reporting_assets,
        load_eval_signal_reporting_summary,
    )
    from scripts.history_sequence_reporting_helpers import (
        build_history_sequence_report_context,
        load_history_sequence_reporting_assets,
    )
except ImportError:
    from eval_report_data_helpers import (  # type: ignore
        load_plot_base64_assets,
    )
    from eval_signal_reporting_helpers import (  # type: ignore
        build_eval_signal_report_context,
        eval_signal_report_rows,
        load_eval_signal_reporting_assets,
        load_eval_signal_reporting_summary,
    )
    from history_sequence_reporting_helpers import (  # type: ignore
        build_history_sequence_report_context,
        load_history_sequence_reporting_assets,
    )


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
HISTORY_DIR = PROJECT_ROOT / "reports" / "eval_history"
PLOTS_DIR = HISTORY_DIR / "plots"
DEFAULT_OUTPUT_DIR = HISTORY_DIR / "report"

def get_git_info() -> Dict[str, str]:
    """Get current git branch and commit."""
    info = {"branch": "unknown", "commit": "unknown", "tag": None}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            info["tag"] = result.stdout.strip()
    except Exception:
        pass
    return info

def generate_css() -> str:
    """Generate inline CSS for the report."""
    return """
    <style>
        :root {
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #ea580c;
            --danger-color: #dc2626;
            --text-color: #1f2937;
            --bg-color: #f9fafb;
            --card-bg: #ffffff;
            --border-color: #e5e7eb;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        h1 {
            color: var(--primary-color);
            margin-bottom: 10px;
            font-size: 2em;
        }

        h2 {
            color: var(--text-color);
            margin: 30px 0 15px 0;
            font-size: 1.5em;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 5px;
        }

        .summary-card {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }

        .score-large {
            font-size: 3em;
            font-weight: bold;
            color: var(--primary-color);
            margin: 10px 0;
        }

        .score-good { color: var(--success-color); }
        .score-warning { color: var(--warning-color); }
        .score-bad { color: var(--danger-color); }

        .meta-info {
            color: #6b7280;
            font-size: 0.9em;
            margin: 5px 0;
        }

        .meta-info code {
            background: #e5e7eb;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        th {
            background: #f3f4f6;
            font-weight: 600;
            color: var(--text-color);
        }

        tr:hover {
            background: #f9fafb;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .trend-section {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }

        .trend-image {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px 0;
        }

        .links {
            background: var(--card-bg);
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }

        .links a {
            color: var(--primary-color);
            text-decoration: none;
            margin-right: 20px;
        }

        .links a:hover {
            text-decoration: underline;
        }

        .empty-notice {
            background: #fef3c7;
            border: 1px solid #fcd34d;
            border-radius: 8px;
            padding: 20px;
            color: #92400e;
            text-align: center;
        }

        .footer {
            text-align: center;
            color: #9ca3af;
            font-size: 0.85em;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }

        @media (max-width: 768px) {
            body { padding: 10px; }
            .score-large { font-size: 2em; }
            table { font-size: 0.9em; }
            th, td { padding: 8px 10px; }
        }
    </style>
    """


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ts


def get_score_class(score: float) -> str:
    """Get CSS class based on score value."""
    if score >= 0.85:
        return "score-good"
    elif score >= 0.70:
        return "score-warning"
    else:
        return "score-bad"


def generate_html_report(
    combined_history: List[Dict[str, Any]],
    ocr_history: List[Dict[str, Any]],
    eval_signal_context: Dict[str, Any],
    combined_trend_b64: Optional[str],
    ocr_trend_b64: Optional[str],
    history_sequence_bundle: Optional[Dict[str, Any]],
    history_sequence_summary: Optional[Dict[str, Any]],
    history_sequence_compare: Optional[Dict[str, Any]],
    history_sequence_trend_b64: Optional[str],
    history_sequence_surface_trend_b64: Optional[str],
    git_info: Dict[str, str],
    redact_commit: bool = False,
    redact_branch: bool = False
) -> str:
    """Generate the complete HTML report."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    latest = (
        eval_signal_context.get("latest_combined_run")
        if isinstance(eval_signal_context.get("latest_combined_run"), dict)
        else None
    )

    # Start HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAD ML Platform - Evaluation Report</title>
    {generate_css()}
</head>
<body>
    <div class="container">
        <h1>CAD ML Platform - Evaluation Report</h1>
        <p class="meta-info">Generated: {now}</p>
"""

    # Summary Card
    if latest:
        c = latest.get("combined", {})
        score = c.get("combined_score", 0)
        v_score = c.get("vision_score", 0)
        o_score = c.get("ocr_score", 0)
        v_weight = c.get("vision_weight", 0.5)
        o_weight = c.get("ocr_weight", 0.5)

        branch = latest.get("branch", "unknown")
        commit = latest.get("commit", "unknown")
        ts = latest.get("timestamp", "unknown")

        if redact_branch:
            branch = "[redacted]"
        if redact_commit:
            commit = "[redacted]"

        score_class = get_score_class(score)

        html += f"""
        <div class="summary-card">
            <h2>Latest Combined Score</h2>
            <div class="score-large {score_class}">{score:.3f}</div>
            <p class="meta-info">
                Vision Score: <strong>{v_score:.3f}</strong> (weight: {v_weight:.0%}) |
                OCR Score: <strong>{o_score:.3f}</strong> (weight: {o_weight:.0%})
            </p>
            <p class="meta-info">
                Branch: <code>{branch}</code> |
                Commit: <code>{commit}</code> |
                Time: {format_timestamp(ts)}
            </p>
        </div>
"""
    else:
        html += """
        <div class="empty-notice">
            <strong>No evaluation history found.</strong><br>
            Run <code>make eval-combined-save</code> to generate evaluation data.
        </div>
"""

    # Health Links
    html += """
        <div class="links">
            <strong>Health Endpoints:</strong>
            <a href="http://localhost:8000/health">/health</a>
            <a href="http://localhost:8000/api/v1/vision/health">/api/v1/vision/health</a>
            <a href="http://localhost:8000/docs">/docs</a>
        </div>
"""

    # Trend Charts (embedded as base64)
    html += """
        <h2>Evaluation Trends</h2>
        <div class="trend-section">
"""

    # Combined trend
    if combined_trend_b64:
        html += f"""
            <h3>Combined Score Trend</h3>
            <img src="{combined_trend_b64}" alt="Combined Score Trend" class="trend-image">
"""
    else:
        html += """
            <p class="meta-info">Combined trend chart not available. Run <code>make eval-trend</code> to generate.</p>
"""

    # OCR trend
    if ocr_trend_b64:
        html += f"""
            <h3>OCR Metrics Trend</h3>
            <img src="{ocr_trend_b64}" alt="OCR Metrics Trend" class="trend-image">
"""
    else:
        html += """
            <p class="meta-info">OCR trend chart not available. Run <code>make eval-trend</code> to generate.</p>
"""

    html += """
        </div>
"""

    html += """
        <h2>History Sequence Reporting</h2>
        <div class="trend-section">
"""
    history_sequence_context = build_history_sequence_report_context(
        history_sequence_bundle,
        history_sequence_summary,
        history_sequence_compare,
    )
    if history_sequence_context["available"]:
        html += f"""
            <p class="meta-info">
                Reports: <strong>{history_sequence_context['report_count']}</strong> |
                Mean Accuracy: <strong>{history_sequence_context['mean_accuracy_overall']:.3f}</strong> |
                Mean Macro F1: <strong>{history_sequence_context['mean_macro_f1_overall']:.3f}</strong> |
                Mean Named Explainability: <strong>{history_sequence_context['mean_named_command_explainability_rate']:.3f}</strong>
            </p>
            <p class="meta-info">
                Latest Surface: <code>{history_sequence_context['latest_sequence_surface_kind']}</code> |
                Vocabulary: <code>{history_sequence_context['latest_named_command_vocabulary_kind']}</code> |
                Best Surface Key: <code>{history_sequence_context['best_surface_key']}</code>
            </p>
"""
        if history_sequence_trend_b64:
            html += f"""
            <h3>History Sequence Trend</h3>
            <img src="{history_sequence_trend_b64}" alt="History Sequence Trend" class="trend-image">
"""
        else:
            html += """
            <p class="meta-info">History-sequence trend chart not available. Run <code>make eval-history</code> or <code>make history-sequence-reporting-bundle</code> to generate.</p>
"""
        if history_sequence_surface_trend_b64:
            html += f"""
            <h3>History Sequence Surface Trend</h3>
            <img src="{history_sequence_surface_trend_b64}" alt="History Sequence Surface Trend" class="trend-image">
"""
        if history_sequence_context["leaderboard_rows"]:
            html += """
            <h3>Top History Sequence Surfaces</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Surface Key</th>
                        <th>Reports</th>
                        <th>Mean Accuracy</th>
                        <th>Mean Macro F1</th>
                        <th>Named Explainability</th>
                    </tr>
                </thead>
                <tbody>
"""
            for row in history_sequence_context["leaderboard_rows"]:
                html += f"""
                    <tr>
                        <td>{row['rank']}</td>
                        <td><code>{row['surface_key']}</code></td>
                        <td>{row['report_count']}</td>
                        <td>{row['mean_accuracy_overall']:.3f}</td>
                        <td>{row['mean_macro_f1_overall']:.3f}</td>
                        <td>{row['mean_named_explainability_rate']:.3f}</td>
                    </tr>
"""
            html += """
                </tbody>
            </table>
"""
    else:
        html += """
            <p class="meta-info">History-sequence reporting bundle not available. Run <code>make eval-history</code> or <code>make history-sequence-reporting-bundle</code> to materialize the canonical summary root.</p>
"""

    html += """
        </div>
"""

    # Hybrid Blind Reporting block
    hybrid_blind_report_count = eval_signal_context.get("hybrid_blind_report_count", 0)
    if hybrid_blind_report_count > 0:
        aggregate = (
            eval_signal_context.get("aggregate_metrics")
            if isinstance(eval_signal_context.get("aggregate_metrics"), dict)
            else {}
        )
        html += """
        <h2>Hybrid Blind Reporting</h2>
        <div class="summary-card">
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
"""
        html += f"""
                    <tr><td>Report count</td><td>{hybrid_blind_report_count}</td></tr>
                    <tr><td>Mean hybrid accuracy</td><td>{aggregate.get('hybrid_blind_accuracy_mean', 0.0):.4f}</td></tr>
                    <tr><td>Mean graph2d accuracy</td><td>{aggregate.get('hybrid_blind_graph2d_accuracy_mean', 0.0):.4f}</td></tr>
                    <tr><td>Mean gain vs graph2d</td><td>{aggregate.get('hybrid_blind_gain_mean', 0.0):.4f}</td></tr>
                    <tr><td>Mean weak-label coverage</td><td>{aggregate.get('hybrid_blind_coverage_mean', 0.0):.4f}</td></tr>
                    <tr><td>Latest label slice count</td><td>{aggregate.get('hybrid_blind_label_slice_count_latest', 0)}</td></tr>
                    <tr><td>Latest family slice count</td><td>{aggregate.get('hybrid_blind_family_slice_count_latest', 0)}</td></tr>
"""
        html += """
                </tbody>
            </table>
        </div>
"""

    # Combined History Table
    if combined_history:
        html += """
        <h2>Combined Evaluation History</h2>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Combined Score</th>
                    <th>Vision</th>
                    <th>OCR</th>
                    <th>Weights (V/O)</th>
                    <th>Branch</th>
                    <th>Commit</th>
                </tr>
            </thead>
            <tbody>
"""
        for entry in reversed(combined_history[-20:]):
            c = entry.get("combined", {})
            score = c.get("combined_score", 0)
            v_score = c.get("vision_score", 0)
            o_score = c.get("ocr_score", 0)
            v_weight = c.get("vision_weight", 0.5)
            o_weight = c.get("ocr_weight", 0.5)

            branch = entry.get("branch", "unknown")
            commit = entry.get("commit", "unknown")
            ts = entry.get("timestamp", "unknown")

            if redact_branch:
                branch = "[redacted]"
            if redact_commit:
                commit = "[redacted]"

            score_class = get_score_class(score)

            html += f"""
                <tr>
                    <td>{format_timestamp(ts)}</td>
                    <td class="{score_class}"><strong>{score:.3f}</strong></td>
                    <td>{v_score:.3f}</td>
                    <td>{o_score:.3f}</td>
                    <td>{v_weight:.0%}/{o_weight:.0%}</td>
                    <td><code>{branch}</code></td>
                    <td><code>{commit}</code></td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""

        if len(combined_history) > 20:
            html += f"""
        <p class="meta-info">Showing 20 of {len(combined_history)} entries.</p>
"""

    # OCR History Table (if available)
    if ocr_history:
        html += """
        <h2>OCR-Only Evaluation History</h2>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Dimension Recall</th>
                    <th>Brier Score</th>
                    <th>Edge F1</th>
                    <th>Branch</th>
                    <th>Commit</th>
                </tr>
            </thead>
            <tbody>
"""
        for entry in reversed(ocr_history[-10:]):
            m = entry.get("metrics", {})
            branch = entry.get("branch", "unknown")
            commit = entry.get("commit", "unknown")
            ts = entry.get("timestamp", "unknown")

            if redact_branch:
                branch = "[redacted]"
            if redact_commit:
                commit = "[redacted]"

            html += f"""
                <tr>
                    <td>{format_timestamp(ts)}</td>
                    <td>{m.get('dimension_recall', 0):.3f}</td>
                    <td>{m.get('brier_score', 0):.3f}</td>
                    <td>{m.get('edge_f1', 0):.3f}</td>
                    <td><code>{branch}</code></td>
                    <td><code>{commit}</code></td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""

    # Footer
    current_branch = git_info.get("branch", "unknown")
    current_commit = git_info.get("commit", "unknown")
    if redact_branch:
        current_branch = "[redacted]"
    if redact_commit:
        current_commit = "[redacted]"

    html += f"""
        <div class="footer">
            <p>CAD ML Platform Evaluation Report</p>
            <p>Current: {current_branch} @ {current_commit}</p>
            <p>Generated with <code>make eval-report</code></p>
        </div>
    </div>
</body>
</html>
"""

    return html


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Generate static HTML evaluation report",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--history-dir",
        type=str,
        default=str(HISTORY_DIR),
        help=f"History directory (default: {HISTORY_DIR})"
    )

    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )

    parser.add_argument(
        "--redact-commit",
        action="store_true",
        help="Hide commit hashes in report (privacy/compliance)"
    )

    parser.add_argument(
        "--redact-branch",
        action="store_true",
        help="Hide branch names in report (privacy/compliance)"
    )

    args = parser.parse_args(argv)

    history_dir = Path(args.history_dir)
    plots_dir = history_dir / "plots"

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Evaluation Report...")
    print(f"  History directory: {history_dir}")
    print(f"  Output directory: {output_dir}")

    # Load canonical eval-signal summary and report rows (bundle-first)
    print("  Loading eval-signal canonical summary (bundle-first)...")
    eval_signal_bundle, eval_signal_summary = load_eval_signal_reporting_assets(history_dir)
    if eval_signal_summary is None:
        eval_signal_summary = load_eval_signal_reporting_summary(history_dir)
    eval_signal_context = build_eval_signal_report_context(
        eval_signal_summary,
        history_dir=history_dir,
    )
    combined_history = eval_signal_report_rows(
        eval_signal_summary,
        history_dir=history_dir,
        report_type="combined",
    )
    print(f"    Found {len(combined_history)} combined evaluation records")

    ocr_history = eval_signal_report_rows(
        eval_signal_summary,
        history_dir=history_dir,
        report_type="ocr",
    )
    print(f"    Found {len(ocr_history)} OCR-only evaluation records")

    print("  Loading history-sequence reporting bundle...")
    history_sequence_bundle, history_sequence_summary, history_sequence_compare = (
        load_history_sequence_reporting_assets(history_dir)
    )
    plot_assets = load_plot_base64_assets(
        plots_dir,
        {
            "combined": "combined_trend.png",
            "ocr": "ocr_trend.png",
            "history_sequence": "history_sequence_trend.png",
            "history_sequence_surface": "history_sequence_surface_trend.png",
        },
    )
    history_sequence_trend_b64 = plot_assets.get("history_sequence")
    history_sequence_surface_trend_b64 = plot_assets.get("history_sequence_surface")
    print(
        "    Found history-sequence reporting summary: "
        f"{'yes' if history_sequence_summary else 'no'}"
    )

    # Get git info
    git_info = get_git_info()

    # Generate HTML
    print("  Generating HTML report...")
    combined_trend_b64 = plot_assets.get("combined")
    ocr_trend_b64 = plot_assets.get("ocr")
    html_content = generate_html_report(
        combined_history,
        ocr_history,
        eval_signal_context,
        combined_trend_b64,
        ocr_trend_b64,
        history_sequence_bundle,
        history_sequence_summary,
        history_sequence_compare,
        history_sequence_trend_b64,
        history_sequence_surface_trend_b64,
        git_info,
        redact_commit=args.redact_commit,
        redact_branch=args.redact_branch
    )

    # Write report
    report_path = output_dir / "index.html"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nReport generated: {report_path}")
    print(f"  Combined records: {len(combined_history)}")
    print(f"  OCR records: {len(ocr_history)}")

    # Check for embedded images
    if (plots_dir / "combined_trend.png").exists():
        print("  Combined trend chart: embedded")
    else:
        print("  Combined trend chart: missing (run 'make eval-trend')")

    if (plots_dir / "ocr_trend.png").exists():
        print("  OCR trend chart: embedded")
    else:
        print("  OCR trend chart: missing (run 'make eval-trend')")

    if history_sequence_summary:
        print("  History-sequence reporting: embedded")
    else:
        print(
            "  History-sequence reporting: missing "
            "(run 'make eval-history' or 'make history-sequence-reporting-bundle')"
        )

    print(f"\nOpen: file://{report_path.resolve()}")

    return 0


if __name__ == "__main__":
    exit(main())
