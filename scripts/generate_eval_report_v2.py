#!/usr/bin/env python3
"""
Enhanced Evaluation Report Generator v2 with Interactive Charts.

Features:
- Three-tier chart fallback: CDN → Local → PNG
- Interactive Chart.js visualizations
- Advanced filtering capabilities
- Responsive design
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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
        history_sequence_chart_rows,
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
        history_sequence_chart_rows,
        load_history_sequence_reporting_assets,
    )


def _format_chart_label(timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%m/%d %H:%M")
    except Exception:
        return "N/A"


def generate_chart_js_config(
    combined_history: List[Dict],
    ocr_history: List[Dict],
    history_sequence_rows: Optional[List[Dict]] = None,
) -> Dict:
    """Generate Chart.js configuration for interactive charts."""

    # Prepare combined chart data
    combined_labels = []
    combined_scores = []
    vision_scores = []
    ocr_scores = []

    for entry in combined_history[-20:]:  # Last 20 entries
        timestamp = entry.get("timestamp", "")
        label = _format_chart_label(timestamp) if timestamp else "N/A"

        combined_labels.append(label)

        if "combined" in entry:
            combined_scores.append(entry["combined"].get("combined_score", 0))
            vision_scores.append(entry["combined"].get("vision_score", 0))
            ocr_scores.append(entry["combined"].get("ocr_score", 0))

    # Prepare OCR chart data
    ocr_labels = []
    dimension_recalls = []
    brier_scores = []
    edge_f1s = []

    for entry in ocr_history[-20:]:  # Last 20 entries
        timestamp = entry.get("timestamp", "")
        label = _format_chart_label(timestamp) if timestamp else "N/A"

        ocr_labels.append(label)

        metrics = entry.get("metrics", {})
        dimension_recalls.append(metrics.get("dimension_recall", 0))
        brier_scores.append(metrics.get("brier_score", 0))
        edge_f1s.append(metrics.get("edge_f1", 0))

    history_labels = []
    history_accuracy = []
    history_macro_f1 = []
    history_named_explainability = []
    for entry in (history_sequence_rows or [])[-20:]:
        timestamp = str(entry.get("timestamp") or "")
        history_labels.append(_format_chart_label(timestamp) if timestamp else "N/A")
        metrics = entry.get("history_metrics", {})
        named_summary = entry.get("named_command_summary", {})
        history_accuracy.append(metrics.get("accuracy_overall", 0))
        history_macro_f1.append(metrics.get("macro_f1_overall", 0))
        history_named_explainability.append(named_summary.get("named_command_explainability_rate", 0))

    return {
        "combined": {
            "type": "line",
            "data": {
                "labels": combined_labels,
                "datasets": [
                    {
                        "label": "Combined Score",
                        "data": combined_scores,
                        "borderColor": "rgb(75, 192, 192)",
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                        "tension": 0.1
                    },
                    {
                        "label": "Vision Score",
                        "data": vision_scores,
                        "borderColor": "rgb(54, 162, 235)",
                        "backgroundColor": "rgba(54, 162, 235, 0.2)",
                        "tension": 0.1
                    },
                    {
                        "label": "OCR Score",
                        "data": ocr_scores,
                        "borderColor": "rgb(255, 99, 132)",
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "tension": 0.1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    },
                    "title": {
                        "display": True,
                        "text": "Combined Evaluation Trends"
                    },
                    "tooltip": {
                        "mode": "index",
                        "intersect": False
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "ticks": {
                            "callback": "function(value) { return (value * 100).toFixed(0) + '%'; }"
                        }
                    }
                },
                "interaction": {
                    "mode": "nearest",
                    "axis": "x",
                    "intersect": False
                }
            }
        },
        "ocr": {
            "type": "line",
            "data": {
                "labels": ocr_labels,
                "datasets": [
                    {
                        "label": "Dimension Recall",
                        "data": dimension_recalls,
                        "borderColor": "rgb(75, 192, 192)",
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                        "tension": 0.1
                    },
                    {
                        "label": "Brier Score (inverted)",
                        "data": [1 - bs for bs in brier_scores],
                        "borderColor": "rgb(255, 206, 86)",
                        "backgroundColor": "rgba(255, 206, 86, 0.2)",
                        "tension": 0.1
                    },
                    {
                        "label": "Edge F1",
                        "data": edge_f1s,
                        "borderColor": "rgb(153, 102, 255)",
                        "backgroundColor": "rgba(153, 102, 255, 0.2)",
                        "tension": 0.1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    },
                    "title": {
                        "display": True,
                        "text": "OCR Metrics Trends"
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "ticks": {
                            "callback": "function(value) { return (value * 100).toFixed(0) + '%'; }"
                        }
                    }
                }
            }
        },
        "history_sequence": {
            "type": "line",
            "data": {
                "labels": history_labels,
                "datasets": [
                    {
                        "label": "Accuracy Overall",
                        "data": history_accuracy,
                        "borderColor": "rgb(16, 185, 129)",
                        "backgroundColor": "rgba(16, 185, 129, 0.2)",
                        "tension": 0.1
                    },
                    {
                        "label": "Macro F1",
                        "data": history_macro_f1,
                        "borderColor": "rgb(245, 158, 11)",
                        "backgroundColor": "rgba(245, 158, 11, 0.2)",
                        "tension": 0.1
                    },
                    {
                        "label": "Named Explainability",
                        "data": history_named_explainability,
                        "borderColor": "rgb(99, 102, 241)",
                        "backgroundColor": "rgba(99, 102, 241, 0.2)",
                        "tension": 0.1
                    }
                ]
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {
                        "display": True,
                        "position": "top"
                    },
                    "title": {
                        "display": True,
                        "text": "History Sequence Trends"
                    }
                },
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "ticks": {
                            "callback": "function(value) { return (value * 100).toFixed(0) + '%'; }"
                        }
                    }
                }
            }
        }
    }


def generate_html_with_charts(
    combined_history: List[Dict],
    ocr_history: List[Dict],
    eval_signal_context: Dict,
    history_sequence_bundle: Optional[Dict],
    history_sequence_summary: Optional[Dict],
    history_sequence_compare: Optional[Dict],
    chart_configs: Dict,
    png_fallbacks: Dict[str, str],
    use_cdn: bool = True,
    local_chartjs_path: Optional[str] = None
) -> str:
    """Generate HTML with interactive charts and fallback support."""

    # Get latest combined score if available
    latest_combined = None
    latest_combined_run = eval_signal_context.get("latest_combined_run")
    if isinstance(latest_combined_run, dict) and "combined" in latest_combined_run:
        latest_combined = latest_combined_run["combined"]
    elif combined_history:
        latest = combined_history[-1]
        if "combined" in latest:
            latest_combined = latest["combined"]

    # Chart.js source selection
    if use_cdn:
        chartjs_src = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
        chartjs_integrity = 'sha384-vxfkwNn5NGXgAZEPcjDEwOXrc6Q3gSYPgfn8R6J0R6qKgNW2pV4s3u8V5Zg7Xk6S'
        chartjs_fallback = f"""
        <script>
            // Test if Chart.js loaded from CDN
            if (typeof Chart === 'undefined') {{
                console.warn('CDN failed, trying local Chart.js...');
                var script = document.createElement('script');
                script.src = '{local_chartjs_path or "chart.min.js"}';
                script.onerror = function() {{
                    console.error('Local Chart.js failed, falling back to static images');
                    document.querySelectorAll('.chart-container').forEach(function(container) {{
                        var canvasId = container.querySelector('canvas').id;
                        var fallbackId = canvasId + '-fallback';
                        var fallback = document.getElementById(fallbackId);
                        if (fallback) {{
                            container.style.display = 'none';
                            fallback.style.display = 'block';
                        }}
                    }});
                }};
                document.head.appendChild(script);
            }}
        </script>
        """
    else:
        chartjs_src = local_chartjs_path or "chart.min.js"
        chartjs_integrity = ""
        chartjs_fallback = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAD ML Platform - Evaluation Report v2</title>

    <!-- Chart.js with fallback strategy -->
    <script src="{chartjs_src}"{' integrity="' + chartjs_integrity + '"' if chartjs_integrity else ''} crossorigin="anonymous"></script>
    {chartjs_fallback}

    <style>
        :root {{
            --primary-color: #3b82f6;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --text-color: #1f2937;
            --bg-color: #f9fafb;
            --card-bg: #ffffff;
            --border-color: #e5e7eb;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary-color), #6366f1);
            color: white;
            padding: 40px 0;
            margin: -20px -20px 20px -20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}

        .summary-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid var(--border-color);
        }}

        .score-display {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}

        .score-item {{
            text-align: center;
            flex: 1;
            min-width: 150px;
        }}

        .score-value {{
            font-size: 3em;
            font-weight: bold;
            line-height: 1;
        }}

        .score-label {{
            color: #6b7280;
            margin-top: 5px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .score-good {{ color: var(--success-color); }}
        .score-warning {{ color: var(--warning-color); }}
        .score-bad {{ color: var(--danger-color); }}

        .chart-section {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}

        .chart-container {{
            position: relative;
            height: 400px;
            margin-top: 20px;
        }}

        .chart-fallback {{
            display: none;
            text-align: center;
        }}

        .chart-fallback img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        .filters {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}

        .filter-row {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-group {{
            flex: 1;
            min-width: 200px;
        }}

        .filter-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #374151;
        }}

        .filter-group select,
        .filter-group input {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background: white;
            font-size: 14px;
        }}

        .btn {{
            padding: 10px 20px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }}

        .btn:hover {{
            background: #2563eb;
        }}

        .btn-secondary {{
            background: #6b7280;
        }}

        .btn-secondary:hover {{
            background: #4b5563;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background: #f3f4f6;
            font-weight: 600;
            color: var(--text-color);
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        tr:hover {{
            background: #f9fafb;
        }}

        .footer {{
            text-align: center;
            color: #6b7280;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid var(--border-color);
        }}

        .footer a {{
            color: var(--primary-color);
            text-decoration: none;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}

            .header {{
                padding: 30px 0;
            }}

            .header h1 {{
                font-size: 2em;
            }}

            .score-value {{
                font-size: 2em;
            }}

            .chart-container {{
                height: 300px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 CAD ML Platform</h1>
            <p>Evaluation Report v2 - Interactive Charts</p>
        </div>
"""

    # Add summary card if we have latest data
    if latest_combined:
        score_class = "score-good" if latest_combined["combined_score"] >= 0.85 else \
                     "score-warning" if latest_combined["combined_score"] >= 0.70 else "score-bad"

        html += f"""
        <div class="summary-card">
            <h2>Latest Evaluation Results</h2>
            <div class="score-display">
                <div class="score-item">
                    <div class="score-value {score_class}">
                        {latest_combined["combined_score"]:.3f}
                    </div>
                    <div class="score-label">Combined Score</div>
                </div>
                <div class="score-item">
                    <div class="score-value">
                        {latest_combined.get("vision_score", 0):.3f}
                    </div>
                    <div class="score-label">Vision Score</div>
                </div>
                <div class="score-item">
                    <div class="score-value">
                        {latest_combined.get("ocr_score", 0):.3f}
                    </div>
                    <div class="score-label">OCR Score</div>
                </div>
            </div>
        </div>
"""

    history_sequence_context = build_history_sequence_report_context(
        history_sequence_bundle,
        history_sequence_summary,
        history_sequence_compare,
    )
    if history_sequence_context["available"]:
        html += f"""
        <div class="summary-card">
            <h2>History Sequence Reporting</h2>
            <div class="score-display">
                <div class="score-item">
                    <div class="score-value">
                        {history_sequence_context["report_count"]}
                    </div>
                    <div class="score-label">Reports</div>
                </div>
                <div class="score-item">
                    <div class="score-value">
                        {history_sequence_context["mean_accuracy_overall"]:.3f}
                    </div>
                    <div class="score-label">Mean Accuracy</div>
                </div>
                <div class="score-item">
                    <div class="score-value">
                        {history_sequence_context["mean_macro_f1_overall"]:.3f}
                    </div>
                    <div class="score-label">Mean Macro F1</div>
                </div>
                <div class="score-item">
                    <div class="score-value">
                        {history_sequence_context["mean_named_command_explainability_rate"]:.3f}
                    </div>
                    <div class="score-label">Mean Explainability</div>
                </div>
            </div>
            <p style="margin-top: 18px; color: #6b7280;">
                Latest Surface:
                <strong>{history_sequence_context["latest_sequence_surface_kind"]}</strong> |
                Vocabulary:
                <strong>{history_sequence_context["latest_named_command_vocabulary_kind"]}</strong> |
                Best Surface Key:
                <strong>{history_sequence_context["best_surface_key"]}</strong>
            </p>
        </div>
"""
    else:
        html += """
        <div class="summary-card">
            <h2>History Sequence Reporting</h2>
            <p>History-sequence reporting bundle not available. Run <code>make eval-history</code> or <code>make history-sequence-reporting-bundle</code> to materialize the canonical summary root.</p>
        </div>
"""

    # Hybrid Blind Reporting block
    hybrid_blind_report_count = eval_signal_context.get("hybrid_blind_report_count", 0)
    if hybrid_blind_report_count > 0:
        _agg = (
            eval_signal_context.get("aggregate_metrics")
            if isinstance(eval_signal_context.get("aggregate_metrics"), dict)
            else {}
        )
        html += f"""
        <div class="summary-card">
            <h2>Hybrid Blind Reporting</h2>
            <div class="score-display">
                <div class="score-item">
                    <div class="score-value">{hybrid_blind_report_count}</div>
                    <div class="score-label">Reports</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{_agg.get('hybrid_blind_accuracy_mean', 0.0):.3f}</div>
                    <div class="score-label">Mean Hybrid Accuracy</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{_agg.get('hybrid_blind_graph2d_accuracy_mean', 0.0):.3f}</div>
                    <div class="score-label">Mean Graph2D Accuracy</div>
                </div>
                <div class="score-item">
                    <div class="score-value">{_agg.get('hybrid_blind_gain_mean', 0.0):.3f}</div>
                    <div class="score-label">Mean Gain vs Graph2D</div>
                </div>
            </div>
            <p style="margin-top: 18px; color: #6b7280;">
                Mean weak-label coverage: <strong>{_agg.get('hybrid_blind_coverage_mean', 0.0):.3f}</strong> |
                Latest label slice count: <strong>{_agg.get('hybrid_blind_label_slice_count_latest', 0)}</strong> |
                Latest family slice count: <strong>{_agg.get('hybrid_blind_family_slice_count_latest', 0)}</strong>
            </p>
        </div>
"""

    # Add filters section
    html += """
        <div class="filters">
            <h3>Filters</h3>
            <div class="filter-row">
                <div class="filter-group">
                    <label for="branch-filter">Branch</label>
                    <select id="branch-filter">
                        <option value="all">All Branches</option>
                        <option value="main">main</option>
                        <option value="develop">develop</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="date-from">From Date</label>
                    <input type="date" id="date-from">
                </div>
                <div class="filter-group">
                    <label for="date-to">To Date</label>
                    <input type="date" id="date-to">
                </div>
                <div class="filter-group" style="align-self: flex-end;">
                    <button class="btn" onclick="applyFilters()">Apply Filters</button>
                    <button class="btn btn-secondary" onclick="resetFilters()">Reset</button>
                </div>
            </div>
        </div>
"""

    # Add combined chart section
    html += f"""
        <div class="chart-section">
            <h2>Combined Evaluation Trends</h2>
            <div class="chart-container">
                <canvas id="combinedChart"></canvas>
            </div>
            <div id="combinedChart-fallback" class="chart-fallback">
                <img src="{png_fallbacks.get('combined', '')}" alt="Combined evaluation trend chart">
                <p>Interactive chart unavailable - showing static image</p>
            </div>
        </div>
"""

    # Add OCR chart section
    html += f"""
        <div class="chart-section">
            <h2>OCR Metrics Trends</h2>
            <div class="chart-container">
                <canvas id="ocrChart"></canvas>
            </div>
            <div id="ocrChart-fallback" class="chart-fallback">
                <img src="{png_fallbacks.get('ocr', '')}" alt="OCR metrics trend chart">
                <p>Interactive chart unavailable - showing static image</p>
            </div>
        </div>
"""

    if history_sequence_context["available"]:
        html += f"""
        <div class="chart-section">
            <h2>History Sequence Trends</h2>
            <div class="chart-container">
                <canvas id="historySequenceChart"></canvas>
            </div>
            <div id="historySequenceChart-fallback" class="chart-fallback">
                <img src="{png_fallbacks.get('history_sequence', '')}" alt="History sequence trend chart">
                <p>Interactive chart unavailable - showing static image</p>
            </div>
        </div>
"""

        if png_fallbacks.get("history_sequence_surface"):
            html += f"""
        <div class="chart-section">
            <h2>History Sequence Surface Trend</h2>
            <div class="chart-fallback" style="display: block;">
                <img src="{png_fallbacks.get('history_sequence_surface', '')}" alt="History sequence surface trend chart">
            </div>
        </div>
"""

        if history_sequence_context["leaderboard_rows"]:
            html += """
        <div class="chart-section">
            <h2>Top History Sequence Surfaces</h2>
            <div style="overflow-x: auto;">
                <table id="historySequenceLeaderboard">
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
                            <td>{row["rank"]}</td>
                            <td>{row["surface_key"]}</td>
                            <td>{row["report_count"]}</td>
                            <td>{row["mean_accuracy_overall"]:.3f}</td>
                            <td>{row["mean_macro_f1_overall"]:.3f}</td>
                            <td>{row["mean_named_explainability_rate"]:.3f}</td>
                        </tr>
"""
            html += """
                    </tbody>
                </table>
            </div>
        </div>
"""

    # Add history table
    html += """
        <div class="chart-section">
            <h2>Evaluation History</h2>
            <div style="overflow-x: auto;">
                <table id="historyTable">
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Branch</th>
                            <th>Combined Score</th>
                            <th>Vision Score</th>
                            <th>OCR Score</th>
                            <th>Runner</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Add table rows
    for entry in reversed(combined_history[-20:]):
        timestamp = entry.get("timestamp", "N/A")
        branch = entry.get("branch", "unknown")
        combined = entry.get("combined", {})
        run_context = entry.get("run_context", {})

        html += f"""
                        <tr data-branch="{branch}" data-timestamp="{timestamp}">
                            <td>{timestamp}</td>
                            <td>{branch}</td>
                            <td>{combined.get("combined_score", 0):.3f}</td>
                            <td>{combined.get("vision_score", 0):.3f}</td>
                            <td>{combined.get("ocr_score", 0):.3f}</td>
                            <td>{run_context.get("runner", "unknown")}</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>
        </div>
"""

    # Add footer
    html += f"""
        <div class="footer">
            <p>Generated on {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
            <p>
                <a href="/health">Health Check</a> |
                <a href="/docs">API Documentation</a> |
                <a href="https://github.com">GitHub Repository</a>
            </p>
        </div>
    </div>

    <script>
        // Store original data for filtering
        const originalCombinedData = {json.dumps(combined_history)};
        const originalOcrData = {json.dumps(ocr_history)};
        const originalHistorySequenceRows = {json.dumps(history_sequence_chart_rows(history_sequence_summary))};

        // Chart instances
        let combinedChart = null;
        let ocrChart = null;
        let historySequenceChart = null;

        // Initialize charts
        function initCharts() {{
            // Check if Chart.js is available
            if (typeof Chart === 'undefined') {{
                console.error('Chart.js not available');
                showFallbackImages();
                return;
            }}

            // Combined chart
            const combinedCtx = document.getElementById('combinedChart');
            if (combinedCtx) {{
                const combinedConfig = {json.dumps(chart_configs["combined"])};

                // Fix callback functions in config
                if (combinedConfig.options && combinedConfig.options.scales && combinedConfig.options.scales.y) {{
                    combinedConfig.options.scales.y.ticks.callback = function(value) {{
                        return (value * 100).toFixed(0) + '%';
                    }};
                }}

                combinedChart = new Chart(combinedCtx, combinedConfig);
            }}

            // OCR chart
            const ocrCtx = document.getElementById('ocrChart');
            if (ocrCtx) {{
                const ocrConfig = {json.dumps(chart_configs["ocr"])};

                // Fix callback functions in config
                if (ocrConfig.options && ocrConfig.options.scales && ocrConfig.options.scales.y) {{
                    ocrConfig.options.scales.y.ticks.callback = function(value) {{
                        return (value * 100).toFixed(0) + '%';
                    }};
                }}

                ocrChart = new Chart(ocrCtx, ocrConfig);
            }}

            // History sequence chart
            const historySequenceCtx = document.getElementById('historySequenceChart');
            if (historySequenceCtx) {{
                const historySequenceConfig = {json.dumps(chart_configs["history_sequence"])};

                if (historySequenceConfig.options && historySequenceConfig.options.scales &&
                    historySequenceConfig.options.scales.y) {{
                    historySequenceConfig.options.scales.y.ticks.callback = function(value) {{
                        return (value * 100).toFixed(0) + '%';
                    }};
                }}

                historySequenceChart = new Chart(historySequenceCtx, historySequenceConfig);
            }}
        }}

        // Show fallback images
        function showFallbackImages() {{
            document.querySelectorAll('.chart-container').forEach(function(container) {{
                container.style.display = 'none';
            }});
            document.querySelectorAll('.chart-fallback').forEach(function(fallback) {{
                fallback.style.display = 'block';
            }});
        }}

        // Apply filters
        function applyFilters() {{
            const branch = document.getElementById('branch-filter').value;
            const dateFrom = document.getElementById('date-from').value;
            const dateTo = document.getElementById('date-to').value;

            // Filter table rows
            const rows = document.querySelectorAll('#historyTable tbody tr');
            rows.forEach(row => {{
                let show = true;

                if (branch !== 'all' && row.dataset.branch !== branch) {{
                    show = false;
                }}

                if (dateFrom && row.dataset.timestamp < dateFrom) {{
                    show = false;
                }}

                if (dateTo && row.dataset.timestamp > dateTo) {{
                    show = false;
                }}

                row.style.display = show ? '' : 'none';
            }});

            // Update charts with filtered data
            updateCharts(branch, dateFrom, dateTo);
        }}

        // Reset filters
        function resetFilters() {{
            document.getElementById('branch-filter').value = 'all';
            document.getElementById('date-from').value = '';
            document.getElementById('date-to').value = '';
            applyFilters();
        }}

        // Update charts with filtered data
        function updateCharts(branch, dateFrom, dateTo) {{
            if (!combinedChart || !ocrChart) return;

            // Filter combined data
            let filteredCombined = originalCombinedData;
            if (branch !== 'all') {{
                filteredCombined = filteredCombined.filter(d => d.branch === branch);
            }}
            if (dateFrom) {{
                filteredCombined = filteredCombined.filter(d => d.timestamp >= dateFrom);
            }}
            if (dateTo) {{
                filteredCombined = filteredCombined.filter(d => d.timestamp <= dateTo);
            }}

            // Update combined chart
            const combinedLabels = [];
            const combinedScores = [];
            const visionScores = [];
            const ocrScores = [];

            filteredCombined.slice(-20).forEach(entry => {{
                const timestamp = entry.timestamp || 'N/A';
                combinedLabels.push(timestamp.substring(5, 16));  // MM-DD HH:mm

                if (entry.combined) {{
                    combinedScores.push(entry.combined.combined_score || 0);
                    visionScores.push(entry.combined.vision_score || 0);
                    ocrScores.push(entry.combined.ocr_score || 0);
                }}
            }});

            combinedChart.data.labels = combinedLabels;
            combinedChart.data.datasets[0].data = combinedScores;
            combinedChart.data.datasets[1].data = visionScores;
            combinedChart.data.datasets[2].data = ocrScores;
            combinedChart.update();

            if (historySequenceChart) {{
                let filteredHistorySequence = originalHistorySequenceRows;
                if (dateFrom) {{
                    filteredHistorySequence = filteredHistorySequence.filter(d => d.timestamp >= dateFrom);
                }}
                if (dateTo) {{
                    filteredHistorySequence = filteredHistorySequence.filter(d => d.timestamp <= dateTo);
                }}

                const historyLabels = [];
                const historyAccuracy = [];
                const historyMacroF1 = [];
                const historyNamedExplainability = [];

                filteredHistorySequence.slice(-20).forEach(entry => {{
                    const timestamp = entry.timestamp || 'N/A';
                    historyLabels.push(timestamp.substring(5, 16));
                    historyAccuracy.push((entry.history_metrics || {{}}).accuracy_overall || 0);
                    historyMacroF1.push((entry.history_metrics || {{}}).macro_f1_overall || 0);
                    historyNamedExplainability.push(
                        (entry.named_command_summary || {{}}).named_command_explainability_rate || 0
                    );
                }});

                historySequenceChart.data.labels = historyLabels;
                historySequenceChart.data.datasets[0].data = historyAccuracy;
                historySequenceChart.data.datasets[1].data = historyMacroF1;
                historySequenceChart.data.datasets[2].data = historyNamedExplainability;
                historySequenceChart.update();
            }}
        }}

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(initCharts, 100);  // Small delay to ensure Chart.js is loaded
        }});
    </script>
</body>
</html>
"""

    return html


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Generate enhanced evaluation report with interactive charts")
    parser.add_argument("--dir", default="reports/eval_history",
                        help="Directory containing evaluation history files")
    parser.add_argument("--out", default="reports/eval_history/report",
                        help="Output directory for the report")
    parser.add_argument("--use-cdn", action="store_true", default=True,
                        help="Use CDN for Chart.js (default: True)")
    parser.add_argument("--local-chartjs", help="Path to local Chart.js file")
    parser.add_argument("--redact-commit", action="store_true",
                        help="Redact commit hashes in report")
    parser.add_argument("--redact-branch", action="store_true",
                        help="Redact branch names in report")

    args = parser.parse_args(argv)

    history_dir = Path(args.dir)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    ocr_history = eval_signal_report_rows(
        eval_signal_summary,
        history_dir=history_dir,
        report_type="ocr",
    )
    history_sequence_bundle, history_sequence_summary, history_sequence_compare = (
        load_history_sequence_reporting_assets(history_dir)
    )

    for data in combined_history + ocr_history:
        if args.redact_commit and "commit" in data:
            data["commit"] = "[redacted]"
        if args.redact_branch and "branch" in data:
            data["branch"] = "[redacted]"

    print(f"Loaded {len(combined_history)} combined and {len(ocr_history)} OCR history entries")
    print(
        "Loaded history-sequence reporting summary: "
        f"{'yes' if history_sequence_summary else 'no'}"
    )

    # Generate Chart.js configurations
    chart_configs = generate_chart_js_config(
        combined_history,
        ocr_history,
        history_sequence_chart_rows(history_sequence_summary),
    )

    # Try to load PNG fallbacks
    plots_dir = history_dir / "plots"
    png_fallbacks = load_plot_base64_assets(
        plots_dir,
        {
            "combined": "combined_trend.png",
            "ocr": "ocr_trend.png",
            "history_sequence": "history_sequence_trend.png",
            "history_sequence_surface": "history_sequence_surface_trend.png",
        },
    )

    # Generate HTML
    html = generate_html_with_charts(
        combined_history,
        ocr_history,
        eval_signal_context,
        history_sequence_bundle,
        history_sequence_summary,
        history_sequence_compare,
        chart_configs,
        png_fallbacks,
        use_cdn=args.use_cdn,
        local_chartjs_path=args.local_chartjs
    )

    # Save report
    output_file = output_dir / "index.html"
    with open(output_file, "w") as f:
        f.write(html)

    print(f"\nEnhanced report generated: {output_file}")
    print(f"  - Interactive charts: {'CDN' if args.use_cdn else 'Local'}")
    print(f"  - Fallback images: {len(png_fallbacks)} available")
    print(f"  - Filtering: Branch and date range enabled")
    print(
        "  - History-sequence reporting: "
        f"{'embedded' if history_sequence_summary else 'missing'}"
    )
    print(f"\nOpen: file://{output_file.absolute()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
