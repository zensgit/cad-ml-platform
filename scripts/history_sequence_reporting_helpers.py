#!/usr/bin/env python3
"""Shared helpers for history-sequence reporting consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scripts.eval_report_data_helpers import load_json_dict
except ImportError:
    from eval_report_data_helpers import load_json_dict  # type: ignore


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_history_sequence_reporting_assets(
    history_dir: Path,
    *,
    bundle_json_path: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    bundle = load_json_dict(bundle_json_path or (history_dir / "history_sequence_reporting_bundle.json"))
    summary = load_json_dict(history_dir / "history_sequence_experiment_summary.json")
    compare = load_json_dict(history_dir / "history_sequence_surface_comparison_report.json")

    if bundle:
        summary_path = Path(str(bundle.get("summary_json") or ""))
        compare_path = Path(str(bundle.get("compare_json") or ""))
        loaded_summary = load_json_dict(summary_path) if str(summary_path).strip() else None
        loaded_compare = load_json_dict(compare_path) if str(compare_path).strip() else None
        summary = loaded_summary or summary
        compare = loaded_compare or compare

    return bundle, summary, compare


def history_sequence_chart_rows(summary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return []
    rows = summary.get("report_rows")
    if not isinstance(rows, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "timestamp": str(row.get("timestamp") or ""),
                "history_metrics": dict(row.get("history_metrics") or {}),
                "named_command_summary": dict(row.get("named_command_summary") or {}),
                "surface_contract": dict(row.get("surface_contract") or {}),
            }
        )
    return normalized


def build_history_sequence_report_context(
    bundle: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    compare: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {
            "available": False,
            "report_count": 0,
            "mean_accuracy_overall": 0.0,
            "mean_macro_f1_overall": 0.0,
            "mean_named_command_explainability_rate": 0.0,
            "latest_sequence_surface_kind": "",
            "latest_named_command_vocabulary_kind": "",
            "best_surface_key": "",
            "leaderboard_rows": [],
        }

    aggregate = summary.get("aggregate_metrics")
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    latest_run = summary.get("latest_run")
    latest_run = latest_run if isinstance(latest_run, dict) else {}
    latest_contract = latest_run.get("surface_contract")
    latest_contract = latest_contract if isinstance(latest_contract, dict) else {}
    leaderboard = compare.get("leaderboard") if isinstance(compare, dict) else []

    leaderboard_rows: List[Dict[str, Any]] = []
    if isinstance(leaderboard, list):
        for row in leaderboard[:5]:
            if not isinstance(row, dict):
                continue
            leaderboard_rows.append(
                {
                    "rank": int(row.get("rank", 0) or 0),
                    "surface_key": str(row.get("surface_key") or ""),
                    "report_count": int(row.get("report_count", 0) or 0),
                    "mean_accuracy_overall": _safe_float(row.get("mean_accuracy_overall"), 0.0),
                    "mean_macro_f1_overall": _safe_float(row.get("mean_macro_f1_overall"), 0.0),
                    "mean_named_explainability_rate": _safe_float(
                        row.get("mean_named_explainability_rate"), 0.0
                    ),
                }
            )

    return {
        "available": True,
        "report_count": int(summary.get("report_count", 0) or 0),
        "mean_accuracy_overall": _safe_float(aggregate.get("mean_accuracy_overall"), 0.0),
        "mean_macro_f1_overall": _safe_float(aggregate.get("mean_macro_f1_overall"), 0.0),
        "mean_named_command_explainability_rate": _safe_float(
            aggregate.get("mean_named_command_explainability_rate"), 0.0
        ),
        "latest_sequence_surface_kind": str(
            latest_contract.get("sequence_surface_kind") or ""
        ),
        "latest_named_command_vocabulary_kind": str(
            latest_contract.get("named_command_vocabulary_kind") or ""
        ),
        "best_surface_key": str(
            (bundle if isinstance(bundle, dict) else {}).get(
                "best_surface_key_by_mean_accuracy_overall"
            )
            or ""
        ),
        "leaderboard_rows": leaderboard_rows,
    }
