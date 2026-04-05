#!/usr/bin/env python3
"""Shared helpers for eval-signal reporting consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scripts import summarize_eval_signal_runs as eval_signal_canonical
    from scripts.eval_report_data_helpers import load_json_dict
except ImportError:
    import summarize_eval_signal_runs as eval_signal_canonical  # type: ignore
    from eval_report_data_helpers import load_json_dict  # type: ignore


def _normalize_report_path(path_text: object, *, root_dir: Path) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = root_dir / path
    return str(path)


def _normalize_summary_row(row: Dict[str, Any], *, root_dir: Path) -> Dict[str, Any]:
    report_path = _normalize_report_path(row.get("report_path"), root_dir=root_dir)
    return {
        "timestamp": str(row.get("timestamp") or ""),
        "report_type": str(row.get("report_type") or ""),
        "branch": str(row.get("branch") or ""),
        "commit": str(row.get("commit") or ""),
        "combined": dict(row.get("combined") or {}),
        "metrics": dict(row.get("metrics") or {}),
        "run_context": dict(row.get("run_context") or {}),
        "report_path": report_path,
        "_file": Path(report_path).name if report_path else "",
    }


def load_eval_signal_reporting_assets(
    history_dir: Path,
    *,
    bundle_json_path: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load eval-signal reporting bundle and summary.

    Returns (bundle, summary) tuple.  When a bundle exists its
    ``summary_json`` pointer is preferred over the default summary path.
    """
    bundle = load_json_dict(
        bundle_json_path or (history_dir / "eval_signal_reporting_bundle.json")
    )
    summary: Optional[Dict[str, Any]] = None
    if isinstance(bundle, dict):
        summary_path_text = str(bundle.get("summary_json") or "").strip()
        if summary_path_text:
            summary = load_json_dict(Path(summary_path_text))

    if summary is None:
        summary = load_eval_signal_reporting_summary(history_dir)

    return bundle, summary


def load_eval_signal_reporting_summary(
    history_dir: Path,
    *,
    summary_json_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    summary = eval_signal_canonical._load_or_build_summary(
        summary_json_path or (history_dir / "eval_signal_experiment_summary.json"),
        eval_history_dir=history_dir,
        report_glob="*.json",
    )
    return summary if isinstance(summary, dict) else None


def eval_signal_report_rows(
    summary: Optional[Dict[str, Any]],
    *,
    history_dir: Path,
    report_type: str,
) -> List[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return []

    rows: List[Dict[str, Any]] = []
    for row in eval_signal_canonical._rows_from_summary(summary):
        if str(row.get("report_type") or "").strip() != report_type:
            continue
        rows.append(_normalize_summary_row(row, root_dir=history_dir))
    return rows


def build_eval_signal_report_context(
    summary: Optional[Dict[str, Any]],
    *,
    history_dir: Path,
) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {
            "available": False,
            "report_count": 0,
            "combined_report_count": 0,
            "ocr_report_count": 0,
            "hybrid_blind_report_count": 0,
            "latest_combined_run": None,
            "latest_ocr_run": None,
            "aggregate_metrics": {},
        }

    report_counts = summary.get("report_counts") if isinstance(summary.get("report_counts"), dict) else {}
    aggregate_metrics = (
        summary.get("aggregate_metrics") if isinstance(summary.get("aggregate_metrics"), dict) else {}
    )
    latest_combined = (
        _normalize_summary_row(summary.get("latest_combined_run"), root_dir=history_dir)
        if isinstance(summary.get("latest_combined_run"), dict)
        else None
    )
    latest_ocr = (
        _normalize_summary_row(summary.get("latest_ocr_run"), root_dir=history_dir)
        if isinstance(summary.get("latest_ocr_run"), dict)
        else None
    )
    return {
        "available": True,
        "report_count": int(summary.get("report_count", 0) or 0),
        "combined_report_count": int(report_counts.get("combined", 0) or 0),
        "ocr_report_count": int(report_counts.get("ocr", 0) or 0),
        "hybrid_blind_report_count": int(report_counts.get("hybrid_blind", 0) or 0),
        "latest_combined_run": latest_combined,
        "latest_ocr_run": latest_ocr,
        "aggregate_metrics": aggregate_metrics,
    }
