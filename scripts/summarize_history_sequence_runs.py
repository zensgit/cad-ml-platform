#!/usr/bin/env python3
"""Summarize history-sequence eval-history records into a canonical experiment report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_ts(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _extract_surface_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    named_summary = (
        payload.get("named_command_summary")
        if isinstance(payload.get("named_command_summary"), dict)
        else {}
    )
    history_metrics = (
        payload.get("history_metrics")
        if isinstance(payload.get("history_metrics"), dict)
        else payload.get("metrics")
        if isinstance(payload.get("metrics"), dict)
        else {}
    )
    return {
        "sequence_surface_kind": str(
            named_summary.get("sequence_surface_kind")
            or history_metrics.get("sequence_surface_kind")
            or ""
        ),
        "named_command_vocabulary_kind": str(
            named_summary.get("named_command_vocabulary_kind") or ""
        ),
        "named_command_authoritative_names_known": bool(
            named_summary.get("named_command_authoritative_names_known", False)
        ),
    }


def _extract_history_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = (
        payload.get("history_metrics")
        if isinstance(payload.get("history_metrics"), dict)
        else payload.get("metrics")
        if isinstance(payload.get("metrics"), dict)
        else {}
    )
    return dict(metrics)


def _extract_named_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    block = payload.get("named_command_summary")
    return dict(block) if isinstance(block, dict) else {}


def _extract_named_error_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    block = payload.get("named_command_error_summary")
    return dict(block) if isinstance(block, dict) else {}


def _load_report(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("type") or "").strip() != "history_sequence":
        return None
    history_metrics = _extract_history_metrics(payload)
    named_summary = _extract_named_summary(payload)
    named_error_summary = _extract_named_error_summary(payload)
    surface_contract = _extract_surface_contract(payload)
    selection_metric_value = _safe_float(history_metrics.get("accuracy_overall"), 0.0)
    selection_metric_kind = "accuracy_overall"
    if selection_metric_value <= 0.0:
        selection_metric_value = _safe_float(history_metrics.get("macro_f1_overall"), 0.0)
        selection_metric_kind = "macro_f1_overall"
    return {
        "report_path": str(path),
        "timestamp": str(payload.get("timestamp") or ""),
        "history_metrics": history_metrics,
        "named_command_summary": named_summary,
        "named_command_error_summary": named_error_summary,
        "surface_contract": surface_contract,
        "selection_metric_kind": selection_metric_kind,
        "selection_metric_value": selection_metric_value,
    }


def _collect_reports(eval_history_dir: Path, report_glob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not eval_history_dir.exists():
        return rows
    for path in sorted(eval_history_dir.glob(str(report_glob))):
        report = _load_report(path)
        if report is not None:
            rows.append(report)
    return rows


def _surface_key_for_row(row: Dict[str, Any]) -> str:
    contract = row.get("surface_contract")
    if not isinstance(contract, dict):
        return ""
    sequence_surface_kind = str(contract.get("sequence_surface_kind") or "")
    vocabulary_kind = str(contract.get("named_command_vocabulary_kind") or "")
    return f"{sequence_surface_kind}::{vocabulary_kind}"


def _count_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        contract = row.get("surface_contract")
        if not isinstance(contract, dict):
            continue
        value = str(contract.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _build_surface_matrix(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = _surface_key_for_row(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _group_rows_by_surface(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_surface_key_for_row(row) or "unknown::unknown", []).append(dict(row))
    return {
        key: sorted(group, key=lambda candidate: _parse_ts(candidate.get("timestamp")))
        for key, group in grouped.items()
    }


def _normalize_report_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "report_path": str(row.get("report_path") or ""),
        "timestamp": str(row.get("timestamp") or ""),
        "selection_metric_kind": str(row.get("selection_metric_kind") or ""),
        "selection_metric_value": _safe_float(row.get("selection_metric_value"), 0.0),
        "history_metrics": dict(row.get("history_metrics") or {}),
        "named_command_summary": dict(row.get("named_command_summary") or {}),
        "named_command_error_summary": dict(row.get("named_command_error_summary") or {}),
        "surface_contract": dict(row.get("surface_contract") or {}),
    }


def _build_report_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        _normalize_report_row(row)
        for row in sorted(rows, key=lambda candidate: _parse_ts(candidate.get("timestamp")))
    ]


def _rows_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_rows = summary.get("report_rows")
    if not isinstance(raw_rows, list):
        return []
    rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            rows.append(_normalize_report_row(row))
    return rows


def _load_summary_artifact(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status") or "").strip() != "ok":
        return None
    if str(payload.get("surface_kind") or "").strip() != "history_sequence_experiment_summary":
        return None
    return payload


def _load_or_build_summary(
    summary_json_path: Path, *, eval_history_dir: Path, report_glob: str
) -> Dict[str, Any]:
    loaded = _load_summary_artifact(summary_json_path)
    if loaded is not None:
        return loaded
    rows = _collect_reports(eval_history_dir, report_glob)
    return _build_summary(rows, eval_history_dir=eval_history_dir, report_glob=report_glob)


def _build_surface_groups(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    grouped = _group_rows_by_surface(rows)
    for surface_key in sorted(grouped):
        group = grouped[surface_key]
        latest = group[-1]
        latest_contract = (
            latest.get("surface_contract")
            if isinstance(latest.get("surface_contract"), dict)
            else {}
        )
        groups.append(
            {
                "surface_key": surface_key,
                "sequence_surface_kind": str(latest_contract.get("sequence_surface_kind") or ""),
                "named_command_vocabulary_kind": str(
                    latest_contract.get("named_command_vocabulary_kind") or ""
                ),
                "report_count": int(len(group)),
                "latest_timestamp": str(latest.get("timestamp") or ""),
                "latest_accuracy_overall": _safe_float(
                    latest.get("history_metrics", {}).get("accuracy_overall"), 0.0
                ),
                "latest_named_explainability_rate": _safe_float(
                    latest.get("named_command_summary", {}).get("named_command_explainability_rate"),
                    0.0,
                ),
                "mean_accuracy_overall": sum(
                    _safe_float(row.get("history_metrics", {}).get("accuracy_overall"), 0.0)
                    for row in group
                )
                / float(len(group)),
                "mean_macro_f1_overall": sum(
                    _safe_float(row.get("history_metrics", {}).get("macro_f1_overall"), 0.0)
                    for row in group
                )
                / float(len(group)),
                "mean_named_explainability_rate": sum(
                    _safe_float(
                        row.get("named_command_summary", {}).get(
                            "named_command_explainability_rate"
                        ),
                        0.0,
                    )
                    for row in group
                )
                / float(len(group)),
                "mean_named_error_rate": sum(
                    _safe_float(
                        row.get("named_command_error_summary", {}).get("overall_incorrect_rate"),
                        0.0,
                    )
                    for row in group
                )
                / float(len(group)),
                "mean_named_low_conf_rate": sum(
                    _safe_float(
                        row.get("named_command_error_summary", {}).get("overall_low_conf_rate"),
                        0.0,
                    )
                    for row in group
                )
                / float(len(group)),
            }
        )
    return groups


def _build_window_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    if not rows_list:
        return {
            "report_count": 0,
            "coverage_mean": 0.0,
            "accuracy_mean": 0.0,
            "macro_f1_mean": 0.0,
            "named_explainability_rate_mean": 0.0,
            "named_error_rate_mean": 0.0,
            "named_low_conf_rate_mean": 0.0,
            "latest_sequence_surface_kind": "",
            "latest_named_vocabulary_kind": "",
            "latest_worst_primary_family": "",
            "latest_worst_primary_reference_surface": "",
            "latest_worst_primary_status": "",
            "surface_group_count": 0,
            "best_surface_key_by_mean_accuracy_overall": "",
            "surface_groups": [],
        }
    latest_row = max(rows_list, key=lambda row: _parse_ts(row.get("timestamp")))
    latest_named_error_summary = (
        latest_row.get("named_command_error_summary")
        if isinstance(latest_row.get("named_command_error_summary"), dict)
        else {}
    )
    latest_worst_family = (
        latest_named_error_summary.get("worst_primary_family")
        if isinstance(latest_named_error_summary.get("worst_primary_family"), dict)
        else {}
    )
    latest_worst_reference_surface = (
        latest_named_error_summary.get("worst_primary_reference_surface")
        if isinstance(latest_named_error_summary.get("worst_primary_reference_surface"), dict)
        else {}
    )
    latest_worst_status = (
        latest_named_error_summary.get("worst_primary_status")
        if isinstance(latest_named_error_summary.get("worst_primary_status"), dict)
        else {}
    )
    surface_groups = _build_surface_groups(rows_list)
    best_surface_key = ""
    best_surface_score = -1.0
    for group in surface_groups:
        score = _safe_float(group.get("mean_accuracy_overall"), 0.0)
        if score > best_surface_score:
            best_surface_score = score
            best_surface_key = str(group.get("surface_key") or "")
    latest_contract = (
        latest_row.get("surface_contract")
        if isinstance(latest_row.get("surface_contract"), dict)
        else {}
    )
    return {
        "report_count": int(len(rows_list)),
        "coverage_mean": sum(
            _safe_float(row.get("history_metrics", {}).get("coverage"), 0.0) for row in rows_list
        )
        / float(len(rows_list)),
        "accuracy_mean": sum(
            _safe_float(row.get("history_metrics", {}).get("accuracy_overall"), 0.0)
            for row in rows_list
        )
        / float(len(rows_list)),
        "macro_f1_mean": sum(
            _safe_float(row.get("history_metrics", {}).get("macro_f1_overall"), 0.0)
            for row in rows_list
        )
        / float(len(rows_list)),
        "named_explainability_rate_mean": sum(
            _safe_float(
                row.get("named_command_summary", {}).get("named_command_explainability_rate"), 0.0
            )
            for row in rows_list
        )
        / float(len(rows_list)),
        "named_error_rate_mean": sum(
            _safe_float(
                row.get("named_command_error_summary", {}).get("overall_incorrect_rate"), 0.0
            )
            for row in rows_list
        )
        / float(len(rows_list)),
        "named_low_conf_rate_mean": sum(
            _safe_float(
                row.get("named_command_error_summary", {}).get("overall_low_conf_rate"), 0.0
            )
            for row in rows_list
        )
        / float(len(rows_list)),
        "latest_sequence_surface_kind": str(latest_contract.get("sequence_surface_kind") or ""),
        "latest_named_vocabulary_kind": str(
            latest_contract.get("named_command_vocabulary_kind") or ""
        ),
        "latest_worst_primary_family": str(latest_worst_family.get("value") or ""),
        "latest_worst_primary_reference_surface": str(
            latest_worst_reference_surface.get("value") or ""
        ),
        "latest_worst_primary_status": str(latest_worst_status.get("value") or ""),
        "surface_group_count": int(len(surface_groups)),
        "best_surface_key_by_mean_accuracy_overall": best_surface_key,
        "surface_groups": surface_groups,
    }


def _build_summary(
    rows: List[Dict[str, Any]], *, eval_history_dir: Path, report_glob: str
) -> Dict[str, Any]:
    if not rows:
        return {
            "status": "ok",
            "surface_kind": "history_sequence_experiment_summary",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eval_history_dir": str(eval_history_dir),
            "report_glob": report_glob,
            "report_count": 0,
            "aggregate_metrics": {},
            "surface_counts": {
                "sequence_surface_kind": {},
                "named_command_vocabulary_kind": {},
                "surface_vocabulary_matrix": {},
            },
            "surface_groups": [],
            "report_rows": [],
            "best_run": None,
            "latest_run": None,
        }
    latest_row = max(rows, key=lambda row: _parse_ts(row.get("timestamp")))
    best_row = max(rows, key=lambda row: _safe_float(row.get("selection_metric_value"), 0.0))
    return {
        "status": "ok",
        "surface_kind": "history_sequence_experiment_summary",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_history_dir": str(eval_history_dir),
        "report_glob": report_glob,
        "report_count": int(len(rows)),
        "aggregate_metrics": {
            "mean_coverage": sum(
                _safe_float(row["history_metrics"].get("coverage"), 0.0) for row in rows
            )
            / float(len(rows)),
            "mean_accuracy_overall": sum(
                _safe_float(row["history_metrics"].get("accuracy_overall"), 0.0) for row in rows
            )
            / float(len(rows)),
            "mean_macro_f1_overall": sum(
                _safe_float(row["history_metrics"].get("macro_f1_overall"), 0.0) for row in rows
            )
            / float(len(rows)),
            "mean_named_command_explainability_rate": sum(
                _safe_float(
                    row["named_command_summary"].get("named_command_explainability_rate"), 0.0
                )
                for row in rows
            )
            / float(len(rows)),
            "mean_named_command_error_rate": sum(
                _safe_float(
                    row["named_command_error_summary"].get("overall_incorrect_rate"), 0.0
                )
                for row in rows
            )
            / float(len(rows)),
            "mean_named_command_low_conf_rate": sum(
                _safe_float(
                    row["named_command_error_summary"].get("overall_low_conf_rate"), 0.0
                )
                for row in rows
            )
            / float(len(rows)),
        },
        "surface_counts": {
            "sequence_surface_kind": _count_by_key(rows, "sequence_surface_kind"),
            "named_command_vocabulary_kind": _count_by_key(rows, "named_command_vocabulary_kind"),
            "surface_vocabulary_matrix": _build_surface_matrix(rows),
        },
        "surface_groups": _build_surface_groups(rows),
        "report_rows": _build_report_rows(rows),
        "best_run": dict(best_row),
        "latest_run": dict(latest_row),
    }


def _build_markdown(summary: Dict[str, Any]) -> str:
    best_run = summary.get("best_run") if isinstance(summary.get("best_run"), dict) else {}
    latest_run = summary.get("latest_run") if isinstance(summary.get("latest_run"), dict) else {}
    best_contract = (
        best_run.get("surface_contract")
        if isinstance(best_run.get("surface_contract"), dict)
        else {}
    )
    latest_contract = (
        latest_run.get("surface_contract")
        if isinstance(latest_run.get("surface_contract"), dict)
        else {}
    )
    aggregate = summary.get("aggregate_metrics") if isinstance(summary.get("aggregate_metrics"), dict) else {}
    surface_counts = summary.get("surface_counts") if isinstance(summary.get("surface_counts"), dict) else {}
    lines = [
        "# History Sequence Experiment Summary",
        "",
        f"- Eval history dir: `{summary.get('eval_history_dir', '')}`",
        f"- Report glob: `{summary.get('report_glob', '')}`",
        f"- Report count: `{summary.get('report_count', 0)}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean coverage | {_safe_float(aggregate.get('mean_coverage'), 0.0):.6f} |",
        f"| Mean accuracy overall | {_safe_float(aggregate.get('mean_accuracy_overall'), 0.0):.6f} |",
        f"| Mean macro F1 overall | {_safe_float(aggregate.get('mean_macro_f1_overall'), 0.0):.6f} |",
        f"| Mean named explainability rate | {_safe_float(aggregate.get('mean_named_command_explainability_rate'), 0.0):.6f} |",
        f"| Mean named error rate | {_safe_float(aggregate.get('mean_named_command_error_rate'), 0.0):.6f} |",
        f"| Mean named low-conf rate | {_safe_float(aggregate.get('mean_named_command_low_conf_rate'), 0.0):.6f} |",
        "",
        "## Best Run",
        "",
        f"- Report: `{best_run.get('report_path', '')}`",
        f"- Selection metric: `{best_run.get('selection_metric_kind', '')}` = `{_safe_float(best_run.get('selection_metric_value'), 0.0):.6f}`",
        f"- Sequence surface: `{best_contract.get('sequence_surface_kind', '')}`",
        f"- Vocabulary kind: `{best_contract.get('named_command_vocabulary_kind', '')}`",
        "",
        "## Latest Run",
        "",
        f"- Report: `{latest_run.get('report_path', '')}`",
        f"- Timestamp: `{latest_run.get('timestamp', '')}`",
        f"- Sequence surface: `{latest_contract.get('sequence_surface_kind', '')}`",
        f"- Vocabulary kind: `{latest_contract.get('named_command_vocabulary_kind', '')}`",
        "",
        "## Surface Counts",
        "",
        f"- Sequence surface counts: `{json.dumps(surface_counts.get('sequence_surface_kind', {}), ensure_ascii=False)}`",
        f"- Vocabulary kind counts: `{json.dumps(surface_counts.get('named_command_vocabulary_kind', {}), ensure_ascii=False)}`",
        f"- Surface/vocabulary matrix: `{json.dumps(surface_counts.get('surface_vocabulary_matrix', {}), ensure_ascii=False)}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize history-sequence eval-history records.")
    parser.add_argument(
        "--eval-history-dir",
        default="reports/eval_history",
        help="Directory containing eval-history JSON records.",
    )
    parser.add_argument(
        "--report-glob",
        default="*.json",
        help="Glob pattern used under --eval-history-dir.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/eval_history/history_sequence_experiment_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/eval_history/history_sequence_experiment_summary.md",
        help="Output Markdown summary path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    rows = _collect_reports(eval_history_dir, str(args.report_glob))
    summary = _build_summary(rows, eval_history_dir=eval_history_dir, report_glob=str(args.report_glob))
    output_json = Path(str(args.output_json))
    output_md = Path(str(args.output_md))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(summary), encoding="utf-8")
    print(f"History sequence summary JSON: {output_json}")
    print(f"History sequence summary Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
