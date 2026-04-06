#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import summarize_history_sequence_runs as canonical


def _surface_key(row: Dict[str, Any]) -> str:
    contract = row.get("surface_contract") if isinstance(row.get("surface_contract"), dict) else {}
    sequence_surface_kind = str(contract.get("sequence_surface_kind") or "")
    vocabulary_kind = str(contract.get("named_command_vocabulary_kind") or "")
    return f"{sequence_surface_kind}::{vocabulary_kind}"


def _surface_contract_from_key(surface_key: str) -> Dict[str, str]:
    sequence_surface_kind, _, vocabulary_kind = surface_key.partition("::")
    return {
        "sequence_surface_kind": sequence_surface_kind,
        "named_command_vocabulary_kind": vocabulary_kind,
    }


def _group_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = _surface_key(row)
        groups.setdefault(key, []).append(dict(row))
    return groups


def _group_sort_key(group: Dict[str, Any]) -> Tuple[float, float, str]:
    aggregate = group.get("aggregate_metrics") if isinstance(group.get("aggregate_metrics"), dict) else {}
    latest_run = group.get("latest_run") if isinstance(group.get("latest_run"), dict) else {}
    return (
        canonical._safe_float(aggregate.get("mean_accuracy_overall"), 0.0),
        canonical._safe_float(aggregate.get("mean_macro_f1_overall"), 0.0),
        str(latest_run.get("timestamp") or ""),
    )


def _build_report(
    rows: List[Dict[str, Any]], *, eval_history_dir: Path, report_glob: str
) -> Dict[str, Any]:
    grouped = _group_rows(rows)
    groups: List[Dict[str, Any]] = []
    for surface_key, group_rows in grouped.items():
        summary = canonical._build_summary(
            group_rows,
            eval_history_dir=eval_history_dir,
            report_glob=report_glob,
        )
        groups.append(
            {
                "surface_key": surface_key,
                "surface_contract": _surface_contract_from_key(surface_key),
                "report_count": int(summary.get("report_count", 0) or 0),
                "aggregate_metrics": dict(summary.get("aggregate_metrics") or {}),
                "best_run": dict(summary.get("best_run") or {}),
                "latest_run": dict(summary.get("latest_run") or {}),
            }
        )
    ordered_groups = sorted(groups, key=_group_sort_key, reverse=True)
    leaderboard = [
        {
            "rank": idx + 1,
            "surface_key": group["surface_key"],
            "report_count": group["report_count"],
            "mean_accuracy_overall": canonical._safe_float(
                group.get("aggregate_metrics", {}).get("mean_accuracy_overall"), 0.0
            ),
            "mean_macro_f1_overall": canonical._safe_float(
                group.get("aggregate_metrics", {}).get("mean_macro_f1_overall"), 0.0
            ),
            "mean_named_explainability_rate": canonical._safe_float(
                group.get("aggregate_metrics", {}).get("mean_named_command_explainability_rate"),
                0.0,
            ),
            "latest_timestamp": str((group.get("latest_run") or {}).get("timestamp") or ""),
        }
        for idx, group in enumerate(ordered_groups)
    ]
    return {
        "status": "ok",
        "surface_kind": "history_sequence_surface_comparison_report",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_history_dir": str(eval_history_dir),
        "report_glob": report_glob,
        "total_reports": int(len(rows)),
        "total_groups": int(len(ordered_groups)),
        "best_surface_key": str(leaderboard[0]["surface_key"]) if leaderboard else "",
        "leaderboard": leaderboard,
        "surface_groups": ordered_groups,
    }


def _build_markdown(report: Dict[str, Any]) -> str:
    leaderboard = report.get("leaderboard") if isinstance(report.get("leaderboard"), list) else []
    groups = report.get("surface_groups") if isinstance(report.get("surface_groups"), list) else []
    lines = [
        "# History Sequence Surface Comparison Report",
        "",
        f"- Eval history dir: `{report.get('eval_history_dir', '')}`",
        f"- Report glob: `{report.get('report_glob', '')}`",
        f"- Total reports: `{report.get('total_reports', 0)}`",
        f"- Total groups: `{report.get('total_groups', 0)}`",
        f"- Best surface key: `{report.get('best_surface_key', '')}`",
        "",
        "## Leaderboard",
        "",
        "| Rank | Surface Key | Reports | Mean Accuracy | Mean Macro F1 | Mean Named Explainability | Latest Timestamp |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in leaderboard:
        lines.append(
            f"| {int(row.get('rank', 0) or 0)} | `{row.get('surface_key', '')}` | "
            f"{int(row.get('report_count', 0) or 0)} | "
            f"{canonical._safe_float(row.get('mean_accuracy_overall'), 0.0):.6f} | "
            f"{canonical._safe_float(row.get('mean_macro_f1_overall'), 0.0):.6f} | "
            f"{canonical._safe_float(row.get('mean_named_explainability_rate'), 0.0):.6f} | "
            f"`{row.get('latest_timestamp', '')}` |"
        )
    lines.extend(["", "## Surface Groups", ""])
    for group in groups:
        aggregate = (
            group.get("aggregate_metrics") if isinstance(group.get("aggregate_metrics"), dict) else {}
        )
        best_run = group.get("best_run") if isinstance(group.get("best_run"), dict) else {}
        latest_run = group.get("latest_run") if isinstance(group.get("latest_run"), dict) else {}
        lines.extend(
            [
                f"### `{group.get('surface_key', '')}`",
                "",
                f"- Reports: `{group.get('report_count', 0)}`",
                f"- Mean coverage: `{canonical._safe_float(aggregate.get('mean_coverage'), 0.0):.6f}`",
                f"- Mean accuracy overall: `{canonical._safe_float(aggregate.get('mean_accuracy_overall'), 0.0):.6f}`",
                f"- Mean macro F1 overall: `{canonical._safe_float(aggregate.get('mean_macro_f1_overall'), 0.0):.6f}`",
                f"- Mean named explainability rate: `{canonical._safe_float(aggregate.get('mean_named_command_explainability_rate'), 0.0):.6f}`",
                f"- Mean named error rate: `{canonical._safe_float(aggregate.get('mean_named_command_error_rate'), 0.0):.6f}`",
                f"- Best run report: `{best_run.get('report_path', '')}`",
                f"- Best run selection metric: `{best_run.get('selection_metric_kind', '')}` = `{canonical._safe_float(best_run.get('selection_metric_value'), 0.0):.6f}`",
                f"- Latest run report: `{latest_run.get('report_path', '')}`",
                f"- Latest run timestamp: `{latest_run.get('timestamp', '')}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate history-sequence surface comparison report.")
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
        "--summary-json",
        default="",
        help=(
            "Optional canonical history-sequence experiment summary JSON. "
            "When present, compare generation prefers the canonical artifact over raw eval-history scans."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="reports/eval_history/history_sequence_surface_comparison_report.json",
        help="Output JSON comparison path.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/eval_history/history_sequence_surface_comparison_report.md",
        help="Output Markdown comparison path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    eval_history_dir = Path(str(args.eval_history_dir))
    summary_json = (
        Path(str(args.summary_json))
        if str(args.summary_json).strip()
        else eval_history_dir / "history_sequence_experiment_summary.json"
    )
    summary = canonical._load_or_build_summary(
        summary_json,
        eval_history_dir=eval_history_dir,
        report_glob=str(args.report_glob),
    )
    rows = canonical._rows_from_summary(summary)
    report = _build_report(rows, eval_history_dir=eval_history_dir, report_glob=str(args.report_glob))
    output_json = Path(str(args.output_json))
    output_md = Path(str(args.output_md))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(report), encoding="utf-8")
    print(f"History sequence comparison JSON: {output_json}")
    print(f"History sequence comparison Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
