#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import summarize_uvnet_training_runs as canonical  # noqa: E402


def _resolve_window_reference_now(
    rows: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> datetime:
    if now is not None:
        return now
    if rows:
        return max(
            (canonical._parse_ts(row.get("generated_at")) for row in rows),
            default=datetime.fromtimestamp(0, tz=timezone.utc),
        )
    return datetime.now(timezone.utc)


def _filter_recent_rows(
    rows: List[Dict[str, Any]], *, days: int, now: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    ref_now = _resolve_window_reference_now(rows, now=now)
    cutoff = ref_now - timedelta(days=max(1, int(days)))
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        ts = canonical._parse_ts(row.get("generated_at"))
        if ts >= cutoff:
            filtered.append(dict(row))
    return filtered


def _build_weekly_summary(
    rows: List[Dict[str, Any]],
    *,
    artifacts_dir: Path,
    artifact_glob: str,
    days: int,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    ref_now = _resolve_window_reference_now(rows, now=now)
    cutoff = ref_now - timedelta(days=max(1, int(days)))
    base = canonical._build_summary(rows, artifacts_dir=artifacts_dir, artifact_glob=artifact_glob)
    latest_run = base.get("latest_run") if isinstance(base.get("latest_run"), dict) else {}
    best_run = base.get("best_run") if isinstance(base.get("best_run"), dict) else {}
    return {
        **base,
        "surface_kind": "uvnet_training_weekly_summary",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days_window": int(max(1, int(days))),
        "cutoff_timestamp": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latest_generated_at": str(latest_run.get("generated_at") or ""),
        "best_run_generated_at": str(best_run.get("generated_at") or ""),
    }


def _branch_kind(row: Dict[str, Any]) -> str:
    contract = row.get("model_surface_contract")
    if not isinstance(contract, dict):
        return ""
    return str(contract.get("grid_branch_surface_kind") or "")


def _surface_key(row: Dict[str, Any]) -> str:
    contract = row.get("model_surface_contract")
    if not isinstance(contract, dict):
        return ""
    branch = str(contract.get("grid_branch_surface_kind") or "")
    tower = str(contract.get("grid_tower_topology_kind") or "")
    return f"{branch}::{tower}"


def _mean_metric(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(
        canonical._safe_float(row.get("training_summary", {}).get(key), 0.0) for row in rows
    ) / float(len(rows))


def _build_branch_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_branch_kind(row) or "unknown", []).append(dict(row))
    branch_groups: List[Dict[str, Any]] = []
    for branch_kind in sorted(grouped):
        group = sorted(grouped[branch_kind], key=lambda row: canonical._parse_ts(row.get("generated_at")))
        latest = group[-1]
        best = max(
            group,
            key=lambda row: canonical._safe_float(
                row.get("training_summary", {}).get("best_val_accuracy"), 0.0
            ),
        )
        latest_contract = (
            latest.get("model_surface_contract")
            if isinstance(latest.get("model_surface_contract"), dict)
            else {}
        )
        best_contract = (
            best.get("model_surface_contract")
            if isinstance(best.get("model_surface_contract"), dict)
            else {}
        )
        branch_groups.append(
            {
                "grid_branch_surface_kind": branch_kind,
                "artifact_count": int(len(group)),
                "latest_generated_at": str(latest.get("generated_at") or ""),
                "latest_grid_tower_topology_kind": str(
                    latest_contract.get("grid_tower_topology_kind") or ""
                ),
                "best_run_generated_at": str(best.get("generated_at") or ""),
                "best_run_grid_tower_topology_kind": str(
                    best_contract.get("grid_tower_topology_kind") or ""
                ),
                "mean_final_val_accuracy": _mean_metric(group, "final_val_accuracy"),
                "mean_best_val_accuracy": _mean_metric(group, "best_val_accuracy"),
            }
        )
    return branch_groups


def _build_branch_tower_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_surface_key(row) or "unknown::unknown", []).append(dict(row))
    surface_groups: List[Dict[str, Any]] = []
    for surface_key in sorted(grouped):
        group = sorted(grouped[surface_key], key=lambda row: canonical._parse_ts(row.get("generated_at")))
        latest = group[-1]
        best = max(
            group,
            key=lambda row: canonical._safe_float(
                row.get("training_summary", {}).get("best_val_accuracy"), 0.0
            ),
        )
        surface_groups.append(
            {
                "surface_key": surface_key,
                "artifact_count": int(len(group)),
                "latest_generated_at": str(latest.get("generated_at") or ""),
                "best_run_generated_at": str(best.get("generated_at") or ""),
                "mean_final_val_accuracy": _mean_metric(group, "final_val_accuracy"),
                "mean_best_val_accuracy": _mean_metric(group, "best_val_accuracy"),
            }
        )
    return surface_groups


def _build_markdown(summary: Dict[str, Any]) -> str:
    aggregate = summary.get("aggregate_metrics") if isinstance(summary.get("aggregate_metrics"), dict) else {}
    surface_counts = summary.get("surface_counts") if isinstance(summary.get("surface_counts"), dict) else {}
    latest_run = summary.get("latest_run") if isinstance(summary.get("latest_run"), dict) else {}
    best_run = summary.get("best_run") if isinstance(summary.get("best_run"), dict) else {}
    latest_contract = (
        latest_run.get("model_surface_contract")
        if isinstance(latest_run.get("model_surface_contract"), dict)
        else {}
    )
    best_contract = (
        best_run.get("model_surface_contract")
        if isinstance(best_run.get("model_surface_contract"), dict)
        else {}
    )
    branch_groups = summary.get("branch_groups") if isinstance(summary.get("branch_groups"), list) else []
    branch_tower_groups = (
        summary.get("branch_tower_groups")
        if isinstance(summary.get("branch_tower_groups"), list)
        else []
    )
    lines = [
        "# UVNet Training Weekly Summary",
        "",
        f"- Artifacts dir: `{summary.get('artifacts_dir', '')}`",
        f"- Days window: `{summary.get('days_window', 0)}`",
        f"- Cutoff timestamp: `{summary.get('cutoff_timestamp', '')}`",
        f"- Artifact count: `{summary.get('artifact_count', 0)}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean final train accuracy | {canonical._safe_float(aggregate.get('mean_final_train_accuracy'), 0.0):.6f} |",
        f"| Mean final val accuracy | {canonical._safe_float(aggregate.get('mean_final_val_accuracy'), 0.0):.6f} |",
        f"| Mean best val accuracy | {canonical._safe_float(aggregate.get('mean_best_val_accuracy'), 0.0):.6f} |",
        "",
        "## Latest Run",
        "",
        f"- Artifact: `{latest_run.get('artifact_path', '')}`",
        f"- Generated at: `{latest_run.get('generated_at', '')}`",
        f"- Grid branch surface: `{latest_contract.get('grid_branch_surface_kind', '')}`",
        f"- Grid tower topology: `{latest_contract.get('grid_tower_topology_kind', '')}`",
        "",
        "## Best Run",
        "",
        f"- Artifact: `{best_run.get('artifact_path', '')}`",
        f"- Selection metric: `{best_run.get('selection_metric_kind', '')}` = `{canonical._safe_float(best_run.get('selection_metric_value'), 0.0):.6f}`",
        f"- Grid branch surface: `{best_contract.get('grid_branch_surface_kind', '')}`",
        f"- Grid tower topology: `{best_contract.get('grid_tower_topology_kind', '')}`",
        "",
        "## Surface Counts",
        "",
        f"- Grid branch counts: `{json.dumps(surface_counts.get('grid_branch_surface_kind', {}), ensure_ascii=False)}`",
        f"- Grid tower counts: `{json.dumps(surface_counts.get('grid_tower_topology_kind', {}), ensure_ascii=False)}`",
        f"- Branch/tower matrix: `{json.dumps(surface_counts.get('branch_tower_matrix', {}), ensure_ascii=False)}`",
        "",
        "## Branch Groups",
        "",
    ]
    for group in branch_groups:
        lines.extend(
            [
                f"- `{group.get('grid_branch_surface_kind', '')}`: runs=`{group.get('artifact_count', 0)}`, "
                f"mean_best_val=`{canonical._safe_float(group.get('mean_best_val_accuracy'), 0.0):.6f}`, "
                f"latest_topology=`{group.get('latest_grid_tower_topology_kind', '')}`",
            ]
        )
    lines.extend(["", "## Branch/Tower Groups", ""])
    for group in branch_tower_groups:
        lines.append(
            f"- `{group.get('surface_key', '')}`: runs=`{group.get('artifact_count', 0)}`, "
            f"mean_best_val=`{canonical._safe_float(group.get('mean_best_val_accuracy'), 0.0):.6f}`"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate UVNet weekly training summary.")
    parser.add_argument(
        "--artifacts-dir",
        default="reports/uvnet_training",
        help="Directory containing UVNet training metrics artifacts.",
    )
    parser.add_argument(
        "--artifact-glob",
        default="*.json",
        help="Glob pattern used under --artifacts-dir.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to include in the weekly summary window.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/uvnet_training/uvnet_training_weekly_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--output-md",
        default="reports/uvnet_training/uvnet_training_weekly_summary.md",
        help="Output Markdown summary path.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    artifacts_dir = Path(str(args.artifacts_dir))
    rows = canonical._collect_artifacts(artifacts_dir, str(args.artifact_glob))
    window_now = _resolve_window_reference_now(rows)
    recent_rows = _filter_recent_rows(rows, days=int(args.days), now=window_now)
    summary = _build_weekly_summary(
        recent_rows,
        artifacts_dir=artifacts_dir,
        artifact_glob=str(args.artifact_glob),
        days=int(args.days),
        now=window_now,
    )
    summary["branch_groups"] = _build_branch_groups(recent_rows)
    summary["branch_tower_groups"] = _build_branch_tower_groups(recent_rows)

    output_json = Path(str(args.output_json))
    output_md = Path(str(args.output_md))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(summary), encoding="utf-8")
    print(f"UVNet training weekly summary JSON: {output_json}")
    print(f"UVNet training weekly summary Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
