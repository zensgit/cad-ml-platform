#!/usr/bin/env python3
"""Summarize UVNet training metrics artifacts into a canonical experiment report."""

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


def _extract_model_surface_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    surface = payload.get("model_surface_contract")
    if isinstance(surface, dict):
        return dict(surface)
    legacy = payload.get("graph_tensor_config")
    if isinstance(legacy, dict):
        return {
            "use_face_grid_features": bool(legacy.get("use_face_grid_features", False)),
            "use_edge_grid_features": bool(legacy.get("use_edge_grid_features", False)),
            "grid_encoder_kind": str(legacy.get("grid_encoder_kind", "")),
            "grid_fusion_mode": str(legacy.get("grid_fusion_mode", "")),
            "grid_branch_surface_kind": str(legacy.get("grid_branch_surface_kind", "")),
            "grid_tower_topology_kind": str(legacy.get("grid_tower_topology_kind", "")),
        }
    return {}


def _build_training_summary_from_epoch_history(epoch_history: Any) -> Dict[str, Any]:
    if not isinstance(epoch_history, list) or not epoch_history:
        return {
            "epochs_completed": 0,
            "final_train_loss": 0.0,
            "final_train_accuracy": 0.0,
            "best_train_accuracy": 0.0,
            "best_train_accuracy_epoch": 0,
            "final_val_loss": 0.0,
            "final_val_accuracy": 0.0,
            "best_val_accuracy": 0.0,
            "best_val_accuracy_epoch": 0,
        }
    best_train_entry = max(epoch_history, key=lambda item: _safe_float(item.get("accuracy"), 0.0))
    best_val_entry = max(
        epoch_history, key=lambda item: _safe_float(item.get("val_accuracy"), 0.0)
    )
    final_entry = dict(epoch_history[-1])
    return {
        "epochs_completed": int(len(epoch_history)),
        "final_train_loss": _safe_float(final_entry.get("loss"), 0.0),
        "final_train_accuracy": _safe_float(final_entry.get("accuracy"), 0.0),
        "best_train_accuracy": _safe_float(best_train_entry.get("accuracy"), 0.0),
        "best_train_accuracy_epoch": int(best_train_entry.get("epoch", 0) or 0),
        "final_val_loss": _safe_float(final_entry.get("val_loss"), 0.0),
        "final_val_accuracy": _safe_float(final_entry.get("val_accuracy"), 0.0),
        "best_val_accuracy": _safe_float(best_val_entry.get("val_accuracy"), 0.0),
        "best_val_accuracy_epoch": int(best_val_entry.get("epoch", 0) or 0),
    }


def _extract_training_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("training_summary")
    if isinstance(summary, dict):
        return dict(summary)
    return _build_training_summary_from_epoch_history(payload.get("epoch_history"))


def _load_artifact(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status") or "").strip() != "ok":
        return None
    surface_kind = str(payload.get("surface_kind") or "")
    if surface_kind and surface_kind != "uvnet_training_metrics_artifact":
        return None
    model_surface_contract = _extract_model_surface_contract(payload)
    training_summary = _extract_training_summary(payload)
    selection_metric_value = _safe_float(training_summary.get("best_val_accuracy"), 0.0)
    selection_metric_kind = "best_val_accuracy"
    if selection_metric_value <= 0.0:
        selection_metric_value = _safe_float(training_summary.get("final_train_accuracy"), 0.0)
        selection_metric_kind = "final_train_accuracy"
    return {
        "artifact_path": str(path),
        "generated_at": str(payload.get("generated_at") or ""),
        "checkpoint_path": str(payload.get("checkpoint_path") or ""),
        "model_surface_contract": model_surface_contract,
        "training_summary": training_summary,
        "selection_metric_kind": selection_metric_kind,
        "selection_metric_value": selection_metric_value,
    }


def _collect_artifacts(artifacts_dir: Path, artifact_glob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not artifacts_dir.exists():
        return rows
    for path in sorted(artifacts_dir.rglob(str(artifact_glob))):
        artifact = _load_artifact(path)
        if artifact is not None:
            rows.append(artifact)
    return rows


def _count_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        contract = row.get("model_surface_contract")
        if not isinstance(contract, dict):
            continue
        value = str(contract.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _build_surface_matrix(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        contract = row.get("model_surface_contract")
        if not isinstance(contract, dict):
            continue
        branch = str(contract.get("grid_branch_surface_kind") or "")
        tower = str(contract.get("grid_tower_topology_kind") or "")
        key = f"{branch}::{tower}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _build_summary(rows: List[Dict[str, Any]], *, artifacts_dir: Path, artifact_glob: str) -> Dict[str, Any]:
    if not rows:
        return {
            "status": "ok",
            "surface_kind": "uvnet_training_experiment_summary",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artifacts_dir": str(artifacts_dir),
            "artifact_glob": artifact_glob,
            "artifact_count": 0,
            "aggregate_metrics": {},
            "surface_counts": {
                "grid_branch_surface_kind": {},
                "grid_tower_topology_kind": {},
                "branch_tower_matrix": {},
            },
            "best_run": None,
            "latest_run": None,
        }
    latest_row = max(rows, key=lambda row: _parse_ts(row.get("generated_at")))
    best_row = max(rows, key=lambda row: _safe_float(row.get("selection_metric_value"), 0.0))
    return {
        "status": "ok",
        "surface_kind": "uvnet_training_experiment_summary",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts_dir": str(artifacts_dir),
        "artifact_glob": artifact_glob,
        "artifact_count": int(len(rows)),
        "aggregate_metrics": {
            "mean_final_train_accuracy": sum(
                _safe_float(row["training_summary"].get("final_train_accuracy"), 0.0)
                for row in rows
            )
            / float(len(rows)),
            "mean_final_val_accuracy": sum(
                _safe_float(row["training_summary"].get("final_val_accuracy"), 0.0) for row in rows
            )
            / float(len(rows)),
            "mean_best_val_accuracy": sum(
                _safe_float(row["training_summary"].get("best_val_accuracy"), 0.0) for row in rows
            )
            / float(len(rows)),
        },
        "surface_counts": {
            "grid_branch_surface_kind": _count_by_key(rows, "grid_branch_surface_kind"),
            "grid_tower_topology_kind": _count_by_key(rows, "grid_tower_topology_kind"),
            "branch_tower_matrix": _build_surface_matrix(rows),
        },
        "best_run": dict(best_row),
        "latest_run": dict(latest_row),
    }


def _build_markdown(summary: Dict[str, Any]) -> str:
    best_run = summary.get("best_run") if isinstance(summary.get("best_run"), dict) else {}
    latest_run = summary.get("latest_run") if isinstance(summary.get("latest_run"), dict) else {}
    best_contract = (
        best_run.get("model_surface_contract")
        if isinstance(best_run.get("model_surface_contract"), dict)
        else {}
    )
    latest_contract = (
        latest_run.get("model_surface_contract")
        if isinstance(latest_run.get("model_surface_contract"), dict)
        else {}
    )
    aggregate = summary.get("aggregate_metrics") if isinstance(summary.get("aggregate_metrics"), dict) else {}
    lines = [
        "# UVNet Training Experiment Summary",
        "",
        f"- Artifacts dir: `{summary.get('artifacts_dir', '')}`",
        f"- Artifact glob: `{summary.get('artifact_glob', '')}`",
        f"- Artifact count: `{summary.get('artifact_count', 0)}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean final train accuracy | { _safe_float(aggregate.get('mean_final_train_accuracy'), 0.0):.6f} |",
        f"| Mean final val accuracy | { _safe_float(aggregate.get('mean_final_val_accuracy'), 0.0):.6f} |",
        f"| Mean best val accuracy | { _safe_float(aggregate.get('mean_best_val_accuracy'), 0.0):.6f} |",
        "",
        "## Best Run",
        "",
        f"- Artifact: `{best_run.get('artifact_path', '')}`",
        f"- Selection metric: `{best_run.get('selection_metric_kind', '')}` = `{_safe_float(best_run.get('selection_metric_value'), 0.0):.6f}`",
        f"- Grid branch surface: `{best_contract.get('grid_branch_surface_kind', '')}`",
        f"- Grid tower topology: `{best_contract.get('grid_tower_topology_kind', '')}`",
        "",
        "## Latest Run",
        "",
        f"- Artifact: `{latest_run.get('artifact_path', '')}`",
        f"- Generated at: `{latest_run.get('generated_at', '')}`",
        f"- Grid branch surface: `{latest_contract.get('grid_branch_surface_kind', '')}`",
        f"- Grid tower topology: `{latest_contract.get('grid_tower_topology_kind', '')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize UVNet training metrics artifacts.")
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
        "--output-json",
        default="",
        help="Output JSON summary path (default: <artifacts-dir>/uvnet_training_experiment_summary.json).",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Output markdown summary path (default: <artifacts-dir>/uvnet_training_experiment_summary.md).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    artifacts_dir = Path(str(args.artifacts_dir))
    output_json = (
        Path(str(args.output_json))
        if str(args.output_json).strip()
        else artifacts_dir / "uvnet_training_experiment_summary.json"
    )
    output_md = (
        Path(str(args.output_md))
        if str(args.output_md).strip()
        else artifacts_dir / "uvnet_training_experiment_summary.md"
    )

    rows = _collect_artifacts(artifacts_dir, str(args.artifact_glob))
    summary = _build_summary(rows, artifacts_dir=artifacts_dir, artifact_glob=str(args.artifact_glob))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(summary), encoding="utf-8")
    print(f"UVNet training summary JSON: {output_json}")
    print(f"UVNet training summary Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
