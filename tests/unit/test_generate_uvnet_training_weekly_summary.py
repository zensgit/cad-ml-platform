from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_uvnet_training_weekly_summary_filters_recent_rows(tmp_path: Path) -> None:
    from scripts.ci.generate_uvnet_training_weekly_summary import (
        _build_weekly_summary,
        _filter_recent_rows,
    )
    from scripts import summarize_uvnet_training_runs as canonical

    artifacts_dir = tmp_path / "uvnet"
    _write_json(
        artifacts_dir / "old.json",
        {
            "surface_kind": "uvnet_training_metrics_artifact",
            "status": "ok",
            "generated_at": "2026-03-20T08:00:00Z",
            "checkpoint_path": "models/old.pth",
            "model_surface_contract": {
                "grid_branch_surface_kind": "graph_only",
                "grid_tower_topology_kind": "graph_only",
            },
            "training_summary": {
                "final_train_accuracy": 0.5,
                "final_val_accuracy": 0.4,
                "best_val_accuracy": 0.45,
            },
        },
    )
    _write_json(
        artifacts_dir / "recent.json",
        {
            "surface_kind": "uvnet_training_metrics_artifact",
            "status": "ok",
            "generated_at": "2026-03-29T08:00:00Z",
            "checkpoint_path": "models/recent.pth",
            "model_surface_contract": {
                "grid_branch_surface_kind": "cnn_pool_concat_projection_dual_branch",
                "grid_tower_topology_kind": "graph_grid_dual_tower_projection",
            },
            "training_summary": {
                "final_train_accuracy": 0.8,
                "final_val_accuracy": 0.7,
                "best_val_accuracy": 0.72,
            },
        },
    )

    rows = canonical._collect_artifacts(artifacts_dir, "*.json")
    recent_rows = _filter_recent_rows(
        rows,
        days=7,
        now=datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert len(recent_rows) == 1
    summary = _build_weekly_summary(
        recent_rows,
        artifacts_dir=artifacts_dir,
        artifact_glob="*.json",
        days=7,
        now=datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert summary["surface_kind"] == "uvnet_training_weekly_summary"
    assert summary["artifact_count"] == 1
    assert summary["latest_run"]["checkpoint_path"] == "models/recent.pth"
    assert summary["surface_counts"]["grid_tower_topology_kind"] == {
        "graph_grid_dual_tower_projection": 1
    }


def test_uvnet_training_weekly_summary_main_writes_outputs(tmp_path: Path) -> None:
    from scripts.ci import generate_uvnet_training_weekly_summary as mod

    artifacts_dir = tmp_path / "uvnet"
    _write_json(
        artifacts_dir / "run.json",
        {
            "surface_kind": "uvnet_training_metrics_artifact",
            "status": "ok",
            "generated_at": "2026-03-29T08:00:00Z",
            "checkpoint_path": "models/run.pth",
            "model_surface_contract": {
                "grid_branch_surface_kind": "cnn_pool_concat_projection_dual_branch",
                "grid_tower_topology_kind": "graph_grid_dual_tower_projection",
            },
            "training_summary": {
                "final_train_accuracy": 0.82,
                "final_val_accuracy": 0.73,
                "best_val_accuracy": 0.75,
            },
        },
    )
    _write_json(
        artifacts_dir / "run_graph.json",
        {
            "surface_kind": "uvnet_training_metrics_artifact",
            "status": "ok",
            "generated_at": "2026-03-29T09:00:00Z",
            "checkpoint_path": "models/run_graph.pth",
            "model_surface_contract": {
                "grid_branch_surface_kind": "graph_only",
                "grid_tower_topology_kind": "graph_only",
            },
            "training_summary": {
                "final_train_accuracy": 0.61,
                "final_val_accuracy": 0.52,
                "best_val_accuracy": 0.57,
            },
        },
    )
    output_json = tmp_path / "weekly.json"
    output_md = tmp_path / "weekly.md"

    rc = mod.main(
        [
            "--artifacts-dir",
            str(artifacts_dir),
            "--days",
            "7",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 0
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["surface_kind"] == "uvnet_training_weekly_summary"
    assert summary["artifact_count"] == 2
    assert summary["latest_generated_at"] == "2026-03-29T09:00:00Z"
    assert len(summary["branch_groups"]) == 2
    assert len(summary["branch_tower_groups"]) == 2
    dual_branch_group = next(
        group
        for group in summary["branch_groups"]
        if group["grid_branch_surface_kind"] == "cnn_pool_concat_projection_dual_branch"
    )
    assert dual_branch_group["artifact_count"] == 1
    assert dual_branch_group["latest_grid_tower_topology_kind"] == (
        "graph_grid_dual_tower_projection"
    )
    text = output_md.read_text(encoding="utf-8")
    assert "UVNet Training Weekly Summary" in text
    assert "graph_grid_dual_tower_projection" in text
    assert "Branch Groups" in text
