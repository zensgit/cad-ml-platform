from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parse_seeds_and_recipes() -> None:
    from scripts.sweep_graph2d_train_recipes import _parse_recipes, _parse_seeds

    assert _parse_seeds("7,21,42") == [7, 21, 42]
    assert _parse_recipes("baseline,focal_balanced") == ["baseline", "focal_balanced"]
    with pytest.raises(ValueError):
        _parse_seeds("7,abc")


def test_build_train_command_includes_recipe_overrides(tmp_path: Path) -> None:
    from scripts.sweep_graph2d_train_recipes import _build_train_command

    checkpoint_path = tmp_path / "model.pth"
    metrics_path = tmp_path / "metrics.json"
    cmd = _build_train_command(
        python_exe="python3",
        train_script="scripts/train_2d_graph.py",
        base_args=["--epochs", "1"],
        recipe_name="focal_balanced",
        seed=22,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )
    assert cmd[:3] == ["python3", "scripts/train_2d_graph.py", "--epochs"]
    assert "--seed" in cmd
    assert "--metrics-out" in cmd
    assert "--loss" in cmd
    assert "focal" in cmd
    assert "--sampler" in cmd
    assert "balanced" in cmd


def test_main_plan_mode_writes_summary(tmp_path: Path) -> None:
    from scripts.sweep_graph2d_train_recipes import main

    work_root = tmp_path / "sweep_out"
    env_path = work_root / "best.env"
    exit_code = main(
        [
            "--recipes",
            "baseline",
            "--seeds",
            "7",
            "--work-root",
            str(work_root),
            "--recommended-env-out",
            str(env_path),
        ]
    )
    assert exit_code == 0
    summary_path = work_root / "train_recipe_sweep_summary.json"
    results_path = work_root / "train_recipe_sweep_results.json"
    assert summary_path.exists()
    assert results_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_runs"] == 1
    assert summary["executed_runs"] == 0
    assert summary["recommended_env_file"] == str(env_path)
    assert summary["best_run_script"] == str(work_root / "run_best_recipe.sh")
    assert isinstance(summary["best_args"], list)
    assert env_path.exists()
    script_path = work_root / "run_best_recipe.sh"
    assert script_path.exists()
    script_text = script_path.read_text(encoding="utf-8")
    assert "TRAIN_SCRIPT=" in script_text
    assert "--metrics-out" in script_text
    env_text = env_path.read_text(encoding="utf-8")
    assert "GRAPH2D_SWEEP_BEST_RECIPE=baseline" in env_text
    assert "GRAPH2D_SWEEP_BEST_ARGS=" in env_text
