from __future__ import annotations

import json
import sys


def test_seed_sweep_skips_when_dxf_dir_missing(tmp_path) -> None:
    from scripts import sweep_graph2d_profile_seeds as mod

    missing_dxf_dir = tmp_path / "not_found_dxf_dir"
    work_root = tmp_path / "seed_sweep_out"

    old_argv = sys.argv
    try:
        sys.argv = [
            "sweep_graph2d_profile_seeds.py",
            "--dxf-dir",
            str(missing_dxf_dir),
            "--work-root",
            str(work_root),
            "--seeds",
            "7,21",
            "--missing-dxf-dir-mode",
            "skip",
        ]
        rc = mod.main()
    finally:
        sys.argv = old_argv

    assert rc == 0

    summary_json = work_root / "seed_sweep_summary.json"
    results_json = work_root / "seed_sweep_results.json"
    assert summary_json.exists()
    assert results_json.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["status"] == "skipped_no_data"
    assert summary["gate"]["skipped"] is True
    assert summary["num_runs"] == 0


def test_seed_sweep_fails_when_mode_is_fail_and_dxf_dir_missing(tmp_path) -> None:
    from scripts import sweep_graph2d_profile_seeds as mod

    missing_dxf_dir = tmp_path / "not_found_dxf_dir"
    work_root = tmp_path / "seed_sweep_out"

    old_argv = sys.argv
    try:
        sys.argv = [
            "sweep_graph2d_profile_seeds.py",
            "--dxf-dir",
            str(missing_dxf_dir),
            "--work-root",
            str(work_root),
            "--seeds",
            "7,21",
            "--missing-dxf-dir-mode",
            "fail",
        ]
        rc = mod.main()
    finally:
        sys.argv = old_argv

    assert rc == 2
    assert not (work_root / "seed_sweep_summary.json").exists()
