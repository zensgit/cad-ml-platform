from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKE_BIN = shutil.which("make")

pytestmark = pytest.mark.skipif(MAKE_BIN is None, reason="make is not available")


def _run_make(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON"] = env.get("PYTHON", "python3")
    return subprocess.run(
        [MAKE_BIN, *args],  # type: ignore[list-item]
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_make_n_graph2d_review_pack_contains_expected_flags() -> None:
    result = _run_make("-n", "graph2d-review-pack")
    assert result.returncode == 0, result.stderr
    assert "scripts/export_hybrid_rejection_review_pack.py" in result.stdout
    assert '--input-csv "' in result.stdout
    assert '--output-csv "' in result.stdout
    assert "--low-confidence-threshold" in result.stdout
    assert "--top-k" in result.stdout


def test_make_n_graph2d_review_pack_gate_contains_expected_flags() -> None:
    result = _run_make("-n", "graph2d-review-pack-gate")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_graph2d_review_pack_gate.py" in result.stdout
    assert "--summary-json" in result.stdout
    assert "--config" in result.stdout
    assert "--output" in result.stdout


def test_make_n_graph2d_review_pack_gate_supports_override_flags() -> None:
    result = _run_make(
        "-n",
        "graph2d-review-pack-gate",
        "GRAPH2D_REVIEW_PACK_GATE_MAX_CANDIDATE_RATE=0.55",
        "GRAPH2D_REVIEW_PACK_GATE_MIN_TOTAL_ROWS=30",
    )
    assert result.returncode == 0, result.stderr
    assert "--max-candidate-rate 0.55" in result.stdout
    assert "--min-total-rows 30" in result.stdout


def test_make_n_graph2d_train_sweep_contains_expected_flags() -> None:
    result = _run_make("-n", "graph2d-train-sweep")
    assert result.returncode == 0, result.stderr
    assert "scripts/sweep_graph2d_train_recipes.py" in result.stdout
    assert '--recipes "' in result.stdout
    assert '--seeds "' in result.stdout
    assert "--base-args-json" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_graph2d_review_pack_gate_strict_e2e_contains_expected_flags() -> None:
    result = _run_make("-n", "graph2d-review-pack-gate-strict-e2e")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/dispatch_graph2d_review_gate_strict_e2e.py" in result.stdout
    assert '--workflow "' in result.stdout
    assert '--ref "' in result.stdout
    assert "--review-pack-input-csv" in result.stdout
    assert "--wait-timeout-seconds" in result.stdout
    assert "--poll-interval-seconds" in result.stdout
    assert "--list-limit" in result.stdout
    assert "--output-json" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_graph2d_review_pack_gate_strict_e2e_print_only_flag() -> None:
    result = _run_make(
        "-n",
        "graph2d-review-pack-gate-strict-e2e",
        "GRAPH2D_REVIEW_PACK_GATE_E2E_PRINT_ONLY=1",
    )
    assert result.returncode == 0, result.stderr
    assert "--print-only" in result.stdout


def test_make_n_validate_graph2d_review_pack_gate_strict_e2e_runs_expected_tests() -> None:
    result = _run_make("-n", "validate-graph2d-review-pack-gate-strict-e2e")
    assert result.returncode == 0, result.stderr
    assert "test_dispatch_graph2d_review_gate_strict_e2e.py" in result.stdout
    assert "test_graph2d_parallel_make_targets.py" in result.stdout
    assert "test_evaluation_report_workflow_graph2d_extensions.py" in result.stdout
