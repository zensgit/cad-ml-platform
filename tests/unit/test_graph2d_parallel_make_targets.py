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


def test_make_n_hybrid_superpass_gate_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-superpass-gate")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_hybrid_superpass_targets.py" in result.stdout
    assert "--config" in result.stdout
    assert "--missing-mode" in result.stdout
    assert "--output" in result.stdout
    assert "--hybrid-blind-gate-report" in result.stdout
    assert "--hybrid-calibration-json" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_hybrid_superpass_e2e_gh_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-superpass-e2e-gh")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/dispatch_hybrid_superpass_workflow.py" in result.stdout
    assert "--workflow" in result.stdout
    assert "hybrid-superpass-e2e.yml" in result.stdout
    assert "--ref" in result.stdout
    assert "--repo" in result.stdout
    assert "--hybrid-superpass-enable" in result.stdout
    assert "--hybrid-superpass-missing-mode" in result.stdout
    assert "--hybrid-superpass-fail-on-failed" in result.stdout
    assert "--expected-conclusion" in result.stdout
    assert "--wait-timeout-seconds" in result.stdout
    assert "--poll-interval-seconds" in result.stdout
    assert "--list-limit" in result.stdout
    assert "--output-json" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_hybrid_superpass_compare_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-superpass-compare")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/compare_hybrid_superpass_reports.py" in result.stdout
    assert "--fail-json" in result.stdout
    assert "--success-json" in result.stdout
    assert "--output-json" in result.stdout
    assert "--output-md" in result.stdout
    assert "--strict" in result.stdout
    assert "--strict-require-distinct-run-ids" in result.stdout


def test_make_n_hybrid_superpass_compare_supports_trace_pair_flag() -> None:
    result = _run_make(
        "-n",
        "hybrid-superpass-compare",
        "HYBRID_SUPERPASS_COMPARE_STRICT_REQUIRE_TRACE_PAIR=1",
    )
    assert result.returncode == 0, result.stderr
    assert "--strict-require-trace-pair" in result.stdout


def test_make_n_hybrid_superpass_e2e_dual_gh_contains_expected_steps() -> None:
    result = _run_make("-n", "hybrid-superpass-e2e-dual-gh")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/run_hybrid_superpass_dual_dispatch.py" in result.stdout
    assert "--fail-output-json" in result.stdout
    assert "--success-output-json" in result.stdout
    assert "--compare-output-json" in result.stdout
    assert "--compare-output-md" in result.stdout
    assert "--strict" in result.stdout
    assert "--strict-require-distinct-run-ids" in result.stdout
    assert "--strict-require-trace-pair" in result.stdout


def test_make_n_hybrid_superpass_e2e_dual_gh_sequential_contains_expected_steps() -> None:
    result = _run_make("-n", "hybrid-superpass-e2e-dual-gh-sequential")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("scripts/ci/dispatch_hybrid_superpass_workflow.py") >= 2
    assert "--hybrid-superpass-missing-mode \"fail\"" in result.stdout
    assert "--hybrid-superpass-missing-mode \"skip\"" in result.stdout
    assert "make hybrid-superpass-compare" in result.stdout.lower()


def test_make_n_hybrid_superpass_nightly_gh_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-superpass-nightly-gh")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/dispatch_hybrid_superpass_nightly_workflow.py" in result.stdout
    assert "--workflow" in result.stdout
    assert "hybrid-superpass-nightly.yml" in result.stdout
    assert "--ref" in result.stdout
    assert "--target-repo" in result.stdout
    assert "--target-ref" in result.stdout
    assert "--target-workflow" in result.stdout
    assert "--expected-conclusion" in result.stdout
    assert "--wait-timeout-seconds" in result.stdout
    assert "--poll-interval-seconds" in result.stdout
    assert "--list-limit" in result.stdout
    assert "--output-json" in result.stdout


def test_make_n_hybrid_superpass_nightly_gh_print_only_flag() -> None:
    result = _run_make(
        "-n",
        "hybrid-superpass-nightly-gh",
        "HYBRID_SUPERPASS_NIGHTLY_PRINT_ONLY=1",
    )
    assert result.returncode == 0, result.stderr
    assert "--print-only" in result.stdout


def test_make_n_hybrid_superpass_apply_gh_vars_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-superpass-apply-gh-vars")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/apply_hybrid_superpass_gh_vars.py" in result.stdout
    assert "--repo" in result.stdout
    assert "--config-path" in result.stdout
    assert "--apply" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_validate_hybrid_superpass_workflow_runs_expected_tests() -> None:
    result = _run_make("-n", "validate-hybrid-superpass-workflow")
    assert result.returncode == 0, result.stderr
    assert "test_dispatch_hybrid_superpass_workflow.py" in result.stdout
    assert "test_run_hybrid_superpass_dual_dispatch.py" in result.stdout
    assert "test_apply_hybrid_superpass_gh_vars.py" in result.stdout
    assert "test_check_hybrid_superpass_targets.py" in result.stdout
    assert "test_compare_hybrid_superpass_reports.py" in result.stdout
    assert "test_evaluation_report_workflow_hybrid_superpass_step.py" in result.stdout
    assert "test_hybrid_superpass_workflow_integration.py" in result.stdout
    assert "test_graph2d_parallel_make_targets.py" in result.stdout


def test_make_n_validate_hybrid_superpass_nightly_workflow_runs_expected_tests() -> None:
    result = _run_make("-n", "validate-hybrid-superpass-nightly-workflow")
    assert result.returncode == 0, result.stderr
    assert "test_dispatch_hybrid_superpass_nightly_workflow.py" in result.stdout
    assert "test_hybrid_superpass_nightly_workflow.py" in result.stdout
    assert "test_graph2d_parallel_make_targets.py" in result.stdout
