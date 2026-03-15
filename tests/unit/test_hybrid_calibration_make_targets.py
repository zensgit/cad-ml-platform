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


def test_make_n_hybrid_calibrate_confidence_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-calibrate-confidence")
    assert result.returncode == 0, result.stderr
    assert "scripts/calibrate_hybrid_confidence.py" in result.stdout
    assert "--input-csv" in result.stdout
    assert "--output-json" in result.stdout
    assert "--method" in result.stdout
    assert "--per-source" in result.stdout
    assert "--confidence-col" in result.stdout
    assert "--correct-col" in result.stdout
    assert "--min-samples" in result.stdout


def test_make_n_hybrid_calibration_gate_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-calibration-gate")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_hybrid_confidence_calibration_gate.py" in result.stdout
    assert "--current-json" in result.stdout
    assert "--baseline-json" in result.stdout
    assert "--config" in result.stdout
    assert "--missing-mode" in result.stdout
    assert "--output-json" in result.stdout


def test_make_n_update_hybrid_calibration_baseline_contains_expected_flags() -> None:
    result = _run_make("-n", "update-hybrid-calibration-baseline")
    assert result.returncode == 0, result.stderr
    assert (
        "scripts/ci/update_hybrid_confidence_calibration_baseline.py" in result.stdout
    )
    assert "--current-json" in result.stdout
    assert "--output-baseline-json" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_refresh_hybrid_calibration_baseline_runs_calibrate_then_update() -> (
    None
):
    result = _run_make("-n", "refresh-hybrid-calibration-baseline")
    assert result.returncode == 0, result.stderr
    assert "make hybrid-calibrate-confidence" in result.stdout.lower()
    assert "make update-hybrid-calibration-baseline" in result.stdout.lower()
    assert "HYBRID_CALIBRATION_MIN_SAMPLES" in result.stdout
    assert "HYBRID_CALIBRATION_BASELINE_SOURCE_JSON" in result.stdout


def test_make_n_validate_hybrid_calibration_workflow_runs_expected_tests() -> None:
    result = _run_make("-n", "validate-hybrid-calibration-workflow")
    assert result.returncode == 0, result.stderr
    assert "test_calibrate_hybrid_confidence_script.py" in result.stdout
    assert "test_hybrid_confidence_calibration_gate_check.py" in result.stdout
    assert "test_hybrid_confidence_calibration_baseline_update.py" in result.stdout
    assert "test_hybrid_calibration_make_targets.py" in result.stdout
    assert "test_evaluation_report_workflow_graph2d_extensions.py" in result.stdout
    assert "test_ci_workflow_hybrid_calibration_regression_step.py" in result.stdout


def test_make_n_hybrid_blind_eval_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-blind-eval", "HYBRID_BLIND_DXF_DIR=/tmp/dxf")
    assert result.returncode == 0, result.stderr
    assert "scripts/batch_analyze_dxf_local.py" in result.stdout
    assert "scripts/ci/build_hybrid_blind_synthetic_dxf_dataset.py" in result.stdout
    assert "--dxf-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--geometry-only" in result.stdout


def test_make_n_hybrid_blind_build_synth_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-blind-build-synth")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/build_hybrid_blind_synthetic_dxf_dataset.py" in result.stdout
    assert "--manifest" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--max-files" in result.stdout


def test_make_n_hybrid_blind_gate_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-blind-gate")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_hybrid_blind_gate.py" in result.stdout
    assert "--summary-json" in result.stdout
    assert "--config" in result.stdout
    assert "--output" in result.stdout


def test_make_n_hybrid_blind_history_bootstrap_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-blind-history-bootstrap")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/bootstrap_hybrid_blind_eval_history.py" in result.stdout
    assert "--summary-json" in result.stdout
    assert "--gate-report-json" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--count" in result.stdout
    assert "--hours-step" in result.stdout
    assert "--hybrid-accuracy-deltas" in result.stdout
    assert "--graph2d-accuracy-deltas" in result.stdout
    assert "--coverage-deltas" in result.stdout
    assert "--label-slice-min-support" in result.stdout
    assert "--family-prefix-len" in result.stdout
    assert "--family-slice-max-slices" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_hybrid_blind_drift_activate_chains_bootstrap_and_drift() -> None:
    result = _run_make("-n", "hybrid-blind-drift-activate")
    assert result.returncode == 0, result.stderr
    assert "make hybrid-blind-history-bootstrap" in result.stdout.lower()
    assert "make hybrid-blind-drift-alert" in result.stdout.lower()


def test_make_n_hybrid_blind_strict_real_contains_expected_steps() -> None:
    result = _run_make("-n", "hybrid-blind-strict-real", "HYBRID_BLIND_DXF_DIR=/tmp")
    assert result.returncode == 0, result.stderr
    assert "Running strict hybrid blind flow with real DXF dataset" in result.stdout
    assert "make hybrid-blind-eval" in result.stdout.lower()
    assert "make hybrid-blind-gate" in result.stdout.lower()


def test_make_n_hybrid_blind_strict_real_e2e_gh_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-blind-strict-real-e2e-gh")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/dispatch_hybrid_blind_strict_real_workflow.py" in result.stdout
    assert "--workflow" in result.stdout
    assert "--ref" in result.stdout
    assert "--repo" in result.stdout
    assert "--hybrid-blind-dxf-dir" in result.stdout
    assert "--strict-fail-on-gate-failed true" in result.stdout
    assert "--strict-require-real-data true" in result.stdout
    assert "--expected-conclusion" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_hybrid_blind_strict_real_template_gh_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-blind-strict-real-template-gh")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/print_hybrid_blind_strict_real_gh_template.py" in result.stdout
    assert "--workflow" in result.stdout
    assert "--ref" in result.stdout
    assert "--dxf-dir" in result.stdout
    assert "--print-vars --print-watch" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_hybrid_blind_strict_real_apply_gh_vars_contains_expected_flags() -> (
    None
):
    result = _run_make("-n", "hybrid-blind-strict-real-apply-gh-vars")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/apply_hybrid_blind_strict_real_gh_vars.py" in result.stdout
    assert "--repo" in result.stdout
    assert "--dxf-dir" in result.stdout
    assert "--apply" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_hybrid_blind_drift_alert_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-blind-drift-alert")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_hybrid_blind_drift_alerts.py" in result.stdout
    assert "--eval-history-dir" in result.stdout
    assert "--output-json" in result.stdout
    assert "--output-md" in result.stdout
    assert "--min-reports" in result.stdout
    assert "--max-hybrid-accuracy-drop" in result.stdout
    assert "--max-gain-drop" in result.stdout
    assert "--max-coverage-drop" in result.stdout
    assert "--consecutive-drop-window" in result.stdout
    assert "--label-slice-enable" in result.stdout
    assert "--label-slice-min-common" in result.stdout
    assert "--label-slice-auto-cap-min-common" in result.stdout
    assert "--no-label-slice-auto-cap-min-common" in result.stdout
    assert "--label-slice-min-support" in result.stdout
    assert "--label-slice-max-hybrid-accuracy-drop" in result.stdout
    assert "--label-slice-max-gain-drop" in result.stdout
    assert "--family-slice-enable" in result.stdout
    assert "--family-slice-min-common" in result.stdout
    assert "--family-slice-auto-cap-min-common" in result.stdout
    assert "--no-family-slice-auto-cap-min-common" in result.stdout
    assert "--family-slice-min-support" in result.stdout
    assert "--family-slice-max-hybrid-accuracy-drop" in result.stdout
    assert "--family-slice-max-gain-drop" in result.stdout
    assert "$extra_flags" in result.stdout


def test_make_n_hybrid_blind_drift_suggest_thresholds_contains_expected_flags() -> None:
    result = _run_make("-n", "hybrid-blind-drift-suggest-thresholds")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/suggest_hybrid_blind_drift_thresholds.py" in result.stdout
    assert "--eval-history-dir" in result.stdout
    assert "--output-json" in result.stdout
    assert "--output-md" in result.stdout
    assert "--quantile" in result.stdout
    assert "--min-reports" in result.stdout
    assert "--safety-multiplier" in result.stdout
    assert "--label-slice-min-support" in result.stdout
    assert "--family-slice-min-support" in result.stdout
    assert "--min-floor-acc-drop" in result.stdout
    assert "--min-floor-gain-drop" in result.stdout
    assert "--min-floor-coverage-drop" in result.stdout
    assert "--floor-label-acc-drop" in result.stdout
    assert "--floor-label-gain-drop" in result.stdout
    assert "--floor-family-acc-drop" in result.stdout
    assert "--floor-family-gain-drop" in result.stdout


def test_make_n_hybrid_blind_drift_apply_suggestion_gh_contains_expected_flags() -> (
    None
):
    result = _run_make("-n", "hybrid-blind-drift-apply-suggestion-gh")
    assert result.returncode == 0, result.stderr
    assert (
        "scripts/ci/apply_hybrid_blind_drift_suggestion_to_gh_vars.py" in result.stdout
    )
    assert "--suggestion-json" in result.stdout
    assert "--repo" in result.stdout
    assert "--apply" in result.stdout


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
    assert "--ref" in result.stdout
    assert "--repo" in result.stdout
    assert "--hybrid-superpass-enable" in result.stdout
    assert "--hybrid-superpass-missing-mode" in result.stdout
    assert "--hybrid-superpass-fail-on-failed" in result.stdout
    assert "--hybrid-superpass-validation-strict" in result.stdout
    assert "--expected-conclusion" in result.stdout
    assert "--wait-timeout-seconds" in result.stdout
    assert "--poll-interval-seconds" in result.stdout
    assert "--list-limit" in result.stdout
    assert "--output-json" in result.stdout
    assert "$extra_flags" in result.stdout


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
    assert "test_apply_hybrid_superpass_gh_vars.py" in result.stdout
    assert "test_check_hybrid_superpass_targets.py" in result.stdout
    assert "test_validate_hybrid_superpass_reports.py" in result.stdout
    assert "test_evaluation_report_workflow_hybrid_superpass_step.py" in result.stdout
    assert "test_hybrid_superpass_workflow_integration.py" in result.stdout
    assert "test_hybrid_calibration_make_targets.py" in result.stdout
    assert "test_evaluation_report_workflow_graph2d_extensions.py" in result.stdout


def test_make_n_validate_hybrid_blind_strict_real_e2e_gh_runs_expected_tests() -> None:
    result = _run_make("-n", "validate-hybrid-blind-strict-real-e2e-gh")
    assert result.returncode == 0, result.stderr
    assert "test_dispatch_hybrid_blind_strict_real_workflow.py" in result.stdout
    assert "test_print_hybrid_blind_strict_real_gh_template.py" in result.stdout
    assert "test_hybrid_calibration_make_targets.py" in result.stdout


def test_make_n_eval_weekly_summary_contains_expected_flags() -> None:
    result = _run_make("-n", "eval-weekly-summary")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/generate_eval_weekly_summary.py" in result.stdout
    assert "--eval-history-dir" in result.stdout
    assert "--output-md" in result.stdout
    assert "--days" in result.stdout


def test_make_n_validate_hybrid_blind_workflow_runs_expected_tests() -> None:
    result = _run_make("-n", "validate-hybrid-blind-workflow")
    assert result.returncode == 0, result.stderr
    assert "test_dispatch_hybrid_blind_strict_real_workflow.py" in result.stdout
    assert "test_print_hybrid_blind_strict_real_gh_template.py" in result.stdout
    assert "test_check_hybrid_blind_drift_alerts.py" in result.stdout
    assert "test_build_hybrid_blind_synthetic_dxf_dataset.py" in result.stdout
    assert "test_hybrid_blind_gate_check.py" in result.stdout
    assert "test_archive_hybrid_blind_eval_history.py" in result.stdout
    assert "test_bootstrap_hybrid_blind_eval_history.py" in result.stdout
    assert "test_apply_hybrid_blind_drift_suggestion_to_gh_vars.py" in result.stdout
    assert "test_suggest_hybrid_blind_drift_thresholds.py" in result.stdout
    assert "test_validate_eval_history_hybrid_blind.py" in result.stdout
    assert "test_generate_eval_weekly_summary.py" in result.stdout
    assert "test_check_hybrid_superpass_targets.py" in result.stdout
    assert "test_evaluation_report_workflow_hybrid_superpass_step.py" in result.stdout
    assert "test_hybrid_calibration_make_targets.py" in result.stdout
    assert "test_evaluation_report_workflow_graph2d_extensions.py" in result.stdout
