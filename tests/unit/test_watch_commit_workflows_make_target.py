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


def test_make_n_watch_commit_workflows_contains_expected_flags() -> None:
    result = _run_make("-n", "watch-commit-workflows")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/watch_commit_workflows.py" in result.stdout
    assert '--sha "HEAD"' in result.stdout
    assert '--events-csv "push"' in result.stdout
    assert '--wait-timeout-seconds "1800"' in result.stdout
    assert '--poll-interval-seconds "20"' in result.stdout
    assert '--heartbeat-interval-seconds "120"' in result.stdout
    assert '--list-limit "100"' in result.stdout
    assert '--max-list-failures "3"' in result.stdout
    assert '--missing-required-mode "fail-fast"' in result.stdout
    assert '--failure-mode "fail-fast"' in result.stdout
    assert '--success-conclusions-csv "success,skipped"' in result.stdout
    assert '--summary-json-out ""' in result.stdout


def test_make_watch_commit_workflows_print_only_outputs_preview() -> None:
    result = _run_make(
        "watch-commit-workflows",
        "CI_WATCH_PRINT_ONLY=1",
        "CI_WATCH_SHA=abc123",
        "CI_WATCH_EVENTS=push,workflow_dispatch",
        "CI_WATCH_REQUIRED_WORKFLOWS=CI,Code Quality",
        "CI_WATCH_TIMEOUT=30",
        "CI_WATCH_POLL_INTERVAL=2",
        "CI_WATCH_LIST_LIMIT=50",
        "CI_WATCH_MAX_LIST_FAILURES=5",
        "CI_WATCH_SUCCESS_CONCLUSIONS=success,skipped,neutral",
        "CI_WATCH_SUMMARY_JSON=/tmp/ci-watch-summary.json",
    )
    assert result.returncode == 0, result.stderr
    assert "gh run list --json" in result.stdout
    assert "# events=['push', 'workflow_dispatch']" in result.stdout
    assert "# required_workflows=['CI', 'Code Quality']" in result.stdout
    assert "# success_conclusions=['neutral', 'skipped', 'success']" in result.stdout
    assert "# missing_required_mode=fail-fast" in result.stdout
    assert "# failure_mode=fail-fast" in result.stdout
    assert "# heartbeat_interval_seconds=120" in result.stdout
    assert "# max_list_failures=5" in result.stdout


def test_make_n_clean_ci_watch_summaries_contains_expected_pattern() -> None:
    result = _run_make("-n", "clean-ci-watch-summaries")
    assert result.returncode == 0, result.stderr
    assert "watch_*_summary.json" in result.stdout


def test_make_n_clean_gh_readiness_summaries_contains_expected_pattern() -> None:
    result = _run_make("-n", "clean-gh-readiness-summaries")
    assert result.returncode == 0, result.stderr
    assert "gh_readiness*.json" in result.stdout


def test_make_n_clean_ci_watch_artifacts_calls_both_targets() -> None:
    result = _run_make("-n", "clean-ci-watch-artifacts")
    assert result.returncode == 0, result.stderr
    assert "make clean-ci-watch-summaries" in result.stdout
    assert "make clean-gh-readiness-summaries" in result.stdout


def test_make_n_watch_commit_workflows_safe_runs_precheck_then_watch() -> None:
    result = _run_make("-n", "watch-commit-workflows-safe")
    assert result.returncode == 0, result.stderr
    assert "make check-gh-actions-ready" in result.stdout
    assert "make check-gh-actions-ready-soft" in result.stdout
    assert "make watch-commit-workflows" in result.stdout
    assert "CI_WATCH_PRECHECK_STRICT" in result.stdout


def test_make_n_check_gh_actions_ready_contains_json_and_skip_flag_logic() -> None:
    result = _run_make("-n", "check-gh-actions-ready")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_gh_actions_ready.py" in result.stdout
    assert '--json-out "reports/ci/gh_readiness_latest.json"' in result.stdout
    assert "$skip_actions_flag" in result.stdout
    assert "--skip-actions-api" in result.stdout


def test_make_n_check_gh_actions_ready_soft_contains_allow_fail() -> None:
    result = _run_make("-n", "check-gh-actions-ready-soft")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_gh_actions_ready.py" in result.stdout
    assert "--allow-fail" in result.stdout
