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


def test_make_n_validate_workflow_file_health_contains_expected_flags() -> None:
    result = _run_make("-n", "validate-workflow-file-health")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_workflow_file_issues.py" in result.stdout
    assert '--glob ".github/workflows/*.yml"' in result.stdout
    assert '--ref "HEAD"' in result.stdout
    assert '--mode "auto"' in result.stdout
    assert '--summary-json-out "reports/ci/workflow_file_health_summary.json"' in result.stdout


def test_make_n_validate_workflow_file_health_tests_contains_expected_files() -> None:
    result = _run_make("-n", "validate-workflow-file-health-tests")
    assert result.returncode == 0, result.stderr
    assert "tests/unit/test_check_workflow_file_issues.py" in result.stdout
    assert "tests/unit/test_stress_workflow_workflow_file_health.py" in result.stdout
    assert "tests/unit/test_workflow_file_health_make_target.py" in result.stdout


def test_make_n_validate_workflow_identity_contains_expected_flags() -> None:
    result = _run_make("-n", "validate-workflow-identity")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/check_workflow_identity_invariants.py" in result.stdout
    assert '--workflow-root ".github/workflows"' in result.stdout
    assert '--ci-watch-required-workflows "' in result.stdout
    assert '--summary-json-out "reports/ci/workflow_identity_summary.json"' in result.stdout


def test_make_n_validate_workflow_identity_tests_contains_expected_files() -> None:
    result = _run_make("-n", "validate-workflow-identity-tests")
    assert result.returncode == 0, result.stderr
    assert "tests/unit/test_check_workflow_identity_invariants.py" in result.stdout
    assert "tests/unit/test_workflow_file_health_make_target.py" in result.stdout


def test_make_n_workflow_inventory_report_contains_expected_flags() -> None:
    result = _run_make("-n", "workflow-inventory-report")
    assert result.returncode == 0, result.stderr
    assert "scripts/ci/generate_workflow_inventory_report.py" in result.stdout
    assert '--workflow-root ".github/workflows"' in result.stdout
    assert '--ci-watch-required-workflows "' in result.stdout
    assert '--output-json "reports/ci/workflow_inventory_report.json"' in result.stdout
    assert '--output-md "reports/ci/workflow_inventory_report.md"' in result.stdout


def test_make_n_validate_workflow_inventory_report_contains_expected_files() -> None:
    result = _run_make("-n", "validate-workflow-inventory-report")
    assert result.returncode == 0, result.stderr
    assert "tests/unit/test_generate_workflow_inventory_report.py" in result.stdout
    assert "tests/unit/test_workflow_file_health_make_target.py" in result.stdout


def test_make_n_validate_ci_watchers_invokes_workflow_file_health_tests() -> None:
    result = _run_make("-n", "validate-ci-watchers")
    assert result.returncode == 0, result.stderr
    assert "make validate-workflow-file-health-tests" in result.stdout
    assert "make validate-workflow-identity-tests" in result.stdout
    assert "make validate-workflow-inventory-report" in result.stdout
