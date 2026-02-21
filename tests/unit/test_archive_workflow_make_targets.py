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


def test_make_n_dry_run_target_contains_wait_timeout_and_poll_flags() -> None:
    result = _run_make("-n", "archive-workflow-dry-run-gh", "ARCHIVE_WORKFLOW_WATCH=1")
    assert result.returncode == 0, result.stderr
    assert '--wait-timeout-seconds "120"' in result.stdout
    assert '--poll-interval-seconds "3"' in result.stdout


def test_make_n_apply_target_contains_required_flags() -> None:
    result = _run_make(
        "-n",
        "archive-workflow-apply-gh",
        "ARCHIVE_APPROVAL_PHRASE=I_UNDERSTAND_DELETE_SOURCE",
        "ARCHIVE_WORKFLOW_WATCH=1",
    )
    assert result.returncode == 0, result.stderr
    assert '--approval-phrase "${ARCHIVE_APPROVAL_PHRASE}"' in result.stdout
    assert '--wait-timeout-seconds "120"' in result.stdout
    assert '--poll-interval-seconds "3"' in result.stdout


def test_make_print_only_dry_run_watch_outputs_dispatch_and_watch_commands() -> None:
    result = _run_make(
        "archive-workflow-dry-run-gh",
        "ARCHIVE_WORKFLOW_PRINT_ONLY=1",
        "ARCHIVE_WORKFLOW_WATCH=1",
        "ARCHIVE_WORKFLOW_WAIT_TIMEOUT=30",
        "ARCHIVE_WORKFLOW_POLL_INTERVAL=2",
    )
    assert result.returncode == 0, result.stderr
    assert "gh workflow run 'Experiment Archive Dry Run'" in result.stdout
    assert "gh run list --workflow 'Experiment Archive Dry Run'" in result.stdout
    assert "gh run watch <run_id> --exit-status" in result.stdout


def test_make_print_only_apply_watch_outputs_dispatch_and_watch_commands() -> None:
    result = _run_make(
        "archive-workflow-apply-gh",
        "ARCHIVE_APPROVAL_PHRASE=I_UNDERSTAND_DELETE_SOURCE",
        "ARCHIVE_WORKFLOW_PRINT_ONLY=1",
        "ARCHIVE_WORKFLOW_WATCH=1",
        "ARCHIVE_WORKFLOW_WAIT_TIMEOUT=30",
        "ARCHIVE_WORKFLOW_POLL_INTERVAL=2",
    )
    assert result.returncode == 0, result.stderr
    assert "gh workflow run 'Experiment Archive Apply'" in result.stdout
    assert "-f approval_phrase=I_UNDERSTAND_DELETE_SOURCE" in result.stdout
    assert "gh run list --workflow 'Experiment Archive Apply'" in result.stdout
    assert "gh run watch <run_id> --exit-status" in result.stdout
