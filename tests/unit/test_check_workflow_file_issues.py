from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["check_workflow_file_issues.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def _write_workflow(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_yaml_mode_success(tmp_path: Path, monkeypatch: Any) -> None:
    from scripts.ci import check_workflow_file_issues as mod

    _write_workflow(
        tmp_path / ".github" / "workflows" / "ci.yml",
        "name: CI\non:\n  push:\n    branches: [main]\n",
    )
    monkeypatch.chdir(tmp_path)

    rc = _invoke_main(mod, ["--mode", "yaml", "--glob", ".github/workflows/*.yml"])
    assert rc == 0


def test_yaml_mode_missing_on_key_fails(tmp_path: Path, monkeypatch: Any) -> None:
    from scripts.ci import check_workflow_file_issues as mod

    _write_workflow(
        tmp_path / ".github" / "workflows" / "ci.yml",
        "name: CI\njobs: {}\n",
    )
    monkeypatch.chdir(tmp_path)

    rc = _invoke_main(mod, ["--mode", "yaml", "--glob", ".github/workflows/*.yml"])
    assert rc == 1


def test_gh_mode_requires_gh_binary(tmp_path: Path, monkeypatch: Any) -> None:
    from scripts.ci import check_workflow_file_issues as mod

    _write_workflow(
        tmp_path / ".github" / "workflows" / "ci.yml",
        "name: CI\non:\n  push:\n",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_is_gh_available", lambda: False)

    rc = _invoke_main(mod, ["--mode", "gh", "--glob", ".github/workflows/*.yml"])
    assert rc == 2


def test_auto_mode_fallbacks_to_yaml_on_auth_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.ci import check_workflow_file_issues as mod

    workflow = tmp_path / ".github" / "workflows" / "ci.yml"
    _write_workflow(
        workflow,
        "name: CI\non:\n  push:\n    branches: [main]\n",
    )
    summary = tmp_path / "summary.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_is_gh_available", lambda: True)
    monkeypatch.setattr(
        mod,
        "_check_with_gh",
        lambda path, ref: mod.WorkflowCheckResult(
            path=str(path),
            mode="gh",
            ok=False,
            message="Failed to log in to github.com account",
        ),
    )

    rc = _invoke_main(
        mod,
        [
            "--mode",
            "auto",
            "--glob",
            ".github/workflows/*.yml",
            "--summary-json-out",
            str(summary),
        ],
    )
    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["requested_mode"] == "auto"
    assert payload["mode_used"] == "yaml"
    assert payload["fallback_reason"] == "gh_auth_or_token_error"
    assert payload["failed_count"] == 0


def test_gh_mode_non_auth_failure_returns_error(tmp_path: Path, monkeypatch: Any) -> None:
    from scripts.ci import check_workflow_file_issues as mod

    _write_workflow(
        tmp_path / ".github" / "workflows" / "ci.yml",
        "name: CI\non:\n  push:\n",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_is_gh_available", lambda: True)
    monkeypatch.setattr(
        mod,
        "_check_with_gh",
        lambda path, ref: mod.WorkflowCheckResult(
            path=str(path),
            mode="gh",
            ok=False,
            message="invalid workflow file",
        ),
    )

    rc = _invoke_main(mod, ["--mode", "gh", "--glob", ".github/workflows/*.yml"])
    assert rc == 1


def test_auto_mode_fallbacks_to_yaml_on_missing_workflow_for_ref(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from scripts.ci import check_workflow_file_issues as mod

    _write_workflow(
        tmp_path / ".github" / "workflows" / "ci.yml",
        "name: CI\non:\n  push:\n",
    )
    summary = tmp_path / "summary.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_is_gh_available", lambda: True)
    monkeypatch.setattr(
        mod,
        "_check_with_gh",
        lambda path, ref: mod.WorkflowCheckResult(
            path=str(path),
            mode="gh",
            ok=False,
            message=(
                "could not find workflow file ci.yml on HEAD, "
                "try specifying a different ref"
            ),
        ),
    )

    rc = _invoke_main(
        mod,
        [
            "--mode",
            "auto",
            "--glob",
            ".github/workflows/*.yml",
            "--summary-json-out",
            str(summary),
        ],
    )
    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["requested_mode"] == "auto"
    assert payload["mode_used"] == "yaml"
    assert payload["fallback_reason"] == "gh_ref_unresolvable_for_local_head"


def test_auto_mode_fallbacks_to_yaml_on_default_branch_404(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A workflow that exists only on a stacked branch yields a GitHub
    "HTTP 404: ... not found on the default branch" error. The fallback
    must still kick in so the file-health check does not red the whole
    stress-tests workflow on stacked PRs.
    """
    from scripts.ci import check_workflow_file_issues as mod

    _write_workflow(
        tmp_path / ".github" / "workflows" / "ci.yml",
        "name: CI\non:\n  push:\n    branches: [main]\n",
    )
    summary = tmp_path / "summary.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_is_gh_available", lambda: True)
    monkeypatch.setattr(
        mod,
        "_check_with_gh",
        lambda path, ref: mod.WorkflowCheckResult(
            path=str(path),
            mode="gh",
            ok=False,
            message=(
                "HTTP 404: workflow .github/workflows/brep-golden-eval.yml "
                "not found on the default branch "
                "(https://api.github.com/repos/owner/repo/actions/workflows/"
                ".github%2Fworkflows%2Fbrep-golden-eval.yml)"
            ),
        ),
    )

    rc = _invoke_main(
        mod,
        [
            "--mode",
            "auto",
            "--glob",
            ".github/workflows/*.yml",
            "--summary-json-out",
            str(summary),
        ],
    )
    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["requested_mode"] == "auto"
    assert payload["mode_used"] == "yaml"
    assert payload["fallback_reason"] == "gh_ref_unresolvable_for_local_head"
    assert payload["failed_count"] == 0


def test_is_missing_workflow_on_ref_error_recognizes_all_variants() -> None:
    from scripts.ci.check_workflow_file_issues import (
        _is_missing_workflow_on_ref_error,
    )

    matches = [
        # Original gh-CLI message on stale ref.
        "could not find workflow file ci.yml on HEAD, try specifying a different ref",
        # Alternate phrasing seen on some gh versions.
        "workflow was not found on the requested ref",
        # NEW: workflow file present only on a stacked branch -> 404 on
        # the default branch (GitHub Actions workflow registry behavior).
        "HTTP 404: workflow .github/workflows/brep-golden-eval.yml not found on the default branch",
    ]
    non_matches = [
        "Failed to log in: HTTP 401 Unauthorized",
        "invalid workflow file: yaml parse error",
    ]
    for msg in matches:
        assert _is_missing_workflow_on_ref_error(msg) is True, msg
    for msg in non_matches:
        assert _is_missing_workflow_on_ref_error(msg) is False, msg


def test_gh_mode_does_not_require_yaml_dependency(tmp_path: Path, monkeypatch: Any) -> None:
    from scripts.ci import check_workflow_file_issues as mod

    _write_workflow(
        tmp_path / ".github" / "workflows" / "ci.yml",
        "name: CI\non:\n  push:\n",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_is_gh_available", lambda: True)
    monkeypatch.setattr(mod, "yaml", None)
    monkeypatch.setattr(
        mod,
        "_check_with_gh",
        lambda path, ref: mod.WorkflowCheckResult(
            path=str(path),
            mode="gh",
            ok=True,
            message="ok",
        ),
    )

    rc = _invoke_main(mod, ["--mode", "gh", "--glob", ".github/workflows/*.yml"])
    assert rc == 0
