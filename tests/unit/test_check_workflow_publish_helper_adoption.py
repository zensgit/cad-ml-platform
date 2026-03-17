from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["check_workflow_publish_helper_adoption.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def _write_workflow(path: Path, script: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "name: Test Workflow",
                "on:",
                "  workflow_dispatch:",
                "jobs:",
                "  test:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - name: Script",
                "        uses: actions/github-script@deadbeef",
                "        with:",
                "          script: |",
            ]
            + [f"            {line}" for line in script.splitlines()]
        )
        + "\n",
        encoding="utf-8",
    )


def test_publish_helper_check_passes_on_helper_imports(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_publish_helper_adoption as mod

    root = tmp_path / ".github" / "workflows"
    _write_workflow(
        root / "code-quality.yml",
        "const { upsertBotIssueComment } = require('./scripts/ci/comment_pr_utils.js');",
    )
    _write_workflow(
        root / "badge-review.yml",
        "const { upsertOpenIssue } = require('./scripts/ci/issue_upsert_utils.js');",
    )
    _write_workflow(root / "misc.yml", "console.log('noop');")

    summary = tmp_path / "summary.json"
    rc = _invoke_main(
        mod,
        ["--workflow-root", str(root), "--summary-json-out", str(summary)],
    )

    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["failed_count"] == 0
    assert payload["checked_count"] == 3


def test_publish_helper_check_fails_on_raw_create_comment(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_publish_helper_adoption as mod

    root = tmp_path / ".github" / "workflows"
    _write_workflow(
        root / "misc.yml",
        "await github.rest.issues.createComment({ body: 'bad' });",
    )

    rc = _invoke_main(mod, ["--workflow-root", str(root)])
    assert rc == 1


def test_publish_helper_check_fails_when_required_comment_helper_missing(
    tmp_path: Path,
) -> None:
    from scripts.ci import check_workflow_publish_helper_adoption as mod

    root = tmp_path / ".github" / "workflows"
    _write_workflow(root / "security-audit.yml", "console.log('missing import');")

    rc = _invoke_main(mod, ["--workflow-root", str(root)])
    assert rc == 1


def test_publish_helper_check_fails_when_required_issue_helper_missing(
    tmp_path: Path,
) -> None:
    from scripts.ci import check_workflow_publish_helper_adoption as mod

    root = tmp_path / ".github" / "workflows"
    _write_workflow(root / "adaptive-rate-limit-monitor.yml", "console.log('missing import');")

    rc = _invoke_main(mod, ["--workflow-root", str(root)])
    assert rc == 1
