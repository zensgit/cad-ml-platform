from __future__ import annotations

import json
from pathlib import Path


def test_render_markdown_includes_attempts_and_pr_comment() -> None:
    from scripts.ci import render_soft_mode_smoke_summary as mod

    markdown = mod.render_markdown(
        {
            "overall_exit_code": 0,
            "dispatch_exit_code": 0,
            "max_dispatch_attempts": 2,
            "retry_sleep_seconds": 15,
            "soft_marker_ok": True,
            "restore_ok": True,
            "dispatch": {
                "run_id": 23126562401,
                "run_url": "https://github.com/example/actions/runs/23126562401",
            },
            "attempts": [
                {
                    "attempt": 1,
                    "dispatch_exit_code": 1,
                    "soft_marker_ok": False,
                    "soft_marker_message": "marker not found",
                }
            ],
            "pr_comment": {
                "requested": True,
                "enabled": True,
                "pr_number": 369,
                "auto_resolve": True,
                "exit_code": 0,
                "error": "comment disabled by dry-run",
            },
        }
    )

    assert "## Evaluation Soft-Mode Smoke" in markdown
    assert "## Smoke Verdict" in markdown
    assert "- verdict: ok" in markdown
    assert "- pr_comment_status: requested=True, enabled=True, exit_code=0" in markdown
    assert "## Smoke Signals" in markdown
    assert "- failed_attempts: 1" in markdown
    assert "- last_attempt_message: marker not found" in markdown
    assert "- pr_comment_error: comment disabled by dry-run" in markdown
    assert "- attempts_total: 1" in markdown
    assert "- attempt #1: dispatch_exit_code=1, soft_marker_ok=False, message=marker not found" in markdown
    assert "- run_id: 23126562401" in markdown
    assert "- pr_comment_pr_number: 369" in markdown
    assert "- pr_comment_auto_resolve: True" in markdown


def test_main_writes_markdown_file_and_stdout(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_soft_mode_smoke_summary as mod

    summary_json = tmp_path / "soft_mode_smoke.json"
    summary_json.write_text(
        json.dumps(
            {
                "overall_exit_code": 0,
                "dispatch_exit_code": 0,
                "attempts": [],
                "soft_marker_ok": True,
                "restore_ok": True,
            }
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "soft_mode_smoke.md"

    rc = mod.main(
        [
            "--summary-json",
            str(summary_json),
            "--output-md",
            str(output_md),
        ]
    )
    assert rc == 0
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Evaluation Soft-Mode Smoke" in rendered
    assert "## Smoke Verdict" in rendered
    assert "- attempts_total: 0" in rendered

    captured = capsys.readouterr()
    assert "## Evaluation Soft-Mode Smoke" in captured.out


def test_main_fails_when_summary_missing(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_soft_mode_smoke_summary as mod

    rc = mod.main(["--summary-json", str(tmp_path / "missing.json")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "summary json does not exist" in captured.out


def test_main_fails_when_summary_json_invalid(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_soft_mode_smoke_summary as mod

    summary_json = tmp_path / "invalid.json"
    summary_json.write_text("{not-json", encoding="utf-8")

    rc = mod.main(["--summary-json", str(summary_json)])
    assert rc == 1
    assert "failed to parse summary json" in capsys.readouterr().out
