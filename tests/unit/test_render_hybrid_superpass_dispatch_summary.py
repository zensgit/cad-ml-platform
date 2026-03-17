from __future__ import annotations

import json
from pathlib import Path


def test_render_markdown_includes_core_fields_and_dispatch_command() -> None:
    from scripts.ci import render_hybrid_superpass_dispatch_summary as mod

    markdown = mod.render_markdown(
        {
            "workflow": "evaluation-report.yml",
            "ref": "main",
            "repo": "zensgit/cad-ml-platform",
            "expected_conclusion": "success",
            "conclusion": "failure",
            "matched_expectation": False,
            "overall_exit_code": 1,
            "watch_exit_code": 1,
            "run_id": 5001,
            "run_url": "https://example.com/r/5001",
            "reason": "superpass_gate_failed",
            "dispatch_command": [
                "gh",
                "workflow",
                "run",
                "evaluation-report.yml",
                "--ref",
                "main",
            ],
            "failure_diagnostics": {
                "available": True,
                "reason": "failed_jobs_detected",
                "failed_job_count": 1,
                "failed_jobs": [
                    {
                        "job_name": "hybrid-superpass",
                        "job_conclusion": "failure",
                        "failed_step_name": "Validate Superpass Reports",
                        "failed_step_conclusion": "failure",
                    }
                ],
            },
        }
    )

    assert "## Hybrid Superpass Dispatch" in markdown
    assert "## Dispatch Verdict" in markdown
    assert "- verdict: expectation_mismatch" in markdown
    assert "- conclusion_pair: expected=success actual=failure" in markdown
    assert "- top_failed_jobs: hybrid-superpass" in markdown
    assert "- top_failed_steps: Validate Superpass Reports" in markdown
    assert "- dispatch_command: gh workflow run evaluation-report.yml --ref main" in markdown
    assert "- matched_expectation: False" in markdown
    assert "## Failure Snapshot" in markdown
    assert "- failed_job_count: 1" in markdown
    assert "- failure_reason: failed_jobs_detected" in markdown
    assert "### Failure Diagnostics" in markdown
    assert "failed_job: hybrid-superpass" in markdown


def test_main_writes_markdown_file_and_stdout(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_superpass_dispatch_summary as mod

    dispatch_json = tmp_path / "hybrid_superpass_dispatch.json"
    dispatch_json.write_text(
        json.dumps(
            {
                "workflow": "evaluation-report.yml",
                "ref": "main",
                "repo": "zensgit/cad-ml-platform",
                "expected_conclusion": "success",
                "conclusion": "success",
                "matched_expectation": True,
                "overall_exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "hybrid_superpass_dispatch.md"

    rc = mod.main(
        [
            "--dispatch-json",
            str(dispatch_json),
            "--output-md",
            str(output_md),
        ]
    )
    assert rc == 0
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Hybrid Superpass Dispatch" in rendered
    assert "## Dispatch Verdict" in rendered
    assert "- overall_exit_code: 0" in rendered
    assert "## Hybrid Superpass Dispatch" in capsys.readouterr().out


def test_main_fails_when_dispatch_missing(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_superpass_dispatch_summary as mod

    rc = mod.main(["--dispatch-json", str(tmp_path / "missing.json")])
    assert rc == 1
    assert "dispatch json does not exist" in capsys.readouterr().out


def test_main_fails_when_dispatch_json_invalid(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_superpass_dispatch_summary as mod

    dispatch_json = tmp_path / "invalid.json"
    dispatch_json.write_text("{not-json", encoding="utf-8")

    rc = mod.main(["--dispatch-json", str(dispatch_json)])
    assert rc == 1
    assert "failed to parse dispatch json" in capsys.readouterr().out
