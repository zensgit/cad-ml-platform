from __future__ import annotations

import json
from pathlib import Path


def test_render_markdown_includes_core_fields_and_failure_diagnostics() -> None:
    from scripts.ci import render_hybrid_blind_strict_real_dispatch_summary as mod

    markdown = mod.render_markdown(
        {
            "workflow": "evaluation-report.yml",
            "ref": "main",
            "repo": "zensgit/cad-ml-platform",
            "hybrid_blind_dxf_dir": "/tmp/dxf",
            "hybrid_blind_manifest_csv": "",
            "hybrid_blind_synth_manifest": "",
            "strict_fail_on_gate_failed": "true",
            "strict_require_real_data": "true",
            "expected_conclusion": "success",
            "conclusion": "failure",
            "matched_expectation": False,
            "overall_exit_code": 1,
            "dispatch_exit_code": 0,
            "watch_exit_code": 1,
            "run_id": 4301,
            "run_url": "https://example.com/r/4301",
            "reason": "strict_gate_blocked",
            "failure_diagnostics": {
                "available": True,
                "reason": "failed_jobs_detected",
                "failed_job_count": 1,
                "failed_jobs": [
                    {
                        "job_name": "strict-real-gate",
                        "job_conclusion": "failure",
                        "failed_step_name": "Check Hybrid Blind Gate",
                        "failed_step_conclusion": "failure",
                    }
                ],
            },
        }
    )

    assert "## Hybrid Blind Strict-Real Dispatch" in markdown
    assert "## Dispatch Verdict" in markdown
    assert "- verdict: expectation_mismatch" in markdown
    assert "- conclusion_pair: expected=success actual=failure" in markdown
    assert "- top_failed_jobs: strict-real-gate" in markdown
    assert "- top_failed_steps: Check Hybrid Blind Gate" in markdown
    assert "- hybrid_blind_dxf_dir: /tmp/dxf" in markdown
    assert "- matched_expectation: False" in markdown
    assert "- run_id: 4301" in markdown
    assert "## Failure Snapshot" in markdown
    assert "- failed_job_count: 1" in markdown
    assert "- failure_reason: failed_jobs_detected" in markdown
    assert "### Failure Diagnostics" in markdown
    assert "failed_job: strict-real-gate" in markdown


def test_main_writes_markdown_file_and_stdout(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_blind_strict_real_dispatch_summary as mod

    dispatch_json = tmp_path / "strict_real_dispatch.json"
    dispatch_json.write_text(
        json.dumps(
            {
                "workflow": "evaluation-report.yml",
                "ref": "main",
                "repo": "zensgit/cad-ml-platform",
                "hybrid_blind_dxf_dir": "/tmp/dxf",
                "expected_conclusion": "success",
                "conclusion": "success",
                "matched_expectation": True,
                "overall_exit_code": 0,
            }
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "strict_real_dispatch.md"

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
    assert "## Hybrid Blind Strict-Real Dispatch" in rendered
    assert "## Dispatch Verdict" in rendered
    assert "- overall_exit_code: 0" in rendered
    assert "## Hybrid Blind Strict-Real Dispatch" in capsys.readouterr().out


def test_main_fails_when_dispatch_missing(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_blind_strict_real_dispatch_summary as mod

    rc = mod.main(["--dispatch-json", str(tmp_path / "missing.json")])
    assert rc == 1
    assert "dispatch json does not exist" in capsys.readouterr().out


def test_main_fails_when_dispatch_json_invalid(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_blind_strict_real_dispatch_summary as mod

    dispatch_json = tmp_path / "invalid.json"
    dispatch_json.write_text("{not-json", encoding="utf-8")

    rc = mod.main(["--dispatch-json", str(dispatch_json)])
    assert rc == 1
    assert "failed to parse dispatch json" in capsys.readouterr().out
