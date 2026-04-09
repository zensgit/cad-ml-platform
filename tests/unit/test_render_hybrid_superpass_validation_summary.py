from __future__ import annotations

import json
from pathlib import Path


def test_render_markdown_includes_summary_and_lists() -> None:
    from scripts.ci import render_hybrid_superpass_validation_summary as mod

    markdown = mod.render_markdown(
        {
            "status": "warn",
            "strict": False,
            "schema_mode": "builtin",
            "overall_exit_code": 0,
            "inputs": {
                "superpass_json": "/tmp/superpass.json",
                "hybrid_blind_gate_report": "/tmp/gate.json",
                "hybrid_calibration_json": "/tmp/calibration.json",
            },
            "summary": {
                "superpass_status": "passed",
                "superpass_check_count": 3,
                "superpass_failure_count": 0,
                "superpass_warning_count": 1,
                "gate_hybrid_accuracy": 0.71,
                "gate_hybrid_gain_vs_graph2d": 0.09,
                "calibration_ece": 0.041,
            },
            "errors": [],
            "warnings": ["hybrid calibration report missing"],
        }
    )

    assert "## Hybrid Superpass Validation" in markdown
    assert "## Validation Verdict" in markdown
    assert "- verdict: ok" in markdown
    assert "- status: warn" in markdown
    assert "- top_errors: (none)" in markdown
    assert "- top_warnings: hybrid calibration report missing" in markdown
    assert "## Validation Snapshot" in markdown
    assert "- inputs.superpass_json: /tmp/superpass.json" in markdown
    assert "- summary.gate_hybrid_accuracy: 0.71" in markdown
    assert "### Warnings" in markdown
    assert "- hybrid calibration report missing" in markdown


def test_main_writes_markdown_file_and_stdout(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_superpass_validation_summary as mod

    validation_json = tmp_path / "hybrid_superpass_validation.json"
    validation_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "strict": False,
                "schema_mode": "builtin",
                "overall_exit_code": 0,
                "summary": {},
                "errors": [],
                "warnings": [],
                "inputs": {},
            }
        ),
        encoding="utf-8",
    )
    output_md = tmp_path / "hybrid_superpass_validation.md"

    rc = mod.main(
        [
            "--validation-json",
            str(validation_json),
            "--output-md",
            str(output_md),
        ]
    )
    assert rc == 0
    rendered = output_md.read_text(encoding="utf-8")
    assert "## Hybrid Superpass Validation" in rendered
    assert "## Validation Verdict" in rendered
    assert "- status: ok" in rendered

    captured = capsys.readouterr()
    assert "## Hybrid Superpass Validation" in captured.out


def test_main_fails_when_validation_missing(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_superpass_validation_summary as mod

    rc = mod.main(["--validation-json", str(tmp_path / "missing.json")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "validation json does not exist" in captured.out


def test_main_fails_when_validation_json_invalid(tmp_path: Path, capsys: object) -> None:
    from scripts.ci import render_hybrid_superpass_validation_summary as mod

    validation_json = tmp_path / "invalid.json"
    validation_json.write_text("{not-json", encoding="utf-8")

    rc = mod.main(["--validation-json", str(validation_json)])
    assert rc == 1
    assert "failed to parse validation json" in capsys.readouterr().out
