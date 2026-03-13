from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_compare_hybrid_superpass_reports_normal(tmp_path: Path) -> None:
    from scripts.ci import compare_hybrid_superpass_reports as mod

    fail_json = tmp_path / "in" / "fail.json"
    success_json = tmp_path / "in" / "success.json"
    output_json = tmp_path / "out" / "summary.json"
    output_md = tmp_path / "out" / "summary.md"

    _write_json(
        fail_json,
        {
            "run_id": 101,
            "conclusion": "failure",
            "expected_conclusion": "failure",
            "matched_expectation": True,
            "dispatch_trace_id": "trace-fail",
            "run_url": "https://example.com/runs/101",
        },
    )
    _write_json(
        success_json,
        {
            "run_id": 102,
            "conclusion": "success",
            "expected_conclusion": "success",
            "matched_expectation": True,
            "dispatch_trace_id": "trace-success",
            "run_url": "https://example.com/runs/102",
        },
    )

    rc = mod.main(
        [
            "--fail-json",
            str(fail_json),
            "--success-json",
            str(success_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 0
    payload = _read_json(output_json)
    assert payload["fail"]["run_id"] == "101"
    assert payload["success"]["run_id"] == "102"
    assert payload["fail"]["conclusion"] == "failure"
    assert payload["success"]["conclusion"] == "success"
    assert payload["run_id_is_different"] is True
    assert payload["checks"]["fail_expected_failure"] is True
    assert payload["checks"]["success_expected_success"] is True
    assert payload["overall_exit_code"] == 0

    markdown = output_md.read_text(encoding="utf-8")
    assert "Runs are different (parallel run isolation check): **YES**" in markdown
    assert "Fail scenario failed as expected: **YES**" in markdown
    assert "Success scenario succeeded as expected: **YES**" in markdown


def test_compare_hybrid_superpass_reports_warns_when_run_id_same(tmp_path: Path) -> None:
    from scripts.ci import compare_hybrid_superpass_reports as mod

    fail_json = tmp_path / "in" / "fail.json"
    success_json = tmp_path / "in" / "success.json"
    output_json = tmp_path / "out" / "summary.json"
    output_md = tmp_path / "out" / "summary.md"

    _write_json(
        fail_json,
        {
            "run_id": 9001,
            "conclusion": "failure",
            "expected_conclusion": "failure",
            "matched_expectation": True,
            "dispatch_trace_id": "trace-a",
            "run_url": "https://example.com/runs/9001",
        },
    )
    _write_json(
        success_json,
        {
            "run_id": 9001,
            "conclusion": "success",
            "expected_conclusion": "success",
            "matched_expectation": True,
            "dispatch_trace_id": "trace-b",
            "run_url": "https://example.com/runs/9001",
        },
    )

    rc = mod.main(
        [
            "--fail-json",
            str(fail_json),
            "--success-json",
            str(success_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 0
    payload = _read_json(output_json)
    assert payload["run_id_is_different"] is False
    assert payload["warnings"]
    assert "share the same run_id" in payload["warnings"][0]

    markdown = output_md.read_text(encoding="utf-8")
    assert "Runs are different (parallel run isolation check): **NO**" in markdown
    assert "## Warnings" in markdown


def test_compare_hybrid_superpass_reports_strict_fails_on_mismatch(tmp_path: Path) -> None:
    from scripts.ci import compare_hybrid_superpass_reports as mod

    fail_json = tmp_path / "in" / "fail.json"
    success_json = tmp_path / "in" / "success.json"
    output_json = tmp_path / "out" / "summary.json"
    output_md = tmp_path / "out" / "summary.md"

    _write_json(
        fail_json,
        {
            "run_id": 201,
            "conclusion": "success",
            "expected_conclusion": "failure",
            "matched_expectation": False,
            "dispatch_trace_id": "trace-fail",
            "run_url": "https://example.com/runs/201",
        },
    )
    _write_json(
        success_json,
        {
            "run_id": 202,
            "conclusion": "success",
            "expected_conclusion": "success",
            "matched_expectation": True,
            "dispatch_trace_id": "trace-success",
            "run_url": "https://example.com/runs/202",
        },
    )

    rc = mod.main(
        [
            "--fail-json",
            str(fail_json),
            "--success-json",
            str(success_json),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ]
    )

    assert rc == 1
    payload = _read_json(output_json)
    assert payload["strict_mode"] is True
    assert payload["strict_failed"] is True
    assert payload["overall_exit_code"] == 1

    markdown = output_md.read_text(encoding="utf-8")
    assert "Strict mode result: **FAILED (exit 1)**" in markdown
