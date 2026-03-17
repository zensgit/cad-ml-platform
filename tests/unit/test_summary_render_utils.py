from __future__ import annotations

import pytest


def test_read_json_object_returns_dict(tmp_path) -> None:
    from scripts.ci.summary_render_utils import read_json_object

    path = tmp_path / "payload.json"
    path.write_text('{"ok": true}', encoding="utf-8")

    payload = read_json_object(path, "dispatch")

    assert payload == {"ok": True}


def test_read_json_object_rejects_invalid_json(tmp_path) -> None:
    from scripts.ci.summary_render_utils import read_json_object

    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed to parse summary json"):
        read_json_object(path, "summary")


def test_read_json_object_rejects_non_object_payload(tmp_path) -> None:
    from scripts.ci.summary_render_utils import read_json_object

    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="validation json must be an object"):
        read_json_object(path, "validation")


def test_boolish_and_is_zeroish_handle_string_forms() -> None:
    from scripts.ci.summary_render_utils import boolish, is_zeroish

    assert boolish(True) is True
    assert boolish("YES") is True
    assert boolish("0") is False
    assert is_zeroish(0) is True
    assert is_zeroish("0") is True
    assert is_zeroish("00") is False


def test_top_nonempty_filters_and_limits() -> None:
    from scripts.ci.summary_render_utils import top_nonempty

    rows = top_nonempty(["", "  ", "alpha", None, "beta", "gamma", "delta"], limit=3)

    assert rows == ["alpha", "beta", "gamma"]


def test_render_inline_items_returns_joined_values_or_empty_marker() -> None:
    from scripts.ci.summary_render_utils import render_inline_items

    assert render_inline_items(["alpha", "", " beta ", None]) == "alpha, beta"
    assert render_inline_items(["", "  ", None]) == "(none)"


def test_append_markdown_section_renders_expected_block() -> None:
    from scripts.ci.summary_render_utils import append_markdown_section

    lines = ["## Root"]

    append_markdown_section(lines, "Dispatch Verdict", [("verdict", "ok"), ("count", 2)])

    assert lines == [
        "## Root",
        "",
        "## Dispatch Verdict",
        "",
        "- verdict: ok",
        "- count: 2",
    ]


def test_append_markdown_section_supports_compact_header_layout() -> None:
    from scripts.ci.summary_render_utils import append_markdown_section

    lines = ["## Root"]

    append_markdown_section(
        lines,
        "Failure Diagnostics",
        [("available", True), ("failed_job_count", 1)],
        level=3,
        blank_after_header=False,
    )

    assert lines == [
        "## Root",
        "",
        "### Failure Diagnostics",
        "- available: True",
        "- failed_job_count: 1",
    ]


def test_append_failure_diagnostics_section_renders_reason_and_failed_jobs() -> None:
    from scripts.ci.summary_render_utils import append_failure_diagnostics_section

    lines = ["## Root"]

    append_failure_diagnostics_section(
        lines,
        {
            "available": True,
            "failed_job_count": 1,
            "reason": "failed_jobs_detected",
        },
        [
            {
                "job_name": "hybrid-superpass",
                "job_conclusion": "failure",
                "failed_step_name": "Validate Superpass Reports",
                "failed_step_conclusion": "failure",
            }
        ],
    )

    assert lines == [
        "## Root",
        "",
        "### Failure Diagnostics",
        "- available: True",
        "- failed_job_count: 1",
        "- reason: failed_jobs_detected",
        "- failed_job: hybrid-superpass job_conclusion=failure failed_step=Validate Superpass Reports step_conclusion=failure",
    ]
