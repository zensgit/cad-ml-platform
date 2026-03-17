from __future__ import annotations


def test_markdown_table_renders_expected_rows() -> None:
    from scripts.ci.comment_markdown_utils import markdown_table

    rendered = markdown_table(["Field", "Value"], [["status", "ok"], ["count", 2]])

    assert "| Field | Value |" in rendered
    assert "| status | ok |" in rendered
    assert "| count | 2 |" in rendered


def test_markdown_section_renders_heading_and_body() -> None:
    from scripts.ci.comment_markdown_utils import markdown_section

    rendered = markdown_section("Attempts", "- attempt 1")

    assert rendered == "### Attempts\n- attempt 1"


def test_markdown_footer_supports_commit_only_and_timestamp() -> None:
    from scripts.ci.comment_markdown_utils import markdown_footer

    assert markdown_footer(commit_sha="abcdef1") == "*Commit: abcdef1*"
    assert markdown_footer(updated_at="2026-03-17 10:00:00", commit_sha="abcdef1") == (
        "*Updated: 2026-03-17 10:00:00 UTC*\n*Commit: abcdef1*"
    )
