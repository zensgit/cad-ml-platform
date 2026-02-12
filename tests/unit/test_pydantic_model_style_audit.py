from __future__ import annotations

from pathlib import Path

from scripts.ci.audit_pydantic_model_style import (
    PATTERNS,
    build_baseline_payload,
    collect_findings,
    find_regressions,
    summarize_counts,
)


def test_collect_findings_and_counts(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "from pydantic import BaseModel, Field",
                "",
                "class X(BaseModel):",
                "    model_config = {}",
                "    a: list[int] = []",
                "    b: dict[str, int] = Field(default={})",
                "    c: int = None",
            ]
        ),
        encoding="utf-8",
    )

    findings = collect_findings([tmp_path])
    counts = summarize_counts(findings)

    assert counts["dict_model_config"] == 1
    assert counts["mutable_literal_default"] == 1
    assert counts["mutable_field_default"] == 1
    assert counts["non_optional_none_default"] == 1
    assert sum(counts.values()) == 4


def test_find_regressions_detects_increase() -> None:
    baseline = {name: 0 for name in PATTERNS}
    current = {name: 0 for name in PATTERNS}
    current["dict_model_config"] = 1
    current["mutable_literal_default"] = 2

    regressions = find_regressions(current, baseline)

    assert regressions["dict_model_config"] == (0, 1)
    assert regressions["mutable_literal_default"] == (0, 2)
    assert "mutable_field_default" not in regressions


def test_build_baseline_payload_includes_all_patterns() -> None:
    counts = {"dict_model_config": 2}
    payload = build_baseline_payload(["src"], counts)

    payload_counts = payload["counts"]
    assert isinstance(payload_counts, dict)
    assert payload_counts["dict_model_config"] == 2
    assert payload_counts["mutable_literal_default"] == 0
    assert payload["total_findings"] == 2
