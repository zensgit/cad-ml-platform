from __future__ import annotations

from pathlib import Path

from scripts.ci.audit_pydantic_v2 import (
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
                "from pydantic.v1 import BaseModel",
                "",
                "class X(BaseModel):",
                "    class Config:",
                "        orm_mode = True",
                "",
                "    @validator('a')",
                "    def check_a(cls, v):",
                "        return v",
                "",
                "obj = X.parse_obj({'a': 1})",
            ]
        ),
        encoding="utf-8",
    )

    findings = collect_findings([tmp_path])
    counts = summarize_counts(findings)

    assert counts["pydantic_v1_import"] == 1
    assert counts["class_config"] == 1
    assert counts["validator_decorator"] == 1
    assert counts["parse_obj_call"] == 1
    assert counts["root_validator_decorator"] == 0
    assert sum(counts.values()) == 4


def test_find_regressions_detects_increase() -> None:
    baseline = {name: 0 for name in PATTERNS}
    current = {name: 0 for name in PATTERNS}
    current["validator_decorator"] = 2
    current["pydantic_v1_import"] = 1

    regressions = find_regressions(current, baseline)

    assert regressions["validator_decorator"] == (0, 2)
    assert regressions["pydantic_v1_import"] == (0, 1)
    assert "class_config" not in regressions


def test_build_baseline_payload_includes_all_patterns() -> None:
    counts = {"validator_decorator": 3}
    payload = build_baseline_payload(["src"], counts)

    payload_counts = payload["counts"]
    assert isinstance(payload_counts, dict)
    assert payload_counts["validator_decorator"] == 3
    assert payload_counts["class_config"] == 0
    assert payload["total_findings"] == 3
