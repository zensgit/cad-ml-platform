from __future__ import annotations

from scripts.ci.summarize_pydantic_style_audit import build_summary


def test_build_summary_with_no_regressions() -> None:
    log = """
Pydantic model style audit summary
----------------------------------
dict_model_config: 0
mutable_literal_default: 0
mutable_field_default: 0
non_optional_none_default: 0
total_findings: 0

No pydantic model-style regressions detected.
""".strip()

    md = build_summary(log, "Pydantic Style Audit")
    assert "Regression gate | ✅" in md
    assert "Total findings | ✅ | `0`" in md
    assert "dict_model_config | ✅ | `0`" in md


def test_build_summary_with_regression_detected() -> None:
    log = """
Pydantic model style audit summary
----------------------------------
dict_model_config: 1
mutable_literal_default: 0
mutable_field_default: 0
non_optional_none_default: 0
total_findings: 1

Regression detected:
- dict_model_config: baseline=0, current=1
""".strip()

    md = build_summary(log, "Pydantic Style Audit")
    assert "Regression gate | ❌" in md
    assert "Total findings | ❌ | `1`" in md
    assert "dict_model_config | ❌ | `1`" in md
