from __future__ import annotations

from scripts.ci.summarize_pydantic_v2_audit import build_summary


def test_build_summary_with_no_regressions() -> None:
    log = """
Pydantic v2 compatibility audit summary
--------------------------------------
pydantic_v1_import: 0
validator_decorator: 0
root_validator_decorator: 0
class_config: 0
parse_obj_call: 0
parse_raw_call: 0
from_orm_call: 0
fields_attr: 0
total_findings: 0

No pydantic v2 compatibility regressions detected.
""".strip()

    md = build_summary(log, "Pydantic V2 Audit")
    assert "Regression gate | ✅" in md
    assert "Total findings | ✅ | `0`" in md
    assert "pydantic_v1_import | ✅ | `0`" in md


def test_build_summary_with_regression_detected() -> None:
    log = """
Pydantic v2 compatibility audit summary
--------------------------------------
pydantic_v1_import: 1
validator_decorator: 0
root_validator_decorator: 0
class_config: 0
parse_obj_call: 0
parse_raw_call: 0
from_orm_call: 0
fields_attr: 0
total_findings: 1

Regression detected:
- pydantic_v1_import: baseline=0, current=1
""".strip()

    md = build_summary(log, "Pydantic V2 Audit")
    assert "Regression gate | ❌" in md
    assert "Total findings | ❌ | `1`" in md
    assert "pydantic_v1_import | ❌ | `1`" in md
