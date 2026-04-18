from __future__ import annotations

from src.api.v1 import analyze
from src.api.v1 import process


def test_analyze_process_rules_audit_reuses_process_endpoint() -> None:
    assert analyze.process_rules_audit is process.process_rules_audit
