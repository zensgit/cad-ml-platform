from __future__ import annotations

from scripts.ci.summarize_core_fast_gate import build_summary


def test_build_summary_parses_all_core_fast_suites() -> None:
    log = """
make validate-core-fast
make validate-tolerance
ISO286 deviations: holes=203 shafts=129 path=data/knowledge/iso286_deviations.json
OK
All required hole symbols present: A, B, C, D, N, P
make test-tolerance
============================== 48 passed in 3.29s ==============================
make validate-openapi
============================== 3 passed in 2.57s ==============================
make test-service-mesh
============================= 103 passed in 2.31s ==============================
make test-provider-core
============================== 59 passed in 2.28s ==============================
make test-provider-contract
================= 4 passed, 20 deselected in 3.01s =================
""".strip()

    md = build_summary(log, "Core Fast Gate")
    assert "ISO286 deviations validator | ✅" in md
    assert "ISO286 hole symbols validator | ✅" in md
    assert "tolerance suite | ✅ | `48 passed in 3.29s`" in md
    assert "openapi-contract suite | ✅ | `3 passed in 2.57s`" in md
    assert "service-mesh suite | ✅ | `103 passed in 2.31s`" in md
    assert "provider-core suite | ✅ | `59 passed in 2.28s`" in md
    assert "provider-contract suite | ✅ | `4 passed, 20 deselected in 3.01s`" in md


def test_build_summary_reports_na_when_suite_missing() -> None:
    log = """
make validate-core-fast
make validate-tolerance
ISO286 deviations: holes=203 shafts=129 path=data/knowledge/iso286_deviations.json
OK
""".strip()
    md = build_summary(log, "Core Fast Gate")
    assert "tolerance suite | ❌ | `N/A`" in md
    assert "provider-contract suite | ❌ | `N/A`" in md
