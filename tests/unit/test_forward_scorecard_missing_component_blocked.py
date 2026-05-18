"""Guard against the "default-empty summary -> release_ready" false-green.

Stage 2c will wire `--brep-summary` and `--manufacturing-evidence-summary`
into evaluation-report.yml. The risk: if either summary file is missing or
empty, `build_forward_scorecard()` must NOT default-empty-dict its way into
`release_ready` — that would let the overall scorecard go misleadingly
green while no real evidence has been measured.

This test exercises `_brep_component` and `_manufacturing_evidence_component`
directly with the same empty-dict inputs the script passes when a CLI
argument is omitted, and asserts the status is `blocked`. The Stage 2c
caller in `scripts/export_forward_scorecard.py` is what feeds these dicts,
so this is the contract the wiring depends on.
"""

from __future__ import annotations

from src.core.benchmark.forward_scorecard import (
    _brep_component,
    _manufacturing_evidence_component,
    build_forward_scorecard,
)


def _ready_model_registry() -> dict:
    """Mirror of test_forward_scorecard._ready_model_registry — kept
    inline (small) to avoid cross-file test fixture coupling."""
    return {
        "ok": True,
        "degraded": False,
        "status": "ready",
        "blocking_reasons": [],
        "degraded_reasons": [],
        "items": {
            "v16_classifier": {"status": "loaded"},
            "graph2d": {"status": "loaded"},
            "uvnet": {"status": "available"},
            "pointnet": {"status": "available"},
            "ocr_provider": {"status": "available"},
            "embedding_model": {"status": "available"},
        },
    }


# --- direct component tests ----------------------------------------------


def test_brep_component_blocked_when_summary_is_empty_dict() -> None:
    component = _brep_component({})
    assert component["status"] == "blocked", (
        f"Empty brep summary must yield blocked, got {component['status']!r}. "
        "Stage 2c wiring depends on this — flipping to release_ready here "
        "would let an unwired evaluation-report pass the release gate with "
        "zero B-Rep evidence."
    )
    assert component["sample_size"] == 0
    assert component["parse_success_count"] == 0


def test_brep_component_blocked_when_sample_size_zero_with_other_fields() -> None:
    # Even if some unrelated fields exist, sample_size=0 must dominate.
    component = _brep_component(
        {"failure_reasons": {"unparseable": 5}, "sample_size": 0}
    )
    assert component["status"] == "blocked"


def test_manufacturing_evidence_component_blocked_when_summary_is_empty() -> None:
    component = _manufacturing_evidence_component({})
    # The exact status taxonomy here ("blocked" vs the manufacturing-specific
    # variant) is enforced by the component; we assert the no-evidence
    # outcome — never `release_ready` / `benchmark_ready_with_gap`.
    assert component["status"] not in ("release_ready", "benchmark_ready_with_gap"), (
        f"Empty manufacturing summary must NOT be release/benchmark; "
        f"got {component['status']!r}."
    )


# --- integration via build_forward_scorecard ------------------------------


def test_overall_scorecard_with_no_brep_or_manufacturing_data_is_not_release_ready() -> None:
    """The combined fail-closed invariant Stage 2c depends on.

    If both data sources are missing (empty summaries), the overall
    forward_status MUST NOT be `release_ready`. This is the *single*
    assertion that protects the release gate from a default-green wiring
    bug in evaluation-report.yml.
    """
    payload = build_forward_scorecard(
        title="Test — no brep, no manufacturing",
        model_readiness=_ready_model_registry(),
        # Empty dicts simulate the CLI default when --brep-summary /
        # --manufacturing-evidence-summary are omitted.
        brep_summary={},
        manufacturing_summary={},
    )

    overall_status = payload["overall_status"]
    assert overall_status != "release_ready", (
        f"build_forward_scorecard returned overall_status={overall_status!r} "
        f"with empty brep + manufacturing summaries. This is the "
        f"default-green failure mode Stage 2c must never re-enable."
    )

    # Cross-check the components themselves are correctly demoted.
    components = payload["components"]
    assert components["brep"]["status"] == "blocked"
    # The component is keyed `manufacturing_evidence` (see
    # build_forward_scorecard); resolve via membership defensively in case
    # an additional manufacturing_* component is added later.
    mfg_keys = [k for k in components if "manufacturing" in k.lower()]
    assert mfg_keys, "expected a manufacturing-evidence component in the scorecard"
    for key in mfg_keys:
        assert components[key]["status"] not in (
            "release_ready",
            "benchmark_ready_with_gap",
        ), f"{key!r} drifted to {components[key]['status']!r} on empty input"
