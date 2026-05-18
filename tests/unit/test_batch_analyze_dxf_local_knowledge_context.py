import json

from scripts.batch_analyze_dxf_local import (
    _extract_decision_context,
    _extract_knowledge_context,
    _extract_manufacturing_evidence,
    _summarize_decision_context,
    _summarize_manufacturing_evidence,
)


def test_extract_knowledge_context_serializes_categories_and_types() -> None:
    payload = _extract_knowledge_context(
        {
            "knowledge_checks": [
                {"category": "material", "item": "304"},
                {"category": "surface_finish", "item": "Ra3.2"},
            ],
            "violations": [
                {"category": "knowledge_conflict", "severity": "warn"},
            ],
            "standards_candidates": [
                {"type": "material", "designation": "304"},
                {"type": "surface_finish", "designation": "Ra 3.2"},
            ],
            "knowledge_hints": [
                {"label": "人孔", "score": 0.8},
            ],
        }
    )

    assert json.loads(payload["knowledge_checks"])[0]["category"] == "material"
    assert payload["knowledge_check_categories"] == "material;surface_finish"
    assert payload["knowledge_violation_categories"] == "knowledge_conflict"
    assert payload["knowledge_standard_types"] == "material;surface_finish"
    assert payload["knowledge_hint_labels"] == "人孔"


def test_extract_decision_context_serializes_decision_service_contract() -> None:
    payload = _extract_decision_context(
        {
            "contract_version": "classification_decision.v1",
            "decision_source": "hybrid",
            "branch_conflicts": {"hybrid_vs_graph2d": True},
            "review_reasons": ["branch_conflict"],
            "fallback_flags": ["rules_baseline"],
            "evidence": [
                {"source": "hybrid", "kind": "prediction", "label": "人孔"},
                {"source": "graph2d", "kind": "prediction", "label": "法兰"},
            ],
            "decision_contract": {
                "fine_part_type": "人孔",
                "coarse_part_type": "开孔件",
                "contract_version": "classification_decision.v1",
            },
        }
    )

    assert payload["decision_contract_present"] is True
    assert payload["decision_contract_version"] == "classification_decision.v1"
    assert payload["decision_source"] == "hybrid"
    assert payload["decision_evidence_count"] == 2
    assert payload["decision_evidence_sources"] == "hybrid;graph2d"
    assert payload["decision_fallback_flags"] == "rules_baseline"
    assert payload["decision_review_reasons"] == "branch_conflict"
    assert json.loads(payload["decision_contract"])["fine_part_type"] == "人孔"
    assert json.loads(payload["decision_branch_conflicts"]) == {
        "hybrid_vs_graph2d": True
    }


def test_summarize_decision_context_counts_exported_contract_coverage() -> None:
    summary = _summarize_decision_context(
        [
            {
                "decision_contract_version": "classification_decision.v1",
                "decision_evidence_count": 2,
                "decision_evidence_sources": "hybrid;knowledge",
                "decision_fallback_flags": "rules_baseline",
                "decision_review_reasons": "branch_conflict",
                "decision_branch_conflicts": json.dumps(
                    {"hybrid_vs_graph2d": True}, ensure_ascii=False
                ),
            },
            {
                "decision_contract_version": "",
                "decision_evidence_count": 0,
                "decision_evidence_sources": "",
                "decision_fallback_flags": "",
                "decision_review_reasons": "",
                "decision_branch_conflicts": "",
            },
        ]
    )

    assert summary["row_count"] == 2
    assert summary["decision_contract_count"] == 1
    assert summary["decision_contract_coverage_rate"] == 0.5
    assert summary["decision_evidence_row_count"] == 1
    assert summary["decision_evidence_total_count"] == 2
    assert summary["decision_evidence_coverage_rate"] == 0.5
    assert summary["branch_conflict_count"] == 1
    assert summary["contract_version_counts"] == {"classification_decision.v1": 1}
    assert summary["evidence_source_counts"] == {"hybrid": 1, "knowledge": 1}
    assert summary["fallback_flag_counts"] == {"rules_baseline": 1}
    assert summary["review_reason_counts"] == {"branch_conflict": 1}


def test_extract_manufacturing_evidence_serializes_required_sources() -> None:
    payload = _extract_manufacturing_evidence(
        {
            "manufacturing_evidence": [
                {"source": "dfm", "kind": "manufacturability_check"},
                {"source": "manufacturing_process", "kind": "process_recommendation"},
                {"source": "manufacturing_cost", "kind": "cost_estimate"},
                {"source": "manufacturing_decision", "kind": "manufacturing_summary"},
            ],
            "classification": {
                "evidence": [{"source": "hybrid", "kind": "prediction"}],
            },
        }
    )

    assert payload["manufacturing_evidence_count"] == 4
    assert payload["manufacturing_evidence_sources"] == (
        "dfm;manufacturing_process;manufacturing_cost;manufacturing_decision"
    )
    assert payload["manufacturing_evidence_required_sources_present"] is True
    assert payload["manufacturing_evidence_has_dfm"] is True
    assert payload["manufacturing_evidence_has_process"] is True
    assert payload["manufacturing_evidence_has_cost"] is True
    assert payload["manufacturing_evidence_has_decision"] is True
    assert json.loads(payload["manufacturing_evidence"])[0]["source"] == "dfm"


def test_summarize_manufacturing_evidence_counts_forward_scorecard_contract() -> None:
    rows = [
        _extract_manufacturing_evidence(
            {
                "manufacturing_evidence": [
                    {"source": "dfm", "kind": "manufacturability_check"},
                    {
                        "source": "manufacturing_process",
                        "kind": "process_recommendation",
                    },
                    {"source": "manufacturing_cost", "kind": "cost_estimate"},
                    {
                        "source": "manufacturing_decision",
                        "kind": "manufacturing_summary",
                    },
                ],
            }
        ),
        _extract_manufacturing_evidence(
            {
                "classification": {
                    "decision_contract": {
                        "evidence": [
                            {"source": "dfm", "kind": "manufacturability_check"},
                            {"source": "hybrid", "kind": "prediction"},
                        ],
                    },
                },
            }
        ),
        _extract_manufacturing_evidence({"classification": {"evidence": []}}),
    ]

    summary = _summarize_manufacturing_evidence(rows)

    assert summary["sample_size"] == 3
    assert summary["records_with_manufacturing_evidence"] == 2
    assert summary["manufacturing_evidence_coverage_rate"] == 0.666667
    assert summary["manufacturing_evidence_total_count"] == 5
    assert summary["source_counts"] == {
        "dfm": 2,
        "manufacturing_process": 1,
        "manufacturing_cost": 1,
        "manufacturing_decision": 1,
    }
    assert summary["source_coverage_rates"]["dfm"] == 0.666667
    assert summary["source_coverage_rates"]["manufacturing_cost"] == 0.333333
    assert summary["sources"] == [
        "dfm",
        "manufacturing_process",
        "manufacturing_cost",
        "manufacturing_decision",
    ]
