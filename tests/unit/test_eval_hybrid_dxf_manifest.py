import json
from pathlib import Path

from scripts.eval_hybrid_dxf_manifest import (
    _coarse_eval_label,
    EvalCase,
    _build_ok_row,
    _collect_manufacturing_evidence_fields,
    _exact_eval_label,
    _load_manifest_cases,
    _manufacturing_correctness_fields,
    _manufacturing_payload_quality_fields,
    _parse_manufacturing_source_tokens,
    _summarize_decision_contract_signals,
    _summarize_knowledge_signals,
    _summarize_manufacturing_evidence,
    _summarize_review_signals,
    _score_rows,
    _summarize_prep_signals,
)


def test_load_manifest_cases_prefers_relative_path(tmp_path: Path) -> None:
    dxf_dir = tmp_path / "dxf"
    dxf_dir.mkdir()
    nested = dxf_dir / "nested"
    nested.mkdir()
    (nested / "part1.dxf").write_text("0", encoding="utf-8")
    (dxf_dir / "part2.dxf").write_text("0", encoding="utf-8")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                (
                    "file_name,label_cn,relative_path,source_dir,"
                    "expected_manufacturing_evidence_sources,"
                    "expected_dfm_status,expected_dfm_detail_mode,"
                    "expected_process_label"
                ),
                (
                    "part1.dxf,过滤组件,nested/part1.dxf,,dfm;process,"
                    "manufacturable,rule,milling"
                ),
                "part2.dxf,轴承件,,,none,,,",
            ]
        ),
        encoding="utf-8",
    )

    cases = _load_manifest_cases(manifest, dxf_dir)

    assert len(cases) == 2
    assert cases[0].file_path == nested / "part1.dxf"
    assert cases[0].expected_manufacturing_evidence_sources == (
        "dfm",
        "manufacturing_process",
    )
    assert cases[0].expected_manufacturing_evidence_payloads == {
        "dfm": {"status": "manufacturable", "details.mode": "rule"},
        "manufacturing_process": {"label": "milling"},
    }
    assert cases[0].manufacturing_evidence_reviewed is True
    assert cases[0].manufacturing_payload_reviewed is True
    assert cases[1].file_path == dxf_dir / "part2.dxf"
    assert cases[1].expected_manufacturing_evidence_sources == ()
    assert cases[1].manufacturing_evidence_reviewed is True


def test_parse_manufacturing_source_tokens_supports_aliases_and_json() -> None:
    sources, reviewed = _parse_manufacturing_source_tokens(
        '["dfm", "process", "cost", "decision"]'
    )

    assert reviewed is True
    assert sources == (
        "dfm",
        "manufacturing_process",
        "manufacturing_cost",
        "manufacturing_decision",
    )

    empty_sources, empty_reviewed = _parse_manufacturing_source_tokens("none")
    assert empty_reviewed is True
    assert empty_sources == ()


def test_score_rows_supports_exact_and_coarse_evaluation() -> None:
    rows = [
        {
            "true_label": "设备",
            "hybrid_label": "再沸器",
            "graph2d_label": "设备",
        },
        {
            "true_label": "传动件",
            "hybrid_label": "搅拌轴组件",
            "graph2d_label": "",
        },
        {
            "true_label": "法兰",
            "hybrid_label": "人孔",
            "graph2d_label": "人孔法兰",
        },
    ]
    alias_map = {"other": "其他"}

    coarse_summary = _score_rows(
        rows,
        branch_to_column={
            "hybrid_label": "hybrid_label",
            "graph2d_label": "graph2d_label",
        },
        alias_map=alias_map,
        normalizer=_coarse_eval_label,
    )
    exact_summary = _score_rows(
        rows,
        branch_to_column={
            "hybrid_label": "hybrid_label",
            "graph2d_label": "graph2d_label",
        },
        alias_map=alias_map,
        normalizer=_exact_eval_label,
    )

    assert coarse_summary["hybrid_label"]["evaluated"] == 3
    assert coarse_summary["hybrid_label"]["correct"] == 2
    assert coarse_summary["hybrid_label"]["accuracy"] == 2 / 3

    assert coarse_summary["graph2d_label"]["evaluated"] == 2
    assert coarse_summary["graph2d_label"]["correct"] == 2
    assert coarse_summary["graph2d_label"]["missing_pred"] == 1

    assert exact_summary["hybrid_label"]["evaluated"] == 3
    assert exact_summary["hybrid_label"]["correct"] == 0
    assert exact_summary["graph2d_label"]["correct"] == 1


def test_build_ok_row_includes_history_and_brep_prep_fields(tmp_path: Path) -> None:
    case = EvalCase(
        file_path=tmp_path / "sample.dxf",
        file_name="sample.dxf",
        true_label="轴承件",
        source_dir="nested",
        relative_path="nested/sample.dxf",
        expected_manufacturing_evidence_sources=(
            "dfm",
            "manufacturing_process",
            "manufacturing_cost",
            "manufacturing_decision",
        ),
        expected_manufacturing_evidence_payloads={
            "dfm": {
                "status": "manufacturable",
                "kind": "manufacturability_check",
                "details.mode": "rule",
            },
            "manufacturing_process": {
                "label": "milling",
                "kind": "process_recommendation",
                "details.rule_version": "process.v1",
            },
            "manufacturing_cost": {
                "label": "CNY",
                "status": "estimated",
                "details.cost_range.low": "90.0",
            },
            "manufacturing_decision": {
                "status": "manufacturable",
                "details.risks_count": "0",
            },
        },
        manufacturing_evidence_reviewed=True,
        manufacturing_payload_reviewed=True,
    )
    results_payload = {
        "classification": {
            "part_type": "轴承件",
            "confidence": 0.81,
            "needs_review": True,
            "confidence_band": "medium",
            "review_priority": "high",
            "review_priority_score": 2.4,
            "review_reasons": ["branch_conflict", "low_confidence"],
            "branch_conflicts": {"hybrid_vs_graph2d": True},
            "contract_version": "classification_decision.v1",
            "decision_source": "hybrid",
            "fallback_flags": ["rules_baseline"],
            "coarse_part_type": "轴承件",
            "fine_part_type": "深沟球轴承",
            "fine_confidence": 0.73,
            "coarse_fine_part_type": "轴承件",
            "graph2d_prediction": {"label": "轴承件", "confidence": 0.7},
            "coarse_graph2d_label": "轴承件",
            "filename_prediction": {"label": "轴承件", "confidence": 0.8},
            "coarse_filename_label": "轴承件",
            "titleblock_prediction": {"label": "深沟球轴承", "confidence": 0.66},
            "coarse_titleblock_label": "轴承件",
            "hybrid_decision": {
                "label": "深沟球轴承",
                "confidence": 0.77,
                "source": "fusion",
                "decision_path": ["history_shadow_only", "fusion_scored"],
            },
            "coarse_hybrid_label": "轴承件",
            "source_contributions": {"history_sequence": 0.2, "graph2d": 0.8},
            "evidence": [
                {"source": "hybrid", "kind": "prediction", "label": "深沟球轴承"},
                {"source": "graph2d", "kind": "prediction", "label": "轴承件"},
            ],
            "decision_contract": {
                "fine_part_type": "深沟球轴承",
                "coarse_part_type": "轴承件",
                "confidence": 0.81,
                "decision_source": "hybrid",
                "branch_conflicts": {"hybrid_vs_graph2d": True},
                "evidence": [
                    {"source": "hybrid", "kind": "prediction", "label": "深沟球轴承"},
                    {"source": "graph2d", "kind": "prediction", "label": "轴承件"},
                ],
                "review_reasons": ["branch_conflict", "low_confidence"],
                "fallback_flags": ["rules_baseline"],
                "contract_version": "classification_decision.v1",
            },
            "hybrid_explanation": {"summary": "graph2d dominated"},
            "history_prediction": {
                "label": "轴承件",
                "confidence": 0.62,
                "status": "ok",
                "source": "history_sequence_prototype",
                "shadow_only": True,
                "used_for_fusion": False,
            },
            "history_sequence_input": {
                "resolved": True,
                "source": "sidecar_exact",
            },
            "knowledge_checks": [
                {"category": "thread_standard", "item": "M8", "status": "ok"},
                {"category": "surface_finish", "item": "Ra3.2", "status": "ok"},
            ],
            "violations": [
                {"category": "knowledge_conflict", "severity": "warn"},
            ],
            "standards_candidates": [
                {"type": "metric_thread", "designation": "M8"},
                {"type": "surface_finish", "designation": "Ra 3.2"},
            ],
            "knowledge_hints": [
                {"label": "轴承件", "score": 0.8},
                {"label": "传动件", "score": 0.3},
            ],
        },
        "manufacturing_evidence": [
            {
                "source": "dfm",
                "kind": "manufacturability_check",
                "status": "manufacturable",
                "details": {"mode": "rule"},
            },
            {
                "source": "manufacturing_process",
                "kind": "process_recommendation",
                "label": "milling",
                "details": {"rule_version": "process.v1"},
            },
            {
                "source": "manufacturing_cost",
                "kind": "cost_estimate",
                "label": "CNY",
                "status": "estimated",
                "details": {"cost_range": {"low": 90.0}},
            },
            {
                "source": "manufacturing_decision",
                "kind": "manufacturing_summary",
                "status": "manufacturable",
                "details": {"risks_count": 0},
            },
        ],
        "features_3d": {
            "valid_3d": True,
            "faces": 20,
            "surface_types": {"cylinder": 15, "plane": 5},
            "embedding_dim": 128,
        },
    }

    row = _build_ok_row(case, results_payload)

    assert row["true_label_exact"] == "轴承件"
    assert row["true_label_coarse"] == "轴承件"
    assert row["coarse_part_type"] == "轴承件"
    assert row["coarse_fine_part_type"] == "轴承件"
    assert row["coarse_graph2d_label"] == "轴承件"
    assert row["coarse_filename_label"] == "轴承件"
    assert row["coarse_titleblock_label"] == "轴承件"
    assert row["coarse_hybrid_label"] == "轴承件"
    assert row["needs_review"] is True
    assert row["confidence_band"] == "medium"
    assert row["review_priority"] == "high"
    assert row["review_priority_score"] == 2.4
    assert row["review_reasons"] == "branch_conflict;low_confidence"
    assert row["decision_contract_present"] is True
    assert row["decision_contract_version"] == "classification_decision.v1"
    assert row["decision_source"] == "hybrid"
    assert row["decision_evidence_count"] == 2
    assert row["decision_evidence_sources"] == "hybrid;graph2d"
    assert row["decision_fallback_flags"] == "rules_baseline"
    assert row["decision_review_reasons"] == "branch_conflict;low_confidence"
    assert json.loads(row["decision_contract"])["fine_part_type"] == "深沟球轴承"
    assert json.loads(row["decision_evidence"])[0]["source"] == "hybrid"
    assert json.loads(row["decision_branch_conflicts"]) == {"hybrid_vs_graph2d": True}
    assert row["manufacturing_evidence_count"] == 4
    assert row["manufacturing_evidence_sources"] == (
        "dfm;manufacturing_process;manufacturing_cost;manufacturing_decision"
    )
    assert row["manufacturing_evidence_required_sources_present"] is True
    assert json.loads(row["manufacturing_evidence"])[0]["source"] == "dfm"
    assert row["expected_manufacturing_evidence_sources"] == (
        "dfm;manufacturing_cost;manufacturing_decision;manufacturing_process"
    )
    assert row["manufacturing_evidence_reviewed"] is True
    assert row["manufacturing_evidence_source_exact_match"] is True
    assert row["manufacturing_evidence_source_precision"] == 1.0
    assert row["manufacturing_evidence_source_recall"] == 1.0
    assert row["manufacturing_evidence_source_f1"] == 1.0
    assert row["manufacturing_evidence_payload_quality_reviewed"] is True
    assert row["manufacturing_evidence_payload_expected_fields"] == 11
    assert row["manufacturing_evidence_payload_matched_fields"] == 11
    assert row["manufacturing_evidence_payload_quality_accuracy"] == 1.0
    assert row["manufacturing_evidence_payload_detail_quality_reviewed"] is True
    assert row["manufacturing_evidence_payload_detail_expected_fields"] == 4
    assert row["manufacturing_evidence_payload_detail_matched_fields"] == 4
    assert row["manufacturing_evidence_payload_detail_quality_accuracy"] == 1.0
    assert row["history_label"] == "轴承件"
    assert row["history_used_for_fusion"] is False
    assert row["history_input_resolved"] is True
    assert row["history_input_source"] == "sidecar_exact"
    assert json.loads(row["knowledge_checks"])[0]["category"] == "thread_standard"
    assert row["knowledge_check_categories"] == "thread_standard;surface_finish"
    assert row["knowledge_violation_categories"] == "knowledge_conflict"
    assert row["knowledge_standard_types"] == "metric_thread;surface_finish"
    assert row["knowledge_hint_labels"] == "轴承件;传动件"
    assert row["brep_valid_3d"] is True
    assert row["brep_faces"] == 20
    assert row["brep_primary_surface_type"] == "cylinder"
    assert row["brep_primary_surface_ratio"] == 0.75
    assert row["brep_feature_hint_top_label"] == "shaft"
    assert row["brep_feature_hint_top_score"] == 0.6
    assert row["brep_embedding_dim"] == 128
    assert json.loads(row["brep_feature_hints"]) == {"bolt": 0.4, "shaft": 0.6}


def test_summarize_decision_contract_signals_counts_coverage_and_sources() -> None:
    summary = _summarize_decision_contract_signals(
        [
            {
                "decision_contract_version": "classification_decision.v1",
                "decision_evidence_count": 2,
                "decision_evidence_sources": "hybrid;graph2d",
                "decision_fallback_flags": "rules_baseline",
                "decision_review_reasons": "branch_conflict",
                "decision_branch_conflicts": json.dumps(
                    {"hybrid_vs_graph2d": True}, ensure_ascii=False
                ),
            },
            {
                "decision_contract_version": "classification_decision.v1",
                "decision_evidence_count": 1,
                "decision_evidence_sources": "baseline",
                "decision_fallback_flags": "",
                "decision_review_reasons": "low_confidence",
                "decision_branch_conflicts": "",
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

    assert summary["row_count"] == 3
    assert summary["decision_contract_count"] == 2
    assert summary["decision_contract_coverage_rate"] == 0.666667
    assert summary["decision_evidence_row_count"] == 2
    assert summary["decision_evidence_total_count"] == 3
    assert summary["decision_evidence_coverage_rate"] == 0.666667
    assert summary["branch_conflict_count"] == 1
    assert summary["contract_version_counts"] == {"classification_decision.v1": 2}
    assert summary["evidence_source_counts"] == {
        "hybrid": 1,
        "graph2d": 1,
        "baseline": 1,
    }
    assert summary["fallback_flag_counts"] == {"rules_baseline": 1}
    assert summary["review_reason_counts"] == {
        "branch_conflict": 1,
        "low_confidence": 1,
    }


def test_collect_manufacturing_evidence_fields_filters_decision_fallback() -> None:
    fields = _collect_manufacturing_evidence_fields(
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
    )

    assert fields["manufacturing_evidence_count"] == 1
    assert fields["manufacturing_evidence_sources"] == "dfm"
    assert fields["manufacturing_evidence_has_dfm"] is True
    assert fields["manufacturing_evidence_has_process"] is False
    assert json.loads(fields["manufacturing_evidence"])[0]["source"] == "dfm"


def test_summarize_manufacturing_evidence_matches_scorecard_input_contract() -> None:
    rows = [
        _collect_manufacturing_evidence_fields(
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
        _collect_manufacturing_evidence_fields(
            {
                "manufacturing_evidence": [
                    {"source": "dfm", "kind": "manufacturability_check"},
                ],
            }
        ),
        _collect_manufacturing_evidence_fields({}),
    ]

    summary = _summarize_manufacturing_evidence(rows)

    assert summary["sample_size"] == 3
    assert summary["records_with_manufacturing_evidence"] == 2
    assert summary["manufacturing_evidence_coverage_rate"] == 0.666667
    assert summary["manufacturing_evidence_total_count"] == 5
    assert summary["source_counts"]["dfm"] == 2
    assert summary["source_counts"]["manufacturing_process"] == 1
    assert summary["source_coverage_rates"]["manufacturing_decision"] == 0.333333
    assert summary["sources"] == [
        "dfm",
        "manufacturing_process",
        "manufacturing_cost",
        "manufacturing_decision",
    ]
    assert summary["source_correctness_available"] is False
    assert summary["reviewed_sample_count"] == 0


def test_manufacturing_source_correctness_counts_reviewed_precision_recall() -> None:
    rows = [
        {
            "manufacturing_evidence_count": 3,
            "manufacturing_evidence_sources": "dfm;manufacturing_process;manufacturing_cost",
            **_manufacturing_correctness_fields(
                ["dfm", "manufacturing_process", "manufacturing_cost"],
                ["dfm", "manufacturing_process"],
                reviewed=True,
            ),
        },
        {
            "manufacturing_evidence_count": 1,
            "manufacturing_evidence_sources": "dfm",
            **_manufacturing_correctness_fields(
                ["dfm"],
                ["dfm", "manufacturing_cost", "manufacturing_decision"],
                reviewed=True,
            ),
        },
        {
            "manufacturing_evidence_count": 1,
            "manufacturing_evidence_sources": "manufacturing_decision",
            **_manufacturing_correctness_fields(
                ["manufacturing_decision"],
                [],
                reviewed=False,
            ),
        },
    ]

    summary = _summarize_manufacturing_evidence(rows)

    assert summary["source_correctness_available"] is True
    assert summary["reviewed_sample_count"] == 2
    assert summary["source_true_positive_total"] == 3
    assert summary["source_false_positive_total"] == 1
    assert summary["source_false_negative_total"] == 2
    assert summary["source_precision"] == 0.75
    assert summary["source_recall"] == 0.6
    assert summary["source_f1"] == 0.666667
    assert summary["source_correctness"]["dfm"] == {
        "expected_count": 2,
        "true_positive": 2,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert summary["source_correctness"]["manufacturing_cost"]["false_positive"] == 1
    assert summary["source_correctness"]["manufacturing_cost"]["false_negative"] == 1
    assert summary["payload_quality_available"] is False
    assert summary["payload_quality_reviewed_sample_count"] == 0


def test_manufacturing_payload_quality_counts_reviewed_fields() -> None:
    rows = [
        {
            "manufacturing_evidence_count": 2,
            "manufacturing_evidence_sources": "dfm;manufacturing_process",
            **_manufacturing_payload_quality_fields(
                [
                    {
                        "source": "dfm",
                        "kind": "manufacturability_check",
                        "status": "manufacturable",
                        "details": {"mode": "rule"},
                    },
                    {
                        "source": "manufacturing_process",
                        "kind": "process_recommendation",
                        "label": "milling",
                        "details": {"rule_version": "process.v1"},
                    },
                ],
                {
                    "dfm": {
                        "kind": "manufacturability_check",
                        "status": "manufacturable",
                        "details.mode": "rule",
                    },
                    "manufacturing_process": {
                        "label": "milling",
                        "details.rule_version": "process.v2",
                    },
                },
                reviewed=True,
            ),
        },
        {
            "manufacturing_evidence_count": 1,
            "manufacturing_evidence_sources": "manufacturing_cost",
            **_manufacturing_payload_quality_fields(
                [
                    {
                        "source": "manufacturing_cost",
                        "kind": "cost_estimate",
                        "label": "USD",
                        "details": {"cost_range": {"low": 90.0}},
                    },
                ],
                {
                    "manufacturing_cost": {
                        "label": "CNY",
                        "status": "estimated",
                        "details.cost_range.low": "90.0",
                        "details.cost_range.high": "110.0",
                    }
                },
                reviewed=True,
            ),
        },
    ]

    summary = _summarize_manufacturing_evidence(rows)

    assert summary["payload_quality_available"] is True
    assert summary["payload_quality_reviewed_sample_count"] == 2
    assert summary["payload_quality_expected_field_total"] == 9
    assert summary["payload_quality_matched_field_total"] == 5
    assert summary["payload_quality_mismatched_field_total"] == 2
    assert summary["payload_quality_missing_field_total"] == 2
    assert summary["payload_quality_accuracy"] == 0.555556
    assert summary["payload_detail_quality_available"] is True
    assert summary["payload_detail_quality_reviewed_sample_count"] == 2
    assert summary["payload_detail_quality_expected_field_total"] == 4
    assert summary["payload_detail_quality_matched_field_total"] == 2
    assert summary["payload_detail_quality_mismatched_field_total"] == 1
    assert summary["payload_detail_quality_missing_field_total"] == 1
    assert summary["payload_detail_quality_accuracy"] == 0.5
    assert summary["payload_quality"]["dfm"]["accuracy"] == 1.0
    assert summary["payload_quality"]["manufacturing_process"]["detail_accuracy"] == 0.0
    assert summary["payload_quality"]["manufacturing_cost"]["accuracy"] == 0.25


def test_summarize_prep_signals_counts_history_and_brep_rows() -> None:
    summary = _summarize_prep_signals(
        [
            {
                "history_label": "轴承件",
                "history_status": "ok",
                "history_input_resolved": True,
                "history_used_for_fusion": False,
                "history_shadow_only": True,
                "brep_valid_3d": True,
                "brep_feature_hint_top_label": "shaft",
            },
            {
                "history_label": "法兰",
                "history_status": "ok",
                "history_input_resolved": False,
                "history_used_for_fusion": True,
                "history_shadow_only": False,
                "brep_valid_3d": True,
                "brep_feature_hint_top_label": "bearing",
            },
            {
                "history_label": "",
                "history_status": "",
                "history_input_resolved": False,
                "history_used_for_fusion": None,
                "history_shadow_only": None,
                "brep_valid_3d": False,
                "brep_feature_hint_top_label": "",
            },
        ]
    )

    assert summary["history_prediction_count"] == 2
    assert summary["history_input_resolved_count"] == 1
    assert summary["history_used_for_fusion_true"] == 1
    assert summary["history_used_for_fusion_false"] == 1
    assert summary["history_shadow_only_true"] == 1
    assert summary["history_status_counts"] == {"ok": 2}
    assert summary["brep_valid_3d_count"] == 2
    assert summary["brep_feature_hints_count"] == 2
    assert summary["brep_top_hint_counts"] == {"shaft": 1, "bearing": 1}


def test_summarize_knowledge_signals_counts_categories_and_candidates() -> None:
    summary = _summarize_knowledge_signals(
        [
            {
                "knowledge_check_categories": "thread_standard;surface_finish",
                "knowledge_violation_categories": "knowledge_conflict",
                "knowledge_standard_types": "metric_thread;surface_finish",
                "knowledge_hint_labels": "轴类;传动件",
            },
            {
                "knowledge_check_categories": "general_tolerance",
                "knowledge_violation_categories": "",
                "knowledge_standard_types": "general_tolerance",
                "knowledge_hint_labels": "壳体类",
            },
        ]
    )

    assert summary["rows_with_checks"] == 2
    assert summary["rows_with_violations"] == 1
    assert summary["rows_with_standards_candidates"] == 2
    assert summary["rows_with_hints"] == 2
    assert summary["total_checks"] == 3
    assert summary["total_violations"] == 1
    assert summary["total_standards_candidates"] == 3
    assert summary["total_hints"] == 3
    assert summary["top_check_categories"]["thread_standard"] == 1
    assert summary["top_standard_types"]["general_tolerance"] == 1
    assert summary["top_violation_categories"]["knowledge_conflict"] == 1
    assert summary["top_hint_labels"]["轴类"] == 1


def test_summarize_review_signals_counts_bands_and_priorities() -> None:
    summary = _summarize_review_signals(
        [
            {
                "needs_review": True,
                "confidence_band": "rejected",
                "review_priority": "critical",
                "review_reasons": "hybrid_rejected:below_min_confidence;knowledge_conflict",
            },
            {
                "needs_review": True,
                "confidence_band": "low",
                "review_priority": "medium",
                "review_reasons": "low_confidence",
            },
            {
                "needs_review": False,
                "confidence_band": "high",
                "review_priority": "none",
                "review_reasons": "",
            },
        ]
    )

    assert summary["needs_review_count"] == 2
    assert summary["confidence_band_counts"] == {"rejected": 1, "low": 1, "high": 1}
    assert summary["review_priority_counts"] == {
        "critical": 1,
        "medium": 1,
        "none": 1,
    }
    assert summary["top_review_reasons"] == {
        "hybrid_rejected:below_min_confidence": 1,
        "knowledge_conflict": 1,
        "low_confidence": 1,
    }
