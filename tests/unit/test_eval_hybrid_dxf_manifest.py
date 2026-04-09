import json
from pathlib import Path

from scripts.eval_hybrid_dxf_manifest import (
    _coarse_eval_label,
    EvalCase,
    _build_ok_row,
    _exact_eval_label,
    _load_manifest_cases,
    _summarize_knowledge_signals,
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
                "file_name,label_cn,relative_path,source_dir",
                "part1.dxf,过滤组件,nested/part1.dxf,",
                "part2.dxf,轴承件,,",
            ]
        ),
        encoding="utf-8",
    )

    cases = _load_manifest_cases(manifest, dxf_dir)

    assert len(cases) == 2
    assert cases[0].file_path == nested / "part1.dxf"
    assert cases[1].file_path == dxf_dir / "part2.dxf"


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
