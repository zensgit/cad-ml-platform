from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_benchmark_scorecard.py"


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_generate_benchmark_scorecard_outputs_files(tmp_path: Path) -> None:
    hybrid = _write_json(
        tmp_path / "hybrid.json",
        {
            "sample_size": 110,
            "exact_accuracy": {
                "hybrid_label": {"accuracy": 0.8727},
                "final_part_type": {"accuracy": 0.5545},
                "graph2d_label": {"accuracy": 0.1182},
            },
            "coarse_accuracy": {
                "hybrid_label": {"accuracy": 0.9},
            },
            "confidence": {
                "graph2d_label": {"low_conf_rate": 1.0},
            },
        },
    )
    graph2d_metrics = _write_json(tmp_path / "graph2d_metrics.json", {"best_val_acc": 0.1538})
    graph2d_diagnose = _write_json(
        tmp_path / "graph2d_diagnose.json",
        {"accuracy": 0.1182, "low_conf_rate": 1.0},
    )
    graph2d_blind = _write_json(
        tmp_path / "graph2d_blind.json",
        {"accuracy": 0.1273, "low_conf_rate": 1.0},
    )
    history = _write_json(
        tmp_path / "history.json",
        {
            "total": 24,
            "coverage": 0.75,
            "accuracy_overall": 0.7,
            "coarse_accuracy_overall": 0.85,
            "low_conf_rate": 0.2,
        },
    )
    brep = _write_json(
        tmp_path / "brep.json",
        {
            "sample_size": 3,
            "valid_3d_count": 3,
            "hint_coverage_count": 2,
            "graph_schema_version_counts": {"v2": 3},
        },
    )
    migration = _write_json(
        tmp_path / "migration.json",
        {
            "plan_ready": True,
            "coverage_complete": True,
            "recommended_from_versions": ["v1"],
            "planned_pending_ratio": 1.0,
            "estimated_total_runs": 2,
        },
    )
    qdrant = _write_json(
        tmp_path / "qdrant.json",
        {
            "backend": "qdrant",
            "backend_health": {
                "reachable": True,
                "collection_exists": True,
                "collection_name": "cad_vectors",
                "collection_status": "green",
                "points_count": 24,
                "indexed_vectors_count": 24,
                "unindexed_vectors_count": 0,
                "indexed_ratio": 1.0,
                "observed_vectors_count": 24,
                "scan_limit": 5000,
                "scan_truncated": False,
                "readiness": "ready",
                "readiness_hints": [],
            },
        },
    )
    assistant = _write_json(
        tmp_path / "assistant.json",
        {
            "total_records": 48,
            "total_evidence_items": 96,
            "average_evidence_count": 2.0,
            "records_with_evidence_pct": 0.92,
            "records_with_decision_path_pct": 0.83,
            "records_with_any_source_signal_pct": 0.88,
            "top_record_kinds": ["assistant", "analyze"],
            "top_evidence_types": ["direct", "derived"],
            "top_structured_sources": ["filename", "titleblock"],
            "top_missing_fields": ["none"],
        },
    )
    review_queue = _write_json(
        tmp_path / "review_queue.json",
        {
            "total": 0,
            "automation_ready_count": 0,
            "automation_ready_ratio": 0.0,
            "evidence_count_total": 0,
            "average_evidence_count": 0.0,
            "records_with_evidence_count": 0,
            "records_with_evidence_ratio": 0.0,
            "top_evidence_sources": [],
            "by_sample_type": {},
            "by_feedback_priority": {},
            "by_decision_source": {},
            "by_review_reason": {},
        },
    )
    feedback = _write_json(
        tmp_path / "feedback.json",
        {
            "total": 12,
            "correction_count": 8,
            "coarse_correction_count": 6,
            "average_rating": 4.3,
            "by_review_outcome": {"updated": 8, "accepted": 4},
            "by_review_reason": {"low_confidence": 5, "branch_conflict": 3},
        },
    )
    finetune = _write_json(
        tmp_path / "finetune.json",
        {
            "sample_count": 8,
            "vector_count": 8,
            "label_distribution": {"人孔": 4, "法兰": 4},
            "coarse_label_distribution": {"开孔件": 4, "法兰": 4},
        },
    )
    metric_train = _write_json(
        tmp_path / "metric_train.json",
        {
            "feedback_entry_count": 12,
            "triplet_count": 6,
            "unique_anchor_count": 6,
            "anchor_label_distribution": {"人孔": 3, "法兰": 3},
            "negative_label_distribution": {"捕集口": 3, "拖轮组件": 3},
        },
    )
    ocr_review = _write_json(
        tmp_path / "ocr_review.json",
        {
            "review_candidate_count": 0,
            "exported_records": 0,
            "automation_ready_count": 0,
            "average_readiness_score": 0.0,
            "average_coverage_ratio": 0.0,
            "review_priority_counts": [],
            "primary_gap_counts": [],
            "top_review_reasons": [],
        },
    )
    engineering = _write_json(
        tmp_path / "engineering.json",
        {
            "sample_size": 110,
            "knowledge_signals": {
                "rows_with_checks": 60,
                "rows_with_violations": 22,
                "rows_with_standards_candidates": 35,
                "rows_with_hints": 50,
                "total_checks": 80,
                "total_violations": 22,
                "total_standards_candidates": 35,
                "total_hints": 70,
                "top_violation_categories": {"standard_conflict": 10},
                "top_standard_types": {"gdt": 20},
                "top_hint_labels": {"flange": 15},
            },
        },
    )
    output_json = tmp_path / "scorecard.json"
    output_md = tmp_path / "scorecard.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--title",
            "CAD Benchmark Scorecard",
            "--hybrid-summary",
            str(hybrid),
            "--graph2d-metrics",
            str(graph2d_metrics),
            "--graph2d-diagnose",
            str(graph2d_diagnose),
            "--graph2d-blind-diagnose",
            str(graph2d_blind),
            "--history-summary",
            str(history),
            "--brep-summary",
            str(brep),
            "--migration-summary",
            str(migration),
            "--qdrant-readiness-summary",
            str(qdrant),
            "--assistant-evidence-summary",
            str(assistant),
            "--review-queue-summary",
            str(review_queue),
            "--feedback-summary",
            str(feedback),
            "--finetune-summary",
            str(finetune),
            "--metric-train-summary",
            str(metric_train),
            "--ocr-review-summary",
            str(ocr_review),
            "--engineering-signals-summary",
            str(engineering),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["overall_status"] == "benchmark_ready_with_multisignal_evidence"
    assert payload["components"]["graph2d"]["status"] == "weak_signal_only"
    assert payload["components"]["hybrid"]["has_output_gap"] is True
    assert payload["components"]["history_sequence"]["status"] == "evidence_ready"
    assert payload["components"]["brep"]["status"] == "graph_ready"
    assert payload["components"]["migration_governance"]["status"] == "operationally_ready"
    assert payload["components"]["qdrant_backend"]["status"] == "ready"
    assert payload["components"]["qdrant_backend"]["readiness"] == "ready"
    assert payload["components"]["qdrant_backend"]["indexed_ratio"] == 1.0
    assert payload["components"]["qdrant_backend"]["unindexed_vectors_count"] == 0
    assert payload["components"]["qdrant_backend"]["scan_truncated"] is False
    assert payload["components"]["assistant_explainability"]["status"] == "explainability_ready"
    assert payload["components"]["review_queue"]["status"] == "under_control"
    assert payload["components"]["review_queue"]["records_with_evidence_ratio"] == 0.0
    assert payload["components"]["review_queue"]["average_evidence_count"] == 0.0
    assert payload["components"]["feedback_flywheel"]["status"] == "closed_loop_ready"
    assert payload["components"]["feedback_flywheel"]["feedback_total"] == 12
    assert payload["components"]["feedback_flywheel"]["metric_triplet_count"] == 6
    assert payload["components"]["ocr_review"]["status"] == "ocr_ready"
    assert payload["components"]["engineering_signals"]["status"] == "engineering_semantics_ready"
    assert output_json.exists()
    assert output_md.exists()
    markdown = output_md.read_text(encoding="utf-8")
    assert "CAD Benchmark Scorecard" in markdown
    assert "weak_signal_only" in markdown
    assert "qdrant_backend" in markdown
    assert "assistant_explainability" in markdown
    assert "review_queue" in markdown
    assert "ocr_review" in markdown
    assert "engineering_signals" in markdown


def test_generate_benchmark_scorecard_handles_missing_optional_inputs(tmp_path: Path) -> None:
    hybrid = _write_json(
        tmp_path / "hybrid.json",
        {
            "sample_size": 10,
            "exact_accuracy": {
                "hybrid_label": {"accuracy": 0.82},
                "final_part_type": {"accuracy": 0.79},
                "graph2d_label": {"accuracy": 0.1},
            },
            "coarse_accuracy": {
                "hybrid_label": {"accuracy": 0.88},
            },
            "confidence": {
                "graph2d_label": {"low_conf_rate": 1.0},
            },
        },
    )
    output_json = tmp_path / "scorecard.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hybrid-summary",
            str(hybrid),
            "--output-json",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["components"]["history_sequence"]["status"] == "missing"
    assert payload["components"]["brep"]["status"] == "missing"
    assert payload["components"]["migration_governance"]["status"] == "missing"
    assert payload["components"]["qdrant_backend"]["status"] == "missing"
    assert payload["components"]["assistant_explainability"]["status"] == "missing"
    assert payload["components"]["review_queue"]["status"] == "missing"
    assert payload["components"]["feedback_flywheel"]["status"] == "missing"
    assert payload["components"]["ocr_review"]["status"] == "missing"
    assert payload["overall_status"] == "benchmark_ready_without_governance"
    assert output_json.exists()


def test_generate_benchmark_scorecard_reports_ocr_gap(tmp_path: Path) -> None:
    hybrid = _write_json(
        tmp_path / "hybrid.json",
        {
            "sample_size": 12,
            "exact_accuracy": {
                "hybrid_label": {"accuracy": 0.84},
                "final_part_type": {"accuracy": 0.8},
                "graph2d_label": {"accuracy": 0.1},
            },
            "coarse_accuracy": {"hybrid_label": {"accuracy": 0.9}},
            "confidence": {"graph2d_label": {"low_conf_rate": 1.0}},
        },
    )
    governance = _write_json(
        tmp_path / "migration.json",
        {
            "plan_ready": True,
            "coverage_complete": True,
        },
    )
    history = _write_json(
        tmp_path / "history.json",
        {
            "total": 20,
            "coverage": 0.8,
            "accuracy_overall": 0.75,
            "coarse_accuracy_overall": 0.86,
            "low_conf_rate": 0.1,
        },
    )
    brep = _write_json(
        tmp_path / "brep.json",
        {
            "sample_size": 3,
            "valid_3d_count": 3,
            "hint_coverage_count": 2,
            "graph_schema_version_counts": {"v2": 3},
        },
    )
    assistant = _write_json(
        tmp_path / "assistant.json",
        {
            "total_records": 10,
            "total_evidence_items": 20,
            "average_evidence_count": 2.0,
            "records_with_evidence_pct": 0.9,
            "records_with_decision_path_pct": 0.8,
            "records_with_any_source_signal_pct": 0.8,
        },
    )
    review_queue = _write_json(
        tmp_path / "review_queue.json",
        {
            "total": 0,
            "automation_ready_count": 0,
            "automation_ready_ratio": 0.0,
            "evidence_count_total": 0,
            "average_evidence_count": 0.0,
            "records_with_evidence_count": 0,
            "records_with_evidence_ratio": 0.0,
            "top_evidence_sources": [],
            "by_sample_type": {},
            "by_feedback_priority": {},
            "by_decision_source": {},
            "by_review_reason": {},
        },
    )
    feedback = _write_json(
        tmp_path / "feedback.json",
        {
            "total": 0,
            "correction_count": 0,
            "coarse_correction_count": 0,
            "average_rating": None,
        },
    )
    ocr_review = _write_json(
        tmp_path / "ocr_review.json",
        {
            "review_candidate_count": 5,
            "exported_records": 5,
            "automation_ready_count": 0,
            "average_readiness_score": 0.2,
            "average_coverage_ratio": 0.3,
            "review_priority_counts": [{"name": "high", "count": 5}],
            "primary_gap_counts": [{"name": "title_block", "count": 5}],
            "top_review_reasons": [{"name": "missing_title_block", "count": 5}],
        },
    )
    output_json = tmp_path / "scorecard.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hybrid-summary",
            str(hybrid),
            "--migration-summary",
            str(governance),
            "--history-summary",
            str(history),
            "--brep-summary",
            str(brep),
            "--assistant-evidence-summary",
            str(assistant),
            "--review-queue-summary",
            str(review_queue),
            "--feedback-summary",
            str(feedback),
            "--ocr-review-summary",
            str(ocr_review),
            "--output-json",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["components"]["feedback_flywheel"]["status"] == "missing"
    assert payload["overall_status"] == "benchmark_ready_with_feedback_gap"
    assert any("feedback" in item.lower() for item in payload["recommendations"])
    assert payload["components"]["ocr_review"]["status"] == "review_heavy"
    assert any("OCR" in item for item in payload["recommendations"])


def test_generate_benchmark_scorecard_reports_review_queue_evidence_gap(
    tmp_path: Path,
) -> None:
    hybrid = _write_json(
        tmp_path / "hybrid.json",
        {
            "sample_size": 20,
            "exact_accuracy": {
                "hybrid_label": {"accuracy": 0.9},
                "final_part_type": {"accuracy": 0.88},
                "graph2d_label": {"accuracy": 0.1},
            },
            "coarse_accuracy": {"hybrid_label": {"accuracy": 0.92}},
            "confidence": {"graph2d_label": {"low_conf_rate": 1.0}},
        },
    )
    governance = _write_json(
        tmp_path / "migration.json",
        {
            "plan_ready": True,
            "coverage_complete": True,
        },
    )
    history = _write_json(
        tmp_path / "history.json",
        {
            "total": 24,
            "coverage": 0.75,
            "accuracy_overall": 0.75,
            "coarse_accuracy_overall": 0.86,
            "low_conf_rate": 0.15,
        },
    )
    brep = _write_json(
        tmp_path / "brep.json",
        {
            "sample_size": 3,
            "valid_3d_count": 3,
            "hint_coverage_count": 2,
            "graph_schema_version_counts": {"v2": 3},
        },
    )
    assistant = _write_json(
        tmp_path / "assistant.json",
        {
            "total_records": 48,
            "total_evidence_items": 96,
            "average_evidence_count": 2.0,
            "records_with_evidence_pct": 0.92,
            "records_with_decision_path_pct": 0.83,
            "records_with_any_source_signal_pct": 0.88,
        },
    )
    review_queue = _write_json(
        tmp_path / "review_queue.json",
        {
            "total": 5,
            "automation_ready_count": 2,
            "automation_ready_ratio": 0.4,
            "evidence_count_total": 2,
            "average_evidence_count": 0.4,
            "records_with_evidence_count": 2,
            "records_with_evidence_ratio": 0.4,
            "top_evidence_sources": [{"name": "filename", "count": 2}],
            "by_sample_type": {"review": 5},
            "by_feedback_priority": {"medium": 5},
            "by_decision_source": {"hybrid": 5},
            "by_review_reason": {"low_confidence": 5},
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hybrid-summary",
            str(hybrid),
            "--migration-summary",
            str(governance),
            "--history-summary",
            str(history),
            "--brep-summary",
            str(brep),
            "--assistant-evidence-summary",
            str(assistant),
            "--review-queue-summary",
            str(review_queue),
            "--output-json",
            str(tmp_path / "scorecard.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["components"]["review_queue"]["status"] == "evidence_gap"
    assert payload["overall_status"] == "benchmark_ready_with_review_gap"
    assert (
        "Raise evidence coverage in active-learning review queue exports"
        in " ".join(payload["recommendations"])
    )


def test_generate_benchmark_scorecard_reports_qdrant_gap(tmp_path: Path) -> None:
    hybrid = _write_json(
        tmp_path / "hybrid.json",
        {
            "sample_size": 12,
            "exact_accuracy": {
                "hybrid_label": {"accuracy": 0.84},
                "final_part_type": {"accuracy": 0.8},
                "graph2d_label": {"accuracy": 0.1},
            },
            "coarse_accuracy": {"hybrid_label": {"accuracy": 0.9}},
            "confidence": {"graph2d_label": {"low_conf_rate": 1.0}},
        },
    )
    governance = _write_json(
        tmp_path / "migration.json",
        {
            "plan_ready": True,
            "coverage_complete": True,
        },
    )
    qdrant = _write_json(
        tmp_path / "qdrant.json",
        {
            "reachable": True,
            "collection_exists": True,
            "collection_name": "cad_vectors",
            "collection_status": "yellow",
            "points_count": 10,
            "indexed_vectors_count": 8,
            "unindexed_vectors_count": 2,
            "indexed_ratio": 0.8,
            "observed_vectors_count": 5,
            "scan_limit": 5,
            "scan_truncated": True,
            "readiness": "partial_scan",
            "readiness_hints": [
                "scan_truncated_use_list_or_migration_for_exact_coverage",
                "vector_index_backfill_in_progress",
            ],
        },
    )
    history = _write_json(
        tmp_path / "history.json",
        {
            "total": 20,
            "coverage": 0.8,
            "accuracy_overall": 0.75,
            "coarse_accuracy_overall": 0.86,
            "low_conf_rate": 0.1,
        },
    )
    brep = _write_json(
        tmp_path / "brep.json",
        {
            "sample_size": 3,
            "valid_3d_count": 3,
            "hint_coverage_count": 2,
            "graph_schema_version_counts": {"v2": 3},
        },
    )
    assistant = _write_json(
        tmp_path / "assistant.json",
        {
            "total_records": 10,
            "total_evidence_items": 20,
            "average_evidence_count": 2.0,
            "records_with_evidence_pct": 0.9,
            "records_with_decision_path_pct": 0.8,
            "records_with_any_source_signal_pct": 0.8,
        },
    )
    review_queue = _write_json(
        tmp_path / "review_queue.json",
        {
            "total": 0,
            "automation_ready_count": 0,
            "automation_ready_ratio": 0.0,
            "evidence_count_total": 0,
            "average_evidence_count": 0.0,
            "records_with_evidence_count": 0,
            "records_with_evidence_ratio": 0.0,
            "top_evidence_sources": [],
            "by_sample_type": {},
            "by_feedback_priority": {},
            "by_decision_source": {},
            "by_review_reason": {},
        },
    )
    feedback = _write_json(
        tmp_path / "feedback.json",
        {
            "total": 8,
            "correction_count": 6,
            "coarse_correction_count": 5,
            "average_rating": 4.1,
        },
    )
    finetune = _write_json(
        tmp_path / "finetune.json",
        {
            "sample_count": 6,
            "vector_count": 6,
            "label_distribution": {"人孔": 3, "法兰": 3},
            "coarse_label_distribution": {"开孔件": 3, "法兰": 3},
        },
    )
    metric_train = _write_json(
        tmp_path / "metric_train.json",
        {
            "feedback_entry_count": 8,
            "triplet_count": 4,
            "unique_anchor_count": 4,
            "anchor_label_distribution": {"人孔": 2, "法兰": 2},
            "negative_label_distribution": {"捕集口": 2, "拖轮组件": 2},
        },
    )
    ocr_review = _write_json(
        tmp_path / "ocr_review.json",
        {
            "review_candidate_count": 0,
            "exported_records": 0,
            "automation_ready_count": 0,
            "average_readiness_score": 0.0,
            "average_coverage_ratio": 0.0,
            "review_priority_counts": [],
            "primary_gap_counts": [],
            "top_review_reasons": [],
        },
    )
    output_json = tmp_path / "scorecard.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hybrid-summary",
            str(hybrid),
            "--migration-summary",
            str(governance),
            "--qdrant-readiness-summary",
            str(qdrant),
            "--history-summary",
            str(history),
            "--brep-summary",
            str(brep),
            "--assistant-evidence-summary",
            str(assistant),
            "--review-queue-summary",
            str(review_queue),
            "--feedback-summary",
            str(feedback),
            "--finetune-summary",
            str(finetune),
            "--metric-train-summary",
            str(metric_train),
            "--ocr-review-summary",
            str(ocr_review),
            "--output-json",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["components"]["qdrant_backend"]["status"] == "partial_scan"
    assert payload["components"]["qdrant_backend"]["indexed_ratio"] == 0.8
    assert payload["components"]["qdrant_backend"]["unindexed_vectors_count"] == 2
    assert payload["components"]["qdrant_backend"]["scan_truncated"] is True
    assert payload["components"]["feedback_flywheel"]["status"] == "closed_loop_ready"
    assert payload["overall_status"] == "benchmark_ready_with_qdrant_gap"
    assert any("Qdrant" in item for item in payload["recommendations"])


def test_generate_benchmark_scorecard_reports_feedback_gap(tmp_path: Path) -> None:
    hybrid = _write_json(
        tmp_path / "hybrid.json",
        {
            "sample_size": 20,
            "exact_accuracy": {
                "hybrid_label": {"accuracy": 0.9},
                "final_part_type": {"accuracy": 0.88},
                "graph2d_label": {"accuracy": 0.1},
            },
            "coarse_accuracy": {"hybrid_label": {"accuracy": 0.92}},
            "confidence": {"graph2d_label": {"low_conf_rate": 1.0}},
        },
    )
    governance = _write_json(
        tmp_path / "migration.json",
        {
            "plan_ready": True,
            "coverage_complete": True,
        },
    )
    history = _write_json(
        tmp_path / "history.json",
        {
            "total": 24,
            "coverage": 0.75,
            "accuracy_overall": 0.75,
            "coarse_accuracy_overall": 0.86,
            "low_conf_rate": 0.15,
        },
    )
    brep = _write_json(
        tmp_path / "brep.json",
        {
            "sample_size": 3,
            "valid_3d_count": 3,
            "hint_coverage_count": 2,
            "graph_schema_version_counts": {"v2": 3},
        },
    )
    assistant = _write_json(
        tmp_path / "assistant.json",
        {
            "total_records": 48,
            "total_evidence_items": 96,
            "average_evidence_count": 2.0,
            "records_with_evidence_pct": 0.92,
            "records_with_decision_path_pct": 0.83,
            "records_with_any_source_signal_pct": 0.88,
        },
    )
    review_queue = _write_json(
        tmp_path / "review_queue.json",
        {
            "total": 0,
            "automation_ready_count": 0,
            "automation_ready_ratio": 0.0,
            "by_sample_type": {},
            "by_feedback_priority": {},
            "by_decision_source": {},
            "by_review_reason": {},
        },
    )
    feedback = _write_json(
        tmp_path / "feedback.json",
        {
            "total": 6,
            "correction_count": 4,
            "coarse_correction_count": 3,
            "average_rating": 4.0,
            "by_review_outcome": {"updated": 4, "accepted": 2},
            "by_review_reason": {"low_confidence": 3, "branch_conflict": 1},
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--hybrid-summary",
            str(hybrid),
            "--migration-summary",
            str(governance),
            "--history-summary",
            str(history),
            "--brep-summary",
            str(brep),
            "--assistant-evidence-summary",
            str(assistant),
            "--review-queue-summary",
            str(review_queue),
            "--feedback-summary",
            str(feedback),
            "--output-json",
            str(tmp_path / "scorecard.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["components"]["feedback_flywheel"]["status"] == "feedback_collected"
    assert payload["overall_status"] == "benchmark_ready_with_feedback_gap"
    assert any("feedback" in item.lower() for item in payload["recommendations"])
