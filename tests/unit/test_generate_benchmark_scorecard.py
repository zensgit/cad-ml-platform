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
            "by_sample_type": {},
            "by_feedback_priority": {},
            "by_decision_source": {},
            "by_review_reason": {},
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
            "--assistant-evidence-summary",
            str(assistant),
            "--review-queue-summary",
            str(review_queue),
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
    assert payload["components"]["assistant_explainability"]["status"] == "explainability_ready"
    assert payload["components"]["review_queue"]["status"] == "under_control"
    assert output_json.exists()
    assert output_md.exists()
    markdown = output_md.read_text(encoding="utf-8")
    assert "CAD Benchmark Scorecard" in markdown
    assert "weak_signal_only" in markdown
    assert "assistant_explainability" in markdown
    assert "review_queue" in markdown


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
    assert payload["components"]["assistant_explainability"]["status"] == "missing"
    assert payload["components"]["review_queue"]["status"] == "missing"
    assert payload["overall_status"] == "benchmark_ready_without_governance"
    assert output_json.exists()
