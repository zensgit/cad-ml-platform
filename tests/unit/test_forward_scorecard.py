from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.core.benchmark.forward_scorecard import (
    build_forward_scorecard,
    render_forward_scorecard_markdown,
)


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_forward_scorecard.py"


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _ready_model_registry() -> dict:
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


def _ready_inputs() -> dict:
    return {
        "model_readiness": _ready_model_registry(),
        "hybrid_summary": {
            "sample_size": 80,
            "coarse_accuracy": {"hybrid_label": {"accuracy": 0.91}},
            "exact_accuracy": {"hybrid_label": {"accuracy": 0.86}},
            "coarse_macro_f1_overall": 0.88,
            "confidence": {"hybrid_label": {"low_conf_rate": 0.05}},
        },
        "graph2d_summary": {
            "sample_size": 80,
            "blind_accuracy": 0.84,
            "low_conf_rate": 0.1,
        },
        "history_summary": {
            "total": 60,
            "coarse_accuracy_overall": 0.83,
            "coarse_macro_f1_overall": 0.81,
            "accuracy_overall": 0.78,
            "low_conf_rate": 0.08,
        },
        "brep_summary": {
            "sample_size": 30,
            "parse_success_count": 30,
            "valid_3d_count": 30,
            "graph_valid_count": 29,
        },
        "qdrant_summary": {
            "backend_health": {
                "readiness": "ready",
                "indexed_ratio": 1.0,
                "unindexed_vectors_count": 0,
                "scan_truncated": False,
            }
        },
        "review_queue_summary": {
            "total": 2,
            "by_feedback_priority": {"low": 2},
            "records_with_evidence_ratio": 1.0,
            "automation_ready_ratio": 1.0,
        },
        "knowledge_summary": {
            "knowledge_readiness": {
                "status": "knowledge_foundation_ready",
                "ready_component_count": 4,
                "partial_component_count": 0,
                "missing_component_count": 0,
                "total_reference_items": 120,
            },
            "knowledge_grounding": {
                "sample_size": 80,
                "records_with_knowledge_evidence": 80,
                "knowledge_evidence_coverage_rate": 1.0,
                "records_with_rule_sources": 80,
                "rule_source_coverage_rate": 1.0,
                "records_with_rule_versions": 80,
                "rule_version_coverage_rate": 1.0,
                "rule_sources": [
                    "materials_catalog",
                    "iso286_it_grade_catalog",
                    "iso1302_surface_finish_catalog",
                ],
                "rule_versions": ["knowledge_grounding.v1"],
            },
        },
        "manufacturing_summary": {
            "sample_size": 80,
            "records_with_manufacturing_evidence": 80,
            "manufacturing_evidence_coverage_rate": 1.0,
            "source_counts": {
                "dfm": 80,
                "manufacturing_process": 80,
                "manufacturing_cost": 80,
                "manufacturing_decision": 80,
            },
            "source_correctness_available": True,
            "reviewed_sample_count": 80,
            "source_exact_match_rate": 1.0,
            "source_precision": 1.0,
            "source_recall": 1.0,
            "source_f1": 1.0,
            "payload_quality_available": True,
            "payload_quality_reviewed_sample_count": 80,
            "payload_quality_accuracy": 1.0,
            "payload_detail_quality_available": True,
            "payload_detail_quality_reviewed_sample_count": 80,
            "payload_detail_quality_accuracy": 1.0,
            "sources": [
                "dfm",
                "manufacturing_process",
                "manufacturing_cost",
                "manufacturing_decision",
            ],
        },
    }


def test_forward_scorecard_can_be_release_ready() -> None:
    payload = build_forward_scorecard(**_ready_inputs(), generated_at=1)

    assert payload["overall_status"] == "release_ready"
    assert payload["components"]["hybrid_dxf"]["status"] == "release_ready"
    assert payload["components"]["brep"]["status"] == "release_ready"
    assert payload["components"]["knowledge"]["rule_version_coverage_rate"] == 1.0
    assert payload["components"]["knowledge"]["grounding_gaps"] == []
    assert payload["components"]["manufacturing_evidence"]["status"] == "release_ready"
    assert payload["components"]["manufacturing_evidence"]["source_precision"] == 1.0
    assert payload["components"]["manufacturing_evidence"]["source_recall"] == 1.0
    assert (
        payload["components"]["manufacturing_evidence"]["payload_quality_accuracy"]
        == 1.0
    )
    assert (
        payload["components"]["manufacturing_evidence"][
            "payload_detail_quality_accuracy"
        ]
        == 1.0
    )
    assert payload["release_claim_rule"].startswith("Release claims")


def test_fallback_model_prevents_release_ready() -> None:
    inputs = _ready_inputs()
    inputs["model_readiness"] = {
        **_ready_model_registry(),
        "degraded": True,
        "status": "degraded",
        "degraded_reasons": ["graph2d:fallback"],
        "items": {
            **_ready_model_registry()["items"],
            "graph2d": {"status": "fallback"},
        },
    }

    payload = build_forward_scorecard(**inputs, generated_at=1)

    assert payload["overall_status"] == "benchmark_ready_with_gap"
    assert payload["components"]["model_readiness"]["fallback_models"] == ["graph2d"]


def test_missing_primary_hybrid_blocks_release_claim() -> None:
    payload = build_forward_scorecard(
        model_readiness=_ready_model_registry(),
        hybrid_summary={},
        generated_at=1,
    )

    assert payload["overall_status"] == "blocked"
    assert payload["components"]["hybrid_dxf"]["status"] == "blocked"


def test_missing_knowledge_grounding_downgrades_release_ready() -> None:
    inputs = _ready_inputs()
    inputs["knowledge_summary"] = {
        "knowledge_readiness": {
            "status": "knowledge_foundation_ready",
            "ready_component_count": 4,
            "partial_component_count": 0,
            "missing_component_count": 0,
            "total_reference_items": 120,
        }
    }

    payload = build_forward_scorecard(**inputs, generated_at=1)

    assert payload["overall_status"] == "benchmark_ready_with_gap"
    assert payload["components"]["knowledge"]["status"] == "benchmark_ready_with_gap"
    assert payload["components"]["knowledge"]["rule_source_coverage_rate"] == 0.0
    assert payload["components"]["knowledge"]["rule_version_coverage_rate"] == 0.0
    assert "rule_source_coverage_below_release" in (
        payload["components"]["knowledge"]["grounding_gaps"]
    )
    assert any("knowledge grounding" in item for item in payload["recommendations"])


def test_missing_manufacturing_evidence_downgrades_release_ready() -> None:
    inputs = _ready_inputs()
    inputs["manufacturing_summary"] = {}

    payload = build_forward_scorecard(**inputs, generated_at=1)

    assert payload["overall_status"] == "benchmark_ready_with_gap"
    assert payload["components"]["manufacturing_evidence"]["status"] == "blocked"
    assert "manufacturing_evidence_sample_missing" in (
        payload["components"]["manufacturing_evidence"]["evidence_gaps"]
    )
    assert any("manufacturing evidence" in item for item in payload["recommendations"])


def test_missing_manufacturing_correctness_downgrades_release_ready() -> None:
    inputs = _ready_inputs()
    inputs["manufacturing_summary"] = {
        key: value
        for key, value in inputs["manufacturing_summary"].items()
        if key
        not in {
            "source_correctness_available",
            "reviewed_sample_count",
            "source_exact_match_rate",
            "source_precision",
            "source_recall",
            "source_f1",
        }
    }

    payload = build_forward_scorecard(**inputs, generated_at=1)

    assert payload["overall_status"] == "benchmark_ready_with_gap"
    assert (
        payload["components"]["manufacturing_evidence"]["status"]
        == "benchmark_ready_with_gap"
    )
    assert "manufacturing_evidence_correctness_review_missing" in (
        payload["components"]["manufacturing_evidence"]["evidence_gaps"]
    )


def test_missing_manufacturing_payload_quality_downgrades_release_ready() -> None:
    inputs = _ready_inputs()
    inputs["manufacturing_summary"] = {
        key: value
        for key, value in inputs["manufacturing_summary"].items()
        if key
        not in {
            "payload_quality_available",
            "payload_quality_reviewed_sample_count",
            "payload_quality_accuracy",
        }
    }

    payload = build_forward_scorecard(**inputs, generated_at=1)

    assert payload["overall_status"] == "benchmark_ready_with_gap"
    assert (
        payload["components"]["manufacturing_evidence"]["status"]
        == "benchmark_ready_with_gap"
    )
    assert "manufacturing_evidence_payload_quality_missing" in (
        payload["components"]["manufacturing_evidence"]["evidence_gaps"]
    )


def test_missing_manufacturing_payload_detail_quality_downgrades_release_ready() -> None:
    inputs = _ready_inputs()
    inputs["manufacturing_summary"] = {
        key: value
        for key, value in inputs["manufacturing_summary"].items()
        if key
        not in {
            "payload_detail_quality_available",
            "payload_detail_quality_reviewed_sample_count",
            "payload_detail_quality_accuracy",
        }
    }

    payload = build_forward_scorecard(**inputs, generated_at=1)

    assert payload["overall_status"] == "benchmark_ready_with_gap"
    assert (
        payload["components"]["manufacturing_evidence"]["status"]
        == "benchmark_ready_with_gap"
    )
    assert "manufacturing_evidence_payload_detail_quality_missing" in (
        payload["components"]["manufacturing_evidence"]["evidence_gaps"]
    )


def test_blocked_manufacturing_review_manifest_validation_downgrades_release_ready() -> None:
    inputs = _ready_inputs()
    inputs["manufacturing_review_manifest_validation"] = {
        "status": "blocked",
        "row_count": 80,
        "min_reviewed_samples": 30,
        "source_reviewed_sample_count": 80,
        "payload_reviewed_sample_count": 12,
        "payload_detail_reviewed_sample_count": 8,
        "approved_review_statuses": ["approved", "confirmed"],
        "approved_review_sample_count": 12,
        "unapproved_review_sample_count": 3,
        "require_reviewer_metadata": True,
        "reviewer_metadata_missing_sample_count": 1,
        "payload_expected_field_total": 64,
        "payload_detail_expected_field_total": 24,
        "blocking_reasons": [
            "payload_reviewed_sample_count_below_minimum",
            "payload_detail_reviewed_sample_count_below_minimum",
        ],
    }

    payload = build_forward_scorecard(**inputs, generated_at=1)

    manufacturing = payload["components"]["manufacturing_evidence"]
    assert payload["overall_status"] == "benchmark_ready_with_gap"
    assert manufacturing["status"] == "benchmark_ready_with_gap"
    assert manufacturing["review_manifest_validation"]["status"] == "blocked"
    assert manufacturing["review_manifest_validation"]["payload_reviewed_sample_count"] == 12
    assert manufacturing["review_manifest_validation"]["approved_review_sample_count"] == 12
    assert manufacturing["review_manifest_validation"]["unapproved_review_sample_count"] == 3
    assert manufacturing["review_manifest_validation"]["require_reviewer_metadata"] is True
    assert (
        manufacturing["review_manifest_validation"][
            "reviewer_metadata_missing_sample_count"
        ]
        == 1
    )
    assert "manufacturing_review_manifest_validation_blocked" in (
        manufacturing["evidence_gaps"]
    )
    assert any("review manifest" in item for item in payload["recommendations"])


def test_forward_scorecard_markdown_contains_component_table() -> None:
    payload = build_forward_scorecard(**_ready_inputs(), generated_at=1)
    rendered = render_forward_scorecard_markdown(payload, "Forward Scorecard")

    assert "# Forward Scorecard" in rendered
    assert "| `hybrid_dxf` | `release_ready` |" in rendered
    assert "rule_ver=1.0" in rendered
    assert "| `manufacturing_evidence` | `release_ready` |" in rendered
    assert "Release claims must cite this scorecard artifact." in rendered


def test_export_forward_scorecard_script_outputs_json_and_markdown(
    tmp_path: Path,
) -> None:
    inputs = _ready_inputs()
    model = _write_json(tmp_path / "model.json", inputs["model_readiness"])
    hybrid = _write_json(tmp_path / "hybrid.json", inputs["hybrid_summary"])
    graph2d = _write_json(tmp_path / "graph2d.json", inputs["graph2d_summary"])
    history = _write_json(tmp_path / "history.json", inputs["history_summary"])
    brep = _write_json(tmp_path / "brep.json", inputs["brep_summary"])
    qdrant = _write_json(tmp_path / "qdrant.json", inputs["qdrant_summary"])
    review = _write_json(tmp_path / "review.json", inputs["review_queue_summary"])
    knowledge = _write_json(tmp_path / "knowledge.json", inputs["knowledge_summary"])
    manufacturing = _write_json(
        tmp_path / "manufacturing.json",
        inputs["manufacturing_summary"],
    )
    review_validation = _write_json(
        tmp_path / "manufacturing_review_validation.json",
        {
            "status": "release_label_ready",
            "row_count": 80,
            "min_reviewed_samples": 30,
            "source_reviewed_sample_count": 80,
            "payload_reviewed_sample_count": 80,
            "payload_detail_reviewed_sample_count": 80,
            "payload_expected_field_total": 320,
            "payload_detail_expected_field_total": 160,
            "blocking_reasons": [],
        },
    )
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model-readiness-summary",
            str(model),
            "--hybrid-summary",
            str(hybrid),
            "--graph2d-summary",
            str(graph2d),
            "--history-summary",
            str(history),
            "--brep-summary",
            str(brep),
            "--qdrant-summary",
            str(qdrant),
            "--review-queue-summary",
            str(review),
            "--knowledge-summary",
            str(knowledge),
            "--manufacturing-evidence-summary",
            str(manufacturing),
            "--manufacturing-review-manifest-validation-summary",
            str(review_validation),
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
    assert payload["overall_status"] == "release_ready"
    assert payload["components"]["manufacturing_evidence"][
        "review_manifest_validation"
    ]["status"] == "release_label_ready"
    assert payload["artifacts"]["manufacturing_review_manifest_validation_summary"] == (
        str(review_validation)
    )
    assert (
        json.loads(output_json.read_text(encoding="utf-8"))["overall_status"]
        == "release_ready"
    )
    assert "`overall_status`: `release_ready`" in output_md.read_text(encoding="utf-8")
