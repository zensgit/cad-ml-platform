from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.ci.check_forward_scorecard_release_gate import evaluate_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_forward_scorecard_release_gate.py"
WRAPPER_SCRIPT = REPO_ROOT / "scripts" / "ci" / "build_forward_scorecard_optional.sh"


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_review_manifest(path: Path, *, reviewed: bool = True) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "label_cn",
                "review_status",
                "reviewer",
                "reviewed_at",
                "reviewed_manufacturing_evidence_sources",
                "reviewed_manufacturing_evidence_payload_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "review_status": "approved" if reviewed else "needs_human_review",
                "reviewer": "manufacturing-reviewer" if reviewed else "",
                "reviewed_at": "2026-05-13" if reviewed else "",
                "reviewed_manufacturing_evidence_sources": (
                    "dfm;process;cost;decision" if reviewed else ""
                ),
                "reviewed_manufacturing_evidence_payload_json": (
                    json.dumps(
                        {
                            "dfm": {
                                "status": "manufacturable",
                                "details": {"mode": "rule"},
                            }
                        }
                    )
                    if reviewed
                    else ""
                ),
            }
        )
    return path


def _write_base_manifest(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "label_cn", "relative_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "shaft.dxf",
                "label_cn": "轴类",
                "relative_path": "release/shaft.dxf",
            }
        )
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


def _ready_artifacts(tmp_path: Path) -> dict[str, Path]:
    return {
        "model": _write_json(tmp_path / "model.json", _ready_model_registry()),
        "hybrid": _write_json(
            tmp_path / "hybrid.json",
            {
                "sample_size": 80,
                "coarse_accuracy": {"hybrid_label": {"accuracy": 0.91}},
                "exact_accuracy": {"hybrid_label": {"accuracy": 0.86}},
                "coarse_macro_f1_overall": 0.88,
                "confidence": {"hybrid_label": {"low_conf_rate": 0.05}},
            },
        ),
        "graph2d": _write_json(
            tmp_path / "graph2d.json",
            {"sample_size": 80, "blind_accuracy": 0.84, "low_conf_rate": 0.1},
        ),
        "history": _write_json(
            tmp_path / "history.json",
            {
                "total": 60,
                "coarse_accuracy_overall": 0.83,
                "coarse_macro_f1_overall": 0.81,
                "accuracy_overall": 0.78,
                "low_conf_rate": 0.08,
            },
        ),
        "brep": _write_json(
            tmp_path / "brep.json",
            {
                "sample_size": 30,
                "parse_success_count": 30,
                "valid_3d_count": 30,
                "graph_valid_count": 29,
            },
        ),
        "qdrant": _write_json(
            tmp_path / "qdrant.json",
            {
                "backend_health": {
                    "readiness": "ready",
                    "indexed_ratio": 1.0,
                    "unindexed_vectors_count": 0,
                    "scan_truncated": False,
                }
            },
        ),
        "review": _write_json(
            tmp_path / "review.json",
            {
                "total": 2,
                "by_feedback_priority": {"low": 2},
                "records_with_evidence_ratio": 1.0,
                "automation_ready_ratio": 1.0,
            },
        ),
        "knowledge": _write_json(
            tmp_path / "knowledge.json",
            {
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
        ),
        "manufacturing": _write_json(
            tmp_path / "manufacturing.json",
            {
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
        ),
    }


def _wrapper_env(**values: str) -> dict[str, str]:
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    return {
        **os.environ,
        "PATH": f"{python_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        **values,
    }


def test_evaluate_gate_blocks_release_label_on_blocked_scorecard() -> None:
    payload = evaluate_gate(
        scorecard={"overall_status": "blocked"},
        labels=["release:candidate"],
        release_label_prefixes=["release"],
        allowed_statuses=["release_ready", "benchmark_ready_with_gap"],
        require_release=False,
        scorecard_path="scorecard.json",
    )

    assert payload["gate_applicable"] is True
    assert payload["should_fail"] is True
    assert payload["release_labels"] == ["release:candidate"]


def test_evaluate_gate_ignores_non_release_labels() -> None:
    payload = evaluate_gate(
        scorecard={"overall_status": "blocked"},
        labels=["docs"],
        release_label_prefixes=["release"],
        allowed_statuses=["release_ready", "benchmark_ready_with_gap"],
        require_release=False,
        scorecard_path="scorecard.json",
    )

    assert payload["gate_applicable"] is False
    assert payload["should_fail"] is False


def test_gate_script_writes_output_json_and_exits_nonzero_for_shadow_release(
    tmp_path: Path,
) -> None:
    scorecard = _write_json(
        tmp_path / "scorecard.json", {"overall_status": "shadow_only"}
    )
    output_json = tmp_path / "gate.json"

    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--scorecard",
            str(scorecard),
            "--labels",
            "release:production",
            "--output-json",
            str(output_json),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["should_fail"] is True
    assert payload["overall_status"] == "shadow_only"


def test_gate_script_allows_benchmark_ready_with_gap_release(tmp_path: Path) -> None:
    scorecard = _write_json(
        tmp_path / "scorecard.json",
        {"overall_status": "benchmark_ready_with_gap"},
    )

    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--scorecard",
            str(scorecard),
            "--labels",
            "release:candidate",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["should_fail"] is False
    assert payload["gate_applicable"] is True


def test_build_forward_scorecard_optional_exports_ready_outputs(tmp_path: Path) -> None:
    artifacts = _ready_artifacts(tmp_path)
    review_manifest = _write_review_manifest(tmp_path / "manufacturing_review.csv")
    base_manifest = _write_base_manifest(tmp_path / "benchmark_manifest.csv")
    review_summary = tmp_path / "manufacturing_review_summary.json"
    review_progress = tmp_path / "manufacturing_review_progress.md"
    review_gap_csv = tmp_path / "manufacturing_review_gaps.csv"
    review_context = tmp_path / "manufacturing_review_context.csv"
    review_batch = tmp_path / "manufacturing_review_batch.csv"
    review_batch_template = tmp_path / "manufacturing_review_batch_template.csv"
    review_assignment = tmp_path / "manufacturing_review_assignment.md"
    reviewer_template = tmp_path / "manufacturing_reviewer_template.csv"
    review_handoff = tmp_path / "manufacturing_review_handoff.md"
    merged_manifest = tmp_path / "benchmark_manifest.reviewed.csv"
    merge_summary = tmp_path / "manufacturing_review_merge_summary.json"
    merge_audit = tmp_path / "manufacturing_review_merge_audit.csv"
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_HYBRID_SUMMARY_JSON=str(artifacts["hybrid"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_BREP_SUMMARY_JSON=str(artifacts["brep"]),
        FORWARD_SCORECARD_QDRANT_SUMMARY_JSON=str(artifacts["qdrant"]),
        FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON=str(artifacts["review"]),
        FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON=str(artifacts["knowledge"]),
        FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON=str(
            artifacts["manufacturing"]
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=str(review_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON=str(
            review_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_PROGRESS_MD=str(
            review_progress
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_GAP_CSV=str(review_gap_csv),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_CONTEXT_CSV=str(review_context),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_CSV=str(review_batch),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV=str(
            review_batch_template
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_ASSIGNMENT_MD=str(
            review_assignment
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_CSV=str(
            reviewer_template
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_HANDOFF_MD=str(review_handoff),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES="1",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA="true",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_BASE_MANIFEST_CSV=str(base_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV=str(
            merged_manifest
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON=str(
            merge_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV=str(
            merge_audit
        ),
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert (
        json.loads(output_json.read_text(encoding="utf-8"))["overall_status"]
        == "release_ready"
    )
    scorecard = json.loads(output_json.read_text(encoding="utf-8"))
    assert scorecard["components"]["manufacturing_evidence"][
        "review_manifest_validation"
    ]["status"] == "release_label_ready"
    assert scorecard["artifacts"][
        "manufacturing_review_manifest_validation_summary"
    ] == str(review_summary)
    outputs = github_output.read_text(encoding="utf-8")
    assert "enabled=true" in outputs
    assert "overall_status=release_ready" in outputs
    assert "gate_applicable=false" in outputs
    assert "manufacturing_evidence_status=release_ready" in outputs
    assert "manufacturing_evidence_summary_available=true" in outputs
    assert f"manufacturing_evidence_summary_json={artifacts['manufacturing']}" in outputs
    assert "manufacturing_review_manifest_available=true" in outputs
    assert f"manufacturing_review_manifest_csv={review_manifest}" in outputs
    assert f"manufacturing_review_manifest_summary_json={review_summary}" in outputs
    assert f"manufacturing_review_manifest_progress_md={review_progress}" in outputs
    assert f"manufacturing_review_manifest_gap_csv={review_gap_csv}" in outputs
    assert f"manufacturing_review_context_csv={review_context}" in outputs
    assert f"manufacturing_review_batch_csv={review_batch}" in outputs
    assert (
        f"manufacturing_review_batch_template_csv={review_batch_template}"
        in outputs
    )
    assert f"manufacturing_review_assignment_md={review_assignment}" in outputs
    assert f"manufacturing_reviewer_template_csv={reviewer_template}" in outputs
    assert f"manufacturing_review_handoff_md={review_handoff}" in outputs
    handoff = review_handoff.read_text(encoding="utf-8")
    assert "Manufacturing Review Handoff" in handoff
    assert (
        f"--validate-reviewer-template {review_batch_template}"
        in handoff
    )
    assert (
        f"--apply-reviewer-template {review_batch_template}"
        in handoff
    )
    assert str(reviewer_template) in handoff
    assert "manufacturing_reviewer_template_preflight_available=false" in outputs
    assert "manufacturing_reviewer_template_preflight_gap_csv=" in outputs
    assert "manufacturing_reviewer_template_preflight_status=missing" in outputs
    assert "manufacturing_reviewer_template_apply_available=false" in outputs
    assert "manufacturing_reviewer_template_apply_audit_csv=" in outputs
    assert "manufacturing_reviewer_template_apply_status=missing" in outputs
    assert "manufacturing_review_manifest_status=release_label_ready" in outputs
    assert "manufacturing_review_manifest_merge_available=true" in outputs
    assert f"manufacturing_review_manifest_base_csv={base_manifest}" in outputs
    assert f"manufacturing_review_manifest_merged_csv={merged_manifest}" in outputs
    assert (
        f"manufacturing_review_manifest_merge_summary_json={merge_summary}"
        in outputs
    )
    assert (
        f"manufacturing_review_manifest_merge_audit_csv={merge_audit}"
        in outputs
    )
    with merge_audit.open("r", encoding="utf-8", newline="") as handle:
        merge_audit_rows = list(csv.DictReader(handle))
    assert merge_audit_rows[0]["merge_status"] == "merged"
    assert "manufacturing_review_manifest_merge_status=merged" in outputs
    assert json.loads(review_summary.read_text(encoding="utf-8"))["status"] == (
        "release_label_ready"
    )
    assert json.loads(review_summary.read_text(encoding="utf-8"))[
        "require_reviewer_metadata"
    ] is True
    assert "Manufacturing Evidence Review Progress" in review_progress.read_text(
        encoding="utf-8"
    )
    with review_gap_csv.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    with review_context.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    with review_batch.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    with review_batch_template.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    assert "Manufacturing Review Assignment Plan" in review_assignment.read_text(
        encoding="utf-8"
    )
    with reviewer_template.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    assert json.loads(merge_summary.read_text(encoding="utf-8"))["status"] == "merged"
    with merged_manifest.open("r", encoding="utf-8", newline="") as handle:
        merged_rows = list(csv.DictReader(handle))
    assert merged_rows[0]["reviewed_manufacturing_evidence_sources"] == (
        "dfm;process;cost;decision"
    )
    assert json.loads(gate_json.read_text(encoding="utf-8"))["should_fail"] is False


def test_build_forward_scorecard_optional_applies_reviewer_template_before_validation(
    tmp_path: Path,
) -> None:
    artifacts = _ready_artifacts(tmp_path)
    review_manifest = _write_review_manifest(
        tmp_path / "manufacturing_review.csv",
        reviewed=False,
    )
    reviewer_template_apply = _write_review_manifest(
        tmp_path / "manufacturing_reviewer_template.filled.csv",
        reviewed=True,
    )
    preflight_summary = tmp_path / "manufacturing_reviewer_template_preflight.json"
    preflight_md = tmp_path / "manufacturing_reviewer_template_preflight.md"
    preflight_gap_csv = tmp_path / "manufacturing_reviewer_template_preflight_gaps.csv"
    review_handoff = tmp_path / "manufacturing_review_handoff.md"
    applied_manifest = tmp_path / "manufacturing_review.applied.csv"
    apply_summary = tmp_path / "manufacturing_reviewer_template_apply.json"
    apply_audit = tmp_path / "manufacturing_reviewer_template_apply_audit.csv"
    review_summary = tmp_path / "manufacturing_review_summary.json"
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_HYBRID_SUMMARY_JSON=str(artifacts["hybrid"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_BREP_SUMMARY_JSON=str(artifacts["brep"]),
        FORWARD_SCORECARD_QDRANT_SUMMARY_JSON=str(artifacts["qdrant"]),
        FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON=str(artifacts["review"]),
        FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON=str(artifacts["knowledge"]),
        FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON=str(
            artifacts["manufacturing"]
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=str(review_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON=str(
            review_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV=str(
            reviewer_template_apply
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON=str(
            preflight_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD=str(
            preflight_md
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV=str(
            preflight_gap_csv
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS="1",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_HANDOFF_MD=str(review_handoff),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV=str(
            applied_manifest
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON=str(
            apply_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV=str(
            apply_audit
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES="1",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA="true",
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    preflight = json.loads(preflight_summary.read_text(encoding="utf-8"))
    assert preflight["status"] == "ready"
    assert preflight["base_manifest_match_required"] is True
    assert preflight["unmatched_template_row_count"] == 0
    assert "Manufacturing Reviewer Template Preflight" in preflight_md.read_text(
        encoding="utf-8"
    )
    assert json.loads(apply_summary.read_text(encoding="utf-8"))["status"] == "applied"
    assert json.loads(review_summary.read_text(encoding="utf-8"))["status"] == (
        "release_label_ready"
    )
    with applied_manifest.open("r", encoding="utf-8", newline="") as handle:
        applied_rows = list(csv.DictReader(handle))
    assert applied_rows[0]["review_status"] == "approved"
    outputs = github_output.read_text(encoding="utf-8")
    assert f"manufacturing_review_manifest_csv={applied_manifest}" in outputs
    assert "manufacturing_reviewer_template_preflight_available=true" in outputs
    assert (
        f"manufacturing_reviewer_template_preflight_summary_json={preflight_summary}"
        in outputs
    )
    assert f"manufacturing_reviewer_template_preflight_md={preflight_md}" in outputs
    assert (
        f"manufacturing_reviewer_template_preflight_gap_csv={preflight_gap_csv}"
        in outputs
    )
    with preflight_gap_csv.open("r", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    assert f"manufacturing_review_handoff_md={review_handoff}" in outputs
    handoff = review_handoff.read_text(encoding="utf-8")
    assert str(preflight_md) in handoff
    assert str(preflight_gap_csv) in handoff
    assert "--min-reviewed-samples 1" in handoff
    assert "manufacturing_reviewer_template_preflight_status=ready" in outputs
    assert "manufacturing_reviewer_template_apply_available=true" in outputs
    assert (
        f"manufacturing_reviewer_template_apply_csv={reviewer_template_apply}"
        in outputs
    )
    assert (
        f"manufacturing_reviewer_template_applied_manifest_csv={applied_manifest}"
        in outputs
    )
    assert (
        f"manufacturing_reviewer_template_apply_summary_json={apply_summary}"
        in outputs
    )
    assert (
        f"manufacturing_reviewer_template_apply_audit_csv={apply_audit}"
        in outputs
    )
    with apply_audit.open("r", encoding="utf-8", newline="") as handle:
        apply_audit_rows = list(csv.DictReader(handle))
    assert apply_audit_rows[0]["apply_status"] == "applied"
    assert "manufacturing_reviewer_template_apply_status=applied" in outputs


def test_build_forward_scorecard_optional_can_preflight_partial_reviewer_template(
    tmp_path: Path,
) -> None:
    artifacts = _ready_artifacts(tmp_path)
    review_manifest = _write_review_manifest(
        tmp_path / "manufacturing_review.csv",
        reviewed=False,
    )
    reviewer_template_apply = _write_review_manifest(
        tmp_path / "manufacturing_reviewer_template.batch.csv",
        reviewed=True,
    )
    preflight_summary = tmp_path / "manufacturing_reviewer_template_preflight.json"
    preflight_md = tmp_path / "manufacturing_reviewer_template_preflight.md"
    preflight_gap_csv = tmp_path / "manufacturing_reviewer_template_preflight_gaps.csv"
    review_handoff = tmp_path / "manufacturing_review_handoff.md"
    applied_manifest = tmp_path / "manufacturing_review.applied.csv"
    apply_summary = tmp_path / "manufacturing_reviewer_template_apply.json"
    apply_audit = tmp_path / "manufacturing_reviewer_template_apply_audit.csv"
    review_summary = tmp_path / "manufacturing_review_summary.json"
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_HYBRID_SUMMARY_JSON=str(artifacts["hybrid"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_BREP_SUMMARY_JSON=str(artifacts["brep"]),
        FORWARD_SCORECARD_QDRANT_SUMMARY_JSON=str(artifacts["qdrant"]),
        FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON=str(artifacts["review"]),
        FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON=str(artifacts["knowledge"]),
        FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON=str(
            artifacts["manufacturing"]
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=str(review_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON=str(
            review_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV=str(
            reviewer_template_apply
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON=str(
            preflight_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD=str(
            preflight_md
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV=str(
            preflight_gap_csv
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_HANDOFF_MD=str(review_handoff),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS="1",
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV=str(
            applied_manifest
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON=str(
            apply_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV=str(
            apply_audit
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES="30",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA="true",
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    preflight = json.loads(preflight_summary.read_text(encoding="utf-8"))
    assert preflight["status"] == "ready"
    assert preflight["min_ready_rows"] == 1
    assert preflight["base_manifest_match_required"] is True
    assert preflight["unmatched_template_row_count"] == 0
    assert json.loads(apply_summary.read_text(encoding="utf-8"))["status"] == "applied"
    validation = json.loads(review_summary.read_text(encoding="utf-8"))
    assert validation["status"] == "blocked"
    assert validation["min_reviewed_samples"] == 30
    assert validation["source_reviewed_sample_count"] == 1
    handoff = review_handoff.read_text(encoding="utf-8")
    assert "--min-reviewed-samples 1" in handoff
    assert "--min-reviewed-samples 30" in handoff
    outputs = github_output.read_text(encoding="utf-8")
    assert "manufacturing_reviewer_template_preflight_status=ready" in outputs
    assert "manufacturing_reviewer_template_apply_status=applied" in outputs


def test_build_forward_scorecard_optional_can_fail_on_blocked_reviewer_template_preflight(
    tmp_path: Path,
) -> None:
    artifacts = _ready_artifacts(tmp_path)
    review_manifest = _write_review_manifest(
        tmp_path / "manufacturing_review.csv",
        reviewed=False,
    )
    reviewer_template_apply = _write_review_manifest(
        tmp_path / "manufacturing_reviewer_template.filled.csv",
        reviewed=False,
    )
    preflight_summary = tmp_path / "manufacturing_reviewer_template_preflight.json"
    preflight_md = tmp_path / "manufacturing_reviewer_template_preflight.md"
    preflight_gap_csv = tmp_path / "manufacturing_reviewer_template_preflight_gaps.csv"
    review_summary = tmp_path / "manufacturing_review_summary.json"
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_HYBRID_SUMMARY_JSON=str(artifacts["hybrid"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_BREP_SUMMARY_JSON=str(artifacts["brep"]),
        FORWARD_SCORECARD_QDRANT_SUMMARY_JSON=str(artifacts["qdrant"]),
        FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON=str(artifacts["review"]),
        FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON=str(artifacts["knowledge"]),
        FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON=str(
            artifacts["manufacturing"]
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=str(review_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON=str(
            review_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV=str(
            reviewer_template_apply
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON=str(
            preflight_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD=str(
            preflight_md
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV=str(
            preflight_gap_csv
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_FAIL_ON_BLOCKED="true",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES="1",
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    result = subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "reviewer template preflight is not ready: blocked" in result.stderr
    assert json.loads(preflight_summary.read_text(encoding="utf-8"))["status"] == (
        "blocked"
    )
    assert "Manufacturing Reviewer Template Preflight" in preflight_md.read_text(
        encoding="utf-8"
    )
    with preflight_gap_csv.open("r", encoding="utf-8", newline="") as handle:
        preflight_gap_rows = list(csv.DictReader(handle))
    assert len(preflight_gap_rows) == 1
    assert (
        "fill reviewed_manufacturing_evidence_sources"
        in preflight_gap_rows[0]["preflight_reasons"]
    )
    assert (
        "fill reviewed_manufacturing_evidence_payload_json"
        in preflight_gap_rows[0]["preflight_reasons"]
    )
    outputs = github_output.read_text(encoding="utf-8")
    assert "manufacturing_reviewer_template_preflight_available=true" in outputs
    assert (
        f"manufacturing_reviewer_template_preflight_gap_csv={preflight_gap_csv}"
        in outputs
    )
    assert "manufacturing_reviewer_template_preflight_status=blocked" in outputs
    assert "manufacturing_reviewer_template_apply_available=false" in outputs
    assert "manufacturing_reviewer_template_apply_status=blocked_preflight" in outputs


def test_build_forward_scorecard_optional_blocks_unmatched_reviewer_template_rows(
    tmp_path: Path,
) -> None:
    artifacts = _ready_artifacts(tmp_path)
    review_manifest = _write_review_manifest(
        tmp_path / "manufacturing_review.csv",
        reviewed=False,
    )
    reviewer_template_apply = tmp_path / "manufacturing_reviewer_template.filled.csv"
    with reviewer_template_apply.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "label_cn",
                "relative_path",
                "review_status",
                "reviewed_manufacturing_evidence_sources",
                "reviewed_manufacturing_evidence_payload_json",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": "missing.dxf",
                "label_cn": "轴类",
                "relative_path": "release/missing.dxf",
                "review_status": "approved",
                "reviewed_manufacturing_evidence_sources": "dfm",
                "reviewed_manufacturing_evidence_payload_json": json.dumps(
                    {
                        "dfm": {
                            "status": "manufacturable",
                            "details": {"mode": "rule"},
                        }
                    }
                ),
            }
        )
    preflight_summary = tmp_path / "manufacturing_reviewer_template_preflight.json"
    preflight_md = tmp_path / "manufacturing_reviewer_template_preflight.md"
    preflight_gap_csv = tmp_path / "manufacturing_reviewer_template_preflight_gaps.csv"
    review_summary = tmp_path / "manufacturing_review_summary.json"
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_HYBRID_SUMMARY_JSON=str(artifacts["hybrid"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_BREP_SUMMARY_JSON=str(artifacts["brep"]),
        FORWARD_SCORECARD_QDRANT_SUMMARY_JSON=str(artifacts["qdrant"]),
        FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON=str(artifacts["review"]),
        FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON=str(artifacts["knowledge"]),
        FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON=str(
            artifacts["manufacturing"]
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=str(review_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON=str(
            review_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV=str(
            reviewer_template_apply
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON=str(
            preflight_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD=str(
            preflight_md
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV=str(
            preflight_gap_csv
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_FAIL_ON_BLOCKED="true",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES="1",
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    result = subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "reviewer template preflight is not ready: blocked" in result.stderr
    preflight = json.loads(preflight_summary.read_text(encoding="utf-8"))
    assert preflight["status"] == "blocked"
    assert preflight["base_manifest_match_required"] is True
    assert preflight["unmatched_template_row_count"] == 1
    assert "unmatched_template_rows" in preflight["blocking_reasons"]
    with preflight_gap_csv.open("r", encoding="utf-8", newline="") as handle:
        preflight_gap_rows = list(csv.DictReader(handle))
    assert len(preflight_gap_rows) == 1
    assert preflight_gap_rows[0]["matched_manifest_row"] == "false"
    assert (
        "match row identity to review manifest"
        in preflight_gap_rows[0]["preflight_reasons"]
    )
    outputs = github_output.read_text(encoding="utf-8")
    assert "manufacturing_reviewer_template_apply_status=blocked_preflight" in outputs


def test_build_forward_scorecard_optional_consumes_brep_golden_step_output(
    tmp_path: Path,
) -> None:
    artifacts = _ready_artifacts(tmp_path)
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    steps_json = json.dumps(
        {
            "brep_golden_manifest": {
                "outputs": {"eval_summary_json": str(artifacts["brep"])}
            }
        }
    )
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        GITHUB_STEPS_JSON=steps_json,
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_HYBRID_SUMMARY_JSON=str(artifacts["hybrid"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_QDRANT_SUMMARY_JSON=str(artifacts["qdrant"]),
        FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON=str(artifacts["review"]),
        FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON=str(artifacts["knowledge"]),
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["components"]["brep"]["status"] == "release_ready"
    assert payload["artifacts"]["brep_summary"] == str(artifacts["brep"])


def test_build_forward_scorecard_optional_gate_blocks_release_label(
    tmp_path: Path,
) -> None:
    artifacts = _ready_artifacts(tmp_path)
    output_json = tmp_path / "blocked.json"
    output_md = tmp_path / "blocked.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_RELEASE_GATE_ENABLE="true",
        FORWARD_SCORECARD_RELEASE_LABELS="release:candidate",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    result = subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert (
        json.loads(output_json.read_text(encoding="utf-8"))["overall_status"]
        == "blocked"
    )
    gate = json.loads(gate_json.read_text(encoding="utf-8"))
    assert gate["should_fail"] is True
    assert "should_fail=true" in github_output.read_text(encoding="utf-8")


def test_build_forward_scorecard_optional_fails_on_blocked_review_manifest_when_required(
    tmp_path: Path,
) -> None:
    artifacts = _ready_artifacts(tmp_path)
    review_manifest = _write_review_manifest(
        tmp_path / "manufacturing_review.csv",
        reviewed=False,
    )
    review_summary = tmp_path / "manufacturing_review_summary.json"
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_HYBRID_SUMMARY_JSON=str(artifacts["hybrid"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_BREP_SUMMARY_JSON=str(artifacts["brep"]),
        FORWARD_SCORECARD_QDRANT_SUMMARY_JSON=str(artifacts["qdrant"]),
        FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON=str(artifacts["review"]),
        FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON=str(artifacts["knowledge"]),
        FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON=str(
            artifacts["manufacturing"]
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=str(review_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON=str(
            review_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES="1",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_FAIL_ON_BLOCKED="true",
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    result = subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "not release-label-ready" in result.stderr
    scorecard = json.loads(output_json.read_text(encoding="utf-8"))
    assert scorecard["overall_status"] == "benchmark_ready_with_gap"
    assert scorecard["components"]["manufacturing_evidence"]["status"] == (
        "benchmark_ready_with_gap"
    )
    assert "manufacturing_review_manifest_validation_blocked" in (
        scorecard["components"]["manufacturing_evidence"]["evidence_gaps"]
    )
    assert json.loads(review_summary.read_text(encoding="utf-8"))["status"] == "blocked"
    outputs = github_output.read_text(encoding="utf-8")
    assert "manufacturing_review_manifest_available=true" in outputs
    assert "manufacturing_review_manifest_status=blocked" in outputs


def test_build_forward_scorecard_optional_fails_on_blocked_review_merge_when_required(
    tmp_path: Path,
) -> None:
    artifacts = _ready_artifacts(tmp_path)
    review_manifest = _write_review_manifest(
        tmp_path / "manufacturing_review.csv",
        reviewed=False,
    )
    base_manifest = _write_base_manifest(tmp_path / "benchmark_manifest.csv")
    review_summary = tmp_path / "manufacturing_review_summary.json"
    merged_manifest = tmp_path / "benchmark_manifest.reviewed.csv"
    merge_summary = tmp_path / "manufacturing_review_merge_summary.json"
    merge_audit = tmp_path / "manufacturing_review_merge_audit.csv"
    output_json = tmp_path / "latest.json"
    output_md = tmp_path / "latest.md"
    gate_json = tmp_path / "gate.json"
    github_output = tmp_path / "github-output.txt"
    env = _wrapper_env(
        GITHUB_OUTPUT=str(github_output),
        FORWARD_SCORECARD_ENABLE="true",
        FORWARD_SCORECARD_MODEL_READINESS_JSON=str(artifacts["model"]),
        FORWARD_SCORECARD_HYBRID_SUMMARY_JSON=str(artifacts["hybrid"]),
        FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON=str(artifacts["graph2d"]),
        FORWARD_SCORECARD_HISTORY_SUMMARY_JSON=str(artifacts["history"]),
        FORWARD_SCORECARD_BREP_SUMMARY_JSON=str(artifacts["brep"]),
        FORWARD_SCORECARD_QDRANT_SUMMARY_JSON=str(artifacts["qdrant"]),
        FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON=str(artifacts["review"]),
        FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON=str(artifacts["knowledge"]),
        FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON=str(
            artifacts["manufacturing"]
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=str(review_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON=str(
            review_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES="1",
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_BASE_MANIFEST_CSV=str(base_manifest),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV=str(
            merged_manifest
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON=str(
            merge_summary
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV=str(
            merge_audit
        ),
        FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_FAIL_ON_BLOCKED="true",
        FORWARD_SCORECARD_OUTPUT_JSON=str(output_json),
        FORWARD_SCORECARD_OUTPUT_MD=str(output_md),
        FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=str(gate_json),
    )

    result = subprocess.run(
        ["bash", str(WRAPPER_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Manufacturing review manifest merge is not ready: blocked" in result.stderr
    assert json.loads(merge_summary.read_text(encoding="utf-8"))["status"] == "blocked"
    with merge_audit.open("r", encoding="utf-8", newline="") as handle:
        merge_audit_rows = list(csv.DictReader(handle))
    assert merge_audit_rows[0]["merge_status"] == "skipped_no_review_content"
    outputs = github_output.read_text(encoding="utf-8")
    assert "manufacturing_review_manifest_merge_available=false" in outputs
    assert "manufacturing_review_manifest_merge_audit_csv=" in outputs
    assert "manufacturing_review_manifest_merge_status=blocked" in outputs
