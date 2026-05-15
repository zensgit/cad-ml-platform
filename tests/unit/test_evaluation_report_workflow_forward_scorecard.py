"""Regression checks for forward scorecard artifact wiring in evaluation-report."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "evaluation-report.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def _step_names(workflow: dict, job_name: str) -> list[str]:
    return [step.get("name", "") for step in workflow["jobs"][job_name]["steps"]]


def test_forward_scorecard_manufacturing_evidence_env_is_wired() -> None:
    workflow = _load_workflow()
    env = workflow["env"]

    assert "BENCHMARK_SCORECARD_MANUFACTURING_EVIDENCE_JSON" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_EVIDENCE_SUMMARY_JSON" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_SUMMARY_JSON" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_PROGRESS_MD" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_GAP_CSV" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_CONTEXT_CSV" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_CSV" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_ASSIGNMENT_MD" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_CSV" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_HANDOFF_MD" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV" in env
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_SUMMARY_JSON"
        in env
    )
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MD" in env
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_MIN_READY_ROWS"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_FAIL_ON_BLOCKED"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MIN_REVIEWED_SAMPLES"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA"
        in env
    )
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_FAIL_ON_BLOCKED" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_BASE_MANIFEST_CSV" in env
    assert "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV" in env
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV"
        in env
    )
    assert (
        "FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_FAIL_ON_BLOCKED"
        in env
    )


def test_manufacturing_evidence_summary_upload_uses_forward_scorecard_output() -> None:
    workflow = _load_workflow()
    step = _get_step(
        workflow,
        "evaluate",
        "Upload manufacturing evidence benchmark summary",
    )

    assert (
        step.get("if")
        == "steps.forward_scorecard.outputs.manufacturing_evidence_summary_available == 'true'"
    )
    assert step.get("uses") == (
        "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert "manufacturing-evidence-benchmark-summary" in step["with"]["name"]
    assert (
        "steps.forward_scorecard.outputs.manufacturing_evidence_summary_json"
        in step["with"]["path"]
    )


def test_manufacturing_evidence_summary_upload_follows_forward_scorecard() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "evaluate")

    forward_idx = names.index("Upload forward scorecard")
    manufacturing_idx = names.index("Upload manufacturing evidence benchmark summary")
    review_manifest_idx = names.index(
        "Upload manufacturing evidence review manifest validation"
    )
    reviewed_manifest_idx = names.index(
        "Upload manufacturing evidence reviewed benchmark manifest"
    )
    operator_idx = names.index("Upload benchmark operator adoption")

    assert (
        forward_idx
        < manufacturing_idx
        < review_manifest_idx
        < reviewed_manifest_idx
        < operator_idx
    )


def test_manufacturing_review_manifest_validation_upload_uses_forward_scorecard_output() -> None:
    workflow = _load_workflow()
    step = _get_step(
        workflow,
        "evaluate",
        "Upload manufacturing evidence review manifest validation",
    )

    assert (
        step.get("if")
        == "steps.forward_scorecard.outputs.manufacturing_review_manifest_available == 'true'"
    )
    assert step.get("uses") == (
        "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert "manufacturing-evidence-review-manifest-validation" in step["with"]["name"]
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_manifest_summary_json"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_manifest_progress_md"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_manifest_gap_csv"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_context_csv"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_batch_csv"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_batch_template_csv"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_assignment_md"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_reviewer_template_csv"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_handoff_md"
        in step["with"]["path"]
    )


def test_reviewed_benchmark_manifest_upload_uses_forward_scorecard_output() -> None:
    workflow = _load_workflow()
    step = _get_step(
        workflow,
        "evaluate",
        "Upload manufacturing evidence reviewed benchmark manifest",
    )

    assert (
        step.get("if")
        == "steps.forward_scorecard.outputs.manufacturing_review_manifest_merge_available == 'true'"
    )
    assert step.get("uses") == (
        "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert "manufacturing-evidence-reviewed-benchmark-manifest" in step["with"]["name"]
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_manifest_merged_csv"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_manifest_merge_summary_json"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_review_manifest_merge_audit_csv"
        in step["with"]["path"]
    )


def test_reviewer_template_apply_upload_uses_forward_scorecard_output() -> None:
    workflow = _load_workflow()
    step = _get_step(
        workflow,
        "evaluate",
        "Upload manufacturing reviewer template apply artifacts",
    )

    assert (
        step.get("if")
        == "steps.forward_scorecard.outputs.manufacturing_reviewer_template_apply_available == 'true'"
    )
    assert step.get("uses") == (
        "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert "manufacturing-reviewer-template-apply" in step["with"]["name"]
    assert (
        "steps.forward_scorecard.outputs.manufacturing_reviewer_template_applied_manifest_csv"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_reviewer_template_apply_summary_json"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_reviewer_template_apply_audit_csv"
        in step["with"]["path"]
    )


def test_reviewer_template_preflight_upload_uses_forward_scorecard_output() -> None:
    workflow = _load_workflow()
    step = _get_step(
        workflow,
        "evaluate",
        "Upload manufacturing reviewer template preflight",
    )

    assert (
        step.get("if")
        == "steps.forward_scorecard.outputs.manufacturing_reviewer_template_preflight_available == 'true'"
    )
    assert step.get("uses") == (
        "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert "manufacturing-reviewer-template-preflight" in step["with"]["name"]
    assert (
        "steps.forward_scorecard.outputs.manufacturing_reviewer_template_preflight_summary_json"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_reviewer_template_preflight_md"
        in step["with"]["path"]
    )
    assert (
        "steps.forward_scorecard.outputs.manufacturing_reviewer_template_preflight_gap_csv"
        in step["with"]["path"]
    )
