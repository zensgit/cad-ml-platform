"""Regression checks for Pages-ready root and deploy-pages wiring."""

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
    return [s.get("name", "") for s in workflow["jobs"][job_name]["steps"]]


# --- evaluate job: assemble + upload ---


def test_assemble_pages_root_step_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Assemble Pages-ready root")

    assert "assemble_eval_reporting_pages_root.py" in step["run"]
    assert "--pages-root" in step["run"]


def test_pages_ready_artifact_upload_exists() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "evaluate", "Upload Pages-ready artifact")

    assert "eval-reporting-pages-" in step["with"]["name"]
    assert "eval_pages" in step["with"]["path"]


# --- deploy-pages job ---


def test_deploy_pages_downloads_pages_artifact() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Download Pages-ready artifact")

    assert "eval-reporting-pages-" in step["with"]["name"]
    assert step["with"]["path"] == "./public"


def test_deploy_pages_no_longer_downloads_old_report_artifact() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    assert "Download report artifact" not in names


def test_deploy_pages_publishes_from_public_dir() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Upload to Pages")

    assert step["with"]["path"] == "./public"


def test_pages_root_contains_landing_plus_reports() -> None:
    """The Pages root must serve landing page as index + both report subdirs."""
    workflow = _load_workflow()
    assemble_step = _get_step(workflow, "evaluate", "Assemble Pages-ready root")
    run_text = assemble_step["run"]

    assert "--eval-history-dir" in run_text
    assert "--pages-root" in run_text


# --- Batch 8B: public discovery surface ---


def test_deploy_pages_has_public_index_generation_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Generate public discovery index")

    assert step.get("if") == "always()"
    assert "generate_eval_reporting_public_index.py" in step["run"]
    assert "--page-url" in step["run"]


def test_deploy_pages_has_public_index_upload_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Upload public discovery index")

    assert step.get("if") == "always()"
    upload_path = str(step.get("with", {}).get("path", ""))
    assert "eval_reporting_public_index" in upload_path


def test_deploy_pages_has_consolidated_summary_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Consolidated eval reporting deploy-pages summary")

    assert step.get("if") == "always()"
    run_text = step["run"]
    assert "GITHUB_STEP_SUMMARY" in run_text
    assert "eval_reporting_public_index.md" in run_text
    assert "eval_reporting_dashboard_payload.md" in run_text
    assert "eval_reporting_webhook_delivery_request.md" in run_text
    assert "eval_reporting_webhook_delivery_result.md" in run_text
    assert "eval_reporting_release_draft_publish_result.md" in run_text


def test_public_index_step_consumes_deployment_page_url() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Generate public discovery index")

    assert "deployment.outputs.page_url" in step["run"]


# --- Batch 10A: dashboard payload ---


def test_deploy_pages_has_dashboard_payload_generation_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Generate eval reporting dashboard payload")

    assert step.get("if") == "always()"
    assert "generate_eval_reporting_dashboard_payload.py" in step["run"]


def test_deploy_pages_has_dashboard_payload_upload_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Upload eval reporting dashboard payload")

    assert step.get("if") == "always()"
    upload_path = str(step.get("with", {}).get("path", ""))
    assert "eval_reporting_dashboard_payload" in upload_path


def test_dashboard_payload_steps_after_public_index() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    public_gen_idx = names.index("Generate public discovery index")
    gen_idx = names.index("Generate eval reporting dashboard payload")

    assert gen_idx > public_gen_idx


def test_deploy_pages_checkout_includes_dashboard_payload_script() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")

    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "generate_eval_reporting_dashboard_payload.py" in sparse


# --- Batch 14A: webhook delivery request (now reads dashboard_payload directly) ---


def test_deploy_pages_checkout_includes_delivery_request_script() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")

    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "generate_eval_reporting_webhook_delivery_request.py" in sparse


def test_deploy_pages_has_delivery_request_generation_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Generate eval reporting webhook delivery request")

    assert step.get("if") == "always()"
    assert "generate_eval_reporting_webhook_delivery_request.py" in step["run"]
    assert "--dashboard-payload-json" in step["run"]


def test_deploy_pages_has_delivery_request_upload_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Upload eval reporting webhook delivery request")

    assert step.get("if") == "always()"
    upload_path = str(step.get("with", {}).get("path", ""))
    assert "eval_reporting_webhook_delivery_request" in upload_path


def test_delivery_request_steps_after_dashboard_payload() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    dp_gen_idx = names.index("Generate eval reporting dashboard payload")
    dr_gen_idx = names.index("Generate eval reporting webhook delivery request")

    assert dr_gen_idx > dp_gen_idx


# --- Batch 14B: webhook delivery result ---


def test_deploy_pages_checkout_includes_webhook_delivery_js() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")

    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "post_eval_reporting_webhook_delivery.js" in sparse


def test_deploy_pages_has_delivery_result_generation_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Generate eval reporting webhook delivery result")

    assert step.get("if") == "always()"
    assert step.get("continue-on-error") in (True, "true")
    script = str(step.get("with", {}).get("script", ""))
    assert "post_eval_reporting_webhook_delivery.js" in script
    assert "postWebhookDelivery" in script
    assert "deliveryEnabled: false" in script


def test_deploy_pages_has_delivery_result_upload_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Upload eval reporting webhook delivery result")

    assert step.get("if") == "always()"
    upload_path = str(step.get("with", {}).get("path", ""))
    assert "eval_reporting_webhook_delivery_result" in upload_path


def test_delivery_result_steps_after_delivery_request() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    dr_gen_idx = names.index("Generate eval reporting webhook delivery request")
    res_gen_idx = names.index("Generate eval reporting webhook delivery result")

    assert res_gen_idx > dr_gen_idx


# --- Batch 17B: Phase 1 baseline hardening (removed surfaces must not reappear) ---


def test_removed_signature_policy_not_in_workflow() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")
    for name in names:
        assert "signature_policy" not in name.lower(), f"Removed surface reappeared: {name}"

    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")
    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "signature_policy" not in sparse


def test_removed_retry_plan_not_in_workflow() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")
    for name in names:
        assert "retry_plan" not in name.lower(), f"Removed surface reappeared: {name}"

    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")
    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "retry_plan" not in sparse


def test_removed_dry_run_not_in_workflow() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")
    for name in names:
        assert "dry_run" not in name.lower() and "dry run" not in name.lower(), f"Removed surface reappeared: {name}"

    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")
    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "dry_run" not in sparse


def test_kept_delivery_result_still_present() -> None:
    workflow = _load_workflow()
    _get_step(workflow, "deploy-pages", "Generate eval reporting webhook delivery result")
    _get_step(workflow, "deploy-pages", "Upload eval reporting webhook delivery result")


def test_kept_publish_result_still_present() -> None:
    workflow = _load_workflow()
    _get_step(workflow, "deploy-pages", "Generate eval reporting release draft publish result")
    _get_step(workflow, "deploy-pages", "Upload eval reporting release draft publish result")


# --- Batch 18B: Phase 2 webhook merge baseline hardening ---


def test_merged_webhook_export_not_in_workflow() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")
    for name in names:
        assert "webhook_export" not in name.lower() and "webhook export" not in name.lower(), (
            f"Merged surface reappeared: {name}"
        )

    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")
    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "webhook_export" not in sparse


def test_kept_delivery_request_still_present_after_merge() -> None:
    workflow = _load_workflow()
    _get_step(workflow, "deploy-pages", "Generate eval reporting webhook delivery request")
    _get_step(workflow, "deploy-pages", "Upload eval reporting webhook delivery request")


def test_kept_delivery_result_still_present_after_merge() -> None:
    workflow = _load_workflow()
    _get_step(workflow, "deploy-pages", "Generate eval reporting webhook delivery result")
    _get_step(workflow, "deploy-pages", "Upload eval reporting webhook delivery result")


def test_delivery_request_reads_dashboard_payload_not_webhook_export() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Generate eval reporting webhook delivery request")

    assert "--dashboard-payload-json" in step["run"]
    assert "--webhook-export-json" not in step["run"]


# --- Batch 19B: Phase 3 release merge baseline hardening ---


def test_merged_release_note_snippet_not_in_workflow() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")
    for name in names:
        assert "release_note_snippet" not in name.lower() and "release note snippet" not in name.lower(), (
            f"Merged surface reappeared: {name}"
        )

    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")
    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "release_note_snippet" not in sparse


def test_merged_release_draft_prefill_not_in_workflow() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")
    for name in names:
        assert "release_draft_prefill" not in name.lower() and "release draft prefill" not in name.lower(), (
            f"Merged surface reappeared: {name}"
        )

    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")
    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "release_draft_prefill" not in sparse


def test_kept_publish_result_still_present_after_merge() -> None:
    workflow = _load_workflow()
    _get_step(workflow, "deploy-pages", "Generate eval reporting release draft publish result")
    _get_step(workflow, "deploy-pages", "Upload eval reporting release draft publish result")


# --- Batch 20B: Phase 4 release publish merge baseline hardening ---


def test_merged_publish_payload_not_in_workflow() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")
    for name in names:
        assert "publish_payload" not in name.lower() and "publish payload" not in name.lower(), (
            f"Merged surface reappeared: {name}"
        )

    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")
    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "publish_payload" not in sparse


def test_kept_publish_result_still_present_after_publish_merge() -> None:
    workflow = _load_workflow()
    _get_step(workflow, "deploy-pages", "Generate eval reporting release draft publish result")
    _get_step(workflow, "deploy-pages", "Upload eval reporting release draft publish result")


# --- Batch 21B: Phase 5 final release baseline hardening ---


def test_merged_draft_payload_not_in_workflow() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")
    for name in names:
        assert "draft_payload" not in name.lower() and "draft payload" not in name.lower(), (
            f"Merged surface reappeared: {name}"
        )

    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")
    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "draft_payload" not in sparse
    assert "generate_eval_reporting_release_draft_payload" not in sparse


def test_kept_publish_result_still_present_after_final_merge() -> None:
    workflow = _load_workflow()
    _get_step(workflow, "deploy-pages", "Generate eval reporting release draft publish result")
    _get_step(workflow, "deploy-pages", "Upload eval reporting release draft publish result")


def test_publish_result_reads_dashboard_not_draft_payload() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Generate eval reporting release draft publish result")

    script = str(step.get("with", {}).get("script", ""))
    assert "dashboardPayloadPath" in script
    assert "draftPayloadPath" not in script
    assert "eval_reporting_dashboard_payload.json" in script
    assert "release_draft_payload.json" not in script


# --- Batch 13B: release draft publish result (now reads dashboard_payload directly) ---


def test_deploy_pages_checkout_includes_publish_js() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Checkout for public index generation")

    sparse = str(step.get("with", {}).get("sparse-checkout", ""))
    assert "post_eval_reporting_release_draft_publish.js" in sparse


def test_deploy_pages_has_publish_result_generation_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Generate eval reporting release draft publish result")

    assert step.get("if") == "always()"
    assert step.get("continue-on-error") in (True, "true")
    script = str(step.get("with", {}).get("script", ""))
    assert "post_eval_reporting_release_draft_publish.js" in script
    assert "postReleaseDraftPublish" in script
    assert "publishEnabled: false" in script


def test_deploy_pages_has_publish_result_upload_step() -> None:
    workflow = _load_workflow()
    step = _get_step(workflow, "deploy-pages", "Upload eval reporting release draft publish result")

    assert step.get("if") == "always()"
    upload_path = str(step.get("with", {}).get("path", ""))
    assert "eval_reporting_release_draft_publish_result" in upload_path


def test_publish_result_steps_after_delivery_result() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    res_gen_idx = names.index("Generate eval reporting webhook delivery result")
    pr_gen_idx = names.index("Generate eval reporting release draft publish result")

    assert pr_gen_idx > res_gen_idx


# --- Batch 22A fix: ordering guards ---


def test_consolidated_summary_after_last_generate() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    generate_names = [
        "Generate public discovery index",
        "Generate eval reporting dashboard payload",
        "Generate eval reporting webhook delivery request",
        "Generate eval reporting webhook delivery result",
        "Generate eval reporting release draft publish result",
    ]
    summary_idx = names.index("Consolidated eval reporting deploy-pages summary")
    for gn in generate_names:
        assert names.index(gn) < summary_idx, f"{gn} must precede consolidated summary"


def test_consolidated_summary_before_first_upload() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    upload_names = [
        "Upload public discovery index",
        "Upload eval reporting dashboard payload",
        "Upload eval reporting webhook delivery request",
        "Upload eval reporting webhook delivery result",
        "Upload eval reporting release draft publish result",
    ]
    summary_idx = names.index("Consolidated eval reporting deploy-pages summary")
    for un in upload_names:
        assert names.index(un) > summary_idx, f"{un} must follow consolidated summary"


def test_upload_block_is_contiguous() -> None:
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    upload_names = [
        "Upload public discovery index",
        "Upload eval reporting dashboard payload",
        "Upload eval reporting webhook delivery request",
        "Upload eval reporting webhook delivery result",
        "Upload eval reporting release draft publish result",
    ]
    upload_indices = sorted(names.index(un) for un in upload_names)
    for i in range(len(upload_indices) - 1):
        assert upload_indices[i + 1] == upload_indices[i] + 1, (
            f"Upload block not contiguous between {names[upload_indices[i]]} and {names[upload_indices[i + 1]]}"
        )


# --- Batch 22B: Phase 6 consolidated deploy-pages baseline hardening ---


def test_old_per_surface_summary_steps_not_in_workflow() -> None:
    """Negative guard: old 5 per-surface summary append steps must not reappear."""
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    removed_names = [
        "Append public URLs to job summary",
        "Append eval reporting dashboard payload to job summary",
        "Append eval reporting webhook delivery request to job summary",
        "Append eval reporting webhook delivery result to job summary",
        "Append eval reporting release draft publish result to job summary",
    ]
    for rn in removed_names:
        assert rn not in names, f"Removed per-surface summary step reappeared: {rn}"


def test_generate_block_fixed_order() -> None:
    """Generate steps must appear in fixed order before consolidated summary."""
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    expected_generate_order = [
        "Generate public discovery index",
        "Generate eval reporting dashboard payload",
        "Generate eval reporting webhook delivery request",
        "Generate eval reporting webhook delivery result",
        "Generate eval reporting release draft publish result",
    ]
    generate_indices = [names.index(gn) for gn in expected_generate_order]
    assert generate_indices == sorted(generate_indices), (
        f"Generate steps not in expected fixed order"
    )


def test_upload_block_fixed_order() -> None:
    """Upload steps must appear in fixed order after consolidated summary."""
    workflow = _load_workflow()
    names = _step_names(workflow, "deploy-pages")

    expected_upload_order = [
        "Upload public discovery index",
        "Upload eval reporting dashboard payload",
        "Upload eval reporting webhook delivery request",
        "Upload eval reporting webhook delivery result",
        "Upload eval reporting release draft publish result",
    ]
    upload_indices = [names.index(un) for un in expected_upload_order]
    assert upload_indices == sorted(upload_indices), (
        f"Upload steps not in expected fixed order"
    )
