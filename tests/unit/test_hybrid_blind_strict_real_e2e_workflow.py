"""Regression checks for hybrid blind strict-real wrapper workflow wiring."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "hybrid-blind-strict-real-e2e.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_workflow_has_expected_dispatch_inputs() -> None:
    workflow = _load_workflow()
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert "ref" in inputs
    assert "expected_conclusion" in inputs
    assert "hybrid_blind_dxf_dir" in inputs
    assert "hybrid_blind_manifest_csv" in inputs
    assert "hybrid_blind_synth_manifest" in inputs
    assert "strict_fail_on_gate_failed" in inputs
    assert "strict_require_real_data" in inputs


def test_workflow_job_permissions_and_steps() -> None:
    workflow = _load_workflow()
    assert workflow["permissions"]["contents"] == "read"
    assert workflow["permissions"]["actions"] == "write"
    assert (
        workflow["concurrency"]["group"]
        == "hybrid-blind-strict-real-e2e-${{ github.ref }}"
    )

    job = workflow["jobs"]["hybrid-blind-strict-real-e2e"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"]["contents"] == "read"
    assert job["permissions"]["actions"] == "write"
    assert job["env"]["GH_TOKEN"] == "${{ github.token }}"

    run_step = _get_step(
        workflow,
        "hybrid-blind-strict-real-e2e",
        "Run hybrid blind strict-real dispatcher",
    )
    run_script = run_step["run"]
    assert "scripts/ci/dispatch_hybrid_blind_strict_real_workflow.py" in run_script
    assert '--workflow "evaluation-report.yml"' in run_script
    assert "--ref \"$REF_INPUT\"" in run_script
    assert "--hybrid-blind-dxf-dir \"$DXF_DIR_INPUT\"" in run_script
    assert "--strict-fail-on-gate-failed" in run_script
    assert "--strict-require-real-data" in run_script
    assert "--expected-conclusion \"$EXPECTED_INPUT\"" in run_script
    assert "--output-json reports/ci/hybrid_blind_strict_real_e2e_summary.json" in run_script
    assert "--hybrid-blind-manifest-csv" in run_script
    assert "--hybrid-blind-synth-manifest" in run_script

    render_step = _get_step(
        workflow,
        "hybrid-blind-strict-real-e2e",
        "Render hybrid blind strict-real summary markdown",
    )
    render_script = render_step["run"]
    assert (
        "scripts/ci/render_hybrid_blind_strict_real_dispatch_summary.py"
        in render_script
    )
    assert (
        "--dispatch-json reports/ci/hybrid_blind_strict_real_e2e_summary.json"
        in render_script
    )
    assert "--output-md reports/ci/hybrid_blind_strict_real_e2e_summary.md" in render_script
    assert ">/dev/null" in render_script

    upload_step = _get_step(
        workflow,
        "hybrid-blind-strict-real-e2e",
        "Upload hybrid blind strict-real summary",
    )
    assert (
        upload_step["uses"]
        == "actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
    )
    assert (
        upload_step["with"]["name"]
        == "hybrid-blind-strict-real-e2e-${{ github.run_number }}"
    )
    upload_path = upload_step["with"]["path"]
    assert "reports/ci/hybrid_blind_strict_real_e2e_summary.json" in upload_path
    assert "reports/ci/hybrid_blind_strict_real_e2e_summary.md" in upload_path

    append_step = _get_step(
        workflow, "hybrid-blind-strict-real-e2e", "Append summary"
    )
    append_script = append_step["run"]
    assert "cat reports/ci/hybrid_blind_strict_real_e2e_summary.md" in append_script
    assert "summary_md missing" in append_script
    assert '>> "$GITHUB_STEP_SUMMARY"' in append_script
