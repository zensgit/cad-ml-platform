from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "action-pin-guard.yml"

CHECKOUT_SHA = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
SETUP_PYTHON_SHA = "a309ff8b426b58ec0e2a45f0f869d46889d02405"


def _load_workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _find_step_by_name(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_action_pin_guard_workflow_has_expected_triggers_and_steps() -> None:
    workflow = _load_workflow(WORKFLOW_PATH)

    assert workflow["name"] == "Action Pin Guard"
    # push stays path-filtered.
    assert ".github/workflows/**" in workflow["on"]["push"]["paths"]
    assert "config/workflow_action_pin_policy.json" in workflow["on"]["push"]["paths"]
    # pull_request must have NO `paths:` filter — this is a REQUIRED status check, so it must run on
    # EVERY PR to produce the context (a paths filter permanently BLOCKS docs-only/CODEOWNERS-only PRs).
    pr_trigger = workflow["on"]["pull_request"]
    assert pr_trigger is None or "paths" not in pr_trigger, (
        f"pull_request must not be path-filtered (required-context deadlock): {pr_trigger}"
    )

    checkout_step = _find_step_by_name(workflow, "action-pin-guard", "Checkout")
    assert checkout_step["uses"] == f"actions/checkout@{CHECKOUT_SHA}"

    setup_step = _find_step_by_name(workflow, "action-pin-guard", "Setup Python")
    assert setup_step["uses"] == f"actions/setup-python@{SETUP_PYTHON_SHA}"

    validate_step = _find_step_by_name(
        workflow, "action-pin-guard", "Validate workflow action pins"
    )
    run_script = validate_step["run"]
    assert "scripts/ci/check_workflow_action_pins.py" in run_script
    assert "--workflows-dir .github/workflows" in run_script
    assert "--policy-json config/workflow_action_pin_policy.json" in run_script
    assert "--require-policy-for-all-external" in run_script


def test_validation_step_has_no_network_dependency() -> None:
    """The validator is pure stdlib, so this REQUIRED check must never install packages.

    A `pip install` here would make every PR depend on PyPI being reachable — a needless
    network-failure surface on a gate whose entire value is being dependable.
    """
    workflow = _load_workflow(WORKFLOW_PATH)
    validate_step = _find_step_by_name(
        workflow, "action-pin-guard", "Validate workflow action pins"
    )
    run_script = validate_step["run"]
    assert "pip install" not in run_script, (
        f"validation step must not install packages (pure stdlib; no PyPI per PR): {run_script!r}"
    )
    # nothing else in the job may reach the network either (only checkout + setup-python + validate).
    steps = workflow["jobs"]["action-pin-guard"]["steps"]
    for step in steps:
        assert "pip install" not in (step.get("run") or ""), f"network dep in step {step.get('name')!r}"
