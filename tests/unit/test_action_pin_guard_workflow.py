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


# The ONLY command the job is allowed to run. This is an ALLOW-list, deliberately not a deny-list:
# a forbidden-token check ("pip install" not in run) is a hand-enumerated list that misses
# `pip  install` (extra space), `pip3 install`, `uv pip install`, `conda install`, `curl … | sh`, …
# Asserting the exact command instead fails on ANY added command, whatever its spelling.
VALIDATOR_CMD = (
    "python scripts/ci/check_workflow_action_pins.py "
    "--workflows-dir .github/workflows "
    "--policy-json config/workflow_action_pin_policy.json "
    "--require-policy-for-all-external"
)


def _shell_command(run: str | None) -> str:
    """Join shell line-continuations and normalise whitespace -> the single command line."""
    return " ".join((run or "").replace("\\\n", " ").split())


def test_validation_step_has_no_network_dependency() -> None:
    """The validator is pure stdlib, so this REQUIRED check must never install packages.

    A package install here would make every PR depend on PyPI being reachable — a needless
    network-failure surface on a gate whose entire value is being dependable.

    Asserted BY CONSTRUCTION: exactly one step runs a command, and that command is exactly the
    stdlib validator. Nothing has to predict how an installer might be spelled.
    """
    workflow = _load_workflow(WORKFLOW_PATH)
    steps = workflow["jobs"]["action-pin-guard"]["steps"]

    running = [
        (step.get("name") or step.get("uses") or f"step {idx}", _shell_command(step.get("run")))
        for idx, step in enumerate(steps)
        if (step.get("run") or "").strip()
    ]
    assert len(running) == 1, (
        "exactly one step may run a command (the pure-stdlib validator); found: "
        f"{[name for name, _ in running]}"
    )
    name, cmd = running[0]
    assert name == "Validate workflow action pins", f"unexpected command-running step: {name!r}"
    assert cmd == VALIDATOR_CMD, (
        "the Validate step must run ONLY the pure-stdlib validator — any extra command (a package "
        "installer in any spelling, a curl, ...) is a network dependency on a REQUIRED check and "
        f"must be reviewed deliberately:\n  got:      {cmd!r}\n  expected: {VALIDATOR_CMD!r}"
    )
