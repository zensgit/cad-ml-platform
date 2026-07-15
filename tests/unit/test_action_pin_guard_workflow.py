from __future__ import annotations

import copy
from pathlib import Path

import pytest
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


# The pin-guard WORKFLOW pinned BY CONSTRUCTION at EVERY level GitHub Actions can run or influence a
# command: the WORKFLOW key set, the JOBS set, the JOB key set, the step LIST, and each STEP key set —
# plus the exact validator command. Pinning less kept losing: a run-only check missed a step `shell:`;
# a job-only check missed a WORKFLOW-level `defaults.run.shell` (the job dict cannot even see it). Exact
# SET EQUALITY (not merely "no added keys") closes the class — any new key at any level (`defaults`,
# `env`, an extra step or job, a `shell:`) fails, without this test predicting how a bootstrap is
# smuggled in.
#
# SCOPE (accurate — earlier drafts overclaimed twice): the invariant is NO PACKAGE BOOTSTRAP / NO PyPI
# DEPENDENCY *at check time*. It is NOT "no network dependency" — actions/checkout and setup-python
# legitimately use the network. And this pins STRUCTURE: some additions (a `shell:` running
# `pip install`) DIRECTLY reintroduce an install the pin-POLICY gate never inspects; others (an extra
# policy-allowed PINNED step, an `env:`) are execution/config entry points that must be a reviewed
# decision even when they do not install a package by themselves.
VALIDATOR_CMD = (
    "python scripts/ci/check_workflow_action_pins.py "
    "--workflows-dir .github/workflows "
    "--policy-json config/workflow_action_pin_policy.json "
    "--require-policy-for-all-external"
)
EXPECTED_WORKFLOW_KEYS = {"name", "on", "permissions", "jobs"}
EXPECTED_JOBS = {"action-pin-guard"}
EXPECTED_JOB_KEYS = {"name", "runs-on", "timeout-minutes", "steps"}
EXPECTED_STEPS = (
    ("Checkout", {"name", "uses"}),
    ("Setup Python", {"name", "uses", "with"}),
    ("Validate workflow action pins", {"name", "run"}),
)


def _shell_command(run: str | None) -> str:
    """Join shell line-continuations and normalise whitespace -> the single command line."""
    return " ".join((run or "").replace("\\\n", " ").split())


def _assert_pin_guard_pinned(workflow: dict) -> None:
    """Raise AssertionError unless the WHOLE workflow is exactly the pinned, bootstrap-free pin guard.

    Exact set equality at every level (workflow / jobs / job / step list / step keys) so nothing new —
    a workflow- or job-level `defaults`/`env`, an extra step or job, a step `shell:`/`env:` — can add
    an execution or config entry point unnoticed.
    """
    assert set(workflow) == EXPECTED_WORKFLOW_KEYS, (
        f"workflow-level keys must be EXACTLY {sorted(EXPECTED_WORKFLOW_KEYS)}; got {sorted(workflow)} "
        "— a top-level `defaults.run.shell` or `env:` runs for EVERY step and the job dict cannot see it"
    )
    assert set(workflow["jobs"]) == EXPECTED_JOBS, (
        f"jobs must be EXACTLY {sorted(EXPECTED_JOBS)}; got {sorted(workflow['jobs'])} — a second job "
        "is another execution surface"
    )
    job = workflow["jobs"]["action-pin-guard"]
    assert set(job) == EXPECTED_JOB_KEYS, (
        f"job keys must be EXACTLY {sorted(EXPECTED_JOB_KEYS)}; got {sorted(job)} — a job-level "
        "`defaults.run.shell` or `env:` bootstraps for every step"
    )
    steps = job["steps"]
    assert [s.get("name") for s in steps] == [n for n, _ in EXPECTED_STEPS], (
        f"steps must be EXACTLY {[n for n, _ in EXPECTED_STEPS]}; got {[s.get('name') for s in steps]} "
        "— an extra step (even a policy-allowed pinned `uses:`) is a new execution entry point"
    )
    for step, (exp_name, allowed) in zip(steps, EXPECTED_STEPS):
        assert set(step) == allowed, (
            f"step {exp_name!r} keys must be EXACTLY {sorted(allowed)}; got {sorted(step)} — `shell:` "
            "and `env:` are extra execution/config entry points"
        )
    cmd = _shell_command(steps[-1]["run"])
    assert cmd == VALIDATOR_CMD, (
        "the Validate step must run ONLY the pure-stdlib validator — any extra command (a package "
        "installer in any spelling, a curl, ...) is a PyPI dependency on a REQUIRED check and must "
        f"be reviewed deliberately:\n  got:      {cmd!r}\n  expected: {VALIDATOR_CMD!r}"
    )


def test_pin_guard_workflow_is_pinned_no_package_bootstrap() -> None:
    """The validator is pure stdlib, so this REQUIRED check must never bootstrap packages.

    A package install here makes every PR depend on PyPI being reachable — a needless failure
    surface on a gate whose entire value is being dependable. (Checkout/Setup-Python DO use the
    network; the invariant is specifically NO PACKAGE BOOTSTRAP at check time, not "no network".)
    """
    _assert_pin_guard_pinned(_load_workflow(WORKFLOW_PATH))


# Each mutation was MISSED by an earlier, less-complete assertion (run-only, then job-only). They fall
# in two honest categories:
#   * DIRECT bootstrap — runs `pip install`; the pin-POLICY gate does NOT inspect shell content, so it
#     would pass THAT gate. Only this structural test stops it.
#   * STRUCTURAL entry point — not a package install by itself, but a new execution/config surface that
#     must be a reviewed decision. The extra step uses a REAL policy-allowed PINNED action, so the
#     pin-policy gate accepts it; only this test rejects it.
@pytest.mark.parametrize("mutate", [
    pytest.param(lambda w: w["jobs"]["action-pin-guard"]["steps"][2].update(
        {"shell": "bash -c 'python -m pip install wheel >/dev/null; bash {0}'"}),
        id="direct-bootstrap:step-shell"),
    pytest.param(lambda w: w["jobs"]["action-pin-guard"].update(
        {"defaults": {"run": {"shell": "bash -c 'pip install x; bash {0}'"}}}),
        id="direct-bootstrap:job-defaults-run-shell"),
    pytest.param(lambda w: w.update(
        {"defaults": {"run": {"shell": "bash -c 'pip install x; bash {0}'"}}}),
        id="direct-bootstrap:workflow-defaults-run-shell"),  # the review-3 escape: invisible to the job dict
    pytest.param(lambda w: w["jobs"]["action-pin-guard"]["steps"].append(
        {"name": "Extra", "uses": f"actions/checkout@{CHECKOUT_SHA}"}),
        id="structural:extra-policy-allowed-pinned-step"),
    pytest.param(lambda w: w["jobs"]["action-pin-guard"]["steps"][2].update(
        {"env": {"PIP_INDEX_URL": "http://example.invalid"}}),
        id="structural:step-env"),
    pytest.param(lambda w: w["jobs"]["action-pin-guard"].update({"env": {"X": "1"}}),
        id="structural:job-env"),
    pytest.param(lambda w: w.update({"env": {"X": "1"}}), id="structural:workflow-env"),
])
def test_added_execution_entry_point_is_red(mutate) -> None:
    """observed-RED: each of these was MISSED by an earlier assertion. Exact set-equality on the FULL
    workflow makes every one RED — direct bootstraps and structural entry points alike."""
    workflow = copy.deepcopy(_load_workflow(WORKFLOW_PATH))
    mutate(workflow)
    with pytest.raises(AssertionError):
        _assert_pin_guard_pinned(workflow)
