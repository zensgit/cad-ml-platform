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


# The pin-guard job pinned BY CONSTRUCTION: the step LIST, every step's KEY SET, and the JOB's key
# set. Pinning only the `run` field was not enough — an earlier version of this test passed while
#     shell: bash -c 'python -m pip install wheel >/dev/null; bash {0}'
# reintroduced a PyPI install, because `shell:` is a second execution entry point. `env:`, an extra
# `uses:` step, and a job-level `defaults.run.shell` are the same escape by other doors. Pinning the
# allowed key sets closes the class instead of the demonstrated instance: anything added fails,
# without this test predicting how a bootstrap might be smuggled in.
#
# SCOPE OF THE CLAIM (accurate — an earlier draft overclaimed): this asserts NO PACKAGE BOOTSTRAP /
# NO PyPI DEPENDENCY *at check time*. It is NOT "no network dependency": actions/checkout and
# actions/setup-python legitimately use the network. What must never come back is installing
# packages when the gate runs.
VALIDATOR_CMD = (
    "python scripts/ci/check_workflow_action_pins.py "
    "--workflows-dir .github/workflows "
    "--policy-json config/workflow_action_pin_policy.json "
    "--require-policy-for-all-external"
)
EXPECTED_JOB_KEYS = {"name", "runs-on", "timeout-minutes", "steps"}
EXPECTED_STEPS = (
    ("Checkout", {"name", "uses"}),
    ("Setup Python", {"name", "uses", "with"}),
    ("Validate workflow action pins", {"name", "run"}),
)


def _shell_command(run: str | None) -> str:
    """Join shell line-continuations and normalise whitespace -> the single command line."""
    return " ".join((run or "").replace("\\\n", " ").split())


def _assert_no_package_bootstrap(job: dict) -> None:
    """Raise AssertionError unless `job` is exactly the pinned, bootstrap-free pin-guard job."""
    extra_job = set(job) - EXPECTED_JOB_KEYS
    assert not extra_job, (
        f"unexpected job-level key(s) {sorted(extra_job)} — e.g. `defaults.run.shell` or `env:` "
        "bootstraps packages for every step; add only after deliberate review"
    )

    steps = job["steps"]
    assert [s.get("name") for s in steps] == [n for n, _ in EXPECTED_STEPS], (
        f"the job must be exactly {[n for n, _ in EXPECTED_STEPS]}; found "
        f"{[s.get('name') for s in steps]} — an extra step (even a policy-allowed `uses:`) is a new "
        "execution entry point"
    )
    for (step, (exp_name, allowed)) in zip(steps, EXPECTED_STEPS):
        extra = set(step) - allowed
        assert not extra, (
            f"step {exp_name!r} has unexpected key(s) {sorted(extra)} — `shell:` and `env:` are "
            "additional execution entry points that can bootstrap packages; add only after review"
        )

    cmd = _shell_command(steps[-1]["run"])
    assert cmd == VALIDATOR_CMD, (
        "the Validate step must run ONLY the pure-stdlib validator — any extra command (a package "
        "installer in any spelling, a curl, ...) is a PyPI dependency on a REQUIRED check and must "
        f"be reviewed deliberately:\n  got:      {cmd!r}\n  expected: {VALIDATOR_CMD!r}"
    )


def test_pin_guard_job_has_no_package_bootstrap() -> None:
    """The validator is pure stdlib, so this REQUIRED check must never bootstrap packages.

    A package install here makes every PR depend on PyPI being reachable — a needless failure
    surface on a gate whose entire value is being dependable. (Checkout/Setup-Python DO use the
    network; the invariant is specifically NO PACKAGE BOOTSTRAP, not "no network".)
    """
    _assert_no_package_bootstrap(_load_workflow(WORKFLOW_PATH)["jobs"]["action-pin-guard"])


@pytest.mark.parametrize("mutate", [
    # the reported escape: `shell:` is a second execution entry point, so the run block stays pristine
    pytest.param(
        lambda j: j["steps"][2].update(
            {"shell": "bash -c 'python -m pip install wheel >/dev/null; bash {0}'"}
        ),
        id="custom-shell-smuggles-an-install",
    ),
    # an extra step, even a policy-allowed `uses:`, is a new execution entry point
    pytest.param(
        lambda j: j["steps"].append({"name": "Extra", "uses": "actions/setup-node@abc123"}),
        id="extra-uses-step",
    ),
    # same class, other doors — proven to escape the run-only assertion too
    pytest.param(
        lambda j: j["steps"][2].update({"env": {"PIP_INDEX_URL": "http://example.invalid"}}),
        id="step-level-env",
    ),
    pytest.param(
        lambda j: j.update({"defaults": {"run": {"shell": "bash -c 'pip install x; bash {0}'"}}}),
        id="job-level-defaults-run-shell",
    ),
])
def test_package_bootstrap_smuggling_is_red(mutate) -> None:
    """observed-RED: every one of these passed the earlier run-only assertion while reintroducing a
    PyPI install (the job still reported ok). Pinning the key sets makes each of them RED."""
    job = copy.deepcopy(_load_workflow(WORKFLOW_PATH)["jobs"]["action-pin-guard"])
    mutate(job)
    with pytest.raises(AssertionError):
        _assert_no_package_bootstrap(job)
