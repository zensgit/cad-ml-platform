"""Regression checks for retry-hardened dependency install in ci-enhanced coverage job."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci-enhanced.yml"


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _get_step(workflow: dict, job_name: str, step_name: str) -> dict:
    steps = workflow["jobs"][job_name]["steps"]
    for step in steps:
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing step {step_name!r} in job {job_name!r}")


def test_coverage_job_retries_pip_install() -> None:
    workflow = _load_workflow()

    step = _get_step(workflow, "coverage", "Install dependencies")
    script = step["run"]

    assert "retry_pip()" in script
    assert "local max_attempts=3" in script
    assert "sleep $((attempt * 5))" in script
    assert "retry_pip pip install -r requirements.txt" in script
    assert "retry_pip pip install -r requirements-dev.txt || true" in script
    assert "retry_pip pip install coverage pytest-cov" in script
