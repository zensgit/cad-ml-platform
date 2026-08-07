"""Unit tests for #476 dependabot action pin sync helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.ci.generate_workflow_action_pin_policy import collect_action_pin_policy
from scripts.ci.rewrite_workflow_action_tags_to_sha import (
    rewrite_workflow_text,
    rewrite_workflows_dir,
)
from scripts.ci.sync_dependabot_action_pin_policy import sync_policy

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "dependabot-action-pin-sync.yml"
CHECKOUT_SHA = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
SETUP_PYTHON_SHA = "a309ff8b426b58ec0e2a45f0f869d46889d02405"


def test_rewrite_leaves_sha_targets_unchanged() -> None:
    text = (
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        f"      - uses: actions/checkout@{CHECKOUT_SHA}\n"
    )
    out, changes = rewrite_workflow_text(
        text, resolve=lambda o, r, ref: "deadbeef" * 5
    )
    assert out == text
    assert changes == []


def test_rewrite_converts_tag_to_sha_and_records_original_ref() -> None:
    text = (
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
    )
    sha = "a" * 40

    def resolve(owner: str, repo: str, ref: str) -> str:
        assert owner == "actions"
        assert repo == "checkout"
        assert ref == "v4"
        return sha

    out, changes = rewrite_workflow_text(text, resolve=resolve)
    assert f"uses: actions/checkout@{sha}  # v4" in out
    assert changes == [
        {
            "status": "rewritten",
            "uses": "actions/checkout@v4",
            "new_uses": f"actions/checkout@{sha}",
            "action": "actions/checkout",
            "ref": "v4",
            "sha": sha,
        }
    ]


def test_rewrite_unresolved_tag_is_fail_closed(tmp_path: Path) -> None:
    wf = tmp_path / "sample.yml"
    wf.write_text(
        "jobs:\n  x:\n    steps:\n      - uses: actions/checkout@v4\n",
        encoding="utf-8",
    )
    report = rewrite_workflows_dir(
        tmp_path,
        dry_run=False,
        resolve_fn=lambda *_a: "",
    )
    assert report["status"] == "partial"
    assert report["unresolved_count"] == 1
    # File must not be rewritten when resolution fails.
    assert "v4" in wf.read_text(encoding="utf-8")


def test_sync_policy_regenerates_allowlist_for_new_sha(tmp_path: Path) -> None:
    """The load-bearing #476 case: workflows already SHA-pinned, policy stale."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    new_sha = "b" * 40
    (workflows / "demo.yml").write_text(
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        f"      - uses: actions/checkout@{new_sha}\n",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    # Stale policy only allows a different SHA.
    policy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "actions": {"actions/checkout": ["c" * 40]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = sync_policy(
        workflows_dir=workflows,
        policy_json=policy_path,
        skip_rewrite=True,
        dry_run=False,
    )
    assert report["status"] == "ok"
    updated = json.loads(policy_path.read_text(encoding="utf-8"))
    assert new_sha in updated["actions"]["actions/checkout"]


def test_sync_policy_rewrite_then_policy_covers_resolved_sha(tmp_path: Path) -> None:
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "demo.yml").write_text(
        "jobs:\n  x:\n    steps:\n      - uses: azure/setup-helm@v5\n",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    sha = "d" * 40

    # Monkey via resolve_fn on rewrite_workflows_dir by using sync with real rewrite
    # and a patched resolver through rewrite_workflows_dir is not exposed on sync_policy;
    # call rewrite + collect directly for this path.
    rewrite_workflows_dir(
        workflows,
        resolve_fn=lambda owner, repo, ref: sha
        if (owner, repo, ref) == ("azure", "setup-helm", "v5")
        else "",
    )
    payload, code = collect_action_pin_policy(workflows_dir=workflows, strict=True)
    assert code == 0
    assert sha in payload["actions"]["azure/setup-helm"]
    text = (workflows / "demo.yml").read_text(encoding="utf-8")
    assert f"azure/setup-helm@{sha}" in text
    assert "v5" in text  # original ref retained as comment


def test_dependabot_pin_sync_workflow_is_sha_pinned_and_scoped() -> None:
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.load(raw, Loader=yaml.BaseLoader)
    assert workflow["name"] == "Dependabot Action Pin Sync"
    assert workflow["on"]["pull_request"]["types"] == [
        "opened",
        "reopened",
        "synchronize",
    ]
    job = workflow["jobs"]["sync-pin-policy"]
    # Only dependabot or manual dispatch — never rewrite arbitrary contributor PRs.
    assert "dependabot[bot]" in job["if"]
    assert "workflow_dispatch" in job["if"]

    uses = []
    for step in job["steps"]:
        if "uses" in step:
            uses.append(step["uses"])
    assert uses == [
        f"actions/checkout@{CHECKOUT_SHA}",
        f"actions/setup-python@{SETUP_PYTHON_SHA}",
    ]
    run_blob = "\n".join(step.get("run", "") for step in job["steps"])
    assert "scripts/ci/sync_dependabot_action_pin_policy.py" in run_blob
    assert "scripts/ci/check_workflow_action_pins.py" in run_blob
    assert "--require-policy-for-all-external" in run_blob
