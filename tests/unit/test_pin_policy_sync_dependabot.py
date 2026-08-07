"""#476 proof: policy regeneration turns unexpected Dependabot SHAs green."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ci.check_workflow_action_pins import (
    DEFAULT_CHECKOUT_SHA,
    DEFAULT_SETUP_PYTHON_SHA,
    _parse_policy_actions,
    scan_workflow_action_pins,
)
from scripts.ci.sync_dependabot_action_pin_policy import sync_policy


def test_sync_policy_turns_unexpected_sha_red_to_green(tmp_path: Path) -> None:
    """Drive the real sync + pin-guard entry points (not reimplemented)."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    # Minimal workflow with a SHA not yet allow-listed.
    new_sha = "a" * 40
    (workflows / "demo.yml").write_text(
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        f"      - uses: actions/checkout@{new_sha}\n",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.json"
    # Stale policy: different SHA only.
    policy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "actions": {"actions/checkout": ["b" * 40]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    policy, _ = _parse_policy_actions(policy_path)
    pre = scan_workflow_action_pins(
        workflows_dir=workflows,
        checkout_sha=DEFAULT_CHECKOUT_SHA,
        setup_python_sha=DEFAULT_SETUP_PYTHON_SHA,
        policy_actions=policy,
        require_policy_for_all_external=True,
    )
    assert pre["status"] == "error"
    assert pre["violations_count"] >= 1

    report = sync_policy(
        workflows_dir=workflows,
        policy_json=policy_path,
        skip_rewrite=True,
        dry_run=False,
    )
    assert report["status"] == "ok"

    policy2, _ = _parse_policy_actions(policy_path)
    post = scan_workflow_action_pins(
        workflows_dir=workflows,
        checkout_sha=DEFAULT_CHECKOUT_SHA,
        setup_python_sha=DEFAULT_SETUP_PYTHON_SHA,
        policy_actions=policy2,
        require_policy_for_all_external=True,
    )
    assert post["status"] == "ok"
    assert post["violations_count"] == 0
    assert new_sha in policy2["actions/checkout"]
