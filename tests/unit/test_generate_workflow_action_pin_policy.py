from __future__ import annotations

import json
from pathlib import Path


def _write_workflow(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_collect_action_pin_policy_collects_sha_refs(tmp_path: Path) -> None:
    from scripts.ci.generate_workflow_action_pin_policy import collect_action_pin_policy

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _write_workflow(
        workflows_dir / "ok.yml",
        [
            "name: W",
            "on: [push]",
            "jobs:",
            "  t:",
            "    runs-on: ubuntu-latest",
            "    steps:",
            "      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
        ],
    )

    payload, exit_code = collect_action_pin_policy(
        workflows_dir=workflows_dir,
        strict=True,
    )
    assert exit_code == 0
    assert payload["non_sha_refs"] == []
    assert "actions/checkout" in payload["actions"]


def test_main_strict_fails_when_non_sha_ref_exists(tmp_path: Path) -> None:
    from scripts.ci import generate_workflow_action_pin_policy as mod

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _write_workflow(
        workflows_dir / "bad.yml",
        [
            "name: W",
            "on: [push]",
            "jobs:",
            "  t:",
            "    runs-on: ubuntu-latest",
            "    steps:",
            "      - uses: actions/checkout@v6",
        ],
    )
    output_json = tmp_path / "policy.json"

    rc = mod.main(
        [
            "--workflows-dir",
            str(workflows_dir),
            "--output-json",
            str(output_json),
            "--strict",
        ]
    )
    assert rc == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(payload["non_sha_refs"]) == 1
