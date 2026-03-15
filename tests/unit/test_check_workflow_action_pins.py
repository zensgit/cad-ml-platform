from __future__ import annotations

import json
from pathlib import Path


def _write_workflow(path: Path, uses_lines: list[str]) -> None:
    lines = [
        "name: Test Workflow",
        "on: [push]",
        "jobs:",
        "  test:",
        "    runs-on: ubuntu-latest",
        "    steps:",
    ]
    for item in uses_lines:
        lines.append(f"      - uses: {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_scan_workflow_action_pins_passes_with_expected_shas(tmp_path: Path) -> None:
    from scripts.ci.check_workflow_action_pins import (
        DEFAULT_CHECKOUT_SHA,
        DEFAULT_SETUP_PYTHON_SHA,
        scan_workflow_action_pins,
    )

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _write_workflow(
        workflows_dir / "ok.yml",
        [
            f"actions/checkout@{DEFAULT_CHECKOUT_SHA}",
            f"actions/setup-python@{DEFAULT_SETUP_PYTHON_SHA}",
        ],
    )

    report = scan_workflow_action_pins(
        workflows_dir=workflows_dir,
        checkout_sha=DEFAULT_CHECKOUT_SHA,
        setup_python_sha=DEFAULT_SETUP_PYTHON_SHA,
    )
    assert report["status"] == "ok"
    assert report["violations_count"] == 0
    assert report["violations"] == []


def test_scan_workflow_action_pins_detects_tag_ref(tmp_path: Path) -> None:
    from scripts.ci.check_workflow_action_pins import (
        DEFAULT_CHECKOUT_SHA,
        DEFAULT_SETUP_PYTHON_SHA,
        scan_workflow_action_pins,
    )

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _write_workflow(
        workflows_dir / "bad_tag.yml",
        [
            "actions/checkout@v6",
            f"actions/setup-python@{DEFAULT_SETUP_PYTHON_SHA}",
        ],
    )

    report = scan_workflow_action_pins(
        workflows_dir=workflows_dir,
        checkout_sha=DEFAULT_CHECKOUT_SHA,
        setup_python_sha=DEFAULT_SETUP_PYTHON_SHA,
    )
    assert report["status"] == "error"
    assert report["violations_count"] == 1
    assert report["violations"][0]["reason"] == "tag_ref_not_allowed"


def test_main_writes_output_json_and_fails_for_unexpected_sha(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_action_pins as mod

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _write_workflow(
        workflows_dir / "bad_sha.yml",
        [
            "actions/checkout@1111111111111111111111111111111111111111",
            f"actions/setup-python@{mod.DEFAULT_SETUP_PYTHON_SHA}",
        ],
    )
    output_json = tmp_path / "report.json"

    rc = mod.main(
        [
            "--workflows-dir",
            str(workflows_dir),
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["violations_count"] == 1
    assert payload["violations"][0]["reason"] == "unexpected_sha"


def test_scan_workflow_action_pins_detects_upload_download_tag_refs(
    tmp_path: Path,
) -> None:
    from scripts.ci.check_workflow_action_pins import (
        DEFAULT_CHECKOUT_SHA,
        DEFAULT_SETUP_PYTHON_SHA,
        scan_workflow_action_pins,
    )

    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _write_workflow(
        workflows_dir / "bad_artifact_tags.yml",
        [
            f"actions/checkout@{DEFAULT_CHECKOUT_SHA}",
            f"actions/setup-python@{DEFAULT_SETUP_PYTHON_SHA}",
            "actions/upload-artifact@v4",
            "actions/download-artifact@v4",
        ],
    )

    report = scan_workflow_action_pins(
        workflows_dir=workflows_dir,
        checkout_sha=DEFAULT_CHECKOUT_SHA,
        setup_python_sha=DEFAULT_SETUP_PYTHON_SHA,
    )
    assert report["status"] == "error"
    assert report["violations_count"] == 2
    reasons = sorted(item["reason"] for item in report["violations"])
    assert reasons == ["tag_ref_not_allowed", "tag_ref_not_allowed"]
