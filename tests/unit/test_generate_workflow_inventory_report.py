from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["generate_workflow_inventory_report.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def _write_workflow(path: Path, *, name: str, inputs: list[str] | None = None) -> None:
    lines = [f"name: {name}", "on:", "  workflow_dispatch:"]
    if inputs is None:
        lines.append("    {}")
    else:
        lines.append("    inputs:")
        for item in inputs:
            lines.extend(
                [
                    f"      {item}:",
                    '        description: "test"',
                    "        required: false",
                    '        default: ""',
                ]
            )
    lines.extend(["jobs:", "  test:", "    runs-on: ubuntu-latest", "    steps: []"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_report_tracks_duplicates_and_missing_required(tmp_path: Path) -> None:
    from scripts.ci import generate_workflow_inventory_report as mod

    workflow_root = tmp_path / ".github" / "workflows"
    _write_workflow(workflow_root / "ci.yml", name="CI", inputs=[])
    _write_workflow(workflow_root / "ci-copy.yml", name="CI", inputs=[])
    _write_workflow(workflow_root / "evaluation-report.yml", name="Evaluation Report", inputs=["x"])

    report = mod.build_report(
        workflow_root=workflow_root,
        ci_watch_required_workflows=["CI", "Evaluation Report", "Code Quality"],
    )

    assert report["workflow_count"] == 3
    assert report["duplicate_name_count"] == 1
    assert report["missing_required_count"] == 1
    assert report["non_unique_required_count"] == 1
    assert report["duplicates"][0]["name"] == "CI"
    assert any(row["name"] == "Code Quality" and row["status"] == "missing" for row in report["required_workflow_mapping"])


def test_render_markdown_contains_required_and_duplicate_sections() -> None:
    from scripts.ci import generate_workflow_inventory_report as mod

    markdown = mod.render_markdown(
        {
            "workflow_root": ".github/workflows",
            "workflow_count": 2,
            "required_count": 1,
            "duplicate_name_count": 1,
            "missing_required_count": 0,
            "non_unique_required_count": 1,
            "required_workflow_mapping": [{"name": "CI", "status": "non_unique", "files": ["ci.yml", "ci-copy.yml"]}],
            "duplicates": [{"name": "CI", "files": ["ci.yml", "ci-copy.yml"]}],
            "workflows": [{"filename": "ci.yml", "name": "CI", "parse_ok": True, "has_workflow_dispatch": True, "dispatch_inputs": []}],
        }
    )

    assert "# Workflow Inventory Audit" in markdown
    assert "## Required Workflow Mapping" in markdown
    assert "status=non_unique" in markdown
    assert "## Duplicate Workflow Names" in markdown
    assert "ci-copy.yml" in markdown


def test_main_writes_json_and_markdown(tmp_path: Path, capsys: Any) -> None:
    from scripts.ci import generate_workflow_inventory_report as mod

    workflow_root = tmp_path / ".github" / "workflows"
    _write_workflow(workflow_root / "ci.yml", name="CI", inputs=[])
    _write_workflow(workflow_root / "evaluation-report.yml", name="Evaluation Report", inputs=["foo"])
    output_json = tmp_path / "workflow_inventory.json"
    output_md = tmp_path / "workflow_inventory.md"

    rc = _invoke_main(
        mod,
        [
            "--workflow-root",
            str(workflow_root),
            "--ci-watch-required-workflows",
            "CI,Evaluation Report",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["workflow_count"] == 2
    assert payload["duplicate_name_count"] == 0
    rendered = output_md.read_text(encoding="utf-8")
    assert "# Workflow Inventory Audit" in rendered
    assert "Evaluation Report" in rendered
    assert "output_json=" in capsys.readouterr().out
