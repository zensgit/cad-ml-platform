from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _invoke_main(module: Any, argv: list[str] | None = None) -> int:
    old_argv = sys.argv
    try:
        sys.argv = ["check_workflow_identity_invariants.py", *(argv or [])]
        return int(module.main())
    except SystemExit as exc:
        return int(exc.code)
    finally:
        sys.argv = old_argv


def _write_workflow(path: Path, *, name: str, inputs: list[str]) -> None:
    lines = [
        f"name: {name}",
        "on:",
        "  workflow_dispatch:",
        "    inputs:",
    ]
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


def _build_valid_repo(tmp_path: Path) -> Path:
    from scripts.ci.check_workflow_identity_invariants import SPECS

    root = tmp_path / ".github" / "workflows"
    for spec in SPECS:
        _write_workflow(
            root / spec.filename,
            name=spec.expected_name,
            inputs=list(spec.required_inputs),
        )
    return root


def _ci_watch_csv() -> str:
    return "CI,Code Quality,Security Audit,Evaluation Report"


def test_identity_check_success_and_summary(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_identity_invariants as mod

    workflow_root = _build_valid_repo(tmp_path)
    summary = tmp_path / "summary.json"

    rc = _invoke_main(
        mod,
        [
            "--workflow-root",
            str(workflow_root),
            "--ci-watch-required-workflows",
            _ci_watch_csv(),
            "--summary-json-out",
            str(summary),
        ],
    )

    assert rc == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["failed_count"] == 0
    assert len(payload["results"]) == len(mod.SPECS) + 1
    assert payload["name_to_files"]["CI"] == ["ci.yml"]
    assert any(row["filename"] == "evaluation-report.yml" for row in payload["results"])
    assert any(
        row["key"] == "ci_watch_required_workflows_mapping" and row["ok"]
        for row in payload["results"]
    )


def test_identity_check_fails_on_name_mismatch(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_identity_invariants as mod

    workflow_root = _build_valid_repo(tmp_path)
    _write_workflow(
        workflow_root / "evaluation-report.yml",
        name="Evaluation Report Drifted",
        inputs=list(next(spec.required_inputs for spec in mod.SPECS if spec.filename == "evaluation-report.yml")),
    )

    rc = _invoke_main(
        mod,
        [
            "--workflow-root",
            str(workflow_root),
            "--ci-watch-required-workflows",
            _ci_watch_csv(),
        ],
    )
    assert rc == 1


def test_identity_check_fails_on_missing_required_input(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_identity_invariants as mod

    workflow_root = _build_valid_repo(tmp_path)
    _write_workflow(
        workflow_root / "hybrid-superpass-e2e.yml",
        name="Hybrid Superpass E2E",
        inputs=[
            item
            for item in next(spec.required_inputs for spec in mod.SPECS if spec.filename == "hybrid-superpass-e2e.yml")
            if item != "hybrid_superpass_missing_mode"
        ],
    )

    rc = _invoke_main(
        mod,
        [
            "--workflow-root",
            str(workflow_root),
            "--ci-watch-required-workflows",
            _ci_watch_csv(),
        ],
    )
    assert rc == 1


def test_identity_check_fails_on_yaml_twin(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_identity_invariants as mod

    workflow_root = _build_valid_repo(tmp_path)
    twin = workflow_root / "evaluation-report.yaml"
    twin.write_text("name: Evaluation Report\non:\n  workflow_dispatch:\n", encoding="utf-8")

    rc = _invoke_main(
        mod,
        [
            "--workflow-root",
            str(workflow_root),
            "--ci-watch-required-workflows",
            _ci_watch_csv(),
        ],
    )
    assert rc == 1


def test_identity_check_fails_when_ci_watch_misses_evaluation_report(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_identity_invariants as mod

    workflow_root = _build_valid_repo(tmp_path)
    rc = _invoke_main(
        mod,
        [
            "--workflow-root",
            str(workflow_root),
            "--ci-watch-required-workflows",
            "CI,Code Quality,Security Audit",
        ],
    )
    assert rc == 1


def test_identity_check_fails_when_ci_name_is_not_unique(tmp_path: Path) -> None:
    from scripts.ci import check_workflow_identity_invariants as mod

    workflow_root = _build_valid_repo(tmp_path)
    _write_workflow(
        workflow_root / "ci-copy.yml",
        name="CI",
        inputs=[],
    )

    rc = _invoke_main(
        mod,
        [
            "--workflow-root",
            str(workflow_root),
            "--ci-watch-required-workflows",
            _ci_watch_csv(),
        ],
    )
    assert rc == 1


def test_identity_check_fails_when_required_workflow_name_is_missing_from_mapping(
    tmp_path: Path,
) -> None:
    from scripts.ci import check_workflow_identity_invariants as mod

    workflow_root = _build_valid_repo(tmp_path)
    rc = _invoke_main(
        mod,
        [
            "--workflow-root",
            str(workflow_root),
            "--ci-watch-required-workflows",
            "CI,Code Quality,Security Audit,Evaluation Report,CI Enhanced",
        ],
    )
    assert rc == 1
