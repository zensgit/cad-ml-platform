#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class WorkflowIdentitySpec:
    key: str
    filename: str
    expected_name: str
    required_inputs: tuple[str, ...] = ()
    require_ci_watch: bool = False


SPECS: tuple[WorkflowIdentitySpec, ...] = (
    WorkflowIdentitySpec(
        key="ci_sentinel",
        filename="ci.yml",
        expected_name="CI",
        require_ci_watch=True,
    ),
    WorkflowIdentitySpec(
        key="code_quality_sentinel",
        filename="code-quality.yml",
        expected_name="Code Quality",
        require_ci_watch=True,
    ),
    WorkflowIdentitySpec(
        key="security_audit",
        filename="security-audit.yml",
        expected_name="Security Audit",
        require_ci_watch=True,
    ),
    WorkflowIdentitySpec(
        key="evaluation_report",
        filename="evaluation-report.yml",
        expected_name="Evaluation Report",
        required_inputs=(
            "hybrid_blind_enable",
            "hybrid_blind_dxf_dir",
            "hybrid_blind_manifest_csv",
            "hybrid_blind_synth_manifest",
            "hybrid_blind_fail_on_gate_failed",
            "hybrid_blind_strict_require_real_data",
            "hybrid_superpass_enable",
            "hybrid_superpass_missing_mode",
            "hybrid_superpass_fail_on_failed",
            "hybrid_calibration_enable",
            "hybrid_calibration_input_csv",
        ),
        require_ci_watch=True,
    ),
    WorkflowIdentitySpec(
        key="soft_mode_smoke",
        filename="evaluation-soft-mode-smoke.yml",
        expected_name="Evaluation Soft-Mode Smoke",
        required_inputs=("ref", "expected_conclusion", "pr_number", "keep_soft", "skip_log_check"),
    ),
    WorkflowIdentitySpec(
        key="hybrid_superpass_wrapper",
        filename="hybrid-superpass-e2e.yml",
        expected_name="Hybrid Superpass E2E",
        required_inputs=(
            "ref",
            "expected_conclusion",
            "hybrid_superpass_enable",
            "hybrid_superpass_missing_mode",
            "hybrid_superpass_fail_on_failed",
            "hybrid_blind_enable",
            "hybrid_blind_dxf_dir",
            "hybrid_blind_fail_on_gate_failed",
            "hybrid_blind_strict_require_real_data",
            "hybrid_calibration_enable",
            "hybrid_calibration_input_csv",
        ),
    ),
    WorkflowIdentitySpec(
        key="hybrid_blind_strict_real_wrapper",
        filename="hybrid-blind-strict-real-e2e.yml",
        expected_name="Hybrid Blind Strict-Real E2E",
        required_inputs=(
            "ref",
            "expected_conclusion",
            "hybrid_blind_dxf_dir",
            "hybrid_blind_manifest_csv",
            "hybrid_blind_synth_manifest",
            "strict_fail_on_gate_failed",
            "strict_require_real_data",
        ),
    ),
    WorkflowIdentitySpec(
        key="archive_dry_run",
        filename="experiment-archive-dry-run.yml",
        expected_name="Experiment Archive Dry Run",
        required_inputs=("experiments_root", "archive_root", "keep_latest_days", "today"),
    ),
    WorkflowIdentitySpec(
        key="archive_apply",
        filename="experiment-archive-apply.yml",
        expected_name="Experiment Archive Apply",
        required_inputs=(
            "approval_phrase",
            "experiments_root",
            "archive_root",
            "keep_latest_days",
            "dirs_csv",
            "today",
            "require_exists",
        ),
    ),
    WorkflowIdentitySpec(
        key="stress_tests",
        filename="stress-tests.yml",
        expected_name="Stress and Observability Checks",
    ),
)


def _split_csv_items(raw: str) -> list[str]:
    items: list[str] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if token:
            items.append(token)
    return items


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if yaml is None:
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return payload if isinstance(payload, dict) else None


def _get_on_block(payload: dict[str, Any]) -> Any:
    if "on" in payload:
        return payload["on"]
    return payload.get(True)


def _get_dispatch_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    on_block = _get_on_block(payload)
    if not isinstance(on_block, dict):
        return {}
    dispatch = on_block.get("workflow_dispatch")
    if not isinstance(dispatch, dict):
        return {}
    inputs = dispatch.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _check_spec(
    *,
    workflow_root: Path,
    spec: WorkflowIdentitySpec,
    ci_watch_required_workflows: set[str],
    name_to_files: dict[str, list[str]],
) -> dict[str, Any]:
    path = workflow_root / spec.filename
    yaml_twin = path.with_suffix(".yaml")
    result = {
        "key": spec.key,
        "path": str(path),
        "filename": spec.filename,
        "expected_name": spec.expected_name,
        "ok": True,
        "issues": [],
    }

    def _issue(message: str) -> None:
        result["ok"] = False
        result["issues"].append(message)

    if not path.is_file():
        _issue("missing required workflow file")
        return result

    if yaml_twin.is_file():
        _issue(f"unexpected .yaml twin exists: {yaml_twin.name}")

    payload = _load_yaml(path)
    if payload is None:
        _issue("failed to parse workflow yaml")
        return result

    actual_name = str(payload.get("name") or "")
    result["actual_name"] = actual_name
    if actual_name != spec.expected_name:
        _issue(f"name mismatch: expected {spec.expected_name!r}, got {actual_name!r}")

    dispatch_inputs = _get_dispatch_inputs(payload)
    result["dispatch_input_count"] = len(dispatch_inputs)
    missing_inputs = [item for item in spec.required_inputs if item not in dispatch_inputs]
    result["missing_inputs"] = missing_inputs
    if missing_inputs:
        _issue(f"missing workflow_dispatch inputs: {', '.join(missing_inputs)}")

    if spec.require_ci_watch and spec.expected_name not in ci_watch_required_workflows:
        _issue("expected workflow name missing from CI_WATCH_REQUIRED_WORKFLOWS")
    if spec.require_ci_watch:
        name_matches = name_to_files.get(spec.expected_name, [])
        if len(name_matches) != 1:
            _issue(
                "expected workflow name must map uniquely to one .yml file: "
                f"{spec.expected_name!r} -> {name_matches}"
            )
        elif name_matches[0] != spec.filename:
            _issue(
                "expected workflow name resolves to unexpected file: "
                f"{spec.expected_name!r} -> {name_matches[0]!r}"
            )

    return result


def _check_ci_watch_required_workflows_mapping(
    *,
    ci_watch_required_workflows: Sequence[str],
    name_to_files: dict[str, list[str]],
) -> dict[str, Any]:
    result = {
        "key": "ci_watch_required_workflows_mapping",
        "path": "CI_WATCH_REQUIRED_WORKFLOWS",
        "filename": "CI_WATCH_REQUIRED_WORKFLOWS",
        "expected_name": "",
        "ok": True,
        "issues": [],
    }

    def _issue(message: str) -> None:
        result["ok"] = False
        result["issues"].append(message)

    checked_names: list[str] = []
    for workflow_name in ci_watch_required_workflows:
        name = str(workflow_name).strip()
        if not name or name in checked_names:
            continue
        checked_names.append(name)
        matches = list(name_to_files.get(name, []))
        if not matches:
            _issue(f"required workflow name missing from .yml workflows: {name!r}")
            continue
        if len(matches) != 1:
            _issue(f"required workflow name is not unique: {name!r} -> {matches}")

    result["checked_names"] = checked_names
    return result


def _write_json(path_value: str, payload: dict[str, Any]) -> None:
    out = Path(path_value).expanduser()
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check critical GitHub workflow identity invariants."
    )
    parser.add_argument(
        "--workflow-root",
        default=".github/workflows",
        help="Workflow directory root (default: .github/workflows).",
    )
    parser.add_argument(
        "--ci-watch-required-workflows",
        default=(
            "CI,CI Enhanced,CI Tiered Tests,Code Quality,"
            "Multi-Architecture Docker Build,Security Audit,"
            "Observability Checks,Self-Check,GHCR Publish,Evaluation Report"
        ),
        help="Comma-separated workflow names expected by CI watcher.",
    )
    parser.add_argument(
        "--summary-json-out",
        default="",
        help="Optional path to write machine-readable summary JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if yaml is None:
        print("error: PyYAML is required for workflow identity checks", flush=True)
        return 2

    workflow_root = Path(str(args.workflow_root)).expanduser()
    ci_watch_required_workflows = set(_split_csv_items(str(args.ci_watch_required_workflows)))
    name_to_files: dict[str, list[str]] = {}
    for path in sorted(workflow_root.glob("*.yml")):
        payload = _load_yaml(path)
        if not isinstance(payload, dict):
            continue
        workflow_name = str(payload.get("name") or "").strip()
        if workflow_name:
            name_to_files.setdefault(workflow_name, []).append(path.name)
    results = [
        _check_spec(
            workflow_root=workflow_root,
            spec=spec,
            ci_watch_required_workflows=ci_watch_required_workflows,
            name_to_files=name_to_files,
        )
        for spec in SPECS
    ]
    results.append(
        _check_ci_watch_required_workflows_mapping(
            ci_watch_required_workflows=sorted(ci_watch_required_workflows),
            name_to_files=name_to_files,
        )
    )
    failed = [row for row in results if not row["ok"]]

    for row in results:
        status = "ok" if row["ok"] else "fail"
        issues = row["issues"] or ["ok"]
        print(f"[{status}] {row['filename']} - {'; '.join(issues)}", flush=True)

    summary_out = str(args.summary_json_out or "").strip()
    if summary_out:
        payload = {
            "version": 1,
            "workflow_root": str(workflow_root),
            "ci_watch_required_workflows": sorted(ci_watch_required_workflows),
            "name_to_files": name_to_files,
            "failed_count": len(failed),
            "results": results,
        }
        _write_json(summary_out, payload)
        print(f"summary_json={summary_out}", flush=True)

    if failed:
        print(f"error: workflow identity check failed for {len(failed)} check(s)", flush=True)
        return 1

    print(f"ok: workflow identity check passed for {len(results)} check(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
