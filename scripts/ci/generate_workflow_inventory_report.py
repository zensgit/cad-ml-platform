#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    yaml = None  # type: ignore[assignment]


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


def collect_workflow_inventory(workflow_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(workflow_root.glob("*.yml")):
        payload = _load_yaml(path)
        if not isinstance(payload, dict):
            rows.append(
                {
                    "filename": path.name,
                    "name": "",
                    "parse_ok": False,
                    "has_workflow_dispatch": False,
                    "dispatch_inputs": [],
                }
            )
            continue

        dispatch_inputs = _get_dispatch_inputs(payload)
        rows.append(
            {
                "filename": path.name,
                "name": str(payload.get("name") or ""),
                "parse_ok": True,
                "has_workflow_dispatch": bool(dispatch_inputs or "workflow_dispatch" in (str(k) for k in (_get_on_block(payload) or {}).keys())),
                "dispatch_inputs": sorted(dispatch_inputs.keys()),
            }
        )
    return rows


def build_report(
    *,
    workflow_root: Path,
    ci_watch_required_workflows: Sequence[str],
) -> dict[str, Any]:
    workflows = collect_workflow_inventory(workflow_root)
    name_to_files: dict[str, list[str]] = {}
    for row in workflows:
        workflow_name = str(row.get("name") or "").strip()
        if workflow_name:
            name_to_files.setdefault(workflow_name, []).append(str(row["filename"]))

    duplicates = [
        {"name": name, "files": sorted(files)}
        for name, files in sorted(name_to_files.items())
        if len(files) > 1
    ]
    required_rows = []
    missing_required: list[str] = []
    non_unique_required: list[dict[str, Any]] = []
    for workflow_name in ci_watch_required_workflows:
        matches = sorted(name_to_files.get(str(workflow_name), []))
        status = "ok"
        if not matches:
            status = "missing"
            missing_required.append(str(workflow_name))
        elif len(matches) > 1:
            status = "non_unique"
            non_unique_required.append({"name": str(workflow_name), "files": matches})
        required_rows.append(
            {
                "name": str(workflow_name),
                "status": status,
                "files": matches,
            }
        )

    return {
        "version": 1,
        "workflow_root": str(workflow_root),
        "ci_watch_required_workflows": list(ci_watch_required_workflows),
        "workflow_count": len(workflows),
        "required_count": len(ci_watch_required_workflows),
        "duplicate_name_count": len(duplicates),
        "missing_required_count": len(missing_required),
        "non_unique_required_count": len(non_unique_required),
        "workflows": workflows,
        "name_to_files": name_to_files,
        "duplicates": duplicates,
        "required_workflow_mapping": required_rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Workflow Inventory Audit",
        "",
        f"- workflow_root: `{report.get('workflow_root')}`",
        f"- workflow_count: `{report.get('workflow_count')}`",
        f"- required_count: `{report.get('required_count')}`",
        f"- duplicate_name_count: `{report.get('duplicate_name_count')}`",
        f"- missing_required_count: `{report.get('missing_required_count')}`",
        f"- non_unique_required_count: `{report.get('non_unique_required_count')}`",
        "",
        "## Required Workflow Mapping",
        "",
    ]
    for row in report.get("required_workflow_mapping", []):
        files = ", ".join(row.get("files") or []) or "(none)"
        lines.append(f"- {row.get('name')}: status={row.get('status')} files={files}")

    duplicates = report.get("duplicates", [])
    lines.extend(["", "## Duplicate Workflow Names", ""])
    if duplicates:
        for row in duplicates:
            lines.append(f"- {row.get('name')}: {', '.join(row.get('files') or [])}")
    else:
        lines.append("- none")

    lines.extend(["", "## Workflow Files", ""])
    for row in report.get("workflows", []):
        lines.append(
            "- "
            f"{row.get('filename')}: "
            f"name={row.get('name') or '(missing)'} "
            f"parse_ok={row.get('parse_ok')} "
            f"workflow_dispatch={row.get('has_workflow_dispatch')} "
            f"inputs={','.join(row.get('dispatch_inputs') or [])}"
        )
    return "\n".join(lines) + "\n"


def _write_text(path_value: str, text: str) -> None:
    out = Path(path_value).expanduser()
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a read-only workflow inventory audit report."
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
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument("--output-md", default="", help="Optional output Markdown path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if yaml is None:
        print("error: PyYAML is required for workflow inventory report", flush=True)
        return 2

    report = build_report(
        workflow_root=Path(str(args.workflow_root)).expanduser(),
        ci_watch_required_workflows=_split_csv_items(str(args.ci_watch_required_workflows)),
    )
    report_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    report_md = render_markdown(report)

    if str(args.output_json or "").strip():
        _write_text(str(args.output_json), report_json)
        print(f"output_json={args.output_json}", flush=True)
    if str(args.output_md or "").strip():
        _write_text(str(args.output_md), report_md)
        print(f"output_md={args.output_md}", flush=True)

    print(report_md, end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
