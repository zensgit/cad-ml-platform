#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on runtime environment
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class WorkflowCheckResult:
    path: str
    mode: str
    ok: bool
    message: str


def _extract_short_error(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    output = (result.stderr or result.stdout or "").strip()
    if not output:
        return fallback
    return output.splitlines()[0]


def _is_auth_or_token_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "failed to log in" in text
        or "authentication" in text
        or "token" in text
        or "gh auth login" in text
        or "http 401" in text
    )


def _is_missing_workflow_on_ref_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "could not find workflow file" in text
        and "try specifying a different ref" in text
    ) or ("workflow was not found" in text and "ref" in text)


def _check_yaml_parse(path: Path) -> WorkflowCheckResult:
    if yaml is None:
        return WorkflowCheckResult(
            path=str(path),
            mode="yaml",
            ok=False,
            message="PyYAML is required for yaml parsing mode",
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return WorkflowCheckResult(path=str(path), mode="yaml", ok=False, message=str(exc))
    except yaml.YAMLError as exc:
        return WorkflowCheckResult(path=str(path), mode="yaml", ok=False, message=str(exc))
    if not isinstance(payload, dict):
        return WorkflowCheckResult(
            path=str(path),
            mode="yaml",
            ok=False,
            message="top-level payload must be an object",
        )
    has_on = "on" in payload or True in payload
    if "name" not in payload or not has_on:
        return WorkflowCheckResult(
            path=str(path),
            mode="yaml",
            ok=False,
            message="missing required top-level keys: name/on",
        )
    return WorkflowCheckResult(path=str(path), mode="yaml", ok=True, message="ok")


def _check_with_gh(path: Path, *, ref: str) -> WorkflowCheckResult:
    command = [
        "gh",
        "workflow",
        "view",
        path.as_posix(),
        "--ref",
        str(ref or "HEAD"),
        "--yaml",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return WorkflowCheckResult(
            path=str(path),
            mode="gh",
            ok=False,
            message=_extract_short_error(result, "gh workflow view failed"),
        )
    return WorkflowCheckResult(path=str(path), mode="gh", ok=True, message="ok")


def _is_gh_available() -> bool:
    result = subprocess.run(
        ["gh", "--version"], capture_output=True, text=True, check=False
    )
    return result.returncode == 0


def _collect_workflow_files(glob_pattern: str) -> list[Path]:
    files = sorted(Path(".").glob(glob_pattern))
    return [path for path in files if path.is_file()]


def _to_summary_payload(
    *,
    ref: str,
    requested_mode: str,
    mode_used: str,
    fallback_reason: str,
    results: Sequence[WorkflowCheckResult],
) -> dict[str, Any]:
    return {
        "version": 1,
        "ref": ref,
        "requested_mode": requested_mode,
        "mode_used": mode_used,
        "fallback_reason": fallback_reason,
        "count": len(results),
        "failed_count": sum(1 for row in results if not row.ok),
        "results": [
            {
                "path": row.path,
                "mode": row.mode,
                "ok": row.ok,
                "message": row.message,
            }
            for row in results
        ],
    }


def _write_json(path_value: str, payload: dict[str, Any]) -> None:
    out = Path(path_value).expanduser()
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check GitHub workflow file health (gh parser + YAML parser)."
    )
    parser.add_argument(
        "--glob",
        default=".github/workflows/*.yml",
        help="Workflow file glob pattern (default: .github/workflows/*.yml).",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git ref used for gh workflow parser checks (default: HEAD).",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "gh", "yaml"),
        default="auto",
        help="Validation mode. auto prefers gh and falls back to yaml on auth/tooling issues.",
    )
    parser.add_argument(
        "--summary-json-out",
        default="",
        help="Optional path to write machine-readable summary JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    files = _collect_workflow_files(str(args.glob))
    if not files:
        print(f"error: no workflow files matched glob: {args.glob}", flush=True)
        return 2

    requested_mode = str(args.mode)
    mode_used = requested_mode
    fallback_reason = ""

    results: list[WorkflowCheckResult] = []
    if requested_mode == "yaml":
        results = [_check_yaml_parse(path) for path in files]
    elif requested_mode == "gh":
        if not _is_gh_available():
            print("error: gh is not available in --mode gh", flush=True)
            return 2
        results = [_check_with_gh(path, ref=str(args.ref)) for path in files]
    else:
        if not _is_gh_available():
            mode_used = "yaml"
            fallback_reason = "gh_unavailable"
            results = [_check_yaml_parse(path) for path in files]
        else:
            gh_results = [_check_with_gh(path, ref=str(args.ref)) for path in files]
            gh_errors = [row for row in gh_results if not row.ok]
            can_fallback = (
                gh_errors
                and all(
                    _is_auth_or_token_error(row.message)
                    or _is_missing_workflow_on_ref_error(row.message)
                    for row in gh_errors
                )
            )
            if can_fallback:
                mode_used = "yaml"
                if any(_is_missing_workflow_on_ref_error(row.message) for row in gh_errors):
                    fallback_reason = "gh_ref_unresolvable_for_local_head"
                else:
                    fallback_reason = "gh_auth_or_token_error"
                print(
                    "warning: gh parser unavailable for current context; fallback to yaml parser",
                    flush=True,
                )
                results = [_check_yaml_parse(path) for path in files]
            else:
                results = gh_results

    failed = [row for row in results if not row.ok]
    for row in results:
        status = "ok" if row.ok else "fail"
        print(f"[{status}] {row.mode} {row.path} - {row.message}", flush=True)

    summary_out = str(args.summary_json_out or "").strip()
    if summary_out:
        payload = _to_summary_payload(
            ref=str(args.ref),
            requested_mode=requested_mode,
            mode_used=mode_used,
            fallback_reason=fallback_reason,
            results=results,
        )
        _write_json(summary_out, payload)
        print(f"summary_json={summary_out}", flush=True)

    if failed:
        print(f"error: workflow health check failed for {len(failed)} file(s)", flush=True)
        return 1
    print(f"ok: workflow health check passed for {len(results)} file(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
