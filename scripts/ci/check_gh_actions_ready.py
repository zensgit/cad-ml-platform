#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    name: str
    message: str


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _extract_error(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    output = f"{result.stderr or ''}\n{result.stdout or ''}".strip()
    if not output:
        return fallback
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return fallback

    keywords = (
        "failed to log in",
        "token",
        "re-authenticate",
        "gh auth login",
        "error connecting",
        "githubstatus.com",
    )
    for line in lines:
        if any(keyword in line.lower() for keyword in keywords):
            return line
    return lines[0]


def _check_gh_version() -> CheckResult:
    result = _run(["gh", "--version"])
    if result.returncode != 0:
        return CheckResult(
            ok=False,
            name="gh_version",
            message=f"gh is unavailable: {_extract_error(result, 'failed to run gh --version')}",
        )
    first_line = (result.stdout or "").splitlines()[0] if result.stdout else "gh available"
    return CheckResult(ok=True, name="gh_version", message=first_line)


def _check_gh_auth() -> CheckResult:
    result = _run(["gh", "auth", "status"])
    if result.returncode != 0:
        return CheckResult(
            ok=False,
            name="gh_auth",
            message=(
                "gh auth is not ready: "
                f"{_extract_error(result, 'failed to run gh auth status')} "
                "(fix: gh auth login -h github.com)"
            ),
        )
    return CheckResult(ok=True, name="gh_auth", message="gh auth status is ready")


def _check_actions_api() -> CheckResult:
    result = _run(["gh", "run", "list", "--limit", "1"])
    if result.returncode != 0:
        summary = _extract_error(result, "failed to query GitHub Actions runs")
        return CheckResult(
            ok=False,
            name="gh_actions_api",
            message=f"cannot access GitHub Actions API: {summary}",
        )
    return CheckResult(ok=True, name="gh_actions_api", message="GitHub Actions API is reachable")


def _write_json(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check local gh CLI readiness for GitHub Actions watcher workflows."
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write structured check results.",
    )
    parser.add_argument(
        "--skip-actions-api",
        action="store_true",
        help="Skip `gh run list` connectivity check (useful for auth-only diagnosis).",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Always exit with code 0 after checks, even when some checks fail.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    checks = [_check_gh_version(), _check_gh_auth()]
    if bool(args.skip_actions_api):
        checks.append(
            CheckResult(
                ok=True,
                name="gh_actions_api",
                message="skipped by --skip-actions-api",
            )
        )
    else:
        checks.append(_check_actions_api())

    for check in checks:
        prefix = "ok" if check.ok else "error"
        print(f"[{prefix}] {check.name}: {check.message}", flush=True)

    payload = {
        "version": 1,
        "skip_actions_api": bool(args.skip_actions_api),
        "allow_fail": bool(args.allow_fail),
        "checks": [
            {"name": check.name, "ok": check.ok, "message": check.message} for check in checks
        ],
        "ok": all(check.ok for check in checks),
    }
    json_out = str(args.json_out or "").strip()
    if json_out:
        _write_json(json_out, payload)
        print(f"json written: {json_out}", flush=True)

    if bool(args.allow_fail):
        return 0
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
