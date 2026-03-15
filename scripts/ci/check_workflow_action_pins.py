#!/usr/bin/env python3
"""Validate that critical GitHub Actions use pinned commit SHAs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CHECKOUT_SHA = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
DEFAULT_SETUP_PYTHON_SHA = "a309ff8b426b58ec0e2a45f0f869d46889d02405"

_USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _normalize_sha(value: str) -> str:
    return str(value or "").strip().lower()


def _iter_workflow_files(workflows_dir: Path) -> Iterable[Path]:
    if not workflows_dir.exists() or not workflows_dir.is_dir():
        return []
    return sorted(path for path in workflows_dir.glob("*.yml") if path.is_file())


def _parse_uses_target(line: str) -> str:
    match = _USES_RE.match(line)
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def _scan_one_file(path: Path, allowed: Dict[str, set[str]]) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line_no, line in enumerate(text.splitlines(), start=1):
        target = _parse_uses_target(line)
        if not target or "@" not in target:
            continue
        action, ref = target.split("@", 1)
        if action not in allowed:
            continue

        normalized_ref = _normalize_sha(ref)
        expected = sorted(allowed[action])
        reason = ""
        if ref.startswith("${{"):
            reason = "dynamic_ref_not_allowed"
        elif _HEX40_RE.fullmatch(ref):
            if normalized_ref not in allowed[action]:
                reason = "unexpected_sha"
        elif str(ref).lower().startswith("v"):
            reason = "tag_ref_not_allowed"
        else:
            reason = "non_sha_ref_not_allowed"

        if reason:
            violations.append(
                {
                    "file": str(path),
                    "line": line_no,
                    "uses": target,
                    "action": action,
                    "ref": ref,
                    "reason": reason,
                    "expected_shas": expected,
                }
            )
    return violations


def scan_workflow_action_pins(
    *,
    workflows_dir: Path,
    checkout_sha: str,
    setup_python_sha: str,
) -> Dict[str, Any]:
    allowed = {
        "actions/checkout": {_normalize_sha(checkout_sha)},
        "actions/setup-python": {_normalize_sha(setup_python_sha)},
    }
    files = list(_iter_workflow_files(workflows_dir))
    violations: List[Dict[str, Any]] = []
    for file_path in files:
        violations.extend(_scan_one_file(file_path, allowed))
    return {
        "status": "ok" if not violations else "error",
        "workflows_dir": str(workflows_dir),
        "files_scanned": len(files),
        "violations_count": len(violations),
        "allowed": {key: sorted(value) for key, value in allowed.items()},
        "violations": violations,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that actions/checkout and actions/setup-python are pinned "
            "to approved commit SHAs in workflow files."
        )
    )
    parser.add_argument("--workflows-dir", default=".github/workflows")
    parser.add_argument("--checkout-sha", default=DEFAULT_CHECKOUT_SHA)
    parser.add_argument("--setup-python-sha", default=DEFAULT_SETUP_PYTHON_SHA)
    parser.add_argument("--output-json", default="")
    return parser


def _write_output_json(path_value: str, payload: Dict[str, Any]) -> None:
    path = Path(path_value).expanduser()
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = scan_workflow_action_pins(
        workflows_dir=Path(str(args.workflows_dir)).expanduser(),
        checkout_sha=str(args.checkout_sha),
        setup_python_sha=str(args.setup_python_sha),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        _write_output_json(str(args.output_json), report)

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
