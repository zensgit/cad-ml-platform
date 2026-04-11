#!/usr/bin/env python3
"""Generate workflow action pin policy from current workflow files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


_USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _iter_workflow_files(workflows_dir: Path) -> List[Path]:
    if not workflows_dir.exists() or not workflows_dir.is_dir():
        return []
    return sorted(path for path in workflows_dir.glob("*.yml") if path.is_file())


def _parse_uses_target(line: str) -> str:
    match = _USES_RE.match(line)
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def collect_action_pin_policy(
    *,
    workflows_dir: Path,
    strict: bool,
) -> Tuple[Dict[str, Any], int]:
    actions: Dict[str, set[str]] = {}
    non_sha_refs: List[Dict[str, Any]] = []
    files = _iter_workflow_files(workflows_dir)

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            target = _parse_uses_target(line)
            if not target or target.startswith("./") or "@" not in target:
                continue
            action, ref = target.split("@", 1)
            if _HEX40_RE.fullmatch(ref):
                actions.setdefault(action, set()).add(str(ref).lower())
                continue
            non_sha_refs.append(
                {
                    "file": str(file_path),
                    "line": line_no,
                    "uses": target,
                    "action": action,
                    "ref": ref,
                }
            )

    payload = {
        "version": 1,
        "workflows_dir": str(workflows_dir),
        "actions": {key: sorted(value) for key, value in sorted(actions.items())},
        "non_sha_refs": non_sha_refs,
    }
    exit_code = 1 if strict and non_sha_refs else 0
    return payload, exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate workflow action pin policy from current workflows."
    )
    parser.add_argument("--workflows-dir", default=".github/workflows")
    parser.add_argument("--output-json", default="config/workflow_action_pin_policy.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when non-sha refs are found.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    payload, exit_code = collect_action_pin_policy(
        workflows_dir=Path(str(args.workflows_dir)).expanduser(),
        strict=bool(args.strict),
    )
    output_path = Path(str(args.output_json)).expanduser()
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
