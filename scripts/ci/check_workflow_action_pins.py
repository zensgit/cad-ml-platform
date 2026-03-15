#!/usr/bin/env python3
"""Validate that workflow actions are pinned to approved commit SHAs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_CHECKOUT_SHA = "de0fac2e4500dabe0009e67214ff5f5447ce83dd"
DEFAULT_SETUP_PYTHON_SHA = "a309ff8b426b58ec0e2a45f0f869d46889d02405"
DEFAULT_UPLOAD_ARTIFACT_SHA = "bbbca2ddaa5d8feaa63e36b76fdaad77386f024f"
DEFAULT_DOWNLOAD_ARTIFACT_SHA = "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"
DEFAULT_POLICY_JSON = "config/workflow_action_pin_policy.json"

_USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _normalize_token(value: str) -> str:
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


def _parse_policy_actions(policy_path: Path) -> Tuple[Dict[str, set[str]], str]:
    if not policy_path.exists() or not policy_path.is_file():
        return ({}, "policy_missing")

    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return ({}, "policy_json_decode_failed")

    if isinstance(payload, dict) and isinstance(payload.get("actions"), dict):
        raw_actions = payload.get("actions")
    elif isinstance(payload, dict):
        raw_actions = payload
    else:
        return ({}, "policy_invalid_payload")

    actions: Dict[str, set[str]] = {}
    for key, value in raw_actions.items():
        action = _normalize_token(str(key))
        if not action:
            continue
        values: List[str] = []
        if isinstance(value, list):
            values = [str(item) for item in value]
        elif isinstance(value, str):
            values = [value]
        for item in values:
            token = _normalize_token(item)
            if _HEX40_RE.fullmatch(token):
                actions.setdefault(action, set()).add(token)
    return (actions, "")


def _base_allowed_actions(
    *,
    checkout_sha: str,
    setup_python_sha: str,
    upload_artifact_sha: str,
    download_artifact_sha: str,
) -> Dict[str, set[str]]:
    return {
        "actions/checkout": {_normalize_token(checkout_sha)},
        "actions/setup-python": {_normalize_token(setup_python_sha)},
        "actions/upload-artifact": {_normalize_token(upload_artifact_sha)},
        "actions/download-artifact": {_normalize_token(download_artifact_sha)},
    }


def _scan_one_file(
    *,
    path: Path,
    allowed: Dict[str, set[str]],
    require_policy_for_all_external: bool,
) -> Tuple[List[Dict[str, Any]], set[str]]:
    violations: List[Dict[str, Any]] = []
    observed_actions: set[str] = set()

    text = path.read_text(encoding="utf-8", errors="ignore")
    for line_no, line in enumerate(text.splitlines(), start=1):
        target = _parse_uses_target(line)
        if not target or target.startswith("./") or "@" not in target:
            continue

        action_raw, ref = target.split("@", 1)
        action = _normalize_token(action_raw)
        observed_actions.add(action)

        if action not in allowed:
            if require_policy_for_all_external:
                violations.append(
                    {
                        "file": str(path),
                        "line": line_no,
                        "uses": target,
                        "action": action_raw,
                        "ref": ref,
                        "reason": "action_not_in_policy",
                        "expected_shas": [],
                    }
                )
            continue

        expected = sorted(allowed[action])
        normalized_ref = _normalize_token(ref)
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
                    "action": action_raw,
                    "ref": ref,
                    "reason": reason,
                    "expected_shas": expected,
                }
            )

    return (violations, observed_actions)


def scan_workflow_action_pins(
    *,
    workflows_dir: Path,
    checkout_sha: str,
    setup_python_sha: str,
    upload_artifact_sha: str = DEFAULT_UPLOAD_ARTIFACT_SHA,
    download_artifact_sha: str = DEFAULT_DOWNLOAD_ARTIFACT_SHA,
    policy_actions: Dict[str, set[str]] | None = None,
    require_policy_for_all_external: bool = False,
) -> Dict[str, Any]:
    allowed = _base_allowed_actions(
        checkout_sha=checkout_sha,
        setup_python_sha=setup_python_sha,
        upload_artifact_sha=upload_artifact_sha,
        download_artifact_sha=download_artifact_sha,
    )
    if policy_actions:
        for action, shas in policy_actions.items():
            key = _normalize_token(action)
            valid = {item for item in shas if _HEX40_RE.fullmatch(item)}
            if valid:
                allowed[key] = set(valid)

    files = list(_iter_workflow_files(workflows_dir))
    violations: List[Dict[str, Any]] = []
    observed: set[str] = set()
    for file_path in files:
        file_violations, file_observed = _scan_one_file(
            path=file_path,
            allowed=allowed,
            require_policy_for_all_external=require_policy_for_all_external,
        )
        violations.extend(file_violations)
        observed.update(file_observed)

    return {
        "status": "ok" if not violations else "error",
        "workflows_dir": str(workflows_dir),
        "files_scanned": len(files),
        "actions_observed_count": len(observed),
        "violations_count": len(violations),
        "require_policy_for_all_external": bool(require_policy_for_all_external),
        "allowed": {key: sorted(value) for key, value in sorted(allowed.items())},
        "violations": violations,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate workflow actions pinning policy. By default checks core actions; "
            "use --policy-json + --require-policy-for-all-external for strict mode."
        )
    )
    parser.add_argument("--workflows-dir", default=".github/workflows")
    parser.add_argument("--checkout-sha", default=DEFAULT_CHECKOUT_SHA)
    parser.add_argument("--setup-python-sha", default=DEFAULT_SETUP_PYTHON_SHA)
    parser.add_argument("--upload-artifact-sha", default=DEFAULT_UPLOAD_ARTIFACT_SHA)
    parser.add_argument(
        "--download-artifact-sha", default=DEFAULT_DOWNLOAD_ARTIFACT_SHA
    )
    parser.add_argument("--policy-json", default=DEFAULT_POLICY_JSON)
    parser.add_argument(
        "--require-policy-for-all-external",
        action="store_true",
        help="Fail when a workflow action is not listed in policy.",
    )
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

    policy_path = Path(str(args.policy_json)).expanduser()
    policy_actions, policy_error = _parse_policy_actions(policy_path)

    report = scan_workflow_action_pins(
        workflows_dir=Path(str(args.workflows_dir)).expanduser(),
        checkout_sha=str(args.checkout_sha),
        setup_python_sha=str(args.setup_python_sha),
        upload_artifact_sha=str(args.upload_artifact_sha),
        download_artifact_sha=str(args.download_artifact_sha),
        policy_actions=policy_actions,
        require_policy_for_all_external=bool(args.require_policy_for_all_external),
    )
    report["policy_json"] = str(policy_path)
    report["policy_loaded"] = bool(policy_actions)
    report["policy_error"] = policy_error

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_json:
        _write_output_json(str(args.output_json), report)

    if policy_error and bool(args.require_policy_for_all_external):
        return 1
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
