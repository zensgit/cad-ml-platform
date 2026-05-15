#!/usr/bin/env python3
"""Gate release labels against the forward scorecard status."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

DEFAULT_ALLOWED_STATUSES = {"release_ready", "benchmark_ready_with_gap"}
DEFAULT_RELEASE_LABEL_PREFIXES = ("release", "production", "deploy", "tag")
LABEL_SPLIT_RE = re.compile(r"[,;\n\r\t ]+")


def _load_json(path_text: str) -> Dict[str, Any]:
    path = Path(path_text).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Forward scorecard not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid forward scorecard JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON in forward scorecard: {path}")
    return payload


def _split_labels(values: Iterable[str]) -> List[str]:
    labels: List[str] = []
    seen = set()
    for value in values:
        for token in LABEL_SPLIT_RE.split(str(value or "")):
            label = token.strip()
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
    return labels


def _event_labels(path_text: str) -> List[str]:
    if not path_text:
        return []
    path = Path(path_text).expanduser()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    labels: List[str] = []

    def add_label(value: Any) -> None:
        if isinstance(value, dict):
            value = value.get("name")
        if isinstance(value, str) and value.strip():
            labels.append(value.strip())

    for key in ("label",):
        add_label(payload.get(key))
    for container_key in ("pull_request", "issue"):
        container = payload.get(container_key)
        if isinstance(container, dict):
            for item in container.get("labels") or []:
                add_label(item)
    for item in payload.get("labels") or []:
        add_label(item)

    ref_type = str(payload.get("ref_type") or os.environ.get("GITHUB_REF_TYPE") or "")
    ref_name = str(payload.get("ref") or os.environ.get("GITHUB_REF_NAME") or "")
    if ref_type == "tag" and ref_name:
        labels.append(f"tag:{ref_name}")
    elif ref_name.startswith("refs/tags/"):
        labels.append(f"tag:{ref_name.removeprefix('refs/tags/')}")

    env_ref_type = str(os.environ.get("GITHUB_REF_TYPE") or "")
    env_ref_name = str(os.environ.get("GITHUB_REF_NAME") or "")
    if env_ref_type == "tag" and env_ref_name:
        labels.append(f"tag:{env_ref_name}")

    return _split_labels(labels)


def _is_true(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _release_labels(labels: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    release_labels: List[str] = []
    prefix_tokens = [prefix.strip().lower() for prefix in prefixes if prefix.strip()]
    for label in labels:
        token = label.strip().lower()
        for prefix in prefix_tokens:
            if token == prefix or token.startswith((f"{prefix}:", f"{prefix}/")):
                release_labels.append(label)
                break
            if token.startswith((f"{prefix}-", f"{prefix}_", f"{prefix}.")):
                release_labels.append(label)
                break
    return release_labels


def _write_json(path_text: str, payload: Dict[str, Any]) -> None:
    if not path_text:
        return
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_github_outputs(payload: Dict[str, Any]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    fields = {
        "gate_applicable": str(payload["gate_applicable"]).lower(),
        "should_fail": str(payload["should_fail"]).lower(),
        "overall_status": payload["overall_status"],
        "release_labels": ",".join(payload["release_labels"]),
        "reason": payload["reason"],
    }
    with open(output_path, "a", encoding="utf-8") as handle:
        for key, value in fields.items():
            handle.write(f"{key}={value}\n")


def evaluate_gate(
    *,
    scorecard: Dict[str, Any],
    labels: Sequence[str],
    release_label_prefixes: Sequence[str],
    allowed_statuses: Sequence[str],
    require_release: bool,
    scorecard_path: str,
) -> Dict[str, Any]:
    """Return a serializable release-gate decision."""
    overall_status = str(scorecard.get("overall_status") or "unknown")
    release_labels = _release_labels(labels, release_label_prefixes)
    gate_applicable = require_release or bool(release_labels)
    allowed = {status.strip() for status in allowed_statuses if status.strip()}
    status_allowed = overall_status in allowed
    should_fail = gate_applicable and not status_allowed
    if not gate_applicable:
        reason = "No release label or required-release mode was detected."
    elif status_allowed:
        reason = f"Forward scorecard status is allowed for release: {overall_status}."
    else:
        reason = f"Forward scorecard status is not allowed for release: {overall_status}."
    return {
        "scorecard_path": scorecard_path,
        "overall_status": overall_status,
        "allowed_statuses": sorted(allowed),
        "gate_applicable": gate_applicable,
        "should_fail": should_fail,
        "release_labels": release_labels,
        "all_labels": list(labels),
        "reason": reason,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail release-labelled runs when the forward scorecard is not release-ready."
    )
    parser.add_argument("--scorecard", required=True)
    parser.add_argument("--labels", action="append", default=[])
    parser.add_argument(
        "--github-event-path",
        default=os.environ.get("GITHUB_EVENT_PATH", ""),
    )
    parser.add_argument(
        "--release-label-prefix",
        action="append",
        default=[],
        help="Release label prefix. Repeat to allow multiple prefixes.",
    )
    parser.add_argument(
        "--allowed-status",
        action="append",
        default=[],
        help="Allowed scorecard status. Repeat to allow multiple statuses.",
    )
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    scorecard = _load_json(args.scorecard)
    env_labels = _split_labels([os.environ.get("FORWARD_SCORECARD_RELEASE_LABELS", "")])
    labels = _split_labels([*args.labels, *env_labels, *_event_labels(args.github_event_path)])
    prefixes = args.release_label_prefix or list(DEFAULT_RELEASE_LABEL_PREFIXES)
    allowed_statuses = args.allowed_status or sorted(DEFAULT_ALLOWED_STATUSES)
    require_release = args.require_release or _is_true(
        os.environ.get("FORWARD_SCORECARD_RELEASE_GATE_REQUIRE_RELEASE", "")
    )
    payload = evaluate_gate(
        scorecard=scorecard,
        labels=labels,
        release_label_prefixes=prefixes,
        allowed_statuses=allowed_statuses,
        require_release=require_release,
        scorecard_path=args.scorecard,
    )
    _write_json(args.output_json, payload)
    _write_github_outputs(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload["should_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
