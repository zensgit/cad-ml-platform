#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_KEY_COUNTS_POLICY: Dict[str, Any] = {
    "recent_runs": 5,
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_yaml_defaults(config_path: str, section: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    data = payload.get(section)
    if not isinstance(data, dict):
        return {}
    return data


def _resolve_recent_runs(config_payload: Dict[str, Any], cli_recent_runs: int | None) -> int:
    if cli_recent_runs is not None:
        return max(1, int(cli_recent_runs))
    candidate = config_payload.get("recent_runs", DEFAULT_KEY_COUNTS_POLICY.get("recent_runs", 5))
    return max(1, _safe_int(candidate, 5))


def _extract_context_diff_keys(report: Dict[str, Any]) -> List[str]:
    meta = _as_dict(report.get("baseline_metadata"))
    diff = _as_dict(meta.get("context_diff"))
    return sorted([str(key) for key in diff.keys() if str(key).strip()])


def _report_row(path: str, report: Dict[str, Any]) -> Dict[str, Any]:
    meta = _as_dict(report.get("baseline_metadata"))
    thresholds = _as_dict(report.get("thresholds"))
    warnings = _as_list(report.get("warnings"))
    failures = _as_list(report.get("failures"))
    diff_keys = _extract_context_diff_keys(report)
    return {
        "path": path,
        "channel": str(report.get("channel", "")),
        "status": str(report.get("status", "")),
        "context_mode": str(thresholds.get("context_mismatch_mode", "")),
        "context_match": meta.get("context_match"),
        "diff_keys": diff_keys,
        "warnings": len(warnings),
        "failures": len(failures),
    }


def build_markdown(
    *,
    reports: List[Tuple[str, Dict[str, Any]]],
    title: str,
    policy_source: Dict[str, Any] | None = None,
) -> str:
    out: List[str] = []
    out.append(f"### {title}")
    out.append("")

    rows = [_report_row(path, payload) for path, payload in reports if payload]
    policy = _as_dict(policy_source)
    resolved_policy = _as_dict(policy.get("resolved_policy"))
    out.append("| Check | Value |")
    out.append("|---|---|")
    out.append(f"| Report count | `{len(rows)}` |")
    out.append(
        "| Policy source | "
        f"`config={policy.get('config', '')}, "
        f"loaded={bool(policy.get('config_loaded', False))}, "
        f"resolved_recent_runs={_safe_int(resolved_policy.get('recent_runs'), 0)}` |"
    )
    out.append("")

    if not rows:
        out.append("No regression reports found.")
        out.append("")
        return "\n".join(out)

    out.append("| Report | Channel | Status | Context mode | Context match | Diff keys | Warn | Fail |")
    out.append("|---|---|---|---|---|---|---:|---:|")
    for row in rows:
        diff_keys = ",".join(row["diff_keys"]) if row["diff_keys"] else "-"
        out.append(
            "| "
            f"`{Path(str(row['path'])).name}` | "
            f"`{row['channel']}` | "
            f"`{row['status']}` | "
            f"`{row['context_mode']}` | "
            f"`{row['context_match']}` | "
            f"`{diff_keys}` | "
            f"{int(row['warnings'])} | "
            f"{int(row['failures'])} |"
        )

    key_counts: Dict[str, int] = {}
    for row in rows:
        for key in row["diff_keys"]:
            key_counts[key] = key_counts.get(key, 0) + 1

    out.append("")
    if not key_counts:
        out.append("Context drift key counts: none.")
        out.append("")
        return "\n".join(out)

    out.append("| Drift key | Count |")
    out.append("|---|---:|")
    for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0])):
        out.append(f"| `{key}` | {int(count)} |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render Graph2D context drift key counts from one or more regression reports."
    )
    parser.add_argument(
        "--config",
        default="config/graph2d_context_drift_alerts.yaml",
        help="YAML config path for drift key count policy metadata.",
    )
    parser.add_argument(
        "--config-section",
        default="graph2d_context_drift_alerts",
        help="Config section name in yaml.",
    )
    parser.add_argument(
        "--recent-runs",
        type=int,
        default=None,
        help="Recent run policy override for metadata output.",
    )
    parser.add_argument(
        "--report-json",
        action="append",
        default=[],
        help="Regression report json path. Can be passed multiple times.",
    )
    parser.add_argument("--title", required=True, help="Section title")
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional markdown output path (stdout when omitted).",
    )
    args = parser.parse_args()

    report_paths = [str(item).strip() for item in list(args.report_json or []) if str(item).strip()]
    reports: List[Tuple[str, Dict[str, Any]]] = []
    for path in report_paths:
        payload = _read_json(Path(path))
        reports.append((path, payload))

    config_payload = _load_yaml_defaults(str(args.config), str(args.config_section))
    recent_runs = _resolve_recent_runs(config_payload, args.recent_runs)
    policy_source = {
        "config": str(args.config),
        "config_section": str(args.config_section),
        "config_loaded": bool(config_payload),
        "resolved_policy": {
            "recent_runs": int(recent_runs),
        },
        "cli_overrides": {
            key: value
            for key, value in {
                "recent_runs": args.recent_runs,
            }.items()
            if value is not None
        },
    }

    markdown = build_markdown(
        reports=reports,
        title=str(args.title),
        policy_source=policy_source,
    )
    if str(args.output_md).strip():
        out_path = Path(str(args.output_md))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
