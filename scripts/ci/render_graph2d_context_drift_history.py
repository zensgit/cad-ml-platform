#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_HISTORY_POLICY: Dict[str, Any] = {
    "recent_runs": 10,
}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
    candidate = config_payload.get("recent_runs", DEFAULT_HISTORY_POLICY.get("recent_runs", 10))
    return max(1, _safe_int(candidate, 10))


def build_markdown(
    *,
    history: List[Dict[str, Any]],
    title: str,
    recent_runs: int = 10,
    policy_source: Dict[str, Any] | None = None,
) -> str:
    out: List[str] = []
    out.append(f"### {title}")
    out.append("")

    policy = _as_dict(policy_source)
    resolved_policy = _as_dict(policy.get("resolved_policy"))
    recent_window = max(1, _safe_int(recent_runs, 10))

    out.append("| Check | Value |")
    out.append("|---|---|")
    out.append(f"| History entries | `{len(history)}` |")
    out.append(f"| Recent runs policy | `{recent_window}` |")
    out.append(
        "| Policy source | "
        f"`config={policy.get('config', '')}, "
        f"loaded={bool(policy.get('config_loaded', False))}, "
        f"resolved_recent_runs={_safe_int(resolved_policy.get('recent_runs'), recent_window)}` |"
    )
    out.append("")

    if not history:
        out.append("No context drift history found.")
        out.append("")
        return "\n".join(out)

    out.append("| Run | Status | Warnings | Failures | Drift keys |")
    out.append("|---|---|---:|---:|---|")
    for item in history:
        one = _as_dict(item)
        run_no = str(one.get("run_number", "")).strip() or str(one.get("run_id", "")).strip() or "unknown"
        status = str(one.get("status", "")).strip() or "unknown"
        warning_count = _safe_int(one.get("warning_count"), 0)
        failure_count = _safe_int(one.get("failure_count"), 0)
        key_counts = _as_dict(one.get("drift_key_counts"))
        keys_sorted = sorted(
            [(str(key), _safe_int(value, 0)) for key, value in key_counts.items() if str(key).strip()],
            key=lambda kv: (-kv[1], kv[0]),
        )
        key_text = ",".join([f"{key}:{count}" for key, count in keys_sorted]) if keys_sorted else "-"
        out.append(
            f"| `#{run_no}` | `{status}` | {warning_count} | {failure_count} | `{key_text}` |"
        )

    recent = history[-recent_window:] if len(history) > recent_window else list(history)
    aggregate: Dict[str, int] = {}
    for item in recent:
        key_counts = _as_dict(_as_dict(item).get("drift_key_counts"))
        for key, value in key_counts.items():
            token = str(key).strip()
            if not token:
                continue
            count = _safe_int(value, 0)
            if count <= 0:
                continue
            aggregate[token] = aggregate.get(token, 0) + count

    out.append("")
    out.append(f"Recent window size: `{len(recent)}`")
    out.append("")
    if not aggregate:
        out.append("Recent drift key totals: none.")
        out.append("")
        return "\n".join(out)

    out.append("| Drift key (recent) | Total |")
    out.append("|---|---:|")
    for key, value in sorted(aggregate.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f"| `{key}` | {value} |")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render Graph2D context drift history markdown."
    )
    parser.add_argument(
        "--config",
        default="config/graph2d_context_drift_alerts.yaml",
        help="YAML config path for history render policy.",
    )
    parser.add_argument(
        "--config-section",
        default="graph2d_context_drift_alerts",
        help="Config section name in yaml.",
    )
    parser.add_argument("--history-json", required=True, help="Path to history json")
    parser.add_argument("--title", required=True, help="Section title")
    parser.add_argument(
        "--recent-runs",
        type=int,
        default=None,
        help="Recent run window size override for aggregate totals.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional markdown output path (stdout when omitted).",
    )
    args = parser.parse_args()

    payload = _read_json(Path(str(args.history_json)))
    history_raw = _as_list(payload)
    history: List[Dict[str, Any]] = []
    for item in history_raw:
        one = _as_dict(item)
        if one:
            history.append(one)

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
        history=history,
        title=str(args.title),
        recent_runs=recent_runs,
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
