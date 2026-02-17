#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_ALERT_POLICY: Dict[str, Any] = {
    "recent_runs": 5,
    "default_key_threshold": 3,
    "key_thresholds": {
        "max_samples": 2,
    },
    "fail_on_alert": False,
}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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


def _parse_key_thresholds(values: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for raw in values:
        text = str(raw).strip()
        if not text or "=" not in text:
            continue
        key, count = text.split("=", 1)
        token = str(key).strip()
        if not token:
            continue
        out[token] = max(1, _safe_int(count, 0))
    return out


def _normalize_key_thresholds(value: Any) -> Dict[str, int]:
    if isinstance(value, dict):
        out: Dict[str, int] = {}
        for key, threshold in value.items():
            token = str(key).strip()
            if not token:
                continue
            out[token] = max(1, _safe_int(threshold, 0))
        return out
    if isinstance(value, list):
        return _parse_key_thresholds([str(item) for item in value])
    if isinstance(value, str):
        return _parse_key_thresholds([str(value)])
    return {}


def _resolve_alert_policy(
    *,
    config_payload: Dict[str, Any],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    recent_runs = _safe_int(
        (
            cli_overrides.get("recent_runs")
            if cli_overrides.get("recent_runs") is not None
            else config_payload.get("recent_runs")
        ),
        _safe_int(DEFAULT_ALERT_POLICY["recent_runs"], 5),
    )
    default_key_threshold = _safe_int(
        (
            cli_overrides.get("default_key_threshold")
            if cli_overrides.get("default_key_threshold") is not None
            else config_payload.get("default_key_threshold")
        ),
        _safe_int(DEFAULT_ALERT_POLICY["default_key_threshold"], 3),
    )

    key_thresholds = _normalize_key_thresholds(
        config_payload.get("key_thresholds", DEFAULT_ALERT_POLICY.get("key_thresholds", {}))
    )
    cli_key_thresholds = _parse_key_thresholds(
        [str(item) for item in list(cli_overrides.get("key_threshold") or [])]
    )
    if cli_key_thresholds:
        key_thresholds.update(cli_key_thresholds)

    fail_on_alert = _safe_bool(
        config_payload.get("fail_on_alert"),
        _safe_bool(DEFAULT_ALERT_POLICY.get("fail_on_alert"), False),
    )
    fail_on_alert_cli = str(cli_overrides.get("fail_on_alert", "auto")).strip().lower()
    if fail_on_alert_cli in {"true", "false"}:
        fail_on_alert = fail_on_alert_cli == "true"

    return {
        "recent_runs": max(1, int(recent_runs)),
        "default_key_threshold": max(1, int(default_key_threshold)),
        "key_thresholds": key_thresholds,
        "fail_on_alert": bool(fail_on_alert),
    }


def evaluate_alerts(
    *,
    history: List[Dict[str, Any]],
    recent_runs: int,
    default_key_threshold: int,
    key_thresholds: Dict[str, int],
) -> Dict[str, Any]:
    rows = [row for row in history if isinstance(row, dict)]
    recent = rows[-max(1, int(recent_runs)) :] if rows else []

    key_totals: Dict[str, int] = {}
    for row in recent:
        one = _as_dict(row.get("drift_key_counts"))
        for key, value in one.items():
            token = str(key).strip()
            if not token:
                continue
            key_totals[token] = key_totals.get(token, 0) + max(0, _safe_int(value, 0))

    alerts: List[Dict[str, Any]] = []
    for key, count in sorted(key_totals.items(), key=lambda item: (-item[1], item[0])):
        threshold = int(key_thresholds.get(key, int(default_key_threshold)))
        if int(count) >= int(threshold):
            alerts.append(
                {
                    "key": key,
                    "count": int(count),
                    "threshold": int(threshold),
                    "message": f"context drift key '{key}' count {int(count)} >= threshold {int(threshold)}",
                }
            )

    return {
        "status": "alerted" if alerts else "clear",
        "history_size": len(rows),
        "recent_runs": len(recent),
        "recent_run_numbers": [
            str(_as_dict(row).get("run_number", "")).strip() for row in recent
        ],
        "default_key_threshold": int(default_key_threshold),
        "key_thresholds": key_thresholds,
        "key_totals": key_totals,
        "alerts": alerts,
    }


def build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    policy = _as_dict(report.get("policy_source"))
    resolved_policy = _as_dict(policy.get("resolved_policy"))
    alerts = [
        _as_dict(item)
        for item in _as_list(report.get("alerts"))
        if _as_dict(item)
    ]
    key_totals = _as_dict(report.get("key_totals"))
    return {
        "status": str(report.get("status", "clear")),
        "history_entries": _safe_int(report.get("history_size"), 0),
        "recent_window": _safe_int(report.get("recent_runs"), 0),
        "alert_count": len(alerts),
        "key_totals": {
            str(key): _safe_int(value, 0)
            for key, value in key_totals.items()
            if str(key).strip()
        },
        "rows": [
            {
                "key": str(item.get("key", "")),
                "count": _safe_int(item.get("count"), 0),
                "threshold": _safe_int(item.get("threshold"), 0),
                "message": str(item.get("message", "")),
            }
            for item in alerts
            if str(item.get("key", "")).strip()
        ],
        "policy_source": {
            "config": str(policy.get("config", "")),
            "config_section": str(policy.get("config_section", "")),
            "config_loaded": bool(policy.get("config_loaded", False)),
            "resolved_policy": {
                "recent_runs": _safe_int(resolved_policy.get("recent_runs"), 0),
                "default_key_threshold": _safe_int(
                    resolved_policy.get("default_key_threshold"), 0
                ),
                "fail_on_alert": _safe_bool(resolved_policy.get("fail_on_alert"), False),
                "key_thresholds": {
                    str(key): _safe_int(value, 0)
                    for key, value in _as_dict(resolved_policy.get("key_thresholds")).items()
                    if str(key).strip()
                },
            },
            "cli_overrides": _as_dict(policy.get("cli_overrides")),
        },
    }


def build_markdown(report: Dict[str, Any], title: str) -> str:
    status = str(report.get("status", "clear"))
    alerts = _as_list(report.get("alerts"))
    key_totals = _as_dict(report.get("key_totals"))
    policy_source = _as_dict(report.get("policy_source"))
    resolved_policy = _as_dict(policy_source.get("resolved_policy"))

    out: List[str] = []
    out.append(f"### {title}")
    out.append("")
    out.append("| Check | Value |")
    out.append("|---|---|")
    out.append(f"| Status | `{status}` |")
    out.append(f"| History size | `{_safe_int(report.get('history_size'), 0)}` |")
    out.append(f"| Recent runs | `{_safe_int(report.get('recent_runs'), 0)}` |")
    out.append(
        f"| Default threshold | `{_safe_int(report.get('default_key_threshold'), 0)}` |"
    )
    out.append(f"| Alert count | `{len(alerts)}` |")
    out.append(
        "| Policy source | "
        f"`config={policy_source.get('config', '')}, "
        f"loaded={bool(policy_source.get('config_loaded', False))}, "
        f"resolved_keys={len(_as_dict(resolved_policy.get('key_thresholds')).keys())}` |"
    )
    out.append("")

    if key_totals:
        out.append("| Drift key | Total |")
        out.append("|---|---:|")
        for key, value in sorted(
            [(str(k), _safe_int(v, 0)) for k, v in key_totals.items()],
            key=lambda item: (-item[1], item[0]),
        ):
            out.append(f"| `{key}` | {int(value)} |")
        out.append("")

    if alerts:
        out.append("Alerts:")
        out.append("```text")
        for item in alerts:
            one = _as_dict(item)
            out.append(str(one.get("message", "")))
        out.append("```")
        out.append("")
    else:
        out.append("No drift alerts in recent window.")
        out.append("")

    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Graph2D context drift key totals against recent-window thresholds."
    )
    parser.add_argument(
        "--config",
        default="config/graph2d_context_drift_alerts.yaml",
        help="YAML config path for drift alert policy.",
    )
    parser.add_argument(
        "--config-section",
        default="graph2d_context_drift_alerts",
        help="Config section name in yaml.",
    )
    parser.add_argument("--history-json", required=True, help="History json path")
    parser.add_argument(
        "--recent-runs", type=int, default=None, help="Recent run window size."
    )
    parser.add_argument(
        "--default-key-threshold",
        type=int,
        default=None,
        help="Default alert threshold for drift key totals in recent window.",
    )
    parser.add_argument(
        "--key-threshold",
        action="append",
        default=[],
        help="Per-key threshold override, format key=count; can be repeated.",
    )
    parser.add_argument("--title", default="Graph2D Context Drift Alerts")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument(
        "--fail-on-alert",
        choices=["auto", "true", "false"],
        default="auto",
        help="Exit non-zero when alerts exist (auto uses config).",
    )
    args = parser.parse_args()

    history_payload = _read_json(Path(str(args.history_json)))
    history_rows = _as_list(history_payload)
    config_payload = _load_yaml_defaults(str(args.config), str(args.config_section))
    policy = _resolve_alert_policy(
        config_payload=config_payload,
        cli_overrides={
            "recent_runs": args.recent_runs,
            "default_key_threshold": args.default_key_threshold,
            "key_threshold": list(args.key_threshold or []),
            "fail_on_alert": args.fail_on_alert,
        },
    )
    report = evaluate_alerts(
        history=[_as_dict(row) for row in history_rows],
        recent_runs=int(policy["recent_runs"]),
        default_key_threshold=int(policy["default_key_threshold"]),
        key_thresholds=_as_dict(policy.get("key_thresholds")),
    )
    report["policy_source"] = {
        "config": str(args.config),
        "config_section": str(args.config_section),
        "config_loaded": bool(config_payload),
        "resolved_policy": policy,
        "cli_overrides": {
            key: value
            for key, value in {
                "recent_runs": args.recent_runs,
                "default_key_threshold": args.default_key_threshold,
                "key_threshold": (
                    list(args.key_threshold or []) if list(args.key_threshold or []) else None
                ),
                "fail_on_alert": (
                    args.fail_on_alert if str(args.fail_on_alert) != "auto" else None
                ),
            }.items()
            if value is not None
        },
    }
    report["summary"] = build_summary(report)

    json_text = json.dumps(report, ensure_ascii=False, indent=2)
    md_text = build_markdown(report, str(args.title))

    if str(args.output_json).strip():
        out_json = Path(str(args.output_json))
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json_text + "\n", encoding="utf-8")
    if str(args.output_md).strip():
        out_md = Path(str(args.output_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md_text + "\n", encoding="utf-8")

    print(json_text)
    if bool(policy.get("fail_on_alert", False)) and _as_list(report.get("alerts")):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
