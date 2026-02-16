#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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


def build_markdown(report: Dict[str, Any], title: str) -> str:
    status = str(report.get("status", "clear"))
    alerts = _as_list(report.get("alerts"))
    key_totals = _as_dict(report.get("key_totals"))

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
    parser.add_argument("--history-json", required=True, help="History json path")
    parser.add_argument(
        "--recent-runs", type=int, default=5, help="Recent run window size (default: 5)"
    )
    parser.add_argument(
        "--default-key-threshold",
        type=int,
        default=3,
        help="Default alert threshold for drift key totals in recent window (default: 3)",
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
        action="store_true",
        help="Exit non-zero when alerts exist (default: non-blocking).",
    )
    args = parser.parse_args()

    history_payload = _read_json(Path(str(args.history_json)))
    history_rows = _as_list(history_payload)
    report = evaluate_alerts(
        history=[_as_dict(row) for row in history_rows],
        recent_runs=max(1, int(args.recent_runs)),
        default_key_threshold=max(1, int(args.default_key_threshold)),
        key_thresholds=_parse_key_thresholds([str(item) for item in list(args.key_threshold)]),
    )

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
    if bool(args.fail_on_alert) and _as_list(report.get("alerts")):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
