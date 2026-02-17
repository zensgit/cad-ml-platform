#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _top_key(key_counts: Dict[str, Any]) -> Dict[str, Any]:
    pairs: List[Tuple[str, int]] = []
    for key, value in key_counts.items():
        token = str(key).strip()
        if not token:
            continue
        pairs.append((token, _safe_int(value, 0)))
    if not pairs:
        return {"key": "", "count": 0}
    key, count = sorted(pairs, key=lambda item: (-item[1], item[0]))[0]
    return {"key": key, "count": int(count)}


def _resolve_severity(
    *,
    status: str,
    alert_count: int,
    drift_key_count: int,
    top_drift_key_count: int,
    required_artifacts_present: int,
    required_artifacts_total: int,
) -> Dict[str, str]:
    state = str(status).strip().lower()
    if required_artifacts_present < required_artifacts_total:
        return {"severity": "failed", "reason": "required artifacts missing"}
    if state == "failed":
        return {"severity": "failed", "reason": "upstream status failed"}
    if state == "alerted" or int(alert_count) > 0:
        return {"severity": "alerted", "reason": "threshold alerts present"}
    if int(drift_key_count) > 0 and int(top_drift_key_count) > 0:
        return {"severity": "warn", "reason": "context drift observed below alert threshold"}
    return {"severity": "clear", "reason": "no context drift signal"}


def _build_alerts_summary(alerts_report: Dict[str, Any]) -> Dict[str, Any]:
    summary = _as_dict(alerts_report.get("summary"))
    if summary:
        return summary
    alerts = _as_list(alerts_report.get("alerts"))
    return {
        "status": str(alerts_report.get("status", "clear")),
        "history_entries": _safe_int(alerts_report.get("history_size"), 0),
        "recent_window": _safe_int(alerts_report.get("recent_runs"), 0),
        "alert_count": len(alerts),
        "key_totals": _as_dict(alerts_report.get("key_totals")),
        "rows": [_as_dict(item) for item in alerts if _as_dict(item)],
        "policy_source": _as_dict(alerts_report.get("policy_source")),
    }


def build_index(
    *,
    alerts_report: Dict[str, Any],
    history_summary: Dict[str, Any],
    key_counts_summary: Dict[str, Any],
    history_payload: Any,
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    alerts = _build_alerts_summary(alerts_report)
    history_rows = _as_list(history_payload)
    history_entry_count = len([_as_dict(row) for row in history_rows if _as_dict(row)])
    key_counts = _as_dict(key_counts_summary.get("key_counts"))
    top_key = _top_key(key_counts)
    artifact_rows = {
        "alerts_report": {
            "path": str(artifact_paths.get("alerts_json", "")),
            "exists": bool(str(artifact_paths.get("alerts_json", "")).strip())
            and Path(str(artifact_paths.get("alerts_json", ""))).exists(),
        },
        "history_summary": {
            "path": str(artifact_paths.get("history_summary_json", "")),
            "exists": bool(str(artifact_paths.get("history_summary_json", "")).strip())
            and Path(str(artifact_paths.get("history_summary_json", ""))).exists(),
        },
        "key_counts_summary": {
            "path": str(artifact_paths.get("key_counts_summary_json", "")),
            "exists": bool(str(artifact_paths.get("key_counts_summary_json", "")).strip())
            and Path(str(artifact_paths.get("key_counts_summary_json", ""))).exists(),
        },
        "history_raw": {
            "path": str(artifact_paths.get("history_json", "")),
            "exists": bool(str(artifact_paths.get("history_json", "")).strip())
            and Path(str(artifact_paths.get("history_json", ""))).exists(),
        },
    }
    required_artifact_names = ["alerts_report", "history_summary", "key_counts_summary"]
    required_present = len(
        [
            name
            for name in required_artifact_names
            if bool(_as_dict(artifact_rows.get(name)).get("exists", False))
        ]
    )
    required_total = len(required_artifact_names)

    status = str(alerts.get("status", "clear"))
    alert_count = _safe_int(alerts.get("alert_count"), 0)
    drift_key_count = len([k for k in key_counts.keys() if str(k).strip()])
    severity_payload = _resolve_severity(
        status=status,
        alert_count=alert_count,
        drift_key_count=drift_key_count,
        top_drift_key_count=_safe_int(top_key.get("count"), 0),
        required_artifacts_present=required_present,
        required_artifacts_total=required_total,
    )

    return {
        "schema_version": "1.0.0",
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "overview": {
            "status": status,
            "severity": str(severity_payload.get("severity", "clear")),
            "severity_reason": str(severity_payload.get("reason", "")),
            "alert_count": alert_count,
            "history_entries": max(
                _safe_int(history_summary.get("history_entries"), 0),
                history_entry_count,
            ),
            "recent_window": _safe_int(history_summary.get("recent_window"), 0),
            "drift_key_count": drift_key_count,
            "top_drift_key": top_key,
            "artifact_coverage": {
                "present": required_present,
                "total": required_total,
            },
        },
        "artifacts": artifact_rows,
        "policy_sources": {
            "alerts": _as_dict(alerts.get("policy_source")),
            "history": _as_dict(history_summary.get("policy_source")),
            "key_counts": _as_dict(key_counts_summary.get("policy_source")),
        },
        "summaries": {
            "alerts": alerts,
            "history": history_summary,
            "key_counts": key_counts_summary,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an index json for Graph2D context drift artifacts."
    )
    parser.add_argument("--alerts-json", required=True, help="Alerts report json path.")
    parser.add_argument(
        "--history-summary-json",
        required=True,
        help="History render summary json path.",
    )
    parser.add_argument(
        "--key-counts-summary-json",
        required=True,
        help="Key-count render summary json path.",
    )
    parser.add_argument(
        "--history-json",
        default="",
        help="Optional raw history json path.",
    )
    parser.add_argument("--output-json", required=True, help="Output index json path.")
    args = parser.parse_args()

    alerts_report = _as_dict(_read_json(Path(str(args.alerts_json))))
    history_summary = _as_dict(_read_json(Path(str(args.history_summary_json))))
    key_counts_summary = _as_dict(_read_json(Path(str(args.key_counts_summary_json))))
    history_payload = _read_json(Path(str(args.history_json))) if str(args.history_json).strip() else []

    index_payload = build_index(
        alerts_report=alerts_report,
        history_summary=history_summary,
        key_counts_summary=key_counts_summary,
        history_payload=history_payload,
        artifact_paths={
            "alerts_json": str(args.alerts_json),
            "history_summary_json": str(args.history_summary_json),
            "key_counts_summary_json": str(args.key_counts_summary_json),
            "history_json": str(args.history_json),
        },
    )

    out_path = Path(str(args.output_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"index_json={out_path}")
    print(f"status={str(_as_dict(index_payload.get('overview')).get('status', 'clear'))}")
    print(f"severity={str(_as_dict(index_payload.get('overview')).get('severity', 'clear'))}")
    print(f"alert_count={_safe_int(_as_dict(index_payload.get('overview')).get('alert_count'), 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
