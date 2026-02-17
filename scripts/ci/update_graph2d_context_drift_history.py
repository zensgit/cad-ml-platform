#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_HISTORY_POLICY: Dict[str, Any] = {
    "max_runs": 20,
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


def _resolve_max_runs(config_payload: Dict[str, Any], cli_max_runs: int | None) -> int:
    if cli_max_runs is not None:
        return max(1, int(cli_max_runs))
    candidate = config_payload.get("max_runs", DEFAULT_HISTORY_POLICY.get("max_runs", 20))
    return max(1, _safe_int(candidate, 20))


def _extract_diff_keys(report: Dict[str, Any]) -> List[str]:
    meta = _as_dict(report.get("baseline_metadata"))
    diff = _as_dict(meta.get("context_diff"))
    out = sorted([str(key) for key in diff.keys() if str(key).strip()])
    return out


def _count_keys(keys: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


def _merge_counts(items: List[Dict[str, int]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for one in items:
        for key, value in one.items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return merged


def _build_snapshot(
    *,
    reports: List[Tuple[str, Dict[str, Any]]],
    run_id: str,
    run_number: str,
    ref_name: str,
    sha: str,
    policy_source: Dict[str, Any],
) -> Dict[str, Any]:
    report_rows: List[Dict[str, Any]] = []
    key_count_items: List[Dict[str, int]] = []
    total_warnings = 0
    total_failures = 0
    for path, payload in reports:
        report = _as_dict(payload)
        if not report:
            continue
        diff_keys = _extract_diff_keys(report)
        warnings = _as_list(report.get("warnings"))
        failures = _as_list(report.get("failures"))
        row = {
            "path": str(path),
            "channel": str(report.get("channel", "")),
            "status": str(report.get("status", "")),
            "context_mode": str(_as_dict(report.get("thresholds")).get("context_mismatch_mode", "")),
            "context_match": _as_dict(report.get("baseline_metadata")).get("context_match"),
            "diff_keys": diff_keys,
            "warning_count": int(len(warnings)),
            "failure_count": int(len(failures)),
        }
        report_rows.append(row)
        key_count_items.append(_count_keys(diff_keys))
        total_warnings += len(warnings)
        total_failures += len(failures)

    merged_counts = _merge_counts(key_count_items)
    if total_failures > 0:
        status = "failed"
    elif total_warnings > 0:
        status = "passed_with_warnings"
    else:
        status = "passed"

    return {
        "timestamp": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "run_id": str(run_id).strip(),
        "run_number": str(run_number).strip(),
        "ref_name": str(ref_name).strip(),
        "sha": str(sha).strip(),
        "status": status,
        "warning_count": int(total_warnings),
        "failure_count": int(total_failures),
        "drift_key_counts": merged_counts,
        "reports": report_rows,
        "policy_source": policy_source,
    }


def update_history(
    *,
    history_payload: Any,
    snapshot: Dict[str, Any],
    max_runs: int,
) -> List[Dict[str, Any]]:
    entries = _as_list(history_payload)
    cleaned: List[Dict[str, Any]] = []
    for item in entries:
        one = _as_dict(item)
        if one:
            cleaned.append(one)

    run_id = str(snapshot.get("run_id", "")).strip()
    if run_id:
        cleaned = [item for item in cleaned if str(item.get("run_id", "")).strip() != run_id]

    cleaned.append(snapshot)
    cleaned = sorted(cleaned, key=lambda x: str(x.get("timestamp", "")))
    limit = max(1, int(max_runs))
    if len(cleaned) > limit:
        cleaned = cleaned[-limit:]
    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update Graph2D context drift history using one or more regression reports."
    )
    parser.add_argument(
        "--config",
        default="config/graph2d_context_drift_alerts.yaml",
        help="YAML config path for history policy.",
    )
    parser.add_argument(
        "--config-section",
        default="graph2d_context_drift_alerts",
        help="Config section name in yaml.",
    )
    parser.add_argument(
        "--report-json",
        action="append",
        default=[],
        help="Regression report json path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--history-json",
        default="",
        help="Existing history json path (optional).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output history json path.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of history entries to keep.",
    )
    parser.add_argument("--run-id", default="", help="Run id (for dedup).")
    parser.add_argument("--run-number", default="", help="Run number.")
    parser.add_argument("--ref-name", default="", help="Git ref name.")
    parser.add_argument("--sha", default="", help="Git commit sha.")
    args = parser.parse_args()

    reports: List[Tuple[str, Dict[str, Any]]] = []
    for path in [str(item).strip() for item in list(args.report_json or []) if str(item).strip()]:
        payload = _as_dict(_read_json(Path(path)))
        reports.append((path, payload))

    config_payload = _load_yaml_defaults(str(args.config), str(args.config_section))
    max_runs = _resolve_max_runs(config_payload, args.max_runs)
    policy_source = {
        "config": str(args.config),
        "config_section": str(args.config_section),
        "config_loaded": bool(config_payload),
        "resolved_policy": {
            "max_runs": int(max_runs),
        },
        "cli_overrides": {
            key: value
            for key, value in {
                "max_runs": args.max_runs,
            }.items()
            if value is not None
        },
    }

    snapshot = _build_snapshot(
        reports=reports,
        run_id=str(args.run_id),
        run_number=str(args.run_number),
        ref_name=str(args.ref_name),
        sha=str(args.sha),
        policy_source=policy_source,
    )

    history_payload = None
    if str(args.history_json).strip():
        history_payload = _read_json(Path(str(args.history_json)))
    updated = update_history(
        history_payload=history_payload,
        snapshot=snapshot,
        max_runs=int(max_runs),
    )

    out_path = Path(str(args.output_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"history_json={out_path}")
    print(f"history_entries={len(updated)}")
    print(f"max_runs={int(max_runs)}")
    print(f"latest_status={snapshot.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
