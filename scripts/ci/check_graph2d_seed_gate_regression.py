#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "max_accuracy_mean_drop": 0.08,
    "max_accuracy_min_drop": 0.08,
    "max_top_pred_ratio_increase": 0.10,
    "max_low_conf_ratio_increase": 0.05,
    "max_distinct_labels_drop": 0,
}

DEFAULT_BASELINE_POLICY: Dict[str, Any] = {
    "max_baseline_age_days": 365,
    "require_snapshot_ref_exists": True,
    "require_snapshot_metrics_match": True,
    "require_integrity_hash_match": True,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _resolve_thresholds(
    *,
    channel: str,
    config_payload: Dict[str, Any],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    thresholds: Dict[str, Any] = dict(DEFAULT_THRESHOLDS)

    for key in DEFAULT_THRESHOLDS.keys():
        if key in config_payload:
            thresholds[key] = config_payload.get(key)
    channel_payload = config_payload.get("channels")
    if isinstance(channel_payload, dict):
        one_channel = channel_payload.get(str(channel))
        if isinstance(one_channel, dict):
            for key in DEFAULT_THRESHOLDS.keys():
                if key in one_channel:
                    thresholds[key] = one_channel.get(key)

    for key, value in cli_overrides.items():
        if value is not None:
            thresholds[key] = value

    return {
        "max_accuracy_mean_drop": _safe_float(
            thresholds.get("max_accuracy_mean_drop"), DEFAULT_THRESHOLDS["max_accuracy_mean_drop"]
        ),
        "max_accuracy_min_drop": _safe_float(
            thresholds.get("max_accuracy_min_drop"), DEFAULT_THRESHOLDS["max_accuracy_min_drop"]
        ),
        "max_top_pred_ratio_increase": _safe_float(
            thresholds.get(
                "max_top_pred_ratio_increase",
                DEFAULT_THRESHOLDS["max_top_pred_ratio_increase"],
            ),
        ),
        "max_low_conf_ratio_increase": _safe_float(
            thresholds.get(
                "max_low_conf_ratio_increase",
                DEFAULT_THRESHOLDS["max_low_conf_ratio_increase"],
            ),
        ),
        "max_distinct_labels_drop": _safe_int(
            thresholds.get(
                "max_distinct_labels_drop",
                DEFAULT_THRESHOLDS["max_distinct_labels_drop"],
            ),
        ),
    }


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_of(value: Any) -> str:
    text = _canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_baseline_policy(
    *,
    config_payload: Dict[str, Any],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    policy: Dict[str, Any] = dict(DEFAULT_BASELINE_POLICY)
    for key in DEFAULT_BASELINE_POLICY.keys():
        if key in config_payload:
            policy[key] = config_payload.get(key)

    if cli_overrides.get("max_baseline_age_days") is not None:
        policy["max_baseline_age_days"] = cli_overrides.get("max_baseline_age_days")

    require_snapshot_cli = str(cli_overrides.get("require_snapshot_ref_exists", "auto"))
    if require_snapshot_cli in {"true", "false"}:
        policy["require_snapshot_ref_exists"] = require_snapshot_cli == "true"

    require_snapshot_match_cli = str(
        cli_overrides.get("require_snapshot_metrics_match", "auto")
    )
    if require_snapshot_match_cli in {"true", "false"}:
        policy["require_snapshot_metrics_match"] = require_snapshot_match_cli == "true"

    require_integrity_cli = str(cli_overrides.get("require_integrity_hash_match", "auto"))
    if require_integrity_cli in {"true", "false"}:
        policy["require_integrity_hash_match"] = require_integrity_cli == "true"

    return {
        "max_baseline_age_days": _safe_int(
            policy.get("max_baseline_age_days"),
            DEFAULT_BASELINE_POLICY["max_baseline_age_days"],
        ),
        "require_snapshot_ref_exists": _safe_bool(
            policy.get("require_snapshot_ref_exists"),
            DEFAULT_BASELINE_POLICY["require_snapshot_ref_exists"],
        ),
        "require_snapshot_metrics_match": _safe_bool(
            policy.get("require_snapshot_metrics_match"),
            DEFAULT_BASELINE_POLICY["require_snapshot_metrics_match"],
        ),
        "require_integrity_hash_match": _safe_bool(
            policy.get("require_integrity_hash_match"),
            DEFAULT_BASELINE_POLICY["require_integrity_hash_match"],
        ),
    }


def _resolve_snapshot_path(snapshot_ref: str, baseline_json_path: str) -> Path:
    ref = Path(str(snapshot_ref))
    if ref.is_absolute():
        return ref
    baseline_parent = Path(baseline_json_path).resolve().parent
    candidate_from_baseline = baseline_parent / ref
    if candidate_from_baseline.exists():
        return candidate_from_baseline
    return Path.cwd() / ref


def _metrics_view(channel_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strict_accuracy_mean": _safe_float(channel_payload.get("strict_accuracy_mean"), -1.0),
        "strict_accuracy_min": _safe_float(channel_payload.get("strict_accuracy_min"), -1.0),
        "strict_top_pred_ratio_max": _safe_float(
            channel_payload.get("strict_top_pred_ratio_max"), -1.0
        ),
        "strict_low_conf_ratio_max": _safe_float(
            channel_payload.get("strict_low_conf_ratio_max"), -1.0
        ),
        "manifest_distinct_labels_min": _safe_int(
            channel_payload.get("manifest_distinct_labels_min"), -1
        ),
    }


def evaluate_regression(
    *,
    summary: Dict[str, Any],
    baseline_channel: Dict[str, Any],
    channel: str,
    max_accuracy_mean_drop: float,
    max_accuracy_min_drop: float,
    max_top_pred_ratio_increase: float,
    max_low_conf_ratio_increase: float,
    max_distinct_labels_drop: int,
    baseline_payload: Dict[str, Any] | None = None,
    baseline_json_path: str = "",
    max_baseline_age_days: int = -1,
    require_snapshot_ref_exists: bool = False,
    require_snapshot_metrics_match: bool = False,
    require_integrity_hash_match: bool = False,
) -> Dict[str, Any]:
    failures: List[str] = []

    def _check_min(
        metric: str,
        max_drop: float,
    ) -> None:
        base = _safe_float(baseline_channel.get(metric), -1.0)
        cur = _safe_float(summary.get(metric), -1.0)
        if base < 0 or cur < 0:
            failures.append(f"{metric}: missing baseline/current value")
            return
        allowed = base - float(max_drop)
        if cur < allowed:
            failures.append(
                f"{metric}: current={cur:.6f} < allowed={allowed:.6f}"
                f" (baseline={base:.6f}, max_drop={float(max_drop):.6f})"
            )

    def _check_max(
        metric: str,
        max_increase: float,
    ) -> None:
        base = _safe_float(baseline_channel.get(metric), -1.0)
        cur = _safe_float(summary.get(metric), -1.0)
        if base < 0 or cur < 0:
            failures.append(f"{metric}: missing baseline/current value")
            return
        allowed = base + float(max_increase)
        if cur > allowed:
            failures.append(
                f"{metric}: current={cur:.6f} > allowed={allowed:.6f}"
                f" (baseline={base:.6f}, max_increase={float(max_increase):.6f})"
            )

    _check_min("strict_accuracy_mean", max_accuracy_mean_drop)
    _check_min("strict_accuracy_min", max_accuracy_min_drop)
    _check_max("strict_top_pred_ratio_max", max_top_pred_ratio_increase)
    _check_max("strict_low_conf_ratio_max", max_low_conf_ratio_increase)

    base_labels_min = _safe_int(baseline_channel.get("manifest_distinct_labels_min"), -1)
    cur_labels_min = _safe_int(summary.get("manifest_distinct_labels_min"), -1)
    if base_labels_min < 0 or cur_labels_min < 0:
        failures.append("manifest_distinct_labels_min: missing baseline/current value")
    else:
        allowed_labels_min = base_labels_min - int(max_distinct_labels_drop)
        if cur_labels_min < allowed_labels_min:
            failures.append(
                "manifest_distinct_labels_min: "
                f"current={cur_labels_min} < allowed={allowed_labels_min} "
                f"(baseline={base_labels_min}, max_drop={int(max_distinct_labels_drop)})"
            )

    metadata = baseline_payload if isinstance(baseline_payload, dict) else {}
    baseline_date = str(metadata.get("date") or "").strip()
    baseline_age_days = -1
    if baseline_date:
        try:
            baseline_age_days = (dt.date.today() - dt.date.fromisoformat(baseline_date)).days
        except Exception:
            failures.append(f"baseline_date: invalid format '{baseline_date}'")
    if int(max_baseline_age_days) >= 0:
        if not baseline_date:
            failures.append("baseline_date: missing date in baseline payload")
        elif baseline_age_days < 0:
            failures.append(f"baseline_date: invalid age={baseline_age_days}")
        elif baseline_age_days > int(max_baseline_age_days):
            failures.append(
                "baseline_date: "
                f"age_days={baseline_age_days} exceeds max={int(max_baseline_age_days)}"
            )

    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    integrity = (
        metadata.get("integrity") if isinstance(metadata.get("integrity"), dict) else {}
    )
    baseline_channel_hash_expected = str(
        integrity.get(f"{str(channel)}_channel_sha256") or ""
    ).strip()
    baseline_channel_hash_actual = _sha256_of(baseline_channel)
    baseline_channel_hash_match = (
        bool(baseline_channel_hash_expected)
        and baseline_channel_hash_expected == baseline_channel_hash_actual
    )
    baseline_core_hash_expected = str(integrity.get("payload_core_sha256") or "").strip()
    baseline_core_hash_actual = _sha256_of(
        {
            "date": metadata.get("date"),
            "source": metadata.get("source"),
            "standard": metadata.get("standard"),
            "strict": metadata.get("strict"),
        }
    )
    baseline_core_hash_match = (
        bool(baseline_core_hash_expected)
        and baseline_core_hash_expected == baseline_core_hash_actual
    )

    if bool(require_integrity_hash_match):
        if not baseline_channel_hash_expected:
            failures.append(f"integrity: missing baseline {channel}_channel_sha256")
        elif not baseline_channel_hash_match:
            failures.append(f"integrity: baseline {channel}_channel_sha256 mismatch")
        if not baseline_core_hash_expected:
            failures.append("integrity: missing baseline payload_core_sha256")
        elif not baseline_core_hash_match:
            failures.append("integrity: baseline payload_core_sha256 mismatch")

    snapshot_ref = str(source.get("snapshot_ref") or "").strip()
    snapshot_path = ""
    snapshot_exists = False
    snapshot_metrics_match: bool | None = None
    snapshot_channel_present = False
    snapshot_metrics_diff: Dict[str, Any] = {}
    snapshot_channel_hash_expected = ""
    snapshot_channel_hash_actual = ""
    snapshot_channel_hash_match: bool | None = None
    snapshot_vs_baseline_hash_match: bool | None = None
    if snapshot_ref:
        resolved = _resolve_snapshot_path(snapshot_ref, baseline_json_path)
        snapshot_path = str(resolved)
        snapshot_exists = bool(resolved.exists())
        if snapshot_exists:
            snapshot_payload = _read_json(resolved)
            snapshot_channel = (
                snapshot_payload.get(str(channel))
                if isinstance(snapshot_payload.get(str(channel)), dict)
                else None
            )
            snapshot_integrity = (
                snapshot_payload.get("integrity")
                if isinstance(snapshot_payload.get("integrity"), dict)
                else {}
            )
            snapshot_channel_present = isinstance(snapshot_channel, dict)
            if snapshot_channel_present:
                left = _metrics_view(baseline_channel)
                right = _metrics_view(snapshot_channel or {})
                diff: Dict[str, Any] = {}
                for key in left.keys():
                    if left.get(key) != right.get(key):
                        diff[key] = {"baseline": left.get(key), "snapshot": right.get(key)}
                snapshot_metrics_diff = diff
                snapshot_metrics_match = len(diff) == 0
                snapshot_channel_hash_expected = str(
                    snapshot_integrity.get(f"{str(channel)}_channel_sha256") or ""
                ).strip()
                snapshot_channel_hash_actual = _sha256_of(snapshot_channel or {})
                snapshot_channel_hash_match = (
                    bool(snapshot_channel_hash_expected)
                    and snapshot_channel_hash_expected == snapshot_channel_hash_actual
                )
                snapshot_vs_baseline_hash_match = (
                    bool(snapshot_channel_hash_expected)
                    and bool(baseline_channel_hash_expected)
                    and snapshot_channel_hash_expected == baseline_channel_hash_expected
                )
            else:
                snapshot_metrics_match = False
                snapshot_channel_hash_match = False
                snapshot_vs_baseline_hash_match = False
    if bool(require_snapshot_ref_exists):
        if not snapshot_ref:
            failures.append("snapshot_ref: missing in baseline source")
        elif not snapshot_exists:
            failures.append(
                f"snapshot_ref: path does not exist ({snapshot_ref} -> {snapshot_path})"
            )
    if bool(require_snapshot_metrics_match):
        if not snapshot_ref:
            failures.append("snapshot_metrics: snapshot_ref missing")
        elif not snapshot_exists:
            failures.append("snapshot_metrics: snapshot file missing")
        elif not snapshot_channel_present:
            failures.append(f"snapshot_metrics: channel '{channel}' missing in snapshot")
        elif snapshot_metrics_match is False:
            failures.append(
                f"snapshot_metrics: channel '{channel}' differs from baseline"
            )
    if bool(require_integrity_hash_match):
        if not snapshot_ref:
            failures.append("integrity: snapshot_ref missing")
        elif not snapshot_exists:
            failures.append("integrity: snapshot file missing")
        elif not snapshot_channel_present:
            failures.append(f"integrity: channel '{channel}' missing in snapshot")
        elif not snapshot_channel_hash_expected:
            failures.append(f"integrity: missing snapshot {channel}_channel_sha256")
        elif snapshot_channel_hash_match is False:
            failures.append(f"integrity: snapshot {channel}_channel_sha256 mismatch")
        elif snapshot_vs_baseline_hash_match is False:
            failures.append(
                f"integrity: snapshot and baseline {channel}_channel_sha256 differ"
            )

    return {
        "channel": channel,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "thresholds": {
            "max_accuracy_mean_drop": float(max_accuracy_mean_drop),
            "max_accuracy_min_drop": float(max_accuracy_min_drop),
            "max_top_pred_ratio_increase": float(max_top_pred_ratio_increase),
            "max_low_conf_ratio_increase": float(max_low_conf_ratio_increase),
            "max_distinct_labels_drop": int(max_distinct_labels_drop),
            "max_baseline_age_days": int(max_baseline_age_days),
            "require_snapshot_ref_exists": bool(require_snapshot_ref_exists),
            "require_snapshot_metrics_match": bool(require_snapshot_metrics_match),
            "require_integrity_hash_match": bool(require_integrity_hash_match),
        },
        "baseline_metadata": {
            "date": baseline_date,
            "age_days": int(baseline_age_days),
            "snapshot_ref": snapshot_ref,
            "snapshot_path": snapshot_path,
            "snapshot_exists": bool(snapshot_exists),
            "snapshot_channel_present": bool(snapshot_channel_present),
            "snapshot_metrics_match": (
                bool(snapshot_metrics_match) if snapshot_metrics_match is not None else None
            ),
            "snapshot_metrics_diff": snapshot_metrics_diff,
            "baseline_channel_hash_expected": baseline_channel_hash_expected,
            "baseline_channel_hash_actual": baseline_channel_hash_actual,
            "baseline_channel_hash_match": bool(baseline_channel_hash_match),
            "baseline_core_hash_expected": baseline_core_hash_expected,
            "baseline_core_hash_actual": baseline_core_hash_actual,
            "baseline_core_hash_match": bool(baseline_core_hash_match),
            "snapshot_channel_hash_expected": snapshot_channel_hash_expected,
            "snapshot_channel_hash_actual": snapshot_channel_hash_actual,
            "snapshot_channel_hash_match": (
                bool(snapshot_channel_hash_match)
                if snapshot_channel_hash_match is not None
                else None
            ),
            "snapshot_vs_baseline_hash_match": (
                bool(snapshot_vs_baseline_hash_match)
                if snapshot_vs_baseline_hash_match is not None
                else None
            ),
        },
        "baseline": {
            "strict_accuracy_mean": _safe_float(
                baseline_channel.get("strict_accuracy_mean"), -1.0
            ),
            "strict_accuracy_min": _safe_float(
                baseline_channel.get("strict_accuracy_min"), -1.0
            ),
            "strict_top_pred_ratio_max": _safe_float(
                baseline_channel.get("strict_top_pred_ratio_max"), -1.0
            ),
            "strict_low_conf_ratio_max": _safe_float(
                baseline_channel.get("strict_low_conf_ratio_max"), -1.0
            ),
            "manifest_distinct_labels_min": _safe_int(
                baseline_channel.get("manifest_distinct_labels_min"), -1
            ),
        },
        "current": {
            "strict_accuracy_mean": _safe_float(summary.get("strict_accuracy_mean"), -1.0),
            "strict_accuracy_min": _safe_float(summary.get("strict_accuracy_min"), -1.0),
            "strict_top_pred_ratio_max": _safe_float(
                summary.get("strict_top_pred_ratio_max"), -1.0
            ),
            "strict_low_conf_ratio_max": _safe_float(
                summary.get("strict_low_conf_ratio_max"), -1.0
            ),
            "manifest_distinct_labels_min": _safe_int(
                summary.get("manifest_distinct_labels_min"), -1
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Graph2D seed gate summary against baseline snapshot."
    )
    parser.add_argument("--summary-json", required=True, help="Current seed sweep summary json")
    parser.add_argument("--baseline-json", required=True, help="Baseline snapshot json")
    parser.add_argument(
        "--config",
        default="config/graph2d_seed_gate_regression.yaml",
        help="YAML config path for regression thresholds.",
    )
    parser.add_argument(
        "--channel", choices=["standard", "strict"], default="standard", help="Baseline channel"
    )
    parser.add_argument(
        "--max-accuracy-mean-drop",
        type=float,
        default=None,
        help="Allowed drop for strict_accuracy_mean",
    )
    parser.add_argument(
        "--max-accuracy-min-drop",
        type=float,
        default=None,
        help="Allowed drop for strict_accuracy_min",
    )
    parser.add_argument(
        "--max-top-pred-ratio-increase",
        type=float,
        default=None,
        help="Allowed increase for strict_top_pred_ratio_max",
    )
    parser.add_argument(
        "--max-low-conf-ratio-increase",
        type=float,
        default=None,
        help="Allowed increase for strict_low_conf_ratio_max",
    )
    parser.add_argument(
        "--max-distinct-labels-drop",
        type=int,
        default=None,
        help="Allowed drop for manifest_distinct_labels_min",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional report output json path",
    )
    parser.add_argument(
        "--max-baseline-age-days",
        type=int,
        default=None,
        help="Allowed baseline age in days (overrides config when set).",
    )
    parser.add_argument(
        "--require-snapshot-ref-exists",
        choices=["auto", "true", "false"],
        default="auto",
        help="Require baseline source.snapshot_ref to exist (auto uses config).",
    )
    parser.add_argument(
        "--require-snapshot-metrics-match",
        choices=["auto", "true", "false"],
        default="auto",
        help="Require snapshot_ref channel metrics to match baseline channel (auto uses config).",
    )
    parser.add_argument(
        "--require-integrity-hash-match",
        choices=["auto", "true", "false"],
        default="auto",
        help="Require integrity hashes to exist and match (auto uses config).",
    )
    args = parser.parse_args()

    summary = _read_json(Path(args.summary_json))
    if not summary:
        print(f"Missing/invalid summary json: {args.summary_json}")
        return 2

    baseline = _read_json(Path(args.baseline_json))
    if not baseline:
        print(f"Missing/invalid baseline json: {args.baseline_json}")
        return 2

    baseline_channel = baseline.get(str(args.channel))
    if not isinstance(baseline_channel, dict):
        print(f"Missing channel '{args.channel}' in baseline json: {args.baseline_json}")
        return 2

    config_payload = _load_yaml_defaults(str(args.config), "graph2d_seed_gate_regression")
    thresholds = _resolve_thresholds(
        channel=str(args.channel),
        config_payload=config_payload,
        cli_overrides={
            "max_accuracy_mean_drop": args.max_accuracy_mean_drop,
            "max_accuracy_min_drop": args.max_accuracy_min_drop,
            "max_top_pred_ratio_increase": args.max_top_pred_ratio_increase,
            "max_low_conf_ratio_increase": args.max_low_conf_ratio_increase,
            "max_distinct_labels_drop": args.max_distinct_labels_drop,
        },
    )
    policy = _resolve_baseline_policy(
        config_payload=config_payload,
        cli_overrides={
            "max_baseline_age_days": args.max_baseline_age_days,
            "require_snapshot_ref_exists": args.require_snapshot_ref_exists,
            "require_snapshot_metrics_match": args.require_snapshot_metrics_match,
            "require_integrity_hash_match": args.require_integrity_hash_match,
        },
    )

    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline_channel,
        channel=str(args.channel),
        max_accuracy_mean_drop=float(thresholds["max_accuracy_mean_drop"]),
        max_accuracy_min_drop=float(thresholds["max_accuracy_min_drop"]),
        max_top_pred_ratio_increase=float(thresholds["max_top_pred_ratio_increase"]),
        max_low_conf_ratio_increase=float(thresholds["max_low_conf_ratio_increase"]),
        max_distinct_labels_drop=int(thresholds["max_distinct_labels_drop"]),
        baseline_payload=baseline,
        baseline_json_path=str(args.baseline_json),
        max_baseline_age_days=int(policy["max_baseline_age_days"]),
        require_snapshot_ref_exists=bool(policy["require_snapshot_ref_exists"]),
        require_snapshot_metrics_match=bool(policy["require_snapshot_metrics_match"]),
        require_integrity_hash_match=bool(policy["require_integrity_hash_match"]),
    )
    report["threshold_source"] = {
        "config": str(args.config),
        "config_loaded": bool(config_payload),
        "cli_overrides": {
            key: value
            for key, value in {
                "max_accuracy_mean_drop": args.max_accuracy_mean_drop,
                "max_accuracy_min_drop": args.max_accuracy_min_drop,
                "max_top_pred_ratio_increase": args.max_top_pred_ratio_increase,
                "max_low_conf_ratio_increase": args.max_low_conf_ratio_increase,
                "max_distinct_labels_drop": args.max_distinct_labels_drop,
                "max_baseline_age_days": args.max_baseline_age_days,
                "require_snapshot_ref_exists": (
                    args.require_snapshot_ref_exists
                    if str(args.require_snapshot_ref_exists) != "auto"
                    else None
                ),
                "require_snapshot_metrics_match": (
                    args.require_snapshot_metrics_match
                    if str(args.require_snapshot_metrics_match) != "auto"
                    else None
                ),
                "require_integrity_hash_match": (
                    args.require_integrity_hash_match
                    if str(args.require_integrity_hash_match) != "auto"
                    else None
                ),
            }.items()
            if value is not None
        },
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if str(report.get("status")) == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
