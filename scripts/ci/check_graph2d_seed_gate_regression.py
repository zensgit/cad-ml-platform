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

DEFAULT_CONTEXT_KEYS: List[str] = [
    "training_profile",
    "manifest_label_mode",
    "max_samples",
    "min_label_confidence",
    "strict_low_conf_threshold",
]

DEFAULT_BASELINE_POLICY: Dict[str, Any] = {
    "max_baseline_age_days": 365,
    "require_snapshot_ref_exists": True,
    "require_snapshot_metrics_match": True,
    "require_integrity_hash_match": True,
    "require_snapshot_date_match": True,
    "require_snapshot_ref_date_match": True,
    "require_context_match": True,
    "context_mismatch_mode": "fail",
    "context_keys": DEFAULT_CONTEXT_KEYS,
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


def _safe_context_keys(value: Any, default: List[str] | None = None) -> List[str]:
    raw: List[str] = []
    if isinstance(value, list):
        raw = [str(item).strip() for item in value]
    elif isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    elif default is not None:
        raw = [str(item).strip() for item in default]
    out: List[str] = []
    for item in raw:
        if not item:
            continue
        if item not in out:
            out.append(item)
    return out


def _safe_context_mismatch_mode(value: Any, default: str = "fail") -> str:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"fail", "warn", "ignore"}:
        return text
    return str(default).strip().lower() or "fail"


def _normalize_context_value(key: str, value: Any) -> Any:
    name = str(key).strip()
    if name in {
        "max_samples",
        "num_runs",
        "force_clean_min_count",
    }:
        return _safe_int(value, -1)
    if name in {"min_label_confidence", "strict_low_conf_threshold"}:
        return round(_safe_float(value, -1.0), 6)
    if name == "seeds":
        if not isinstance(value, list):
            return []
        out: List[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return ""
    return str(value).strip()


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

    require_snapshot_date_cli = str(cli_overrides.get("require_snapshot_date_match", "auto"))
    if require_snapshot_date_cli in {"true", "false"}:
        policy["require_snapshot_date_match"] = require_snapshot_date_cli == "true"

    require_snapshot_ref_date_cli = str(
        cli_overrides.get("require_snapshot_ref_date_match", "auto")
    )
    if require_snapshot_ref_date_cli in {"true", "false"}:
        policy["require_snapshot_ref_date_match"] = require_snapshot_ref_date_cli == "true"

    require_context_cli = str(cli_overrides.get("require_context_match", "auto"))
    if require_context_cli in {"true", "false"}:
        policy["require_context_match"] = require_context_cli == "true"

    context_mode_cli = str(cli_overrides.get("context_mismatch_mode", "auto")).strip().lower()
    if context_mode_cli in {"fail", "warn", "ignore"}:
        policy["context_mismatch_mode"] = context_mode_cli

    if cli_overrides.get("context_keys") is not None:
        policy["context_keys"] = cli_overrides.get("context_keys")

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
        "require_snapshot_date_match": _safe_bool(
            policy.get("require_snapshot_date_match"),
            DEFAULT_BASELINE_POLICY["require_snapshot_date_match"],
        ),
        "require_snapshot_ref_date_match": _safe_bool(
            policy.get("require_snapshot_ref_date_match"),
            DEFAULT_BASELINE_POLICY["require_snapshot_ref_date_match"],
        ),
        "require_context_match": _safe_bool(
            policy.get("require_context_match"),
            DEFAULT_BASELINE_POLICY["require_context_match"],
        ),
        "context_mismatch_mode": _safe_context_mismatch_mode(
            policy.get("context_mismatch_mode"),
            default=str(DEFAULT_BASELINE_POLICY["context_mismatch_mode"]),
        ),
        "context_keys": _safe_context_keys(
            policy.get("context_keys"),
            default=DEFAULT_CONTEXT_KEYS,
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


def _extract_date_stamp_from_snapshot_ref(snapshot_ref: str) -> str:
    name = Path(str(snapshot_ref)).name
    marker = "graph2d_seed_gate_baseline_snapshot_"
    suffix = ".json"
    if marker not in name or not name.endswith(suffix):
        return ""
    token = name.split(marker, 1)[1][:8]
    if len(token) != 8 or not token.isdigit():
        return ""
    return token


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


def _resolve_current_summary(
    *,
    use_baseline_as_current: bool,
    baseline_channel: Dict[str, Any],
    summary_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if bool(use_baseline_as_current):
        return _metrics_view(baseline_channel)
    return summary_payload if isinstance(summary_payload, dict) else {}


def _resolve_current_context(
    *,
    use_baseline_as_current: bool,
    baseline_channel: Dict[str, Any],
    summary_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if bool(use_baseline_as_current):
        baseline_context = baseline_channel.get("context")
        return baseline_context if isinstance(baseline_context, dict) else {}
    return summary_payload if isinstance(summary_payload, dict) else {}


def evaluate_regression(
    *,
    summary: Dict[str, Any],
    current_context: Dict[str, Any] | None = None,
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
    require_snapshot_date_match: bool = False,
    require_snapshot_ref_date_match: bool = False,
    require_context_match: bool = False,
    context_mismatch_mode: str = "fail",
    context_keys: List[str] | None = None,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []

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

    effective_context_keys = _safe_context_keys(
        context_keys,
        default=DEFAULT_CONTEXT_KEYS,
    )
    baseline_context = (
        baseline_channel.get("context")
        if isinstance(baseline_channel.get("context"), dict)
        else {}
    )
    cur_context = current_context if isinstance(current_context, dict) else {}
    context_diff: Dict[str, Dict[str, Any]] = {}
    context_match: bool | None = None
    effective_context_mode = _safe_context_mismatch_mode(context_mismatch_mode, "fail")

    def _record_context_violation(message: str) -> None:
        if effective_context_mode == "ignore":
            return
        if effective_context_mode == "warn":
            warnings.append(message)
            return
        failures.append(message)

    if bool(require_context_match):
        if not baseline_context:
            _record_context_violation("context: missing baseline context")
            context_match = False
        elif not cur_context:
            _record_context_violation("context: missing current context")
            context_match = False
        else:
            for key in effective_context_keys:
                has_base = key in baseline_context
                has_cur = key in cur_context
                if not has_base or not has_cur:
                    context_diff[key] = {
                        "baseline": baseline_context.get(key, "<missing>"),
                        "current": cur_context.get(key, "<missing>"),
                    }
                    continue
                base_value = _normalize_context_value(key, baseline_context.get(key))
                cur_value = _normalize_context_value(key, cur_context.get(key))
                if base_value != cur_value:
                    context_diff[key] = {
                        "baseline": base_value,
                        "current": cur_value,
                    }
            context_match = len(context_diff) == 0
            if not context_match:
                _record_context_violation(
                    "context: mismatch on keys "
                    f"[{', '.join(sorted(context_diff.keys()))}]"
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
    snapshot_payload_date = ""
    snapshot_date_match: bool | None = None
    snapshot_ref_date_stamp = _extract_date_stamp_from_snapshot_ref(snapshot_ref)
    expected_date_stamp = baseline_date.replace("-", "") if baseline_date else ""
    snapshot_ref_date_match: bool | None = None
    if snapshot_ref_date_stamp:
        snapshot_ref_date_match = bool(expected_date_stamp) and (
            snapshot_ref_date_stamp == expected_date_stamp
        )
    if snapshot_ref:
        resolved = _resolve_snapshot_path(snapshot_ref, baseline_json_path)
        snapshot_path = str(resolved)
        snapshot_exists = bool(resolved.exists())
        if snapshot_exists:
            snapshot_payload = _read_json(resolved)
            snapshot_payload_date = str(snapshot_payload.get("date") or "").strip()
            if snapshot_payload_date and baseline_date:
                snapshot_date_match = snapshot_payload_date == baseline_date
            elif snapshot_payload_date or baseline_date:
                snapshot_date_match = False
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
    if bool(require_snapshot_date_match):
        if not snapshot_ref:
            failures.append("snapshot_date: snapshot_ref missing")
        elif not snapshot_exists:
            failures.append("snapshot_date: snapshot file missing")
        elif snapshot_date_match is not True:
            failures.append(
                f"snapshot_date: snapshot date '{snapshot_payload_date}' != baseline date '{baseline_date}'"
            )
    if bool(require_snapshot_ref_date_match):
        if not snapshot_ref:
            failures.append("snapshot_ref_date: snapshot_ref missing")
        elif not expected_date_stamp:
            failures.append("snapshot_ref_date: baseline date missing")
        elif not snapshot_ref_date_stamp:
            failures.append("snapshot_ref_date: cannot parse date from snapshot_ref name")
        elif snapshot_ref_date_match is not True:
            failures.append(
                f"snapshot_ref_date: snapshot_ref stamp '{snapshot_ref_date_stamp}' != baseline stamp '{expected_date_stamp}'"
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

    status = "failed" if failures else ("passed_with_warnings" if warnings else "passed")
    return {
        "channel": channel,
        "status": status,
        "failures": failures,
        "warnings": warnings,
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
            "require_snapshot_date_match": bool(require_snapshot_date_match),
            "require_snapshot_ref_date_match": bool(require_snapshot_ref_date_match),
            "require_context_match": bool(require_context_match),
            "context_mismatch_mode": effective_context_mode,
            "context_keys": effective_context_keys,
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
            "snapshot_payload_date": snapshot_payload_date,
            "snapshot_date_match": (
                bool(snapshot_date_match) if snapshot_date_match is not None else None
            ),
            "snapshot_ref_date_stamp": snapshot_ref_date_stamp,
            "expected_date_stamp": expected_date_stamp,
            "snapshot_ref_date_match": (
                bool(snapshot_ref_date_match)
                if snapshot_ref_date_match is not None
                else None
            ),
            "baseline_context_present": bool(baseline_context),
            "current_context_present": bool(cur_context),
            "context_match": bool(context_match) if context_match is not None else None,
            "context_diff": context_diff,
            "context_mismatch_mode": effective_context_mode,
            "context_keys": effective_context_keys,
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
        "baseline_context": baseline_context,
        "current_context": cur_context,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Graph2D seed gate summary against baseline snapshot."
    )
    parser.add_argument("--summary-json", default="", help="Current seed sweep summary json")
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
    parser.add_argument(
        "--require-snapshot-date-match",
        choices=["auto", "true", "false"],
        default="auto",
        help="Require snapshot payload date to match baseline date (auto uses config).",
    )
    parser.add_argument(
        "--require-snapshot-ref-date-match",
        choices=["auto", "true", "false"],
        default="auto",
        help="Require snapshot_ref filename date stamp to match baseline date (auto uses config).",
    )
    parser.add_argument(
        "--require-context-match",
        choices=["auto", "true", "false"],
        default="auto",
        help="Require current summary context to match baseline context (auto uses config).",
    )
    parser.add_argument(
        "--context-keys",
        default="",
        help=(
            "Optional comma-separated context keys for context match check "
            "(overrides config when set)."
        ),
    )
    parser.add_argument(
        "--context-mismatch-mode",
        choices=["auto", "fail", "warn", "ignore"],
        default="auto",
        help=(
            "Behavior when context check fails and require_context_match is enabled "
            "(auto uses config)."
        ),
    )
    parser.add_argument(
        "--use-baseline-as-current",
        action="store_true",
        help="Use baseline channel metrics as current summary (baseline health check mode).",
    )
    args = parser.parse_args()

    baseline = _read_json(Path(args.baseline_json))
    if not baseline:
        print(f"Missing/invalid baseline json: {args.baseline_json}")
        return 2

    baseline_channel = baseline.get(str(args.channel))
    if not isinstance(baseline_channel, dict):
        print(f"Missing channel '{args.channel}' in baseline json: {args.baseline_json}")
        return 2

    summary_payload: Dict[str, Any] = {}
    if not bool(args.use_baseline_as_current):
        if not str(args.summary_json).strip():
            print("summary-json is required unless --use-baseline-as-current is set")
            return 2
        summary_payload = _read_json(Path(args.summary_json))
        if not summary_payload:
            print(f"Missing/invalid summary json: {args.summary_json}")
            return 2
    summary = _resolve_current_summary(
        use_baseline_as_current=bool(args.use_baseline_as_current),
        baseline_channel=baseline_channel,
        summary_payload=summary_payload,
    )
    current_context = _resolve_current_context(
        use_baseline_as_current=bool(args.use_baseline_as_current),
        baseline_channel=baseline_channel,
        summary_payload=summary_payload,
    )

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
            "require_snapshot_date_match": args.require_snapshot_date_match,
            "require_snapshot_ref_date_match": args.require_snapshot_ref_date_match,
            "require_context_match": args.require_context_match,
            "context_mismatch_mode": args.context_mismatch_mode,
            "context_keys": (
                args.context_keys if str(args.context_keys).strip() else None
            ),
        },
    )

    report = evaluate_regression(
        summary=summary,
        current_context=current_context,
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
        require_snapshot_date_match=bool(policy["require_snapshot_date_match"]),
        require_snapshot_ref_date_match=bool(policy["require_snapshot_ref_date_match"]),
        require_context_match=bool(policy["require_context_match"]),
        context_mismatch_mode=str(policy.get("context_mismatch_mode", "fail")),
        context_keys=list(policy.get("context_keys") or []),
    )
    report["threshold_source"] = {
        "config": str(args.config),
        "config_loaded": bool(config_payload),
        "current_source": (
            "baseline_channel"
            if bool(args.use_baseline_as_current)
            else str(args.summary_json)
        ),
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
                "require_snapshot_date_match": (
                    args.require_snapshot_date_match
                    if str(args.require_snapshot_date_match) != "auto"
                    else None
                ),
                "require_snapshot_ref_date_match": (
                    args.require_snapshot_ref_date_match
                    if str(args.require_snapshot_ref_date_match) != "auto"
                    else None
                ),
                "require_context_match": (
                    args.require_context_match
                    if str(args.require_context_match) != "auto"
                    else None
                ),
                "context_keys": (
                    _safe_context_keys(str(args.context_keys))
                    if str(args.context_keys).strip()
                    else None
                ),
                "context_mismatch_mode": (
                    args.context_mismatch_mode
                    if str(args.context_mismatch_mode) != "auto"
                    else None
                ),
                "use_baseline_as_current": (
                    bool(args.use_baseline_as_current)
                    if bool(args.use_baseline_as_current)
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
    return 0 if str(report.get("status")) != "failed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
