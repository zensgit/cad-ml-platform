#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline_channel,
        channel=str(args.channel),
        max_accuracy_mean_drop=float(thresholds["max_accuracy_mean_drop"]),
        max_accuracy_min_drop=float(thresholds["max_accuracy_min_drop"]),
        max_top_pred_ratio_increase=float(thresholds["max_top_pred_ratio_increase"]),
        max_low_conf_ratio_increase=float(thresholds["max_low_conf_ratio_increase"]),
        max_distinct_labels_drop=int(thresholds["max_distinct_labels_drop"]),
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
