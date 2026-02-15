#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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
        "--channel", choices=["standard", "strict"], default="standard", help="Baseline channel"
    )
    parser.add_argument(
        "--max-accuracy-mean-drop",
        type=float,
        default=0.08,
        help="Allowed drop for strict_accuracy_mean",
    )
    parser.add_argument(
        "--max-accuracy-min-drop",
        type=float,
        default=0.08,
        help="Allowed drop for strict_accuracy_min",
    )
    parser.add_argument(
        "--max-top-pred-ratio-increase",
        type=float,
        default=0.10,
        help="Allowed increase for strict_top_pred_ratio_max",
    )
    parser.add_argument(
        "--max-low-conf-ratio-increase",
        type=float,
        default=0.05,
        help="Allowed increase for strict_low_conf_ratio_max",
    )
    parser.add_argument(
        "--max-distinct-labels-drop",
        type=int,
        default=0,
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

    report = evaluate_regression(
        summary=summary,
        baseline_channel=baseline_channel,
        channel=str(args.channel),
        max_accuracy_mean_drop=float(args.max_accuracy_mean_drop),
        max_accuracy_min_drop=float(args.max_accuracy_min_drop),
        max_top_pred_ratio_increase=float(args.max_top_pred_ratio_increase),
        max_low_conf_ratio_increase=float(args.max_low_conf_ratio_increase),
        max_distinct_labels_drop=int(args.max_distinct_labels_drop),
    )
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if str(report.get("status")) == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())

