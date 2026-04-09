#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_iso_ts(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_slices(
    metrics: Dict[str, Any],
    *,
    source_field: str,
    id_field: str,
) -> Dict[str, Dict[str, Any]]:
    raw = metrics.get(source_field)
    if not isinstance(raw, list):
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get(id_field) or "").strip()
        if not key:
            continue
        rows[key] = {
            "support": _safe_int(item.get("support"), 0),
            "hybrid_accuracy": _optional_float(item.get("hybrid_accuracy")),
            "hybrid_gain_vs_graph2d": _optional_float(item.get("hybrid_gain_vs_graph2d")),
        }
    return rows


def _load_history(eval_history_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not eval_history_dir.exists():
        return rows
    for path in sorted(eval_history_dir.glob("*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("type") or "").strip() != "hybrid_blind":
            continue
        ts = _parse_iso_ts(payload.get("timestamp"))
        if ts is None:
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        rows.append(
            {
                "timestamp": ts,
                "timestamp_text": str(payload.get("timestamp") or ""),
                "hybrid_accuracy": _safe_float(metrics.get("hybrid_accuracy"), 0.0),
                "hybrid_gain_vs_graph2d": _safe_float(metrics.get("hybrid_gain_vs_graph2d"), 0.0),
                "weak_label_coverage": _safe_float(metrics.get("weak_label_coverage"), 0.0),
                "label_slices": _normalize_slices(
                    metrics, source_field="label_slices", id_field="label"
                ),
                "family_slices": _normalize_slices(
                    metrics, source_field="family_slices", id_field="family"
                ),
            }
        )
    rows.sort(key=lambda item: item["timestamp"])
    return rows


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    q = min(1.0, max(0.0, float(q)))
    sorted_values = sorted(float(v) for v in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _summary(values: List[float], quantile: float) -> Dict[str, Any]:
    p = _quantile(values, quantile)
    return {
        "count": len(values),
        "mean": round(mean(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
        "quantile": float(quantile),
        "quantile_value": round(p, 6) if p is not None else None,
    }


def _max_drop(prev_value: Optional[float], latest_value: Optional[float]) -> Optional[float]:
    if prev_value is None or latest_value is None:
        return None
    return max(0.0, float(prev_value) - float(latest_value))


def _pairwise_slice_stats(
    *,
    previous: Dict[str, Dict[str, Any]],
    latest: Dict[str, Dict[str, Any]],
    min_support: int,
) -> Tuple[Optional[float], Optional[float], int]:
    common = 0
    worst_acc_drop: Optional[float] = None
    worst_gain_drop: Optional[float] = None
    for key in sorted(set(previous.keys()) & set(latest.keys())):
        prev_support = _safe_int(previous[key].get("support"), 0)
        latest_support = _safe_int(latest[key].get("support"), 0)
        if prev_support < int(min_support) or latest_support < int(min_support):
            continue
        common += 1

        acc_drop = _max_drop(
            _optional_float(previous[key].get("hybrid_accuracy")),
            _optional_float(latest[key].get("hybrid_accuracy")),
        )
        if acc_drop is not None:
            if worst_acc_drop is None or acc_drop > worst_acc_drop:
                worst_acc_drop = acc_drop

        gain_drop = _max_drop(
            _optional_float(previous[key].get("hybrid_gain_vs_graph2d")),
            _optional_float(latest[key].get("hybrid_gain_vs_graph2d")),
        )
        if gain_drop is not None:
            if worst_gain_drop is None or gain_drop > worst_gain_drop:
                worst_gain_drop = gain_drop

    return worst_acc_drop, worst_gain_drop, common


def _recommend_threshold(
    *,
    values: List[float],
    quantile: float,
    floor_value: float,
    safety_multiplier: float,
) -> float:
    qv = _quantile(values, quantile)
    base = float(qv) if qv is not None else float(floor_value)
    return round(max(float(floor_value), base * float(safety_multiplier)), 6)


def build_report(
    *,
    history: List[Dict[str, Any]],
    quantile: float,
    min_reports: int,
    safety_multiplier: float,
    label_slice_min_support: int,
    family_slice_min_support: int,
    floor_acc_drop: float,
    floor_gain_drop: float,
    floor_coverage_drop: float,
    floor_label_acc_drop: float,
    floor_label_gain_drop: float,
    floor_family_acc_drop: float,
    floor_family_gain_drop: float,
) -> Dict[str, Any]:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: Dict[str, Any] = {
        "status": "insufficient",
        "generated_at": generated_at,
        "inputs": {
            "quantile": float(quantile),
            "min_reports": int(min_reports),
            "safety_multiplier": float(safety_multiplier),
            "label_slice_min_support": int(label_slice_min_support),
            "family_slice_min_support": int(family_slice_min_support),
        },
        "history": {
            "report_count": len(history),
            "pair_count": max(0, len(history) - 1),
            "oldest_timestamp": str(history[0]["timestamp_text"]) if history else "",
            "latest_timestamp": str(history[-1]["timestamp_text"]) if history else "",
        },
        "distributions": {},
        "recommended_thresholds": {},
        "notes": [],
    }

    if len(history) < int(min_reports):
        report["notes"] = [
            f"insufficient reports: {len(history)} < min_reports {int(min_reports)}"
        ]
        return report

    acc_drops: List[float] = []
    gain_drops: List[float] = []
    coverage_drops: List[float] = []
    label_worst_acc_drops: List[float] = []
    label_worst_gain_drops: List[float] = []
    label_common_counts: List[int] = []
    family_worst_acc_drops: List[float] = []
    family_worst_gain_drops: List[float] = []
    family_common_counts: List[int] = []

    for idx in range(1, len(history)):
        previous = history[idx - 1]
        latest = history[idx]
        acc_drops.append(
            max(
                0.0,
                _safe_float(previous.get("hybrid_accuracy"), 0.0)
                - _safe_float(latest.get("hybrid_accuracy"), 0.0),
            )
        )
        gain_drops.append(
            max(
                0.0,
                _safe_float(previous.get("hybrid_gain_vs_graph2d"), 0.0)
                - _safe_float(latest.get("hybrid_gain_vs_graph2d"), 0.0),
            )
        )
        coverage_drops.append(
            max(
                0.0,
                _safe_float(previous.get("weak_label_coverage"), 0.0)
                - _safe_float(latest.get("weak_label_coverage"), 0.0),
            )
        )

        prev_label = (
            previous.get("label_slices") if isinstance(previous.get("label_slices"), dict) else {}
        )
        new_label = latest.get("label_slices") if isinstance(latest.get("label_slices"), dict) else {}
        worst_label_acc_drop, worst_label_gain_drop, label_common = _pairwise_slice_stats(
            previous=prev_label,
            latest=new_label,
            min_support=int(label_slice_min_support),
        )
        label_common_counts.append(int(label_common))
        if worst_label_acc_drop is not None:
            label_worst_acc_drops.append(float(worst_label_acc_drop))
        if worst_label_gain_drop is not None:
            label_worst_gain_drops.append(float(worst_label_gain_drop))

        prev_family = (
            previous.get("family_slices")
            if isinstance(previous.get("family_slices"), dict)
            else {}
        )
        new_family = (
            latest.get("family_slices") if isinstance(latest.get("family_slices"), dict) else {}
        )
        worst_family_acc_drop, worst_family_gain_drop, family_common = _pairwise_slice_stats(
            previous=prev_family,
            latest=new_family,
            min_support=int(family_slice_min_support),
        )
        family_common_counts.append(int(family_common))
        if worst_family_acc_drop is not None:
            family_worst_acc_drops.append(float(worst_family_acc_drop))
        if worst_family_gain_drop is not None:
            family_worst_gain_drops.append(float(worst_family_gain_drop))

    report["distributions"] = {
        "global": {
            "hybrid_accuracy_drop": _summary(acc_drops, quantile),
            "hybrid_gain_vs_graph2d_drop": _summary(gain_drops, quantile),
            "weak_label_coverage_drop": _summary(coverage_drops, quantile),
        },
        "label_slice": {
            "worst_hybrid_accuracy_drop": _summary(label_worst_acc_drops, quantile),
            "worst_hybrid_gain_drop": _summary(label_worst_gain_drops, quantile),
            "common_count": _summary([float(v) for v in label_common_counts], quantile),
        },
        "family_slice": {
            "worst_hybrid_accuracy_drop": _summary(family_worst_acc_drops, quantile),
            "worst_hybrid_gain_drop": _summary(family_worst_gain_drops, quantile),
            "common_count": _summary([float(v) for v in family_common_counts], quantile),
        },
    }

    label_min_common_q = _quantile([float(v) for v in label_common_counts], 0.5)
    family_min_common_q = _quantile([float(v) for v in family_common_counts], 0.5)

    report["recommended_thresholds"] = {
        "min_reports": int(min_reports),
        "max_hybrid_accuracy_drop": _recommend_threshold(
            values=acc_drops,
            quantile=quantile,
            floor_value=float(floor_acc_drop),
            safety_multiplier=float(safety_multiplier),
        ),
        "max_gain_drop": _recommend_threshold(
            values=gain_drops,
            quantile=quantile,
            floor_value=float(floor_gain_drop),
            safety_multiplier=float(safety_multiplier),
        ),
        "max_coverage_drop": _recommend_threshold(
            values=coverage_drops,
            quantile=quantile,
            floor_value=float(floor_coverage_drop),
            safety_multiplier=float(safety_multiplier),
        ),
        "label_slice_enable": bool(label_worst_acc_drops or label_worst_gain_drops),
        "label_slice_min_common": max(1, int(round(label_min_common_q or 1.0))),
        "label_slice_auto_cap_min_common": True,
        "label_slice_min_support": int(label_slice_min_support),
        "label_slice_max_hybrid_accuracy_drop": _recommend_threshold(
            values=label_worst_acc_drops,
            quantile=quantile,
            floor_value=float(floor_label_acc_drop),
            safety_multiplier=float(safety_multiplier),
        ),
        "label_slice_max_gain_drop": _recommend_threshold(
            values=label_worst_gain_drops,
            quantile=quantile,
            floor_value=float(floor_label_gain_drop),
            safety_multiplier=float(safety_multiplier),
        ),
        "family_slice_enable": bool(family_worst_acc_drops or family_worst_gain_drops),
        "family_slice_min_common": max(1, int(round(family_min_common_q or 1.0))),
        "family_slice_auto_cap_min_common": True,
        "family_slice_min_support": int(family_slice_min_support),
        "family_slice_max_hybrid_accuracy_drop": _recommend_threshold(
            values=family_worst_acc_drops,
            quantile=quantile,
            floor_value=float(floor_family_acc_drop),
            safety_multiplier=float(safety_multiplier),
        ),
        "family_slice_max_gain_drop": _recommend_threshold(
            values=family_worst_gain_drops,
            quantile=quantile,
            floor_value=float(floor_family_gain_drop),
            safety_multiplier=float(safety_multiplier),
        ),
    }
    report["status"] = "ok"
    report["notes"] = [
        "thresholds are derived from consecutive-drop quantiles with safety multiplier",
        "slice min_common uses median observed overlap and relies on auto-cap in runtime checker",
    ]
    return report


def build_markdown(report: Dict[str, Any]) -> str:
    status = str(report.get("status") or "unknown")
    history = report.get("history") if isinstance(report.get("history"), dict) else {}
    thresholds = (
        report.get("recommended_thresholds")
        if isinstance(report.get("recommended_thresholds"), dict)
        else {}
    )
    distributions = (
        report.get("distributions") if isinstance(report.get("distributions"), dict) else {}
    )
    notes = report.get("notes") if isinstance(report.get("notes"), list) else []

    lines: List[str] = []
    lines.append("# Hybrid Blind Drift Threshold Suggestions")
    lines.append("")
    lines.append(f"- Status: `{status}`")
    lines.append(f"- Generated at: `{report.get('generated_at', '')}`")
    lines.append(
        f"- Reports: `{history.get('report_count', 0)}` "
        f"(pairs=`{history.get('pair_count', 0)}`)"
    )
    lines.append("")
    if status != "ok":
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {str(note)}")
        lines.append("")
        return "\n".join(lines)

    global_dist = distributions.get("global") if isinstance(distributions.get("global"), dict) else {}
    label_dist = (
        distributions.get("label_slice") if isinstance(distributions.get("label_slice"), dict) else {}
    )
    family_dist = (
        distributions.get("family_slice")
        if isinstance(distributions.get("family_slice"), dict)
        else {}
    )

    lines.append("## Suggested Thresholds")
    lines.append("")
    lines.append("| Field | Suggested |")
    lines.append("|---|---:|")
    for key in [
        "max_hybrid_accuracy_drop",
        "max_gain_drop",
        "max_coverage_drop",
        "label_slice_min_common",
        "label_slice_min_support",
        "label_slice_max_hybrid_accuracy_drop",
        "label_slice_max_gain_drop",
        "family_slice_min_common",
        "family_slice_min_support",
        "family_slice_max_hybrid_accuracy_drop",
        "family_slice_max_gain_drop",
    ]:
        lines.append(f"| {key} | {thresholds.get(key, 'n/a')} |")
    lines.append("")
    lines.append("## Distribution Snapshot")
    lines.append("")
    lines.append("| Metric | Count | Mean | Max | Quantile |")
    lines.append("|---|---:|---:|---:|---:|")
    for section, key in [
        (global_dist, "hybrid_accuracy_drop"),
        (global_dist, "hybrid_gain_vs_graph2d_drop"),
        (global_dist, "weak_label_coverage_drop"),
        (label_dist, "worst_hybrid_accuracy_drop"),
        (label_dist, "worst_hybrid_gain_drop"),
        (label_dist, "common_count"),
        (family_dist, "worst_hybrid_accuracy_drop"),
        (family_dist, "worst_hybrid_gain_drop"),
        (family_dist, "common_count"),
    ]:
        row = section.get(key) if isinstance(section.get(key), dict) else {}
        lines.append(
            f"| {key} | {row.get('count', 0)} | {row.get('mean', 'n/a')} | "
            f"{row.get('max', 'n/a')} | {row.get('quantile_value', 'n/a')} |"
        )
    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {str(note)}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Suggest Hybrid blind drift thresholds from eval_history snapshots."
    )
    parser.add_argument("--eval-history-dir", default="reports/eval_history")
    parser.add_argument(
        "--output-json",
        default="reports/eval_history/hybrid_blind_drift_threshold_suggestion.json",
    )
    parser.add_argument(
        "--output-md",
        default="reports/eval_history/hybrid_blind_drift_threshold_suggestion.md",
    )
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--min-reports", type=int, default=4)
    parser.add_argument("--safety-multiplier", type=float, default=1.20)
    parser.add_argument("--label-slice-min-support", type=int, default=3)
    parser.add_argument("--family-slice-min-support", type=int, default=5)
    parser.add_argument(
        "--floor-acc-drop",
        "--min-floor-acc-drop",
        dest="floor_acc_drop",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--floor-gain-drop",
        "--min-floor-gain-drop",
        dest="floor_gain_drop",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--floor-coverage-drop",
        "--min-floor-coverage-drop",
        dest="floor_coverage_drop",
        type=float,
        default=0.10,
    )
    parser.add_argument("--floor-label-acc-drop", type=float, default=0.15)
    parser.add_argument("--floor-label-gain-drop", type=float, default=0.15)
    parser.add_argument("--floor-family-acc-drop", type=float, default=0.20)
    parser.add_argument("--floor-family-gain-drop", type=float, default=0.20)
    args = parser.parse_args(argv)

    history = _load_history(Path(args.eval_history_dir))
    report = build_report(
        history=history,
        quantile=max(0.0, min(1.0, float(args.quantile))),
        min_reports=max(2, int(args.min_reports)),
        safety_multiplier=max(1.0, float(args.safety_multiplier)),
        label_slice_min_support=max(1, int(args.label_slice_min_support)),
        family_slice_min_support=max(1, int(args.family_slice_min_support)),
        floor_acc_drop=max(0.0, float(args.floor_acc_drop)),
        floor_gain_drop=max(0.0, float(args.floor_gain_drop)),
        floor_coverage_drop=max(0.0, float(args.floor_coverage_drop)),
        floor_label_acc_drop=max(0.0, float(args.floor_label_acc_drop)),
        floor_label_gain_drop=max(0.0, float(args.floor_label_gain_drop)),
        floor_family_acc_drop=max(0.0, float(args.floor_family_acc_drop)),
        floor_family_gain_drop=max(0.0, float(args.floor_family_gain_drop)),
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
