#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def _parse_ts(value: Any) -> Optional[datetime]:
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


def _default_metrics() -> Dict[str, Any]:
    return {
        "report_count": 0,
        "latest_timestamp": "",
        "previous_timestamp": "",
        "latest_hybrid_accuracy": None,
        "previous_hybrid_accuracy": None,
        "delta_hybrid_accuracy": None,
        "latest_hybrid_gain_vs_graph2d": None,
        "previous_hybrid_gain_vs_graph2d": None,
        "delta_hybrid_gain_vs_graph2d": None,
        "latest_weak_label_coverage": None,
        "previous_weak_label_coverage": None,
        "delta_weak_label_coverage": None,
        "label_slice_enabled": False,
        "latest_label_slice_count": 0,
        "previous_label_slice_count": 0,
        "common_label_slice_count": 0,
        "effective_label_slice_min_common": 0,
        "worst_label_slice_hybrid_accuracy_drop": None,
        "worst_label_slice_hybrid_accuracy_drop_label": "",
        "worst_label_slice_hybrid_gain_drop": None,
        "worst_label_slice_hybrid_gain_drop_label": "",
        "family_slice_enabled": False,
        "latest_family_slice_count": 0,
        "previous_family_slice_count": 0,
        "common_family_slice_count": 0,
        "effective_family_slice_min_common": 0,
        "worst_family_slice_hybrid_accuracy_drop": None,
        "worst_family_slice_hybrid_accuracy_drop_family": "",
        "worst_family_slice_hybrid_gain_drop": None,
        "worst_family_slice_hybrid_gain_drop_family": "",
    }


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def build_drift_markdown(report: Dict[str, Any]) -> str:
    status = str(report.get("status") or "unknown")
    failures = report.get("failures") if isinstance(report.get("failures"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    thresholds = report.get("thresholds") if isinstance(report.get("thresholds"), dict) else {}

    lines: List[str] = []
    lines.append("# Hybrid Blind Drift Alert Report")
    lines.append("")
    lines.append(f"- Status: `{status}`")
    lines.append(f"- Generated at: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    lines.append(
        f"- Report count: `{int(metrics.get('report_count') or 0)}` "
        f"(min: `{int(thresholds.get('min_reports') or 0)}`)"
    )
    lines.append(
        f"- Consecutive window: `{int(thresholds.get('consecutive_drop_window') or 1)}`"
    )
    lines.append("")
    lines.append("## Drift Comparison")
    lines.append("")
    lines.append("| Metric | Previous | Latest | Delta | Max Drop |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        "| Hybrid Accuracy | "
        f"{_fmt_metric(metrics.get('previous_hybrid_accuracy'))} | "
        f"{_fmt_metric(metrics.get('latest_hybrid_accuracy'))} | "
        f"{_fmt_metric(metrics.get('delta_hybrid_accuracy'))} | "
        f"{_fmt_metric(thresholds.get('max_hybrid_accuracy_drop'))} |"
    )
    lines.append(
        "| Hybrid Gain vs Graph2D | "
        f"{_fmt_metric(metrics.get('previous_hybrid_gain_vs_graph2d'))} | "
        f"{_fmt_metric(metrics.get('latest_hybrid_gain_vs_graph2d'))} | "
        f"{_fmt_metric(metrics.get('delta_hybrid_gain_vs_graph2d'))} | "
        f"{_fmt_metric(thresholds.get('max_gain_drop'))} |"
    )
    lines.append(
        "| Weak Label Coverage | "
        f"{_fmt_metric(metrics.get('previous_weak_label_coverage'))} | "
        f"{_fmt_metric(metrics.get('latest_weak_label_coverage'))} | "
        f"{_fmt_metric(metrics.get('delta_weak_label_coverage'))} | "
        f"{_fmt_metric(thresholds.get('max_coverage_drop'))} |"
    )
    lines.append(
        "| Consecutive Breach Count (acc/gain/cov) | n/a | n/a | "
        f"{int(metrics.get('consecutive_breach_hybrid_accuracy') or 0)}/"
        f"{int(metrics.get('consecutive_breach_hybrid_gain_vs_graph2d') or 0)}/"
        f"{int(metrics.get('consecutive_breach_weak_label_coverage') or 0)} | n/a |"
    )
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append(
        f"- Latest timestamp: `{str(metrics.get('latest_timestamp') or '') or 'n/a'}`"
    )
    lines.append(
        f"- Previous timestamp: `{str(metrics.get('previous_timestamp') or '') or 'n/a'}`"
    )
    if bool(thresholds.get("label_slice_enable")):
        lines.append(
            f"- Label-slice enabled: `true` "
            f"(min_support={int(thresholds.get('label_slice_min_support') or 0)}, "
            f"min_common={int(thresholds.get('label_slice_min_common') or 0)}, "
            f"auto_cap_min_common={str(bool(thresholds.get('label_slice_auto_cap_min_common'))).lower()})"
        )
        lines.append(
            f"- Label-slice counts (latest/previous/common): "
            f"`{int(metrics.get('latest_label_slice_count') or 0)}/"
            f"{int(metrics.get('previous_label_slice_count') or 0)}/"
            f"{int(metrics.get('common_label_slice_count') or 0)}`"
        )
        lines.append(
            f"- Label-slice effective min_common: "
            f"`{int(metrics.get('effective_label_slice_min_common') or 0)}`"
        )
        lines.append(
            f"- Worst label hybrid-accuracy drop: "
            f"`{str(metrics.get('worst_label_slice_hybrid_accuracy_drop_label') or 'n/a')}` "
            f"({ _fmt_metric(metrics.get('worst_label_slice_hybrid_accuracy_drop')) })"
        )
        lines.append(
            f"- Worst label gain drop: "
            f"`{str(metrics.get('worst_label_slice_hybrid_gain_drop_label') or 'n/a')}` "
            f"({ _fmt_metric(metrics.get('worst_label_slice_hybrid_gain_drop')) })"
        )
    if bool(thresholds.get("family_slice_enable")):
        lines.append(
            f"- Family-slice enabled: `true` "
            f"(min_support={int(thresholds.get('family_slice_min_support') or 0)}, "
            f"min_common={int(thresholds.get('family_slice_min_common') or 0)}, "
            f"auto_cap_min_common={str(bool(thresholds.get('family_slice_auto_cap_min_common'))).lower()})"
        )
        lines.append(
            f"- Family-slice counts (latest/previous/common): "
            f"`{int(metrics.get('latest_family_slice_count') or 0)}/"
            f"{int(metrics.get('previous_family_slice_count') or 0)}/"
            f"{int(metrics.get('common_family_slice_count') or 0)}`"
        )
        lines.append(
            f"- Family-slice effective min_common: "
            f"`{int(metrics.get('effective_family_slice_min_common') or 0)}`"
        )
        lines.append(
            f"- Worst family hybrid-accuracy drop: "
            f"`{str(metrics.get('worst_family_slice_hybrid_accuracy_drop_family') or 'n/a')}` "
            f"({ _fmt_metric(metrics.get('worst_family_slice_hybrid_accuracy_drop')) })"
        )
        lines.append(
            f"- Worst family gain drop: "
            f"`{str(metrics.get('worst_family_slice_hybrid_gain_drop_family') or 'n/a')}` "
            f"({ _fmt_metric(metrics.get('worst_family_slice_hybrid_gain_drop')) })"
        )
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for item in failures:
            lines.append(f"- {str(item)}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for item in warnings:
            lines.append(f"- {str(item)}")
    lines.append("")
    return "\n".join(lines)


def _normalize_named_slices(
    metrics: Dict[str, Any],
    *,
    source_field: str,
    id_field: str,
) -> Dict[str, Dict[str, Any]]:
    raw_slices = metrics.get(source_field)
    if not isinstance(raw_slices, list):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for item in raw_slices:
        if not isinstance(item, dict):
            continue
        key = str(item.get(id_field) or "").strip()
        if not key:
            continue
        normalized[key] = {
            "support": _safe_int(item.get("support"), 0),
            "hybrid_accuracy": _optional_float(item.get("hybrid_accuracy")),
            "graph2d_accuracy": _optional_float(item.get("graph2d_accuracy")),
            "hybrid_gain_vs_graph2d": _optional_float(item.get("hybrid_gain_vs_graph2d")),
        }
    return normalized


def _normalize_label_slices(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return _normalize_named_slices(metrics, source_field="label_slices", id_field="label")


def _normalize_family_slices(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return _normalize_named_slices(metrics, source_field="family_slices", id_field="family")


def load_hybrid_blind_history(eval_history_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not eval_history_dir.exists():
        return rows

    for path in sorted(eval_history_dir.glob("*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("type") or "").strip() != "hybrid_blind":
            continue
        ts = _parse_ts(payload.get("timestamp"))
        if ts is None:
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        rows.append(
            {
                "timestamp": ts,
                "timestamp_text": str(payload.get("timestamp") or ""),
                "hybrid_accuracy": _safe_float(metrics.get("hybrid_accuracy"), 0.0),
                "hybrid_gain_vs_graph2d": _safe_float(
                    metrics.get("hybrid_gain_vs_graph2d"), 0.0
                ),
                "weak_label_coverage": _safe_float(metrics.get("weak_label_coverage"), 0.0),
                "label_slices": _normalize_label_slices(metrics),
                "family_slices": _normalize_family_slices(metrics),
            }
        )

    rows.sort(key=lambda item: item["timestamp"])
    return rows


def evaluate_hybrid_blind_drift(
    *,
    history: List[Dict[str, Any]],
    min_reports: int,
    max_hybrid_accuracy_drop: float,
    max_gain_drop: float,
    max_coverage_drop: float,
    consecutive_drop_window: int,
    label_slice_enable: bool,
    label_slice_min_common: int,
    label_slice_auto_cap_min_common: bool,
    label_slice_min_support: int,
    label_slice_max_hybrid_accuracy_drop: float,
    label_slice_max_gain_drop: float,
    family_slice_enable: bool,
    family_slice_min_common: int,
    family_slice_auto_cap_min_common: bool,
    family_slice_min_support: int,
    family_slice_max_hybrid_accuracy_drop: float,
    family_slice_max_gain_drop: float,
    allow_missing: bool,
    include_missing_dir_warning: bool = False,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    metrics = _default_metrics()
    metrics["report_count"] = len(history)
    metrics["label_slice_enabled"] = bool(label_slice_enable)
    metrics["family_slice_enabled"] = bool(family_slice_enable)

    if include_missing_dir_warning:
        warnings.append("eval_history directory does not exist.")

    if history:
        latest = history[-1]
        metrics["latest_timestamp"] = str(latest.get("timestamp_text") or "")
        metrics["latest_hybrid_accuracy"] = round(
            _safe_float(latest.get("hybrid_accuracy"), 0.0), 6
        )
        metrics["latest_hybrid_gain_vs_graph2d"] = round(
            _safe_float(latest.get("hybrid_gain_vs_graph2d"), 0.0), 6
        )
        metrics["latest_weak_label_coverage"] = round(
            _safe_float(latest.get("weak_label_coverage"), 0.0), 6
        )

    if len(history) >= 2:
        previous = history[-2]
        metrics["previous_timestamp"] = str(previous.get("timestamp_text") or "")
        metrics["previous_hybrid_accuracy"] = round(
            _safe_float(previous.get("hybrid_accuracy"), 0.0), 6
        )
        metrics["previous_hybrid_gain_vs_graph2d"] = round(
            _safe_float(previous.get("hybrid_gain_vs_graph2d"), 0.0), 6
        )
        metrics["previous_weak_label_coverage"] = round(
            _safe_float(previous.get("weak_label_coverage"), 0.0), 6
        )

    required_reports = max(int(min_reports), int(consecutive_drop_window) + 1)
    if len(history) < required_reports:
        message = f"hybrid_blind reports {len(history)} < min_reports {min_reports}"
        if required_reports != min_reports:
            message = (
                f"hybrid_blind reports {len(history)} < required_reports {required_reports} "
                f"(min_reports={min_reports}, consecutive_drop_window={consecutive_drop_window})"
            )
        warnings.append(message)
        if allow_missing:
            status = "skipped"
        else:
            failures.append(message)
            status = "failed"
        return {
            "status": status,
            "failures": failures,
            "warnings": warnings,
            "metrics": metrics,
            "thresholds": {
                "min_reports": int(min_reports),
                "max_hybrid_accuracy_drop": float(max_hybrid_accuracy_drop),
                "max_gain_drop": float(max_gain_drop),
                "max_coverage_drop": float(max_coverage_drop),
                "consecutive_drop_window": int(consecutive_drop_window),
                "label_slice_enable": bool(label_slice_enable),
                "label_slice_min_common": int(label_slice_min_common),
                "label_slice_auto_cap_min_common": bool(label_slice_auto_cap_min_common),
                "label_slice_min_support": int(label_slice_min_support),
                "label_slice_max_hybrid_accuracy_drop": float(
                    label_slice_max_hybrid_accuracy_drop
                ),
                "label_slice_max_gain_drop": float(label_slice_max_gain_drop),
                "family_slice_enable": bool(family_slice_enable),
                "family_slice_min_common": int(family_slice_min_common),
                "family_slice_auto_cap_min_common": bool(family_slice_auto_cap_min_common),
                "family_slice_min_support": int(family_slice_min_support),
                "family_slice_max_hybrid_accuracy_drop": float(
                    family_slice_max_hybrid_accuracy_drop
                ),
                "family_slice_max_gain_drop": float(family_slice_max_gain_drop),
            },
        }

    previous = history[-2]
    latest = history[-1]
    previous_accuracy = _safe_float(previous.get("hybrid_accuracy"), 0.0)
    latest_accuracy = _safe_float(latest.get("hybrid_accuracy"), 0.0)
    previous_gain = _safe_float(previous.get("hybrid_gain_vs_graph2d"), 0.0)
    latest_gain = _safe_float(latest.get("hybrid_gain_vs_graph2d"), 0.0)
    previous_coverage = _safe_float(previous.get("weak_label_coverage"), 0.0)
    latest_coverage = _safe_float(latest.get("weak_label_coverage"), 0.0)

    delta_accuracy = latest_accuracy - previous_accuracy
    delta_gain = latest_gain - previous_gain
    delta_coverage = latest_coverage - previous_coverage

    metrics["delta_hybrid_accuracy"] = round(delta_accuracy, 6)
    metrics["delta_hybrid_gain_vs_graph2d"] = round(delta_gain, 6)
    metrics["delta_weak_label_coverage"] = round(delta_coverage, 6)

    def _count_consecutive_breaches(metric_key: str, threshold: float) -> int:
        count = 0
        for idx in range(len(history) - 1, 0, -1):
            newer = _safe_float(history[idx].get(metric_key), 0.0)
            older = _safe_float(history[idx - 1].get(metric_key), 0.0)
            drop = older - newer
            if drop > threshold:
                count += 1
            else:
                break
        return count

    accuracy_breach_count = _count_consecutive_breaches("hybrid_accuracy", max_hybrid_accuracy_drop)
    gain_breach_count = _count_consecutive_breaches("hybrid_gain_vs_graph2d", max_gain_drop)
    coverage_breach_count = _count_consecutive_breaches("weak_label_coverage", max_coverage_drop)
    metrics["consecutive_breach_hybrid_accuracy"] = int(accuracy_breach_count)
    metrics["consecutive_breach_hybrid_gain_vs_graph2d"] = int(gain_breach_count)
    metrics["consecutive_breach_weak_label_coverage"] = int(coverage_breach_count)

    accuracy_drop = previous_accuracy - latest_accuracy
    gain_drop = previous_gain - latest_gain
    coverage_drop = previous_coverage - latest_coverage

    if accuracy_drop > max_hybrid_accuracy_drop and accuracy_breach_count >= int(consecutive_drop_window):
        failures.append(
            "hybrid_accuracy drop "
            f"{accuracy_drop:.6f} > {max_hybrid_accuracy_drop:.6f} "
            f"(consecutive={accuracy_breach_count})"
        )
    elif accuracy_drop > max_hybrid_accuracy_drop:
        warnings.append(
            "hybrid_accuracy drop observed but consecutive window not met: "
            f"drop={accuracy_drop:.6f}, consecutive={accuracy_breach_count}, "
            f"required={int(consecutive_drop_window)}"
        )

    if gain_drop > max_gain_drop and gain_breach_count >= int(consecutive_drop_window):
        failures.append(
            "hybrid_gain_vs_graph2d drop "
            f"{gain_drop:.6f} > {max_gain_drop:.6f} "
            f"(consecutive={gain_breach_count})"
        )
    elif gain_drop > max_gain_drop:
        warnings.append(
            "hybrid_gain_vs_graph2d drop observed but consecutive window not met: "
            f"drop={gain_drop:.6f}, consecutive={gain_breach_count}, "
            f"required={int(consecutive_drop_window)}"
        )

    if coverage_drop > max_coverage_drop and coverage_breach_count >= int(consecutive_drop_window):
        failures.append(
            "weak_label_coverage drop "
            f"{coverage_drop:.6f} > {max_coverage_drop:.6f} "
            f"(consecutive={coverage_breach_count})"
        )
    elif coverage_drop > max_coverage_drop:
        warnings.append(
            "weak_label_coverage drop observed but consecutive window not met: "
            f"drop={coverage_drop:.6f}, consecutive={coverage_breach_count}, "
            f"required={int(consecutive_drop_window)}"
        )

    if bool(label_slice_enable):
        previous_slices = (
            previous.get("label_slices")
            if isinstance(previous.get("label_slices"), dict)
            else {}
        )
        latest_slices = (
            latest.get("label_slices")
            if isinstance(latest.get("label_slices"), dict)
            else {}
        )
        metrics["previous_label_slice_count"] = len(previous_slices)
        metrics["latest_label_slice_count"] = len(latest_slices)
        required_label_common = int(label_slice_min_common)
        if bool(label_slice_auto_cap_min_common) and previous_slices and latest_slices:
            required_label_common = min(
                required_label_common,
                len(previous_slices),
                len(latest_slices),
            )
        required_label_common = max(1, int(required_label_common))
        metrics["effective_label_slice_min_common"] = int(required_label_common)

        common_labels: List[str] = []
        for label in sorted(set(previous_slices.keys()) & set(latest_slices.keys())):
            prev_support = _safe_int(previous_slices[label].get("support"), 0)
            latest_support = _safe_int(latest_slices[label].get("support"), 0)
            if prev_support >= int(label_slice_min_support) and latest_support >= int(
                label_slice_min_support
            ):
                common_labels.append(label)

        metrics["common_label_slice_count"] = len(common_labels)
        if len(common_labels) < int(required_label_common):
            message = (
                "label-slice drift check has insufficient overlap: "
                f"common={len(common_labels)} < min_common={int(required_label_common)} "
                f"(min_support={int(label_slice_min_support)})"
            )
            if allow_missing:
                warnings.append(message)
            else:
                failures.append(message)
        else:
            worst_acc_drop: Optional[float] = None
            worst_acc_drop_label = ""
            worst_gain_drop: Optional[float] = None
            worst_gain_drop_label = ""
            for label in common_labels:
                prev_metrics = previous_slices[label]
                latest_metrics = latest_slices[label]

                prev_hybrid_acc = _optional_float(prev_metrics.get("hybrid_accuracy"))
                latest_hybrid_acc = _optional_float(latest_metrics.get("hybrid_accuracy"))
                if prev_hybrid_acc is not None and latest_hybrid_acc is not None:
                    acc_drop = prev_hybrid_acc - latest_hybrid_acc
                    if worst_acc_drop is None or acc_drop > worst_acc_drop:
                        worst_acc_drop = acc_drop
                        worst_acc_drop_label = label

                prev_hybrid_gain = _optional_float(
                    prev_metrics.get("hybrid_gain_vs_graph2d")
                )
                latest_hybrid_gain = _optional_float(
                    latest_metrics.get("hybrid_gain_vs_graph2d")
                )
                if prev_hybrid_gain is not None and latest_hybrid_gain is not None:
                    gain_drop_label = prev_hybrid_gain - latest_hybrid_gain
                    if worst_gain_drop is None or gain_drop_label > worst_gain_drop:
                        worst_gain_drop = gain_drop_label
                        worst_gain_drop_label = label

            metrics["worst_label_slice_hybrid_accuracy_drop"] = (
                round(worst_acc_drop, 6) if worst_acc_drop is not None else None
            )
            metrics["worst_label_slice_hybrid_accuracy_drop_label"] = worst_acc_drop_label
            metrics["worst_label_slice_hybrid_gain_drop"] = (
                round(worst_gain_drop, 6) if worst_gain_drop is not None else None
            )
            metrics["worst_label_slice_hybrid_gain_drop_label"] = worst_gain_drop_label

            if (
                worst_acc_drop is not None
                and worst_acc_drop > float(label_slice_max_hybrid_accuracy_drop)
            ):
                failures.append(
                    "label-slice worst hybrid_accuracy drop "
                    f"{worst_acc_drop:.6f} > {float(label_slice_max_hybrid_accuracy_drop):.6f} "
                    f"(label={worst_acc_drop_label or 'n/a'})"
                )
            if worst_gain_drop is not None and worst_gain_drop > float(
                label_slice_max_gain_drop
            ):
                failures.append(
                    "label-slice worst hybrid_gain_vs_graph2d drop "
                    f"{worst_gain_drop:.6f} > {float(label_slice_max_gain_drop):.6f} "
                    f"(label={worst_gain_drop_label or 'n/a'})"
                )

    if bool(family_slice_enable):
        previous_family_slices = (
            previous.get("family_slices")
            if isinstance(previous.get("family_slices"), dict)
            else {}
        )
        latest_family_slices = (
            latest.get("family_slices")
            if isinstance(latest.get("family_slices"), dict)
            else {}
        )
        metrics["previous_family_slice_count"] = len(previous_family_slices)
        metrics["latest_family_slice_count"] = len(latest_family_slices)
        required_family_common = int(family_slice_min_common)
        if bool(family_slice_auto_cap_min_common) and previous_family_slices and latest_family_slices:
            required_family_common = min(
                required_family_common,
                len(previous_family_slices),
                len(latest_family_slices),
            )
        required_family_common = max(1, int(required_family_common))
        metrics["effective_family_slice_min_common"] = int(required_family_common)

        common_families: List[str] = []
        for family in sorted(
            set(previous_family_slices.keys()) & set(latest_family_slices.keys())
        ):
            prev_support = _safe_int(previous_family_slices[family].get("support"), 0)
            latest_support = _safe_int(latest_family_slices[family].get("support"), 0)
            if prev_support >= int(family_slice_min_support) and latest_support >= int(
                family_slice_min_support
            ):
                common_families.append(family)

        metrics["common_family_slice_count"] = len(common_families)
        if len(common_families) < int(required_family_common):
            message = (
                "family-slice drift check has insufficient overlap: "
                f"common={len(common_families)} < min_common={int(required_family_common)} "
                f"(min_support={int(family_slice_min_support)})"
            )
            if allow_missing:
                warnings.append(message)
            else:
                failures.append(message)
        else:
            worst_acc_drop: Optional[float] = None
            worst_acc_drop_family = ""
            worst_gain_drop: Optional[float] = None
            worst_gain_drop_family = ""
            for family in common_families:
                prev_metrics = previous_family_slices[family]
                latest_metrics = latest_family_slices[family]

                prev_hybrid_acc = _optional_float(prev_metrics.get("hybrid_accuracy"))
                latest_hybrid_acc = _optional_float(latest_metrics.get("hybrid_accuracy"))
                if prev_hybrid_acc is not None and latest_hybrid_acc is not None:
                    acc_drop = prev_hybrid_acc - latest_hybrid_acc
                    if worst_acc_drop is None or acc_drop > worst_acc_drop:
                        worst_acc_drop = acc_drop
                        worst_acc_drop_family = family

                prev_hybrid_gain = _optional_float(
                    prev_metrics.get("hybrid_gain_vs_graph2d")
                )
                latest_hybrid_gain = _optional_float(
                    latest_metrics.get("hybrid_gain_vs_graph2d")
                )
                if prev_hybrid_gain is not None and latest_hybrid_gain is not None:
                    gain_drop_family = prev_hybrid_gain - latest_hybrid_gain
                    if worst_gain_drop is None or gain_drop_family > worst_gain_drop:
                        worst_gain_drop = gain_drop_family
                        worst_gain_drop_family = family

            metrics["worst_family_slice_hybrid_accuracy_drop"] = (
                round(worst_acc_drop, 6) if worst_acc_drop is not None else None
            )
            metrics["worst_family_slice_hybrid_accuracy_drop_family"] = (
                worst_acc_drop_family
            )
            metrics["worst_family_slice_hybrid_gain_drop"] = (
                round(worst_gain_drop, 6) if worst_gain_drop is not None else None
            )
            metrics["worst_family_slice_hybrid_gain_drop_family"] = (
                worst_gain_drop_family
            )

            if (
                worst_acc_drop is not None
                and worst_acc_drop > float(family_slice_max_hybrid_accuracy_drop)
            ):
                failures.append(
                    "family-slice worst hybrid_accuracy drop "
                    f"{worst_acc_drop:.6f} > {float(family_slice_max_hybrid_accuracy_drop):.6f} "
                    f"(family={worst_acc_drop_family or 'n/a'})"
                )
            if worst_gain_drop is not None and worst_gain_drop > float(
                family_slice_max_gain_drop
            ):
                failures.append(
                    "family-slice worst hybrid_gain_vs_graph2d drop "
                    f"{worst_gain_drop:.6f} > {float(family_slice_max_gain_drop):.6f} "
                    f"(family={worst_gain_drop_family or 'n/a'})"
                )

    status = "passed" if not failures else "failed"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
        "thresholds": {
            "min_reports": int(min_reports),
            "max_hybrid_accuracy_drop": float(max_hybrid_accuracy_drop),
            "max_gain_drop": float(max_gain_drop),
            "max_coverage_drop": float(max_coverage_drop),
            "consecutive_drop_window": int(consecutive_drop_window),
            "label_slice_enable": bool(label_slice_enable),
            "label_slice_min_common": int(label_slice_min_common),
            "label_slice_auto_cap_min_common": bool(label_slice_auto_cap_min_common),
            "label_slice_min_support": int(label_slice_min_support),
            "label_slice_max_hybrid_accuracy_drop": float(
                label_slice_max_hybrid_accuracy_drop
            ),
            "label_slice_max_gain_drop": float(label_slice_max_gain_drop),
            "family_slice_enable": bool(family_slice_enable),
            "family_slice_min_common": int(family_slice_min_common),
            "family_slice_auto_cap_min_common": bool(family_slice_auto_cap_min_common),
            "family_slice_min_support": int(family_slice_min_support),
            "family_slice_max_hybrid_accuracy_drop": float(
                family_slice_max_hybrid_accuracy_drop
            ),
            "family_slice_max_gain_drop": float(family_slice_max_gain_drop),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check drift alerts for hybrid blind eval history."
    )
    parser.add_argument("--eval-history-dir", default="reports/eval_history")
    parser.add_argument(
        "--output-json",
        default="reports/eval_history/hybrid_blind_drift_alert_report.json",
    )
    parser.add_argument(
        "--output-md",
        default="reports/eval_history/hybrid_blind_drift_alert_report.md",
    )
    parser.add_argument("--min-reports", type=int, default=2)
    parser.add_argument("--max-hybrid-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--max-gain-drop", type=float, default=0.05)
    parser.add_argument("--max-coverage-drop", type=float, default=0.1)
    parser.add_argument("--consecutive-drop-window", type=int, default=1)
    parser.add_argument("--label-slice-enable", action="store_true")
    parser.add_argument("--label-slice-min-common", type=int, default=3)
    parser.add_argument(
        "--label-slice-auto-cap-min-common",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--label-slice-min-support", type=int, default=3)
    parser.add_argument(
        "--label-slice-max-hybrid-accuracy-drop", type=float, default=0.15
    )
    parser.add_argument("--label-slice-max-gain-drop", type=float, default=0.15)
    parser.add_argument("--family-slice-enable", action="store_true")
    parser.add_argument("--family-slice-min-common", type=int, default=2)
    parser.add_argument(
        "--family-slice-auto-cap-min-common",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--family-slice-min-support", type=int, default=5)
    parser.add_argument(
        "--family-slice-max-hybrid-accuracy-drop", type=float, default=0.20
    )
    parser.add_argument("--family-slice-max-gain-drop", type=float, default=0.20)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    eval_history_dir = Path(args.eval_history_dir)
    include_missing_dir_warning = not eval_history_dir.exists()
    history = load_hybrid_blind_history(eval_history_dir)
    report = evaluate_hybrid_blind_drift(
        history=history,
        min_reports=max(1, int(args.min_reports)),
        max_hybrid_accuracy_drop=float(args.max_hybrid_accuracy_drop),
        max_gain_drop=float(args.max_gain_drop),
        max_coverage_drop=float(args.max_coverage_drop),
        consecutive_drop_window=max(1, int(args.consecutive_drop_window)),
        label_slice_enable=bool(args.label_slice_enable),
        label_slice_min_common=max(1, int(args.label_slice_min_common)),
        label_slice_auto_cap_min_common=bool(args.label_slice_auto_cap_min_common),
        label_slice_min_support=max(1, int(args.label_slice_min_support)),
        label_slice_max_hybrid_accuracy_drop=float(
            args.label_slice_max_hybrid_accuracy_drop
        ),
        label_slice_max_gain_drop=float(args.label_slice_max_gain_drop),
        family_slice_enable=bool(args.family_slice_enable),
        family_slice_min_common=max(1, int(args.family_slice_min_common)),
        family_slice_auto_cap_min_common=bool(args.family_slice_auto_cap_min_common),
        family_slice_min_support=max(1, int(args.family_slice_min_support)),
        family_slice_max_hybrid_accuracy_drop=float(
            args.family_slice_max_hybrid_accuracy_drop
        ),
        family_slice_max_gain_drop=float(args.family_slice_max_gain_drop),
        allow_missing=bool(args.allow_missing),
        include_missing_dir_warning=include_missing_dir_warning,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_drift_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") in {"passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
