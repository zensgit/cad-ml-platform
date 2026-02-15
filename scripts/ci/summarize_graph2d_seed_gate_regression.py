#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _bool_mark(ok: bool) -> str:
    return "✅" if ok else "❌"


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


def build_summary(report: Dict[str, Any], title: str) -> str:
    status = str(report.get("status", "failed"))
    failures = report.get("failures") if isinstance(report.get("failures"), list) else []
    channel = str(report.get("channel", "unknown"))
    baseline = report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    current = report.get("current") if isinstance(report.get("current"), dict) else {}
    thresholds = (
        report.get("thresholds") if isinstance(report.get("thresholds"), dict) else {}
    )
    threshold_source = (
        report.get("threshold_source")
        if isinstance(report.get("threshold_source"), dict)
        else {}
    )
    baseline_metadata = (
        report.get("baseline_metadata")
        if isinstance(report.get("baseline_metadata"), dict)
        else {}
    )

    out: list[str] = []
    out.append(f"## {title}")
    out.append("")
    out.append("| Check | Status | Evidence |")
    out.append("|---|---|---|")
    out.append(f"| Channel | ✅ | `{channel}` |")
    out.append(f"| Regression status | {_bool_mark(status == 'passed')} | `{status}` |")
    out.append(
        "| strict_accuracy_mean (cur/base) | "
        f"{_bool_mark(_safe_float(current.get('strict_accuracy_mean'), -1) >= 0)} | "
        f"`{_safe_float(current.get('strict_accuracy_mean'), -1):.6f} / "
        f"{_safe_float(baseline.get('strict_accuracy_mean'), -1):.6f}` |"
    )
    out.append(
        "| strict_accuracy_min (cur/base) | "
        f"{_bool_mark(_safe_float(current.get('strict_accuracy_min'), -1) >= 0)} | "
        f"`{_safe_float(current.get('strict_accuracy_min'), -1):.6f} / "
        f"{_safe_float(baseline.get('strict_accuracy_min'), -1):.6f}` |"
    )
    out.append(
        "| strict_top_pred_ratio_max (cur/base) | "
        f"{_bool_mark(_safe_float(current.get('strict_top_pred_ratio_max'), -1) >= 0)} | "
        f"`{_safe_float(current.get('strict_top_pred_ratio_max'), -1):.6f} / "
        f"{_safe_float(baseline.get('strict_top_pred_ratio_max'), -1):.6f}` |"
    )
    out.append(
        "| strict_low_conf_ratio_max (cur/base) | "
        f"{_bool_mark(_safe_float(current.get('strict_low_conf_ratio_max'), -1) >= 0)} | "
        f"`{_safe_float(current.get('strict_low_conf_ratio_max'), -1):.6f} / "
        f"{_safe_float(baseline.get('strict_low_conf_ratio_max'), -1):.6f}` |"
    )
    out.append(
        "| manifest_distinct_labels_min (cur/base) | "
        f"{_bool_mark(_safe_int(current.get('manifest_distinct_labels_min'), -1) >= 0)} | "
        f"`{_safe_int(current.get('manifest_distinct_labels_min'), -1)} / "
        f"{_safe_int(baseline.get('manifest_distinct_labels_min'), -1)}` |"
    )
    out.append(
        "| Thresholds | ✅ | "
        f"`mean_drop<={_safe_float(thresholds.get('max_accuracy_mean_drop'), -1):.3f}, "
        f"min_drop<={_safe_float(thresholds.get('max_accuracy_min_drop'), -1):.3f}, "
        f"top_pred_inc<={_safe_float(thresholds.get('max_top_pred_ratio_increase'), -1):.3f}, "
        f"low_conf_inc<={_safe_float(thresholds.get('max_low_conf_ratio_increase'), -1):.3f}, "
        f"labels_drop<={_safe_int(thresholds.get('max_distinct_labels_drop'), -1)}, "
        f"baseline_age<={_safe_int(thresholds.get('max_baseline_age_days'), -1)}d, "
        f"snapshot_exists={bool(thresholds.get('require_snapshot_ref_exists', False))}, "
        f"snapshot_match={bool(thresholds.get('require_snapshot_metrics_match', False))}` |"
    )
    out.append(
        "| Threshold source | ✅ | "
        f"`config={threshold_source.get('config', '')}, "
        f"loaded={bool(threshold_source.get('config_loaded', False))}, "
        f"cli_overrides={len(threshold_source.get('cli_overrides') or {})}` |"
    )
    out.append(
        "| Baseline metadata | "
        f"{_bool_mark(_safe_int(baseline_metadata.get('age_days'), -1) >= 0 and bool(baseline_metadata.get('snapshot_exists', False)) and bool(baseline_metadata.get('snapshot_metrics_match', False)))} | "
        f"`date={baseline_metadata.get('date', '')}, "
        f"age_days={_safe_int(baseline_metadata.get('age_days'), -1)}, "
        f"snapshot_exists={bool(baseline_metadata.get('snapshot_exists', False))}, "
        f"snapshot_metrics_match={baseline_metadata.get('snapshot_metrics_match')}` |"
    )
    out.append("")
    if failures:
        out.append("Regression failures:")
        out.append("```text")
        out.extend([str(item) for item in failures])
        out.append("```")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Graph2D seed gate regression report into markdown."
    )
    parser.add_argument("--report-json", required=True, help="Path to regression report json")
    parser.add_argument("--title", required=True, help="Section title")
    args = parser.parse_args()

    report = _read_json(Path(args.report_json))
    if not report:
        print(f"## {args.title}\n\nNo regression report found.\n")
        return 0

    print(build_summary(report, args.title), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
