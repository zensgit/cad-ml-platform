#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def build_markdown(report: Dict[str, Any], title: str) -> str:
    status = str(report.get("status") or "unknown")
    reason = str(report.get("reason") or "")
    failures = (
        report.get("failures") if isinstance(report.get("failures"), list) else []
    )
    warnings = (
        report.get("warnings") if isinstance(report.get("warnings"), list) else []
    )
    current = report.get("current") if isinstance(report.get("current"), dict) else {}
    baseline = (
        report.get("baseline") if isinstance(report.get("baseline"), dict) else {}
    )
    thresholds = (
        report.get("thresholds") if isinstance(report.get("thresholds"), dict) else {}
    )

    cur_before = (
        current.get("metrics_before")
        if isinstance(current.get("metrics_before"), dict)
        else {}
    )
    cur_after = (
        current.get("metrics_after")
        if isinstance(current.get("metrics_after"), dict)
        else {}
    )
    base_after = (
        baseline.get("metrics_after")
        if isinstance(baseline.get("metrics_after"), dict)
        else {}
    )

    out: list[str] = []
    out.append(f"## {title}")
    out.append("")
    out.append("| Check | Value |")
    out.append("|---|---|")
    out.append(f"| status | `{status}` |")
    out.append(f"| reason | `{reason}` |")
    out.append(f"| failures | `{len(failures)}` |")
    out.append(f"| warnings | `{len(warnings)}` |")
    out.append(f"| n_samples | `{_safe_int(current.get('n_samples'), 0)}` |")
    out.append(
        "| current_before (ece/brier/mce) | "
        f"`{_safe_float(cur_before.get('ece'), 0.0):.6f} / "
        f"{_safe_float(cur_before.get('brier_score'), 0.0):.6f} / "
        f"{_safe_float(cur_before.get('mce'), 0.0):.6f}` |"
    )
    out.append(
        "| current_after (ece/brier/mce) | "
        f"`{_safe_float(cur_after.get('ece'), 0.0):.6f} / "
        f"{_safe_float(cur_after.get('brier_score'), 0.0):.6f} / "
        f"{_safe_float(cur_after.get('mce'), 0.0):.6f}` |"
    )
    out.append(
        "| baseline_after (ece/brier/mce) | "
        f"`{_safe_float(base_after.get('ece'), 0.0):.6f} / "
        f"{_safe_float(base_after.get('brier_score'), 0.0):.6f} / "
        f"{_safe_float(base_after.get('mce'), 0.0):.6f}` |"
    )
    out.append(
        "| thresholds | "
        f"`min_samples={_safe_int(thresholds.get('min_samples'), 0)}, "
        f"max_ece_inc={_safe_float(thresholds.get('max_ece_increase'), 0.0):.4f}, "
        f"max_brier_inc={_safe_float(thresholds.get('max_brier_increase'), 0.0):.4f}, "
        f"max_mce_inc={_safe_float(thresholds.get('max_mce_increase'), 0.0):.4f}` |"
    )

    if failures:
        out.append("")
        out.append("Failures:")
        out.append("```text")
        out.extend([str(item) for item in failures])
        out.append("```")
    if warnings:
        out.append("")
        out.append("Warnings:")
        out.append("```text")
        out.extend([str(item) for item in warnings])
        out.append("```")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render Hybrid confidence calibration gate report as markdown."
    )
    parser.add_argument("--report-json", required=True, help="Report json path.")
    parser.add_argument("--title", required=True, help="Markdown section title.")
    args = parser.parse_args(argv)

    report = _read_json(Path(args.report_json))
    if not report:
        print(f"## {args.title}\n\nNo hybrid calibration gate report found.\n")
        return 0
    print(build_markdown(report, args.title), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
