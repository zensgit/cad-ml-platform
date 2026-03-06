#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_annotation_lines(report: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    failures = report.get("failures")
    if isinstance(failures, list):
        for item in failures:
            text = str(item or "").strip()
            if text:
                lines.append(f"::warning title=Graph2D Review Gate Failure::{text}")
    warnings = report.get("warnings")
    if isinstance(warnings, list):
        for item in warnings:
            text = str(item or "").strip()
            if text:
                lines.append(f"::warning title=Graph2D Review Gate Warning::{text}")
    status = str(report.get("status") or "").strip().lower()
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    if status and metrics:
        lines.append(
            " ".join(
                [
                    "::notice title=Graph2D Review Gate Metrics::",
                    f"status={status}",
                    f"candidate_rate={metrics.get('candidate_rate', 0)}",
                    f"hybrid_rejected_rate={metrics.get('hybrid_rejected_rate', 0)}",
                    f"conflict_rate={metrics.get('conflict_rate', 0)}",
                    f"low_confidence_rate={metrics.get('low_confidence_rate', 0)}",
                ]
            )
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit GitHub warning annotations from Graph2D review-pack gate report."
    )
    parser.add_argument("--report-json", required=True, help="Gate report json path")
    args = parser.parse_args()

    report = _read_json(Path(str(args.report_json)))
    lines = build_annotation_lines(report)
    for line in lines:
        print(line)
    print(f"warning_annotations={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
