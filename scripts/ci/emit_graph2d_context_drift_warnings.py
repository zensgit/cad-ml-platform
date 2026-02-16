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


def build_warning_lines(report: Dict[str, Any]) -> List[str]:
    alerts = report.get("alerts")
    if not isinstance(alerts, list):
        return []
    lines: List[str] = []
    for item in alerts:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message", "")).strip()
        if not message:
            continue
        lines.append(f"::warning title=Graph2D Context Drift::{message}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit GitHub warning annotations from Graph2D context drift alert report."
    )
    parser.add_argument("--report-json", required=True, help="Alert report json path")
    args = parser.parse_args()

    report = _read_json(Path(str(args.report_json)))
    lines = build_warning_lines(report)
    for line in lines:
        print(line)
    print(f"warning_annotations={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
