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


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_annotation_line(index_payload: Dict[str, Any]) -> str:
    overview = _as_dict(index_payload.get("overview"))
    severity = str(overview.get("severity", "clear")).strip().lower()
    status = str(overview.get("status", "clear")).strip().lower()
    alert_count = _safe_int(overview.get("alert_count"), 0)
    reason = str(overview.get("severity_reason", "")).strip()
    message = (
        f"severity={severity}, status={status}, alert_count={alert_count}"
        + (f", reason={reason}" if reason else "")
    )
    if severity == "failed":
        return f"::error title=Graph2D Context Drift Index::{message}"
    if severity in {"warn", "alerted"}:
        return f"::warning title=Graph2D Context Drift Index::{message}"
    return f"::notice title=Graph2D Context Drift Index::{message}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit GitHub annotation from Graph2D context drift index json."
    )
    parser.add_argument("--index-json", required=True, help="Path to index json.")
    args = parser.parse_args()

    payload = _read_json(Path(str(args.index_json)))
    if not payload:
        print("index_annotation_emitted=0")
        return 0
    print(build_annotation_line(payload))
    print("index_annotation_emitted=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
