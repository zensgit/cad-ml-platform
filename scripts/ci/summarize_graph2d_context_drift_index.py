#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _bool_mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def build_summary(index_payload: Dict[str, Any], title: str) -> str:
    overview = _as_dict(index_payload.get("overview"))
    artifacts = _as_dict(index_payload.get("artifacts"))
    summaries = _as_dict(index_payload.get("summaries"))

    status = str(overview.get("status", "clear"))
    alert_count = _safe_int(overview.get("alert_count"), 0)
    history_entries = _safe_int(overview.get("history_entries"), 0)
    recent_window = _safe_int(overview.get("recent_window"), 0)
    drift_key_count = _safe_int(overview.get("drift_key_count"), 0)
    top_key = _as_dict(overview.get("top_drift_key"))

    artifact_rows = [
        _as_dict(value)
        for value in artifacts.values()
        if _as_dict(value)
    ]
    artifact_total = len(artifact_rows)
    artifact_present = len([row for row in artifact_rows if bool(row.get("exists", False))])

    out: List[str] = []
    out.append(f"## {title}")
    out.append("")
    out.append("| Check | Status | Evidence |")
    out.append("|---|---|---|")
    out.append(
        f"| Status | {_bool_mark(status != 'failed')} | `{status}` |"
    )
    out.append(
        f"| Alert count | {_bool_mark(alert_count == 0)} | `{alert_count}` |"
    )
    out.append(
        f"| History entries | {_bool_mark(history_entries > 0)} | `{history_entries}` |"
    )
    out.append(
        f"| Recent window | {_bool_mark(recent_window >= 0)} | `{recent_window}` |"
    )
    out.append(
        f"| Drift key count | {_bool_mark(drift_key_count >= 0)} | `{drift_key_count}` |"
    )
    out.append(
        "| Top drift key | ✅ | "
        f"`{str(top_key.get('key', '')).strip() or '-'}:{_safe_int(top_key.get('count'), 0)}` |"
    )
    out.append(
        f"| Artifact coverage | {_bool_mark(artifact_total == artifact_present)} | "
        f"`{artifact_present}/{artifact_total}` |"
    )
    out.append("")

    if artifact_rows:
        out.append("| Artifact | Exists | Path |")
        out.append("|---|---|---|")
        for name, raw in artifacts.items():
            row = _as_dict(raw)
            out.append(
                f"| `{name}` | `{bool(row.get('exists', False))}` | `{str(row.get('path', ''))}` |"
            )
        out.append("")

    alert_rows = _as_list(_as_dict(summaries.get("alerts")).get("rows"))
    if alert_rows:
        out.append("Alert rows:")
        out.append("```text")
        for item in alert_rows:
            one = _as_dict(item)
            key = str(one.get("key", "")).strip()
            count = _safe_int(one.get("count"), 0)
            threshold = _safe_int(one.get("threshold"), 0)
            out.append(f"{key}: {count} >= {threshold}")
        out.append("```")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Graph2D context drift index into markdown."
    )
    parser.add_argument("--index-json", required=True, help="Path to index json")
    parser.add_argument("--title", required=True, help="Section title")
    args = parser.parse_args()

    index_payload = _read_json(Path(str(args.index_json)))
    if not index_payload:
        print(f"## {args.title}\n\nNo Graph2D context drift index found.\n")
        return 0

    print(build_summary(index_payload, str(args.title)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
