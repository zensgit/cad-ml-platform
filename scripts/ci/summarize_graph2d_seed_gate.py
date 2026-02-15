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


def build_summary(summary: Dict[str, Any], title: str, log_text: str = "") -> str:
    gate = summary.get("gate") if isinstance(summary.get("gate"), dict) else {}
    gate_enabled = bool(gate.get("enabled", False))
    gate_passed = bool(gate.get("passed", False))
    gate_failures = gate.get("failures") if isinstance(gate.get("failures"), list) else []

    strict_mean = _safe_float(summary.get("strict_accuracy_mean"), -1.0)
    strict_min = _safe_float(summary.get("strict_accuracy_min"), -1.0)
    strict_max = _safe_float(summary.get("strict_accuracy_max"), -1.0)
    manifest_labels_min = _safe_int(summary.get("manifest_distinct_labels_min"), 0)
    manifest_labels_max = _safe_int(summary.get("manifest_distinct_labels_max"), 0)

    out: list[str] = []
    out.append(f"## {title}")
    out.append("")
    out.append("| Check | Status | Evidence |")
    out.append("|---|---|---|")
    out.append(
        f"| Seed gate enabled | {_bool_mark(gate_enabled)} | `{gate_enabled}` |"
    )
    out.append(
        f"| Seed gate passed | {_bool_mark((not gate_enabled) or gate_passed)} | `{gate_passed}` |"
    )
    out.append(
        f"| Runs OK / Total | {_bool_mark(_safe_int(summary.get('num_error_runs'), 0) == 0)} | "
        f"`{_safe_int(summary.get('num_success_runs'), 0)} / {_safe_int(summary.get('num_runs'), 0)}` |"
    )
    out.append(
        f"| Strict accuracy (mean/min/max) | {_bool_mark(strict_mean >= 0 and strict_min >= 0)} | "
        f"`{strict_mean:.6f} / {strict_min:.6f} / {strict_max:.6f}` |"
    )
    out.append(
        f"| Manifest distinct labels (min/max) | {_bool_mark(manifest_labels_min > 0)} | "
        f"`{manifest_labels_min} / {manifest_labels_max}` |"
    )
    out.append(
        f"| Config | ✅ | `{summary.get('config', 'N/A')}` |"
    )
    out.append(
        f"| Profile / Label mode | ✅ | "
        f"`{summary.get('training_profile', 'N/A')} / {summary.get('manifest_label_mode', 'N/A')}` |"
    )
    out.append("")
    if gate_failures:
        out.append("Gate failures:")
        out.append("```text")
        out.extend([str(item) for item in gate_failures])
        out.append("```")
    out.append("Tail:")
    out.append("```text")
    lines = log_text.splitlines()
    tail = lines[-20:] if lines else ["<no log provided>"]
    out.extend(tail)
    out.append("```")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Graph2D seed gate output into markdown table."
    )
    parser.add_argument(
        "--summary-json", required=True, help="Path to seed_sweep_summary.json"
    )
    parser.add_argument("--title", required=True, help="Section title")
    parser.add_argument(
        "--log-file", default="", help="Optional log file path for tail display"
    )
    args = parser.parse_args()

    summary = _read_json(Path(args.summary_json))
    if not summary:
        print(f"## {args.title}\n\nNo Graph2D seed gate summary found.\n")
        return 0

    log_text = ""
    if args.log_file:
        log_path = Path(args.log_file)
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")

    print(build_summary(summary, args.title, log_text=log_text), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
