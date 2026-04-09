#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from scripts.ci import archive_hybrid_blind_eval_history as archive_mod
except ModuleNotFoundError:
    # Allow running this file directly via "python scripts/ci/....py".
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.ci import archive_hybrid_blind_eval_history as archive_mod


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_file_text(value: Any) -> str:
    text = _safe_str(value, "unknown")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "unknown"


def _normalize_commit(value: Any, default: str = "[redacted]") -> str:
    text = _safe_str(value, "")
    if not text:
        return default
    if text == "[redacted]":
        return text
    lowered = text.lower()
    if re.fullmatch(r"[a-f0-9]{6,40}", lowered):
        return lowered
    return default


def _parse_iso_ts(value: str) -> Optional[datetime]:
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


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _parse_delta_series(raw: str, count: int) -> List[float]:
    text = str(raw or "").strip()
    if not text:
        return [0.0] * max(1, count)
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(_safe_float(token, 0.0))
    if not values:
        return [0.0] * max(1, count)
    if len(values) == 1:
        return [values[0]] * max(1, count)
    if len(values) < count:
        values.extend([values[-1]] * (count - len(values)))
    return values[:count]


def _apply_deltas(
    payload: Dict[str, Any],
    *,
    delta_hybrid_accuracy: float,
    delta_graph2d_accuracy: float,
    delta_coverage: float,
) -> Dict[str, Any]:
    cloned = json.loads(json.dumps(payload, ensure_ascii=False))
    metrics = cloned.get("metrics") if isinstance(cloned.get("metrics"), dict) else {}

    hybrid_accuracy = _clamp01(
        _safe_float(metrics.get("hybrid_accuracy"), 0.0) + float(delta_hybrid_accuracy)
    )
    graph2d_accuracy = _clamp01(
        _safe_float(metrics.get("graph2d_accuracy"), 0.0) + float(delta_graph2d_accuracy)
    )
    coverage = _clamp01(
        _safe_float(metrics.get("weak_label_coverage"), 0.0) + float(delta_coverage)
    )
    metrics["hybrid_accuracy"] = hybrid_accuracy
    metrics["graph2d_accuracy"] = graph2d_accuracy
    metrics["weak_label_coverage"] = coverage
    metrics["hybrid_gain_vs_graph2d"] = hybrid_accuracy - graph2d_accuracy

    def _adjust_slices(field: str, key: str) -> None:
        raw = metrics.get(field)
        if not isinstance(raw, list):
            return
        adjusted: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            if row.get(key) is None:
                continue
            h = row.get("hybrid_accuracy")
            g = row.get("graph2d_accuracy")
            if h is not None:
                h = _clamp01(_safe_float(h, 0.0) + float(delta_hybrid_accuracy))
                row["hybrid_accuracy"] = h
            if g is not None:
                g = _clamp01(_safe_float(g, 0.0) + float(delta_graph2d_accuracy))
                row["graph2d_accuracy"] = g
            if h is not None and g is not None:
                row["hybrid_gain_vs_graph2d"] = h - g
            adjusted.append(row)
        metrics[field] = adjusted

    _adjust_slices("label_slices", "label")
    _adjust_slices("family_slices", "family")
    cloned["metrics"] = metrics
    return cloned


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap hybrid_blind eval history snapshots for drift activation."
    )
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--gate-report-json", default="")
    parser.add_argument("--output-dir", default="reports/eval_history")
    parser.add_argument("--branch", default="local-dev")
    parser.add_argument("--commit", default="bootstrap")
    parser.add_argument("--runner", default="local")
    parser.add_argument("--machine", default="local")
    parser.add_argument("--os-info", default="unknown")
    parser.add_argument("--python-version", default="unknown")
    parser.add_argument("--ci-job-id", default="")
    parser.add_argument("--ci-workflow", default="")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--hours-step", type=int, default=24)
    parser.add_argument("--end-timestamp", default="")
    parser.add_argument("--hybrid-accuracy-deltas", default="0,-0.01,-0.02")
    parser.add_argument("--graph2d-accuracy-deltas", default="0,0,0")
    parser.add_argument("--coverage-deltas", default="0,-0.01,-0.02")
    parser.add_argument("--label-slice-min-support", type=int, default=1)
    parser.add_argument("--label-slice-max-slices", type=int, default=20)
    parser.add_argument("--family-prefix-len", type=int, default=2)
    parser.add_argument("--family-map-json", default="")
    parser.add_argument("--family-slice-max-slices", type=int, default=20)
    args = parser.parse_args(argv)

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise SystemExit(f"invalid summary json: {summary_path}")

    gate_report: Dict[str, Any] = {}
    if str(args.gate_report_json).strip():
        gate_path = Path(args.gate_report_json)
        if gate_path.exists():
            payload = json.loads(gate_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                gate_report = payload

    count = max(1, int(args.count))
    step_hours = max(1, int(args.hours_step))
    end_ts = _parse_iso_ts(args.end_timestamp) or datetime.now(timezone.utc)

    delta_hybrid = _parse_delta_series(args.hybrid_accuracy_deltas, count)
    delta_graph2d = _parse_delta_series(args.graph2d_accuracy_deltas, count)
    delta_coverage = _parse_delta_series(args.coverage_deltas, count)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    branch = _safe_file_text(args.branch)
    normalized_commit = _normalize_commit(args.commit)
    commit = _safe_file_text(normalized_commit)

    written_paths: List[Path] = []
    for idx in range(count):
        ts = end_ts - timedelta(hours=step_hours * (count - idx - 1))
        payload = archive_mod.build_payload(
            summary=summary,
            gate_report=gate_report,
            branch=_safe_str(args.branch, "local-dev"),
            commit=normalized_commit,
            runner=_safe_str(args.runner, "local"),
            machine=_safe_str(args.machine, "local"),
            os_info=_safe_str(args.os_info, "unknown"),
            python_version=_safe_str(args.python_version, "unknown"),
            ci_job_id=_safe_str(args.ci_job_id, ""),
            ci_workflow=_safe_str(args.ci_workflow, ""),
            timestamp=ts,
            label_slice_min_support=max(1, int(args.label_slice_min_support)),
            label_slice_max_slices=max(0, int(args.label_slice_max_slices)),
            family_prefix_len=max(0, int(args.family_prefix_len)),
            family_map_json=_safe_str(args.family_map_json, ""),
            family_slice_max_slices=max(0, int(args.family_slice_max_slices)),
        )
        payload = _apply_deltas(
            payload,
            delta_hybrid_accuracy=delta_hybrid[idx],
            delta_graph2d_accuracy=delta_graph2d[idx],
            delta_coverage=delta_coverage[idx],
        )
        stamp = ts.strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"{stamp}_{branch}_{commit}_hybrid_blind.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written_paths.append(out_path)

    print(f"written={len(written_paths)}")
    for path in written_paths:
        print(f"path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
