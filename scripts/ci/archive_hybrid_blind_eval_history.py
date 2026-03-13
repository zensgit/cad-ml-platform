#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_COMMIT_RE = re.compile(r"^[a-f0-9]{6,40}$")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_json_text(value: Any) -> str:
    text = _safe_str(value, "unknown")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "unknown"


def _normalize_commit(value: Any, default: str = "[redacted]") -> str:
    text = _safe_str(value, "")
    if not text:
        return default
    if text == "[redacted]":
        return text
    lowered = text.lower()
    if _COMMIT_RE.fullmatch(lowered):
        return lowered
    return default


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _extract_accuracy(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    if "accuracy" in payload:
        return _safe_float(payload.get("accuracy"), 0.0)
    evaluated = _safe_int(payload.get("evaluated"), 0)
    if evaluated <= 0:
        return None
    correct = _safe_int(payload.get("correct"), 0)
    return float(correct) / float(max(1, evaluated))


def _load_family_map(path: str) -> Dict[str, str]:
    text = str(path or "").strip()
    if not text:
        return {}
    map_path = Path(text)
    if not map_path.exists():
        return {}
    try:
        payload = _read_json(map_path)
    except Exception:
        return {}
    if isinstance(payload.get("label_to_family"), dict):
        return {
            str(k).strip(): str(v).strip()
            for k, v in payload.get("label_to_family", {}).items()
            if str(k).strip() and str(v).strip()
        }
    if isinstance(payload, dict):
        return {
            str(k).strip(): str(v).strip()
            for k, v in payload.items()
            if str(k).strip() and str(v).strip()
        }
    return {}


def _derive_family_key(label: str, *, prefix_len: int, label_to_family: Dict[str, str]) -> str:
    direct = label_to_family.get(label)
    if direct:
        return direct
    normalized = re.sub(r"[\\s\\-_()（）\\[\\]{}]+", "", str(label or "").strip())
    if not normalized:
        return "unknown"
    if prefix_len <= 0:
        return normalized
    if len(normalized) <= prefix_len:
        return normalized
    return normalized[:prefix_len]


def _build_label_slices(
    *,
    summary: Dict[str, Any],
    min_support: int,
    max_slices: int,
) -> List[Dict[str, Any]]:
    weak = summary.get("weak_labels") if isinstance(summary.get("weak_labels"), dict) else {}
    label_counts = weak.get("label_counts") if isinstance(weak.get("label_counts"), dict) else {}
    by_true = (
        weak.get("by_true_label_accuracy")
        if isinstance(weak.get("by_true_label_accuracy"), dict)
        else {}
    )
    hybrid_by_label = by_true.get("hybrid_label") if isinstance(by_true.get("hybrid_label"), dict) else {}
    graph2d_by_label = (
        by_true.get("graph2d_label") if isinstance(by_true.get("graph2d_label"), dict) else {}
    )

    labels = set(str(k) for k in label_counts.keys())
    labels.update(str(k) for k in hybrid_by_label.keys())
    labels.update(str(k) for k in graph2d_by_label.keys())

    rows: List[Dict[str, Any]] = []
    for label in sorted(labels):
        support = _safe_int(label_counts.get(label), 0)
        if support < int(min_support):
            continue
        hybrid_acc = _extract_accuracy(hybrid_by_label.get(label))
        graph2d_acc = _extract_accuracy(graph2d_by_label.get(label))
        gain = None
        if hybrid_acc is not None and graph2d_acc is not None:
            gain = hybrid_acc - graph2d_acc
        rows.append(
            {
                "label": label,
                "support": support,
                "hybrid_accuracy": hybrid_acc,
                "graph2d_accuracy": graph2d_acc,
                "hybrid_gain_vs_graph2d": gain,
            }
        )

    rows.sort(
        key=lambda item: (
            -_safe_int(item.get("support"), 0),
            str(item.get("label") or ""),
        )
    )
    limit = max(0, int(max_slices))
    if limit > 0:
        rows = rows[:limit]
    return rows


def _build_family_slices(
    *,
    label_slices: List[Dict[str, Any]],
    family_prefix_len: int,
    family_map_path: str,
    max_slices: int,
) -> List[Dict[str, Any]]:
    label_to_family = _load_family_map(family_map_path)
    buckets: Dict[str, Dict[str, float]] = {}
    for row in label_slices:
        label = _safe_str(row.get("label"), "")
        if not label:
            continue
        family = _derive_family_key(
            label,
            prefix_len=max(0, int(family_prefix_len)),
            label_to_family=label_to_family,
        )
        bucket = buckets.setdefault(
            family,
            {
                "support": 0.0,
                "hybrid_weighted_sum": 0.0,
                "hybrid_support": 0.0,
                "graph2d_weighted_sum": 0.0,
                "graph2d_support": 0.0,
            },
        )
        support = float(max(0, _safe_int(row.get("support"), 0)))
        bucket["support"] += support

        hybrid_acc = row.get("hybrid_accuracy")
        if hybrid_acc is not None:
            bucket["hybrid_weighted_sum"] += _safe_float(hybrid_acc, 0.0) * support
            bucket["hybrid_support"] += support

        graph2d_acc = row.get("graph2d_accuracy")
        if graph2d_acc is not None:
            bucket["graph2d_weighted_sum"] += _safe_float(graph2d_acc, 0.0) * support
            bucket["graph2d_support"] += support

    rows: List[Dict[str, Any]] = []
    for family, agg in buckets.items():
        support = int(round(agg["support"]))
        hybrid_acc = None
        if agg["hybrid_support"] > 0:
            hybrid_acc = agg["hybrid_weighted_sum"] / agg["hybrid_support"]
        graph2d_acc = None
        if agg["graph2d_support"] > 0:
            graph2d_acc = agg["graph2d_weighted_sum"] / agg["graph2d_support"]
        gain = None
        if hybrid_acc is not None and graph2d_acc is not None:
            gain = hybrid_acc - graph2d_acc
        rows.append(
            {
                "family": family,
                "support": support,
                "hybrid_accuracy": hybrid_acc,
                "graph2d_accuracy": graph2d_acc,
                "hybrid_gain_vs_graph2d": gain,
            }
        )

    rows.sort(
        key=lambda item: (
            -_safe_int(item.get("support"), 0),
            str(item.get("family") or ""),
        )
    )
    limit = max(0, int(max_slices))
    if limit > 0:
        rows = rows[:limit]
    return rows


def build_payload(
    *,
    summary: Dict[str, Any],
    gate_report: Dict[str, Any],
    branch: str,
    commit: str,
    runner: str,
    machine: str,
    os_info: str,
    python_version: str,
    ci_job_id: str,
    ci_workflow: str,
    timestamp: datetime,
    label_slice_min_support: int,
    label_slice_max_slices: int,
    family_prefix_len: int,
    family_map_json: str,
    family_slice_max_slices: int,
) -> Dict[str, Any]:
    weak = summary.get("weak_labels") if isinstance(summary.get("weak_labels"), dict) else {}
    acc = weak.get("accuracy") if isinstance(weak.get("accuracy"), dict) else {}
    hybrid = acc.get("hybrid_label") if isinstance(acc.get("hybrid_label"), dict) else {}
    graph2d = acc.get("graph2d_label") if isinstance(acc.get("graph2d_label"), dict) else {}
    gate_metrics = gate_report.get("metrics") if isinstance(gate_report.get("metrics"), dict) else {}

    hybrid_accuracy = _safe_float(hybrid.get("accuracy"), _safe_float(gate_metrics.get("hybrid_accuracy"), 0.0))
    graph2d_accuracy = _safe_float(graph2d.get("accuracy"), _safe_float(gate_metrics.get("graph2d_accuracy"), 0.0))
    gain = _safe_float(
        gate_metrics.get("hybrid_gain_vs_graph2d"),
        hybrid_accuracy - graph2d_accuracy,
    )
    label_slices = _build_label_slices(
        summary=summary,
        min_support=label_slice_min_support,
        max_slices=label_slice_max_slices,
    )
    family_slices = _build_family_slices(
        label_slices=label_slices,
        family_prefix_len=family_prefix_len,
        family_map_path=family_map_json,
        max_slices=family_slice_max_slices,
    )

    ci_job: Any = None if not ci_job_id else ci_job_id
    ci_wf: Any = None if not ci_workflow else ci_workflow
    normalized_commit = _normalize_commit(commit)
    return {
        "schema_version": "1.0.0",
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": branch,
        "commit": normalized_commit,
        "type": "hybrid_blind",
        "metrics": {
            "sample_size": int(summary.get("sample_size") or 0),
            "weak_label_coverage": _safe_float(
                weak.get("covered_rate"), _safe_float(gate_metrics.get("weak_label_coverage"), 0.0)
            ),
            "hybrid_accuracy": hybrid_accuracy,
            "graph2d_accuracy": graph2d_accuracy,
            "hybrid_gain_vs_graph2d": gain,
            "gate_status": _safe_str(gate_report.get("status"), "unknown"),
            "label_slices": label_slices,
            "label_slice_meta": {
                "source": "weak_labels.by_true_label_accuracy",
                "min_support": int(label_slice_min_support),
                "max_slices": int(label_slice_max_slices),
                "slice_count": len(label_slices),
            },
            "family_slices": family_slices,
            "family_slice_meta": {
                "source": "label_slices.aggregate",
                "prefix_len": int(family_prefix_len),
                "family_map_json": _safe_str(family_map_json, ""),
                "max_slices": int(family_slice_max_slices),
                "slice_count": len(family_slices),
            },
        },
        "run_context": {
            "runner": runner,
            "machine": machine,
            "os": os_info,
            "python": python_version,
            "start_time": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ci_job_id": ci_job,
            "ci_workflow": ci_wf,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive hybrid blind evaluation snapshot into eval history.")
    parser.add_argument("--summary-json", required=True, help="Hybrid blind summary JSON path.")
    parser.add_argument("--gate-report-json", default="", help="Hybrid blind gate report JSON path.")
    parser.add_argument("--output-dir", default="reports/eval_history", help="Eval history output dir.")
    parser.add_argument("--branch", default="unknown", help="Git branch name.")
    parser.add_argument("--commit", default="unknown", help="Git short commit.")
    parser.add_argument("--runner", default="unknown", help="Runner context.")
    parser.add_argument("--machine", default="unknown", help="Machine context.")
    parser.add_argument("--os-info", default="unknown", help="OS context.")
    parser.add_argument("--python-version", default="unknown", help="Python version.")
    parser.add_argument("--ci-job-id", default="", help="CI job ID (optional).")
    parser.add_argument("--ci-workflow", default="", help="CI workflow name (optional).")
    parser.add_argument(
        "--label-slice-min-support",
        type=int,
        default=1,
        help="Minimum weak-label support required to archive a per-label slice.",
    )
    parser.add_argument(
        "--label-slice-max-slices",
        type=int,
        default=20,
        help="Max number of per-label slices to persist in history.",
    )
    parser.add_argument(
        "--family-prefix-len",
        type=int,
        default=2,
        help="Fallback family grouping prefix length when label-family map is not provided.",
    )
    parser.add_argument(
        "--family-map-json",
        default="",
        help="Optional JSON path for label->family mapping.",
    )
    parser.add_argument(
        "--family-slice-max-slices",
        type=int,
        default=20,
        help="Max number of aggregated family slices to persist in history.",
    )
    args = parser.parse_args(argv)

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f"summary not found: {summary_path}")
    summary = _read_json(summary_path)

    gate_report: Dict[str, Any] = {}
    if str(args.gate_report_json).strip():
        gate_path = Path(args.gate_report_json)
        if gate_path.exists():
            gate_report = _read_json(gate_path)

    now = datetime.now(timezone.utc)
    branch = _safe_json_text(args.branch)
    normalized_commit = _normalize_commit(args.commit)
    commit = _safe_json_text(normalized_commit)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_{branch}_{commit}_hybrid_blind.json"

    payload = build_payload(
        summary=summary,
        gate_report=gate_report,
        branch=_safe_str(args.branch, "unknown"),
        commit=normalized_commit,
        runner=_safe_str(args.runner, "unknown"),
        machine=_safe_str(args.machine, "unknown"),
        os_info=_safe_str(args.os_info, "unknown"),
        python_version=_safe_str(args.python_version, "unknown"),
        ci_job_id=_safe_str(args.ci_job_id, ""),
        ci_workflow=_safe_str(args.ci_workflow, ""),
        timestamp=now,
        label_slice_min_support=max(1, int(args.label_slice_min_support)),
        label_slice_max_slices=max(0, int(args.label_slice_max_slices)),
        family_prefix_len=max(0, int(args.family_prefix_len)),
        family_map_json=_safe_str(args.family_map_json, ""),
        family_slice_max_slices=max(0, int(args.family_slice_max_slices)),
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
