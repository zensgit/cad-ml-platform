#!/usr/bin/env python3
"""Export active-learning review queue summaries for benchmark/governance reporting."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.active_learning import get_active_learner, reset_active_learner  # noqa: E402


PRIORITY_ORDER = ("critical", "high", "medium", "normal")
AUTOMATION_READY_DECISION_SOURCES = {"filename", "titleblock", "hybrid"}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if isinstance(value, tuple):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple)):
            return [_clean_text(item) for item in parsed if _clean_text(item)]
    if ";" in text:
        return [_clean_text(part) for part in text.split(";") if _clean_text(part)]
    if "," in text:
        return [_clean_text(part) for part in text.split(",") if _clean_text(part)]
    return [text]


def _ranked(counter: Counter[str]) -> List[Dict[str, Any]]:
    return [{"name": name, "count": int(count)} for name, count in counter.most_common()]


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _load_csv_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_input_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Input path not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _load_json_records(path)
    if suffix == ".csv":
        return _load_csv_records(path)
    raise SystemExit(f"Unsupported input format: {path.suffix}")


def _normalize_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    review_reasons = _coerce_list(payload.get("review_reasons"))
    return {
        "id": _clean_text(payload.get("id") or payload.get("sample_id")),
        "doc_id": _clean_text(payload.get("doc_id")),
        "status": _clean_text(payload.get("status")) or "pending",
        "sample_type": _clean_text(payload.get("sample_type")) or "review",
        "feedback_priority": _clean_text(payload.get("feedback_priority")) or "normal",
        "decision_source": _clean_text(payload.get("decision_source")) or "unknown",
        "uncertainty_reason": _clean_text(payload.get("uncertainty_reason")) or "unknown",
        "review_reasons": review_reasons,
        "confidence": payload.get("confidence"),
    }


def _rows_from_data_dir(
    data_dir: Path,
    *,
    status: str,
    sample_type: Optional[str],
    feedback_priority: Optional[str],
    sort_by: str,
) -> List[Dict[str, Any]]:
    old_data_dir = os.environ.get("ACTIVE_LEARNING_DATA_DIR")
    old_store = os.environ.get("ACTIVE_LEARNING_STORE")
    os.environ["ACTIVE_LEARNING_DATA_DIR"] = str(data_dir)
    os.environ["ACTIVE_LEARNING_STORE"] = "file"
    reset_active_learner()
    try:
        learner = get_active_learner()
        payload = learner.get_review_queue(
            limit=1000000,
            offset=0,
            status=status,
            sample_type=sample_type,
            feedback_priority=feedback_priority,
            sort_by=sort_by,
        )
    finally:
        reset_active_learner()
        if old_data_dir is None:
            os.environ.pop("ACTIVE_LEARNING_DATA_DIR", None)
        else:
            os.environ["ACTIVE_LEARNING_DATA_DIR"] = old_data_dir
        if old_store is None:
            os.environ.pop("ACTIVE_LEARNING_STORE", None)
        else:
            os.environ["ACTIVE_LEARNING_STORE"] = old_store
    rows: List[Dict[str, Any]] = []
    for item in payload.get("items", []):
        review_reasons = item.score_breakdown.get("review_reasons")
        if not isinstance(review_reasons, list):
            review_reasons = []
        rows.append(
            {
                "id": item.id,
                "doc_id": item.doc_id,
                "status": item.status.value,
                "sample_type": item.sample_type or "review",
                "feedback_priority": item.feedback_priority or "normal",
                "decision_source": str(
                    item.score_breakdown.get("final_decision_source")
                    or item.score_breakdown.get("decision_source")
                    or "unknown"
                ),
                "uncertainty_reason": item.uncertainty_reason or "unknown",
                "review_reasons": review_reasons,
                "confidence": item.confidence,
            }
        )
    return rows


def build_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [_normalize_row(record) for record in records]
    by_sample_type: Counter[str] = Counter()
    by_feedback_priority: Counter[str] = Counter()
    by_decision_source: Counter[str] = Counter()
    by_uncertainty_reason: Counter[str] = Counter()
    by_review_reason: Counter[str] = Counter()

    for row in rows:
        by_sample_type[row["sample_type"]] += 1
        by_feedback_priority[row["feedback_priority"]] += 1
        by_decision_source[row["decision_source"]] += 1
        by_uncertainty_reason[row["uncertainty_reason"]] += 1
        review_reasons = row["review_reasons"] or [row["uncertainty_reason"]]
        for reason in review_reasons:
            by_review_reason[_clean_text(reason) or "unknown"] += 1

    total = len(rows)
    high_priority_count = by_feedback_priority.get("high", 0) + by_feedback_priority.get(
        "critical", 0
    )
    critical_priority_count = by_feedback_priority.get("critical", 0)
    high_priority_ratio = float(high_priority_count) / float(total) if total else 0.0
    automation_ready_count = sum(
        1
        for row in rows
        if row["decision_source"] in AUTOMATION_READY_DECISION_SOURCES
        and row["feedback_priority"] not in {"critical", "high"}
        and row["sample_type"] not in {"knowledge_conflict", "branch_conflict"}
    )
    automation_ready_ratio = (
        float(automation_ready_count) / float(total) if total else 0.0
    )

    if total == 0:
        status = "under_control"
    elif critical_priority_count > 0:
        status = "critical_backlog"
    elif high_priority_ratio >= 0.5:
        status = "managed_backlog"
    else:
        status = "routine_backlog"

    return {
        "status": status,
        "total": total,
        "high_priority_count": int(high_priority_count),
        "critical_priority_count": int(critical_priority_count),
        "high_priority_ratio": round(high_priority_ratio, 6),
        "automation_ready_count": int(automation_ready_count),
        "automation_ready_ratio": round(automation_ready_ratio, 6),
        "by_sample_type": dict(by_sample_type),
        "by_feedback_priority": dict(by_feedback_priority),
        "by_decision_source": dict(by_decision_source),
        "by_uncertainty_reason": dict(by_uncertainty_reason),
        "by_review_reason": dict(by_review_reason),
        "top_feedback_priorities": _ranked(by_feedback_priority),
        "top_decision_sources": _ranked(by_decision_source),
        "top_review_reasons": _ranked(by_review_reason),
        "top_uncertainty_reasons": _ranked(by_uncertainty_reason),
        "recommended_actions": _recommended_actions(
            status=status,
            by_feedback_priority=by_feedback_priority,
            by_review_reason=by_review_reason,
        ),
    }


def _recommended_actions(
    *,
    status: str,
    by_feedback_priority: Counter[str],
    by_review_reason: Counter[str],
) -> List[str]:
    items: List[str] = []
    if status == "critical_backlog":
        items.append("prioritize_critical_review_queue")
    elif status == "managed_backlog":
        items.append("drain_high_priority_review_queue")
    if by_review_reason.get("knowledge_conflict", 0) > 0:
        items.append("review_knowledge_conflict_samples")
    if by_feedback_priority.get("high", 0) > 0 or by_feedback_priority.get("critical", 0) > 0:
        items.append("schedule_manual_triage_session")
    if not items:
        items.append("review_queue_within_expected_band")
    return items


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export active-learning review queue summary for benchmark reporting."
    )
    parser.add_argument("--input-path", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--status", default="pending")
    parser.add_argument("--sample-type", default="")
    parser.add_argument("--feedback-priority", default="")
    parser.add_argument("--sort-by", default="priority")
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args(argv)

    input_path = str(args.input_path or "").strip()
    data_dir = str(args.data_dir or "").strip()
    if not input_path and not data_dir:
        raise SystemExit("One of --input-path or --data-dir is required")

    if input_path:
        records = _load_input_records(Path(input_path).expanduser())
    else:
        records = _rows_from_data_dir(
            Path(data_dir).expanduser(),
            status=str(args.status or "pending"),
            sample_type=str(args.sample_type or "").strip() or None,
            feedback_priority=str(args.feedback_priority or "").strip() or None,
            sort_by=str(args.sort_by or "priority"),
        )

    summary = build_summary(records)
    output_path = Path(args.summary_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
