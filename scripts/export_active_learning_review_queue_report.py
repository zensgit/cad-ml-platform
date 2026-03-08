#!/usr/bin/env python3
"""Export active-learning review queue records into benchmark-friendly summaries."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from src.core.active_learning import (
    ActiveLearningSample,
    _derive_feedback_priority,
    _derive_sample_type,
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return None


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = _clean_text(value)
    if not text:
        return {}
    parsed = _safe_json_loads(text)
    if isinstance(parsed, dict):
        return dict(parsed)
    return {}


def _coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    text = _clean_text(value)
    if not text:
        return []
    parsed = _safe_json_loads(text)
    if isinstance(parsed, list):
        return list(parsed)
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value).lower()
    return text in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _automation_ready(score_breakdown: Dict[str, Any]) -> bool:
    return bool(
        score_breakdown.get("automation_ready")
        or score_breakdown.get("review_automation_ready")
    )


def _normalize_review_reasons(value: Any, uncertainty_reason: str) -> List[str]:
    reasons = [token for token in (_clean_text(item) for item in _coerce_list(value)) if token]
    if reasons:
        return reasons
    if uncertainty_reason:
        return [uncertainty_reason]
    return []


def _compact_top(counter: Counter[str], top_k: int) -> List[Dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common(top_k)]


def _row_from_sample(sample: ActiveLearningSample) -> Dict[str, Any]:
    score_breakdown = dict(sample.score_breakdown or {})
    decision_source = _clean_text(
        score_breakdown.get("final_decision_source")
        or score_breakdown.get("decision_source")
        or "unknown"
    )
    uncertainty_reason = _clean_text(sample.uncertainty_reason)
    review_reasons = _normalize_review_reasons(
        score_breakdown.get("review_reasons"),
        uncertainty_reason,
    )
    return {
        "id": sample.id,
        "doc_id": sample.doc_id,
        "status": sample.status.value,
        "sample_type": _derive_sample_type(sample),
        "feedback_priority": _derive_feedback_priority(sample),
        "decision_source": decision_source,
        "uncertainty_reason": uncertainty_reason or "unknown",
        "review_reasons": review_reasons,
        "confidence": float(sample.confidence),
        "predicted_type": _clean_text(sample.predicted_type),
        "predicted_fine_type": _clean_text(sample.predicted_fine_type),
        "predicted_coarse_type": _clean_text(sample.predicted_coarse_type),
        "automation_ready": _automation_ready(score_breakdown),
        "evidence_count": int(sample.evidence_count or 0),
        "evidence_sources": list(sample.evidence_sources or []),
        "evidence_summary": _clean_text(sample.evidence_summary),
        "score_breakdown": score_breakdown,
    }


def _row_from_export_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    score_breakdown = _coerce_dict(mapping.get("score_breakdown"))
    uncertainty_reason = _clean_text(mapping.get("uncertainty_reason")) or "unknown"
    return {
        "id": _clean_text(mapping.get("id")),
        "doc_id": _clean_text(mapping.get("doc_id")),
        "status": _clean_text(mapping.get("status")) or "pending",
        "sample_type": _clean_text(mapping.get("sample_type")) or "review",
        "feedback_priority": _clean_text(mapping.get("feedback_priority")) or "normal",
        "decision_source": _clean_text(mapping.get("decision_source")) or "unknown",
        "uncertainty_reason": uncertainty_reason,
        "review_reasons": _normalize_review_reasons(
            mapping.get("review_reasons"),
            uncertainty_reason,
        ),
        "confidence": float(mapping.get("confidence") or 0.0),
        "predicted_type": _clean_text(mapping.get("predicted_type")),
        "predicted_fine_type": _clean_text(mapping.get("predicted_fine_type")),
        "predicted_coarse_type": _clean_text(mapping.get("predicted_coarse_type")),
        "automation_ready": _coerce_bool(mapping.get("automation_ready"))
        or _automation_ready(score_breakdown),
        "evidence_count": _coerce_int(mapping.get("evidence_count")),
        "evidence_sources": [
            _clean_text(item)
            for item in _coerce_list(mapping.get("evidence_sources"))
            if _clean_text(item)
        ],
        "evidence_summary": _clean_text(mapping.get("evidence_summary")),
        "score_breakdown": score_breakdown,
    }


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                continue
            yield payload


def _iter_csv(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _iter_candidate_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        yield input_path
        return
    if not input_path.is_dir():
        return
    patterns = ("samples.jsonl", "review_queue_*.jsonl", "review_queue_*.csv")
    seen: set[Path] = set()
    for pattern in patterns:
        for candidate in sorted(input_path.rglob(pattern)):
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _load_rows(input_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candidate in _iter_candidate_files(input_path):
        if candidate.name == "samples.jsonl":
            for payload in _iter_jsonl(candidate):
                rows.append(_row_from_sample(ActiveLearningSample(**payload)))
            continue
        if candidate.suffix.lower() == ".jsonl":
            for payload in _iter_jsonl(candidate):
                rows.append(_row_from_export_mapping(payload))
            continue
        if candidate.suffix.lower() == ".csv":
            for payload in _iter_csv(candidate):
                rows.append(_row_from_export_mapping(payload))
    return rows


def _build_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    input_path: Path,
) -> Dict[str, Any]:
    by_sample_type: Counter[str] = Counter()
    by_feedback_priority: Counter[str] = Counter()
    by_decision_source: Counter[str] = Counter()
    by_review_reason: Counter[str] = Counter()
    by_evidence_source: Counter[str] = Counter()

    critical_count = 0
    high_count = 0
    automation_ready_count = 0
    evidence_count_total = 0
    records_with_evidence_count = 0

    for row in rows:
        sample_type = _clean_text(row.get("sample_type")) or "review"
        feedback_priority = _clean_text(row.get("feedback_priority")) or "normal"
        decision_source = _clean_text(row.get("decision_source")) or "unknown"
        by_sample_type[sample_type] += 1
        by_feedback_priority[feedback_priority] += 1
        by_decision_source[decision_source] += 1
        if feedback_priority == "critical":
            critical_count += 1
        if feedback_priority == "high":
            high_count += 1
        if bool(row.get("automation_ready")):
            automation_ready_count += 1
        evidence_count = _coerce_int(row.get("evidence_count"))
        evidence_count_total += evidence_count
        if evidence_count > 0:
            records_with_evidence_count += 1
        for source in row.get("evidence_sources") or []:
            source_text = _clean_text(source) or "unknown"
            by_evidence_source[source_text] += 1
        for reason in row.get("review_reasons") or []:
            reason_text = _clean_text(reason) or "unknown"
            by_review_reason[reason_text] += 1

    total = len(rows)
    denom = max(total, 1)
    if total <= 0:
        operational_status = "under_control"
    elif critical_count > 0:
        operational_status = "critical_backlog"
    elif high_count > 0:
        operational_status = "managed_backlog"
    else:
        operational_status = "routine_backlog"

    return {
        "input_path": str(input_path),
        "total": total,
        "by_sample_type": dict(by_sample_type),
        "by_feedback_priority": dict(by_feedback_priority),
        "by_decision_source": dict(by_decision_source),
        "by_review_reason": dict(by_review_reason),
        "critical_count": critical_count,
        "high_count": high_count,
        "automation_ready_count": automation_ready_count,
        "evidence_count_total": evidence_count_total,
        "average_evidence_count": round(evidence_count_total / denom, 6),
        "records_with_evidence_count": records_with_evidence_count,
        "records_with_evidence_ratio": round(records_with_evidence_count / denom, 6),
        "critical_ratio": round(critical_count / denom, 6),
        "high_ratio": round(high_count / denom, 6),
        "automation_ready_ratio": round(automation_ready_count / denom, 6),
        "operational_status": operational_status,
        "top_sample_types": _compact_top(by_sample_type, top_k),
        "top_feedback_priorities": _compact_top(by_feedback_priority, top_k),
        "top_decision_sources": _compact_top(by_decision_source, top_k),
        "top_review_reasons": _compact_top(by_review_reason, top_k),
        "top_evidence_sources": _compact_top(by_evidence_source, top_k),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "doc_id",
        "status",
        "sample_type",
        "feedback_priority",
        "decision_source",
        "uncertainty_reason",
        "review_reasons",
        "confidence",
        "predicted_type",
        "predicted_fine_type",
        "predicted_coarse_type",
        "automation_ready",
        "evidence_count",
        "evidence_sources",
        "evidence_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                    if key not in {"review_reasons", "evidence_sources"}
                }
                | {
                    "review_reasons": json.dumps(
                        row.get("review_reasons") or [],
                        ensure_ascii=False,
                    ),
                    "evidence_sources": json.dumps(
                        row.get("evidence_sources") or [],
                        ensure_ascii=False,
                    ),
                }
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export active-learning review queue records into a benchmark summary."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    rows = _load_rows(input_path)
    summary = _build_summary(rows, top_k=max(args.top_k, 1), input_path=input_path)

    output_json = Path(args.output_json).expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.output_csv:
        _write_csv(Path(args.output_csv).expanduser(), rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
