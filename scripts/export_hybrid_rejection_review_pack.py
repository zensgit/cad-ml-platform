#!/usr/bin/env python3
"""Export a prioritized review pack from batch Graph2D/Hybrid CSV results."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return False
    return text in {"1", "true", "yes", "y", "on"}


def _clean_label(value: Any) -> str:
    return str(value or "").strip()


def _parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = _clean_label(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _primary_sources(value: Any, top_k: int = 3) -> str:
    payload = _parse_json_object(value)
    if not payload:
        return ""
    ranked: List[tuple[str, float]] = []
    for key, raw_value in payload.items():
        try:
            ranked.append((str(key), float(raw_value)))
        except (TypeError, ValueError):
            continue
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ";".join(source for source, _ in ranked[: max(1, int(top_k))])


def _split_semicolon_tokens(value: Any) -> List[str]:
    text = _clean_label(value)
    if not text:
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def _top_named_counts(counter: Counter[str], limit: int = 5) -> List[Dict[str, Any]]:
    return [
        {"name": name, "count": int(count)}
        for name, count in counter.most_common(max(0, int(limit)))
    ]


def _shadow_sources(value: Any) -> str:
    payload = _parse_json_object(value)
    if not payload:
        return ""
    names = sorted(str(key).strip() for key in payload.keys() if str(key).strip())
    return ";".join(names)


def _has_hybrid_rejection(row: Dict[str, Any]) -> bool:
    if _to_bool(row.get("hybrid_rejected")):
        return True
    return bool(_clean_label(row.get("hybrid_rejection_reason")))


def _has_conflict(row: Dict[str, Any]) -> bool:
    hybrid_label = _clean_label(row.get("hybrid_label") or row.get("fine_part_type"))
    graph2d_label = _clean_label(row.get("graph2d_label"))
    if not hybrid_label or not graph2d_label:
        return False
    return hybrid_label != graph2d_label


def _priority_score(
    *,
    has_rejection: bool,
    has_conflict: bool,
    low_confidence: bool,
    confidence: float,
    low_confidence_threshold: float,
) -> float:
    score = 0.0
    if has_rejection:
        score += 3.0
    if has_conflict:
        score += 2.0
    if low_confidence:
        gap = max(0.0, low_confidence_threshold - confidence)
        score += 1.0 + min(1.0, gap)
    return score


def _collect_candidates(
    rows: Iterable[Dict[str, Any]],
    *,
    low_confidence_threshold: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "").strip().lower() != "ok":
            continue
        confidence = _to_float(row.get("confidence"), default=0.0)
        has_rejection = _has_hybrid_rejection(row)
        has_conflict = _has_conflict(row)
        low_confidence = confidence <= low_confidence_threshold
        if not (has_rejection or has_conflict or low_confidence):
            continue

        reason_parts: List[str] = []
        if has_rejection:
            reason_parts.append(
                f"hybrid_rejected:{_clean_label(row.get('hybrid_rejection_reason')) or 'unknown'}"
            )
        if has_conflict:
            reason_parts.append("hybrid_graph2d_conflict")
        if low_confidence:
            reason_parts.append("low_confidence")

        score = _priority_score(
            has_rejection=has_rejection,
            has_conflict=has_conflict,
            low_confidence=low_confidence,
            confidence=confidence,
            low_confidence_threshold=low_confidence_threshold,
        )
        out = dict(row)
        out["review_priority_score"] = f"{score:.3f}"
        out["review_reasons"] = ";".join(reason_parts)
        out["review_has_hybrid_rejection"] = has_rejection
        out["review_has_hybrid_graph2d_conflict"] = has_conflict
        out["review_is_low_confidence"] = low_confidence
        out["review_primary_sources"] = _primary_sources(
            row.get("hybrid_source_contributions")
            or row.get("source_contributions")
        )
        out["review_explanation_summary"] = _clean_label(
            row.get("hybrid_explanation_summary")
            or _parse_json_object(row.get("hybrid_explanation")).get("summary")
        )
        out["review_decision_path"] = _clean_label(
            row.get("hybrid_path") or row.get("decision_path")
        )
        out["review_fusion_strategy"] = _clean_label(
            row.get("hybrid_fusion_strategy")
        )
        out["review_shadow_sources"] = _shadow_sources(
            row.get("hybrid_shadow_predictions")
        )
        out["review_history_shadow_only"] = _to_bool(row.get("history_shadow_only"))
        out["review_history_shadow_label"] = _clean_label(row.get("history_label"))
        out["review_history_shadow_confidence"] = _clean_label(
            row.get("history_confidence")
        )
        candidates.append(out)

    candidates.sort(
        key=lambda r: (
            -_to_float(r.get("review_priority_score"), 0.0),
            _to_float(r.get("confidence"), 0.0),
            str(r.get("file") or ""),
        )
    )
    return candidates


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    keys: set[str] = set()
    for row in rows:
        keys.update(str(k) for k in row.keys())
    fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    *,
    input_csv: Path,
    output_csv: Path,
    total_rows: int,
    candidate_rows: List[Dict[str, Any]],
    low_confidence_threshold: float,
) -> Dict[str, Any]:
    review_reason_counter: Counter[str] = Counter()
    primary_source_counter: Counter[str] = Counter()
    shadow_source_counter: Counter[str] = Counter()
    sample_explanations: List[str] = []
    sample_candidates: List[Dict[str, Any]] = []

    for row in candidate_rows:
        review_reason_counter.update(_split_semicolon_tokens(row.get("review_reasons")))
        primary_source_counter.update(
            _split_semicolon_tokens(row.get("review_primary_sources"))
        )
        shadow_source_counter.update(
            _split_semicolon_tokens(row.get("review_shadow_sources"))
        )

        explanation_summary = _clean_label(row.get("review_explanation_summary"))
        if explanation_summary and explanation_summary not in sample_explanations:
            sample_explanations.append(explanation_summary)

        if len(sample_candidates) < 3:
            sample_candidates.append(
                {
                    "file": _clean_label(row.get("file")),
                    "reasons": _clean_label(row.get("review_reasons")),
                    "primary_sources": _clean_label(row.get("review_primary_sources")),
                    "shadow_sources": _clean_label(row.get("review_shadow_sources")),
                    "explanation_summary": explanation_summary,
                }
            )

    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "total_rows": int(total_rows),
        "candidate_rows": int(len(candidate_rows)),
        "low_confidence_threshold": float(low_confidence_threshold),
        "hybrid_rejected_count": sum(
            1 for row in candidate_rows if _to_bool(row.get("review_has_hybrid_rejection"))
        ),
        "conflict_count": sum(
            1
            for row in candidate_rows
            if _to_bool(row.get("review_has_hybrid_graph2d_conflict"))
        ),
        "low_confidence_count": sum(
            1 for row in candidate_rows if _to_bool(row.get("review_is_low_confidence"))
        ),
        "top_review_reasons": _top_named_counts(review_reason_counter),
        "top_primary_sources": _top_named_counts(primary_source_counter),
        "top_shadow_sources": _top_named_counts(shadow_source_counter),
        "sample_explanations": sample_explanations[:3],
        "sample_candidates": sample_candidates,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a prioritized CSV review pack containing hybrid rejections, "
            "low-confidence cases, and hybrid-vs-graph2d conflicts."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Input batch_results.csv path")
    parser.add_argument("--output-csv", required=True, help="Output prioritized CSV path")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional summary JSON path (default: <output-csv>.summary.json).",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.6,
        help="Include rows whose final confidence <= threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Keep only top K rows by priority (0 means no limit).",
    )
    args = parser.parse_args(argv)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    candidates = _collect_candidates(
        rows, low_confidence_threshold=float(args.low_confidence_threshold)
    )
    if int(args.top_k) > 0:
        candidates = candidates[: int(args.top_k)]

    _write_csv(output_csv, candidates)

    summary = _build_summary(
        input_csv=input_csv,
        output_csv=output_csv,
        total_rows=len(rows),
        candidate_rows=candidates,
        low_confidence_threshold=float(args.low_confidence_threshold),
    )
    summary_path = (
        Path(args.summary_json)
        if str(args.summary_json).strip()
        else output_csv.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Exported {len(candidates)} review candidates to {output_csv} "
        f"(summary: {summary_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
