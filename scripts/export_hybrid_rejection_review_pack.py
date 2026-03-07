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


def _parse_json_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    text = _clean_label(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _knowledge_tokens(value: Any, *, key: str) -> str:
    tokens: List[str] = []
    for item in _parse_json_list(value):
        token = str(item.get(key) or "").strip()
        if token:
            tokens.append(token)
    return ";".join(tokens)


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


def _first_nonempty(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = _clean_label(row.get(key))
        if text:
            return text
    return ""


def _normalized_optional_signal(value: Any, *, truthy_label: str = "present") -> str:
    if isinstance(value, bool):
        return truthy_label if value else ""
    payload = _parse_json_object(value)
    if payload:
        for key in ("level", "status", "name", "type", "reason"):
            token = _normalized_optional_signal(payload.get(key), truthy_label=truthy_label)
            if token:
                return token
        return truthy_label
    text = _clean_label(value)
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"0", "false", "n/a", "na", "no", "none", "null", "off"}:
        return ""
    if lowered in {"1", "true", "yes", "y", "on"}:
        return truthy_label
    return text


def _coarse_label(row: Dict[str, Any]) -> str:
    return _first_nonempty(row, "graph2d_label", "part_type")


def _fine_label(row: Dict[str, Any]) -> str:
    return _first_nonempty(row, "hybrid_label", "fine_part_type")


def _rejection_reason(row: Dict[str, Any]) -> str:
    reason = _clean_label(row.get("hybrid_rejection_reason"))
    if reason:
        return reason
    return "unknown" if _has_hybrid_rejection(row) else ""


def _knowledge_conflict(row: Dict[str, Any]) -> str:
    for key in (
        "knowledge_conflict",
        "knowledge_conflict_level",
        "fusion_consistency_check",
        "consistency_check",
    ):
        token = _normalized_optional_signal(row.get(key), truthy_label="present")
        if token:
            return token
    return ""


def _knowledge_conflict_note(row: Dict[str, Any]) -> str:
    return _first_nonempty(
        row,
        "knowledge_conflict_reason",
        "knowledge_conflict_note",
        "knowledge_conflict_notes",
        "fusion_consistency_notes",
        "consistency_notes",
    )


def _has_hybrid_rejection(row: Dict[str, Any]) -> bool:
    if _to_bool(row.get("hybrid_rejected")):
        return True
    return bool(_clean_label(row.get("hybrid_rejection_reason")))


def _has_conflict(row: Dict[str, Any]) -> bool:
    hybrid_label = _fine_label(row)
    graph2d_label = _clean_label(row.get("graph2d_label"))
    if not hybrid_label or not graph2d_label:
        return False
    return hybrid_label != graph2d_label


def _existing_review_reasons(row: Dict[str, Any]) -> List[str]:
    return _split_semicolon_tokens(row.get("review_reasons"))


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
        low_confidence = _to_bool(row.get("review_is_low_confidence")) or (
            confidence <= low_confidence_threshold
        )
        existing_needs_review = _to_bool(row.get("needs_review"))
        if not (existing_needs_review or has_rejection or has_conflict or low_confidence):
            continue

        reason_parts = _existing_review_reasons(row)
        if not reason_parts:
            if has_rejection:
                rejection_reason = (
                    _clean_label(row.get("hybrid_rejection_reason")) or "unknown"
                )
                reason_parts.append(
                    f"hybrid_rejected:{rejection_reason}"
                )
            if has_conflict:
                reason_parts.append("hybrid_graph2d_conflict")
            if low_confidence:
                reason_parts.append("low_confidence")

        existing_score = row.get("review_priority_score")
        score = _to_float(existing_score, default=-1.0)
        if score < 0.0:
            score = _priority_score(
                has_rejection=has_rejection,
                has_conflict=has_conflict,
                low_confidence=low_confidence,
                confidence=confidence,
                low_confidence_threshold=low_confidence_threshold,
            )
        out = dict(row)
        out["review_priority_score"] = f"{score:.3f}"
        out["review_priority"] = _clean_label(row.get("review_priority"))
        out["review_reasons"] = ";".join(reason_parts)
        out["review_has_hybrid_rejection"] = has_rejection
        out["review_has_hybrid_graph2d_conflict"] = has_conflict
        out["review_is_low_confidence"] = low_confidence
        out["review_confidence_band"] = _clean_label(
            row.get("confidence_band") or row.get("review_confidence_band")
        )
        out["review_coarse_label"] = _coarse_label(row)
        out["review_fine_label"] = _fine_label(row)
        out["review_rejection_reason"] = _rejection_reason(row)
        out["review_knowledge_conflict"] = _knowledge_conflict(row)
        out["review_knowledge_conflict_note"] = _knowledge_conflict_note(row)
        out["review_has_knowledge_conflict"] = bool(out["review_knowledge_conflict"])
        out["review_knowledge_check_categories"] = _knowledge_tokens(
            row.get("knowledge_checks"), key="category"
        )
        out["review_standard_candidate_types"] = _knowledge_tokens(
            row.get("standards_candidates"), key="type"
        )
        out["review_knowledge_hint_labels"] = _knowledge_tokens(
            row.get("knowledge_hints"), key="label"
        )
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
    review_priority_counter: Counter[str] = Counter()
    confidence_band_counter: Counter[str] = Counter()
    coarse_label_counter: Counter[str] = Counter()
    fine_label_counter: Counter[str] = Counter()
    rejection_reason_counter: Counter[str] = Counter()
    knowledge_conflict_counter: Counter[str] = Counter()
    knowledge_check_category_counter: Counter[str] = Counter()
    standard_candidate_type_counter: Counter[str] = Counter()
    knowledge_hint_label_counter: Counter[str] = Counter()
    primary_source_counter: Counter[str] = Counter()
    shadow_source_counter: Counter[str] = Counter()
    sample_explanations: List[str] = []
    sample_candidates: List[Dict[str, Any]] = []

    for row in candidate_rows:
        review_reason_counter.update(_split_semicolon_tokens(row.get("review_reasons")))
        review_priority = _clean_label(row.get("review_priority"))
        if review_priority:
            review_priority_counter[review_priority] += 1
        confidence_band = _clean_label(row.get("review_confidence_band"))
        if confidence_band:
            confidence_band_counter[confidence_band] += 1
        coarse_label_counter.update(_split_semicolon_tokens(row.get("review_coarse_label")))
        fine_label_counter.update(_split_semicolon_tokens(row.get("review_fine_label")))
        rejection_reason_counter.update(
            _split_semicolon_tokens(row.get("review_rejection_reason"))
        )
        knowledge_conflict_counter.update(
            _split_semicolon_tokens(row.get("review_knowledge_conflict"))
        )
        knowledge_check_category_counter.update(
            _split_semicolon_tokens(row.get("review_knowledge_check_categories"))
        )
        standard_candidate_type_counter.update(
            _split_semicolon_tokens(row.get("review_standard_candidate_types"))
        )
        knowledge_hint_label_counter.update(
            _split_semicolon_tokens(row.get("review_knowledge_hint_labels"))
        )
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
                    "coarse_label": _clean_label(row.get("review_coarse_label")),
                    "fine_label": _clean_label(row.get("review_fine_label")),
                    "review_priority": _clean_label(row.get("review_priority")),
                    "confidence_band": _clean_label(row.get("review_confidence_band")),
                    "rejection_reason": _clean_label(row.get("review_rejection_reason")),
                    "knowledge_conflict": _clean_label(
                        row.get("review_knowledge_conflict")
                    ),
                    "knowledge_conflict_note": _clean_label(
                        row.get("review_knowledge_conflict_note")
                    ),
                    "knowledge_check_categories": _clean_label(
                        row.get("review_knowledge_check_categories")
                    ),
                    "standard_candidate_types": _clean_label(
                        row.get("review_standard_candidate_types")
                    ),
                    "knowledge_hint_labels": _clean_label(
                        row.get("review_knowledge_hint_labels")
                    ),
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
        "knowledge_conflict_count": sum(
            1 for row in candidate_rows if _to_bool(row.get("review_has_knowledge_conflict"))
        ),
        "knowledge_check_row_count": sum(
            1
            for row in candidate_rows
            if _clean_label(row.get("review_knowledge_check_categories"))
        ),
        "standards_candidate_row_count": sum(
            1
            for row in candidate_rows
            if _clean_label(row.get("review_standard_candidate_types"))
        ),
        "top_review_reasons": _top_named_counts(review_reason_counter),
        "top_review_priorities": _top_named_counts(review_priority_counter),
        "top_confidence_bands": _top_named_counts(confidence_band_counter),
        "top_coarse_labels": _top_named_counts(coarse_label_counter),
        "top_fine_labels": _top_named_counts(fine_label_counter),
        "top_rejection_reasons": _top_named_counts(rejection_reason_counter),
        "top_knowledge_conflicts": _top_named_counts(knowledge_conflict_counter),
        "top_knowledge_check_categories": _top_named_counts(
            knowledge_check_category_counter
        ),
        "top_standard_candidate_types": _top_named_counts(
            standard_candidate_type_counter
        ),
        "top_knowledge_hint_labels": _top_named_counts(knowledge_hint_label_counter),
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
