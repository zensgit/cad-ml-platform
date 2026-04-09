"""Reusable benchmark helpers for engineering/knowledge signals."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ranked_counts_to_map(value: Any) -> Dict[str, int]:
    rows = value if isinstance(value, list) else []
    out: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out[name] = _to_int(row.get("count"))
    return out


def _ranked_total(value: Any) -> int:
    return sum(_ranked_counts_to_map(value).values())


def _coverage_ratio(count: int, sample_size: int) -> float:
    if sample_size <= 0:
        return 0.0
    return round(float(count) / float(sample_size), 6)


def build_engineering_signals_status(
    hybrid_summary: Dict[str, Any],
    ocr_review_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize knowledge/engineering signals for benchmark reporting."""
    if not hybrid_summary and not ocr_review_summary:
        return {"status": "missing"}

    knowledge = hybrid_summary.get("knowledge_signals") or {}
    sample_size = _to_int(hybrid_summary.get("sample_size"))
    rows_with_checks = _to_int(knowledge.get("rows_with_checks"))
    rows_with_violations = _to_int(knowledge.get("rows_with_violations"))
    rows_with_standards = _to_int(knowledge.get("rows_with_standards_candidates"))
    rows_with_hints = _to_int(knowledge.get("rows_with_hints"))
    total_checks = _to_int(knowledge.get("total_checks"))
    total_violations = _to_int(knowledge.get("total_violations"))
    total_standards = _to_int(knowledge.get("total_standards_candidates"))
    total_hints = _to_int(knowledge.get("total_hints"))

    top_check_categories = knowledge.get("top_check_categories") or {}
    top_violation_categories = knowledge.get("top_violation_categories") or {}
    top_standard_types = knowledge.get("top_standard_types") or {}
    top_hint_labels = knowledge.get("top_hint_labels") or {}

    ocr_review_candidates = _to_int(ocr_review_summary.get("review_candidate_count"))
    ocr_exported_records = _to_int(ocr_review_summary.get("exported_records"))
    ocr_automation_ready = _to_int(ocr_review_summary.get("automation_ready_count"))
    ocr_top_standard_candidates = _ranked_counts_to_map(
        ocr_review_summary.get("top_standards_candidates")
    )
    ocr_top_primary_gaps = _ranked_counts_to_map(
        ocr_review_summary.get("primary_gap_counts")
    )
    ocr_top_review_reasons = _ranked_counts_to_map(
        ocr_review_summary.get("top_review_reasons")
    )

    signal_rows_upper_bound = max(
        rows_with_checks,
        rows_with_violations,
        rows_with_standards,
        rows_with_hints,
    )
    coverage_ratio = _coverage_ratio(signal_rows_upper_bound, sample_size)
    standards_ratio = _coverage_ratio(rows_with_standards, sample_size)
    violations_ratio = _coverage_ratio(rows_with_violations, sample_size)
    hints_ratio = _coverage_ratio(rows_with_hints, sample_size)
    ocr_standard_signal_count = _ranked_total(ocr_review_summary.get("top_standards_candidates"))

    if sample_size <= 0 and ocr_exported_records <= 0 and ocr_review_candidates <= 0:
        status = "missing"
    elif (
        coverage_ratio >= 0.4
        and rows_with_violations > 0
        and (rows_with_standards > 0 or ocr_standard_signal_count > 0)
    ):
        status = "engineering_semantics_ready"
    elif (
        signal_rows_upper_bound > 0
        or ocr_standard_signal_count > 0
        or total_checks > 0
        or total_violations > 0
        or total_standards > 0
    ):
        status = "partial_engineering_semantics"
    else:
        status = "weak_engineering_semantics"

    return {
        "status": status,
        "sample_size": sample_size,
        "signal_rows_upper_bound": signal_rows_upper_bound,
        "coverage_ratio": coverage_ratio,
        "rows_with_checks": rows_with_checks,
        "rows_with_violations": rows_with_violations,
        "rows_with_standards_candidates": rows_with_standards,
        "rows_with_hints": rows_with_hints,
        "checks_coverage_ratio": _coverage_ratio(rows_with_checks, sample_size),
        "violations_coverage_ratio": violations_ratio,
        "standards_coverage_ratio": standards_ratio,
        "hints_coverage_ratio": hints_ratio,
        "total_checks": total_checks,
        "total_violations": total_violations,
        "total_standards_candidates": total_standards,
        "total_hints": total_hints,
        "top_check_categories": top_check_categories,
        "top_violation_categories": top_violation_categories,
        "top_standard_types": top_standard_types,
        "top_hint_labels": top_hint_labels,
        "ocr_review_candidate_count": ocr_review_candidates,
        "ocr_exported_records": ocr_exported_records,
        "ocr_automation_ready_count": ocr_automation_ready,
        "ocr_standard_signal_count": ocr_standard_signal_count,
        "ocr_top_standards_candidates": ocr_top_standard_candidates,
        "ocr_top_primary_gaps": ocr_top_primary_gaps,
        "ocr_top_review_reasons": ocr_top_review_reasons,
        "average_ocr_readiness_score": _to_float(
            ocr_review_summary.get("average_readiness_score")
        ),
    }


def engineering_signals_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = str(summary.get("status") or "").strip().lower()
    if status == "missing":
        return [
            "Produce hybrid knowledge summaries and OCR engineering review summaries before "
            "claiming benchmark engineering-semantic readiness."
        ]

    items: List[str] = []
    if not summary.get("rows_with_violations"):
        items.append(
            "Surface violation/rule-conflict evidence in benchmark artifacts instead of only "
            "positive matches."
        )
    if not summary.get("rows_with_standards_candidates") and not summary.get(
        "ocr_standard_signal_count"
    ):
        items.append(
            "Raise standards/tolerance/GD&T candidate visibility in benchmark evidence."
        )
    if _to_float(summary.get("coverage_ratio")) < 0.4:
        items.append(
            "Increase engineering signal coverage across hybrid benchmark samples."
        )
    if _to_int(summary.get("ocr_review_candidate_count")) > 0 and _to_int(
        summary.get("ocr_automation_ready_count")
    ) <= 0:
        items.append(
            "Reduce OCR engineering review backlog or raise automation-ready coverage."
        )
    if status == "weak_engineering_semantics":
        items.append(
            "Treat engineering checks as a benchmark gap until standards, violations, and hints "
            "appear in enough samples."
        )
    elif status == "partial_engineering_semantics":
        items.append(
            "Promote engineering signals into scorecard, companion, and release decision views."
        )
    return items


def render_engineering_signals_markdown(payload: Dict[str, Any], title: str) -> str:
    component = payload.get("engineering_signals") or {}
    recommendations = payload.get("recommendations") or []
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        f"- `status`: `{component.get('status', 'missing')}`",
        f"- `sample_size`: `{component.get('sample_size', 0)}`",
        f"- `coverage_ratio`: `{component.get('coverage_ratio', 0.0)}`",
        f"- `rows_with_checks`: `{component.get('rows_with_checks', 0)}`",
        f"- `rows_with_violations`: `{component.get('rows_with_violations', 0)}`",
        f"- `rows_with_standards_candidates`: "
        f"`{component.get('rows_with_standards_candidates', 0)}`",
        f"- `rows_with_hints`: `{component.get('rows_with_hints', 0)}`",
        "",
        "## Hybrid Signal Leaders",
        "",
        f"- `top_violation_categories`: `{component.get('top_violation_categories', {})}`",
        f"- `top_standard_types`: `{component.get('top_standard_types', {})}`",
        f"- `top_hint_labels`: `{component.get('top_hint_labels', {})}`",
        "",
        "## OCR Engineering Review",
        "",
        f"- `ocr_review_candidate_count`: `{component.get('ocr_review_candidate_count', 0)}`",
        f"- `ocr_automation_ready_count`: `{component.get('ocr_automation_ready_count', 0)}`",
        f"- `ocr_top_standards_candidates`: "
        f"`{component.get('ocr_top_standards_candidates', {})}`",
        "",
        "## Recommendations",
        "",
    ]
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- Engineering signals are ready for benchmark consumption.")
    lines.append("")
    return "\n".join(lines)
