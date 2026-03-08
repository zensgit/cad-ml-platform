"""Reusable benchmark helpers for applied knowledge-domain signals."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


DOMAIN_SPECS: Dict[str, Dict[str, Any]] = {
    "tolerance": {
        "label": "Tolerance & Fits",
        "check_categories": ["general_tolerance"],
        "violation_categories": ["general_tolerance"],
        "standard_types": ["general_tolerance"],
        "ocr_keywords": ["1800", "2768", "tolerance", "fit", "it"],
        "action": (
            "Increase applied tolerance evidence in benchmark outputs and keep ISO 286 / "
            "GB-T 1800 tables wired into review, companion, and release surfaces."
        ),
    },
    "standards": {
        "label": "Standards & Design Tables",
        "check_categories": ["thread_standard", "standard_part"],
        "violation_categories": ["thread_standard", "standard_conflict"],
        "standard_types": ["metric_thread", "general_tolerance", "standard_part"],
        "ocr_keywords": ["gb", "iso", "din", "ansi", "jis", "thread"],
        "action": (
            "Promote thread/standard-part evidence and design-table coverage into benchmark "
            "surfaces so standards readiness becomes visible as applied capability."
        ),
    },
    "gdt": {
        "label": "GD&T & Datums",
        "check_categories": ["gdt"],
        "violation_categories": ["gdt"],
        "standard_types": ["gdt"],
        "ocr_keywords": ["gdt", "datum", "position", "profile", "flatness"],
        "action": (
            "Increase GD&T extraction/application evidence and surface datum/tolerance "
            "guidance in benchmark release artifacts."
        ),
    },
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_count_map(value: Any) -> Dict[str, int]:
    rows = value if isinstance(value, dict) else {}
    out: Dict[str, int] = {}
    for key, raw in rows.items():
        name = str(key or "").strip()
        if not name:
            continue
        out[name] = _to_int(raw)
    return out


def _sum_exact(values: Dict[str, int], keys: Iterable[str]) -> int:
    return sum(max(values.get(key, 0), 0) for key in keys)


def _sum_keyword_matches(values: Dict[str, int], keywords: Iterable[str]) -> int:
    lowered = {str(key or "").strip().lower(): count for key, count in values.items()}
    total = 0
    for key, count in lowered.items():
        if any(keyword in key for keyword in keywords):
            total += max(count, 0)
    return total


def _evidence_status(signal_count: int) -> str:
    if signal_count >= 3:
        return "ready"
    if signal_count > 0:
        return "partial"
    return "missing"


def _domain_status(readiness_status: str, evidence_status: str) -> str:
    if readiness_status == "ready" and evidence_status == "ready":
        return "ready"
    if readiness_status == "missing" and evidence_status == "missing":
        return "missing"
    return "partial"


def _priority_for_domain(readiness_status: str, evidence_status: str) -> str:
    if readiness_status == "missing" or evidence_status == "missing":
        return "high"
    return "medium"


def build_knowledge_application_status(
    engineering_signals_summary: Dict[str, Any],
    knowledge_readiness_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a benchmark-oriented summary of applied knowledge domains."""
    engineering = (
        engineering_signals_summary.get("engineering_signals")
        or engineering_signals_summary
        or {}
    )
    readiness = (
        knowledge_readiness_summary.get("knowledge_readiness")
        or knowledge_readiness_summary
        or {}
    )
    readiness_domains = readiness.get("domains") or {}

    top_check_categories = _normalize_count_map(engineering.get("top_check_categories"))
    top_violation_categories = _normalize_count_map(
        engineering.get("top_violation_categories")
    )
    top_standard_types = _normalize_count_map(engineering.get("top_standard_types"))
    ocr_candidates = _normalize_count_map(engineering.get("ocr_top_standards_candidates"))

    domains: Dict[str, Dict[str, Any]] = {}
    focus_areas_detail: List[Dict[str, Any]] = []
    ready_count = 0
    partial_count = 0
    missing_count = 0

    for name, spec in DOMAIN_SPECS.items():
        readiness_row = readiness_domains.get(name) or {}
        readiness_status = str(readiness_row.get("status") or "missing")
        check_count = _sum_exact(top_check_categories, spec["check_categories"])
        violation_count = _sum_exact(
            top_violation_categories,
            spec["violation_categories"],
        )
        standards_count = _sum_exact(top_standard_types, spec["standard_types"])
        ocr_count = _sum_keyword_matches(ocr_candidates, spec["ocr_keywords"])
        signal_count = check_count + violation_count + standards_count + ocr_count
        evidence_status = _evidence_status(signal_count)
        status = _domain_status(readiness_status, evidence_status)
        priority = _priority_for_domain(readiness_status, evidence_status)
        gap_reason = (
            "missing_foundation_and_evidence"
            if readiness_status == "missing" and evidence_status == "missing"
            else "missing_foundation"
            if readiness_status == "missing"
            else "missing_application_evidence"
            if evidence_status == "missing"
            else "partial_coverage"
        )

        row = {
            "domain": name,
            "label": str(spec["label"]),
            "status": status,
            "priority": priority,
            "gap_reason": gap_reason if status != "ready" else "none",
            "readiness_status": readiness_status,
            "evidence_status": evidence_status,
            "signal_count": signal_count,
            "signal_breakdown": {
                "check_categories": check_count,
                "violation_categories": violation_count,
                "standard_types": standards_count,
                "ocr_candidates": ocr_count,
            },
            "focus_components": list(readiness_row.get("focus_components") or []),
            "missing_metrics": list(readiness_row.get("missing_metrics") or []),
            "action": (
                str(readiness_row.get("action") or "").strip()
                if readiness_status == "missing"
                else str(spec["action"])
            ),
        }
        domains[name] = row

        if status == "ready":
            ready_count += 1
        elif status == "partial":
            partial_count += 1
            focus_areas_detail.append(row)
        else:
            missing_count += 1
            focus_areas_detail.append(row)

    if ready_count == len(DOMAIN_SPECS):
        overall_status = "knowledge_application_ready"
    elif ready_count > 0 or partial_count > 0:
        overall_status = "knowledge_application_partial"
    else:
        overall_status = "knowledge_application_missing"

    priority_domains = [
        row["domain"]
        for row in focus_areas_detail
        if str(row.get("priority") or "medium") == "high"
    ]

    return {
        "status": overall_status,
        "ready_domain_count": ready_count,
        "partial_domain_count": partial_count,
        "missing_domain_count": missing_count,
        "total_domain_count": len(DOMAIN_SPECS),
        "focus_areas": [row["domain"] for row in focus_areas_detail],
        "priority_domains": priority_domains,
        "focus_areas_detail": focus_areas_detail,
        "domains": domains,
    }


def knowledge_application_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = str(summary.get("status") or "").strip().lower()
    if status == "knowledge_application_missing":
        return [
            "Produce applied tolerance, standards, and GD&T evidence before claiming "
            "benchmark knowledge-application readiness."
        ]

    items: List[str] = []
    for row in summary.get("focus_areas_detail") or []:
        domain = str(row.get("domain") or "unknown")
        readiness_status = str(row.get("readiness_status") or "missing")
        evidence_status = str(row.get("evidence_status") or "missing")
        action = str(row.get("action") or "").strip()
        if readiness_status == "missing":
            items.append(f"Backfill {domain} foundation: {action}")
            continue
        if evidence_status == "missing":
            items.append(
                f"Promote {domain} application evidence into benchmark surfaces: {action}"
            )
            continue
        items.append(f"Raise {domain} application coverage: {action}")

    if status == "knowledge_application_partial":
        items.append(
            "Wire knowledge application status into companion, bundle, release decision, "
            "and runbook surfaces before treating benchmark knowledge coverage as complete."
        )
    return items


def render_knowledge_application_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = payload.get("knowledge_application") or {}
    domains = component.get("domains") or {}
    recommendations = payload.get("recommendations") or []
    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        f"- `status`: `{component.get('status', 'knowledge_application_missing')}`",
        f"- `ready_domain_count`: `{component.get('ready_domain_count', 0)}`",
        f"- `partial_domain_count`: `{component.get('partial_domain_count', 0)}`",
        f"- `missing_domain_count`: `{component.get('missing_domain_count', 0)}`",
        f"- `priority_domains`: "
        f"`{', '.join(component.get('priority_domains') or []) or 'none'}`",
        "",
        "## Domains",
        "",
    ]
    for name, row in domains.items():
        lines.append(
            "- "
            f"`{name}` "
            f"status=`{row.get('status')}` "
            f"readiness=`{row.get('readiness_status')}` "
            f"evidence=`{row.get('evidence_status')}` "
            f"signal_count=`{row.get('signal_count')}` "
            f"missing_metrics=`{', '.join(row.get('missing_metrics') or []) or 'none'}`"
        )
    lines.extend(["", "## Focus Areas", ""])
    focus_areas = component.get("focus_areas_detail") or []
    if focus_areas:
        for row in focus_areas:
            lines.append(
                "- "
                f"`{row.get('domain')}` "
                f"status=`{row.get('status')}` "
                f"priority=`{row.get('priority')}` "
                f"gap_reason=`{row.get('gap_reason')}` "
                f"action=`{row.get('action')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Recommendations", ""])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- Knowledge application is ready for benchmark release surfaces.")
    lines.append("")
    return "\n".join(lines)
