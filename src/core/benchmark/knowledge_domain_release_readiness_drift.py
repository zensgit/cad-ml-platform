"""Drift helpers for benchmark knowledge-domain release-readiness matrix."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


STATUS_RANK = {
    "ready": 3,
    "partial": 2,
    "blocked": 1,
    "unknown": 0,
    "knowledge_domain_release_readiness_ready": 3,
    "knowledge_domain_release_readiness_partial": 2,
    "knowledge_domain_release_readiness_blocked": 1,
    "knowledge_domain_release_readiness_unavailable": 0,
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _status(value: Any) -> str:
    return _text(value) or "unknown"


def _rank(value: Any) -> int:
    return STATUS_RANK.get(_status(value), 0)


def _compact(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = _text(item)
        if text and text not in out:
            out.append(text)
    return out


def _component(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("knowledge_domain_release_readiness_drift") or summary or {}


def _matrix(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("knowledge_domain_release_readiness_matrix") or summary or {}


def _domains(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = _matrix(summary).get("domains") or {}
    return rows if isinstance(rows, dict) else {}


def build_knowledge_domain_release_readiness_drift_status(
    current_summary: Dict[str, Any],
    previous_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current and previous release-readiness summaries."""
    current_component = _matrix(current_summary)
    previous_component = _matrix(previous_summary)

    if not previous_component:
        return {
            "status": "baseline_missing",
            "summary": "baseline missing",
            "current_status": _status(current_component.get("status")),
            "previous_status": "baseline_missing",
            "ready_domain_delta": int(current_component.get("ready_domain_count") or 0),
            "partial_domain_delta": int(
                current_component.get("partial_domain_count") or 0
            ),
            "blocked_domain_delta": int(
                current_component.get("blocked_domain_count") or 0
            ),
            "regressions": [],
            "improvements": [],
            "resolved_priority_domains": [],
            "new_priority_domains": _compact(
                current_component.get("priority_domains") or []
            ),
            "resolved_releasable_domains": [],
            "new_releasable_domains": _compact(
                current_component.get("releasable_domains") or []
            ),
            "resolved_blocked_domains": [],
            "new_blocked_domains": _compact(current_component.get("blocked_domains") or []),
            "domain_regressions": [],
            "domain_improvements": [],
            "domain_changes": [],
        }

    current_domains = _domains(current_summary)
    previous_domains = _domains(previous_summary)
    domain_names = sorted(set(current_domains) | set(previous_domains))
    domain_regressions: List[str] = []
    domain_improvements: List[str] = []
    domain_changes: List[Dict[str, Any]] = []

    for name in domain_names:
        current_row = current_domains.get(name) or {}
        previous_row = previous_domains.get(name) or {}
        current_status = _status(current_row.get("status"))
        previous_status = _status(previous_row.get("status"))
        current_rank = _rank(current_status)
        previous_rank = _rank(previous_status)
        if current_rank > previous_rank:
            trend = "improved"
            domain_improvements.append(name)
        elif current_rank < previous_rank:
            trend = "regressed"
            domain_regressions.append(name)
        else:
            trend = "stable"
        domain_changes.append(
            {
                "domain": name,
                "label": _text(current_row.get("label"))
                or _text(previous_row.get("label"))
                or name,
                "previous_status": previous_status,
                "current_status": current_status,
                "trend": trend,
                "previous_priority": _text(previous_row.get("priority")) or "unknown",
                "current_priority": _text(current_row.get("priority")) or "unknown",
                "previous_blocking_reasons": _compact(
                    previous_row.get("blocking_reasons") or []
                ),
                "current_blocking_reasons": _compact(
                    current_row.get("blocking_reasons") or []
                ),
                "previous_warning_reasons": _compact(
                    previous_row.get("warning_reasons") or []
                ),
                "current_warning_reasons": _compact(
                    current_row.get("warning_reasons") or []
                ),
            }
        )

    current_status = _status(current_component.get("status"))
    previous_status = _status(previous_component.get("status"))

    if domain_regressions and domain_improvements:
        status = "mixed"
    elif domain_regressions:
        status = "regressed"
    elif domain_improvements:
        status = "improved"
    else:
        status = "stable"

    summary = (
        f"current={current_status}; previous={previous_status}; "
        f"regressions={len(domain_regressions)}; improvements={len(domain_improvements)}"
    )

    current_priority = set(_compact(current_component.get("priority_domains") or []))
    previous_priority = set(_compact(previous_component.get("priority_domains") or []))
    current_releasable = set(
        _compact(current_component.get("releasable_domains") or [])
    )
    previous_releasable = set(
        _compact(previous_component.get("releasable_domains") or [])
    )
    current_blocked = set(_compact(current_component.get("blocked_domains") or []))
    previous_blocked = set(_compact(previous_component.get("blocked_domains") or []))

    return {
        "status": status,
        "summary": summary,
        "current_status": current_status,
        "previous_status": previous_status,
        "ready_domain_delta": int(current_component.get("ready_domain_count") or 0)
        - int(previous_component.get("ready_domain_count") or 0),
        "partial_domain_delta": int(current_component.get("partial_domain_count") or 0)
        - int(previous_component.get("partial_domain_count") or 0),
        "blocked_domain_delta": int(current_component.get("blocked_domain_count") or 0)
        - int(previous_component.get("blocked_domain_count") or 0),
        "regressions": _compact(domain_regressions),
        "improvements": _compact(domain_improvements),
        "resolved_priority_domains": sorted(previous_priority - current_priority),
        "new_priority_domains": sorted(current_priority - previous_priority),
        "resolved_releasable_domains": sorted(previous_releasable - current_releasable),
        "new_releasable_domains": sorted(current_releasable - previous_releasable),
        "resolved_blocked_domains": sorted(previous_blocked - current_blocked),
        "new_blocked_domains": sorted(current_blocked - previous_blocked),
        "domain_regressions": domain_regressions,
        "domain_improvements": domain_improvements,
        "domain_changes": domain_changes,
    }


def knowledge_domain_release_readiness_drift_recommendations(
    summary: Dict[str, Any],
) -> List[str]:
    status = _status(summary.get("status"))
    if status == "baseline_missing":
        return [
            "Persist the current release-readiness matrix as the new benchmark baseline."
        ]
    if status == "regressed":
        items = [
            "Restore regressed knowledge-domain release-readiness signals before promotion."
        ]
        regressed = _compact(summary.get("domain_regressions") or [])
        if regressed:
            items.append("Regressed domains: " + ", ".join(regressed))
        blocked = _compact(summary.get("new_blocked_domains") or [])
        if blocked:
            items.append("New blocked domains: " + ", ".join(blocked))
        return items
    if status == "improved":
        items = [
            "Promote the improved release-readiness matrix after refreshing release surfaces."
        ]
        improved = _compact(summary.get("domain_improvements") or [])
        if improved:
            items.append("Improved domains: " + ", ".join(improved))
        return items
    if status == "mixed":
        return [
            "Keep the previous release-readiness baseline until regressions are cleared.",
            "Regressed domains: "
            + (", ".join(_compact(summary.get("domain_regressions") or [])) or "none"),
            "Improved domains: "
            + (", ".join(_compact(summary.get("domain_improvements") or [])) or "none"),
        ]
    if summary.get("new_priority_domains"):
        return [
            "Watch new priority domains: "
            + ", ".join(_compact(summary.get("new_priority_domains") or []))
        ]
    return ["Release-readiness benchmark is stable against the previous baseline."]


def render_knowledge_domain_release_readiness_drift_markdown(
    payload: Dict[str, Any],
    title: str,
) -> str:
    component = _component(payload)
    lines = [
        f"# {title}",
        "",
        f"- `status`: `{component.get('status') or 'unknown'}`",
        f"- `summary`: `{component.get('summary') or 'none'}`",
        f"- `current_status`: `{component.get('current_status') or 'unknown'}`",
        f"- `previous_status`: `{component.get('previous_status') or 'unknown'}`",
        "",
        "## Domain Changes",
        "",
    ]
    for row in component.get("domain_changes") or []:
        lines.extend(
            [
                (
                    f"### {row.get('label') or row.get('domain')}"
                    f" (`{row.get('domain') or 'unknown'}`)"
                ),
                "",
                f"- `trend`: `{row.get('trend') or 'stable'}`",
                (
                    f"- `status`: `{row.get('previous_status') or 'unknown'}` -> "
                    f"`{row.get('current_status') or 'unknown'}`"
                ),
                (
                    f"- `priority`: `{row.get('previous_priority') or 'unknown'}` -> "
                    f"`{row.get('current_priority') or 'unknown'}`"
                ),
                (
                    f"- `blocking_reasons`: "
                    f"`{', '.join(row.get('current_blocking_reasons') or []) or 'none'}`"
                ),
                (
                    f"- `warning_reasons`: "
                    f"`{', '.join(row.get('current_warning_reasons') or []) or 'none'}`"
                ),
                "",
            ]
        )
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend(["## Recommendations", ""])
        for item in recommendations:
            for domain in _compact(component.get("domain_regressions") or []):
                item = item.replace(domain, f"`{domain}`")
            for domain in _compact(component.get("domain_improvements") or []):
                item = item.replace(domain, f"`{domain}`")
            for domain in _compact(component.get("new_blocked_domains") or []):
                item = item.replace(domain, f"`{domain}`")
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"
