"""Trend helpers for benchmark competitive-surpass index."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


STATUS_RANK = {
    "competitive_surpass_ready": 3,
    "competitive_surpass_partial": 2,
    "competitive_surpass_attention_required": 1,
    "competitive_surpass_blocked": 0,
    "unknown": 0,
}

PILLAR_TIERS = ("ready_pillars", "partial_pillars", "blocked_pillars")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _component(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("competitive_surpass_index") or summary or {}


def _compact(items: Iterable[Any]) -> List[str]:
    values: List[str] = []
    for item in items:
        text = _text(item)
        if text and text not in values:
            values.append(text)
    return values


def _status_rank(value: Any) -> int:
    return STATUS_RANK.get(_text(value) or "unknown", 0)


def _tier_sets(component: Dict[str, Any]) -> Dict[str, set[str]]:
    return {name: set(_compact(component.get(name) or [])) for name in PILLAR_TIERS}


def _pillar_tier_map(component: Dict[str, Any]) -> Dict[str, str]:
    tiers = _tier_sets(component)
    mapping: Dict[str, str] = {}
    for tier_name, pillars in tiers.items():
        for pillar in pillars:
            mapping[pillar] = tier_name
    return mapping


def _status(
    current_status: str,
    previous_status: str,
    score_delta: int,
    pillar_improvements: List[str],
    pillar_regressions: List[str],
    resolved_primary_gaps: List[str],
    new_primary_gaps: List[str],
) -> str:
    if previous_status == "baseline_missing":
        return "baseline_missing"

    improved = bool(
        score_delta > 0 or pillar_improvements or resolved_primary_gaps
    )
    regressed = bool(
        score_delta < 0
        or pillar_regressions
        or new_primary_gaps
        or _status_rank(current_status) < _status_rank(previous_status)
    )

    if improved and regressed:
        return "mixed"
    if regressed:
        return "regressed"
    if improved:
        return "improved"
    return "stable"


def build_competitive_surpass_trend_status(
    current_summary: Dict[str, Any],
    previous_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current and previous competitive-surpass summaries."""
    current_component = _component(current_summary)
    previous_component = _component(previous_summary)

    current_status = _text(current_component.get("status")) or "unknown"
    previous_status = _text(previous_component.get("status")) or "baseline_missing"
    current_score = int(current_component.get("score") or 0)
    previous_score = int(previous_component.get("score") or 0)

    current_primary_gaps = set(_compact(current_component.get("primary_gaps") or []))
    previous_primary_gaps = set(_compact(previous_component.get("primary_gaps") or []))

    current_pillars = _pillar_tier_map(current_component)
    previous_pillars = _pillar_tier_map(previous_component)
    pillar_names = sorted(set(current_pillars) | set(previous_pillars))
    pillar_improvements: List[str] = []
    pillar_regressions: List[str] = []
    pillar_changes: List[Dict[str, Any]] = []

    for pillar in pillar_names:
        current_tier = current_pillars.get(pillar, "missing")
        previous_tier = previous_pillars.get(pillar, "missing")
        current_rank = {"ready_pillars": 3, "partial_pillars": 2, "blocked_pillars": 1}.get(
            current_tier, 0
        )
        previous_rank = {"ready_pillars": 3, "partial_pillars": 2, "blocked_pillars": 1}.get(
            previous_tier, 0
        )
        if current_rank > previous_rank:
            trend = "improved"
            pillar_improvements.append(pillar)
        elif current_rank < previous_rank:
            trend = "regressed"
            pillar_regressions.append(pillar)
        else:
            trend = "stable"
        pillar_changes.append(
            {
                "name": pillar,
                "current_tier": current_tier,
                "previous_tier": previous_tier,
                "trend": trend,
            }
        )

    status = _status(
        current_status=current_status,
        previous_status=previous_status,
        score_delta=current_score - previous_score,
        pillar_improvements=pillar_improvements,
        pillar_regressions=pillar_regressions,
        resolved_primary_gaps=sorted(previous_primary_gaps - current_primary_gaps),
        new_primary_gaps=sorted(current_primary_gaps - previous_primary_gaps),
    )

    return {
        "status": status,
        "current_status": current_status,
        "previous_status": previous_status,
        "current_score": current_score,
        "previous_score": previous_score,
        "score_delta": current_score - previous_score,
        "ready_pillar_delta": len(_compact(current_component.get("ready_pillars") or []))
        - len(_compact(previous_component.get("ready_pillars") or [])),
        "partial_pillar_delta": len(_compact(current_component.get("partial_pillars") or []))
        - len(_compact(previous_component.get("partial_pillars") or [])),
        "blocked_pillar_delta": len(_compact(current_component.get("blocked_pillars") or []))
        - len(_compact(previous_component.get("blocked_pillars") or [])),
        "pillar_improvements": pillar_improvements,
        "pillar_regressions": pillar_regressions,
        "resolved_primary_gaps": sorted(previous_primary_gaps - current_primary_gaps),
        "new_primary_gaps": sorted(current_primary_gaps - previous_primary_gaps),
        "pillar_changes": pillar_changes,
    }


def competitive_surpass_trend_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = _text(summary.get("status")) or "unknown"
    if status == "baseline_missing":
        return [
            "Persist the current competitive surpass index as the next benchmark baseline."
        ]
    if status == "regressed":
        items = [
            "Resolve competitive surpass regressions before claiming benchmark progress."
        ]
        if summary.get("new_primary_gaps"):
            items.append(
                "New primary gaps: " + ", ".join(_compact(summary.get("new_primary_gaps") or []))
            )
        if summary.get("pillar_regressions"):
            items.append(
                "Regressed pillars: "
                + ", ".join(_compact(summary.get("pillar_regressions") or []))
            )
        return items
    if status == "improved":
        items = [
            "Promote the improved competitive surpass posture after CI surfaces refresh."
        ]
        if summary.get("resolved_primary_gaps"):
            items.append(
                "Resolved primary gaps: "
                + ", ".join(_compact(summary.get("resolved_primary_gaps") or []))
            )
        if summary.get("pillar_improvements"):
            items.append(
                "Improved pillars: "
                + ", ".join(_compact(summary.get("pillar_improvements") or []))
            )
        return items
    if status == "mixed":
        return [
            (
                "Keep the current competitive surpass rollout under review until "
                "regressions are cleared."
            ),
            "Regressed pillars: "
            + (", ".join(_compact(summary.get("pillar_regressions") or [])) or "none"),
            "Improved pillars: "
            + (", ".join(_compact(summary.get("pillar_improvements") or [])) or "none"),
        ]
    return [
        "Competitive surpass index is stable against the previous benchmark baseline."
    ]


def render_competitive_surpass_trend_markdown(
    payload: Dict[str, Any],
    title: str = "Benchmark Competitive Surpass Trend",
) -> str:
    component = payload.get("competitive_surpass_trend") or {}
    lines = [
        f"# {title}",
        "",
        f"- `status`: `{component.get('status') or 'unknown'}`",
        f"- `current_status`: `{component.get('current_status') or 'unknown'}`",
        f"- `previous_status`: `{component.get('previous_status') or 'unknown'}`",
        f"- `score_delta`: `{component.get('score_delta') or 0}`",
        "",
        "## Deltas",
        "",
        f"- `ready_pillar_delta`: `{component.get('ready_pillar_delta') or 0}`",
        f"- `partial_pillar_delta`: `{component.get('partial_pillar_delta') or 0}`",
        f"- `blocked_pillar_delta`: `{component.get('blocked_pillar_delta') or 0}`",
        "",
        "## Changes",
        "",
        (
            "- `pillar_improvements`: "
            + (
                ", ".join(_compact(component.get("pillar_improvements") or []))
                or "none"
            )
        ),
        (
            "- `pillar_regressions`: "
            + (
                ", ".join(_compact(component.get("pillar_regressions") or []))
                or "none"
            )
        ),
        (
            "- `resolved_primary_gaps`: "
            + (
                ", ".join(_compact(component.get("resolved_primary_gaps") or []))
                or "none"
            )
        ),
        (
            "- `new_primary_gaps`: "
            + (", ".join(_compact(component.get("new_primary_gaps") or [])) or "none")
        ),
        "",
        "## Recommendations",
        "",
    ]
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.extend([f"- {item}" for item in recommendations])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
