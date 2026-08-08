"""Review-workflow metrics from the task ledger (not Track E model metrics)."""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, List, Optional

from .models import CandidateState, ReviewReuseTask, TaskEventType, TaskStatus
from .store import ReviewReuseStoreProtocol


def compute_review_metrics(
    store: ReviewReuseStoreProtocol, tenant_id: str
) -> Dict[str, Any]:
    """Aggregate operator metrics for one tenant's ReviewReuse tasks."""
    tasks = store.list_for_tenant(tenant_id)
    by_status: Dict[str, int] = {}
    by_decision: Dict[str, int] = {}
    insufficient = 0
    candidate_total = 0
    times_to_evidence: List[float] = []
    reviewers: set = set()

    for t in tasks:
        st = t.status.value if isinstance(t.status, TaskStatus) else str(t.status)
        by_status[st] = by_status.get(st, 0) + 1

        if t.human_decision is not None:
            ds = t.human_decision.state.value
            by_decision[ds] = by_decision.get(ds, 0) + 1
            reviewers.add(t.human_decision.reviewer_id)

        for c in t.candidates:
            candidate_total += 1
            if c.state == CandidateState.insufficient_evidence:
                insufficient += 1

        tte = _time_to_evidence(t)
        if tte is not None:
            times_to_evidence.append(tte)

    return {
        "schema_version": "review-reuse-metrics-v1",
        "metric_family": "review_workflow",  # NOT track_e_model_release
        "tenant_id": tenant_id,
        "task_count": len(tasks),
        "by_status": by_status,
        "by_decision": by_decision,
        "accepted_reuse": by_decision.get("reuse", 0),
        "candidate_total": candidate_total,
        "insufficient_evidence_count": insufficient,
        "insufficient_evidence_rate": (
            (insufficient / candidate_total) if candidate_total else 0.0
        ),
        "median_time_to_evidence_seconds": (
            float(median(times_to_evidence)) if times_to_evidence else None
        ),
        "reviewer_coverage": len(reviewers),
        "notes": [
            "Human false-duplicate / missed-reuse / top-5 usefulness require pilot labels",
            "Does not replace Track E model-release metrics or eval_integrity_gate",
        ],
    }


def format_metrics_markdown(metrics_dict: Dict[str, Any]) -> str:
    """Render review-workflow metrics as operator-facing markdown (pure)."""
    m = metrics_dict or {}
    lines: List[str] = [
        "# ReviewReuse workflow metrics",
        "",
        f"- schema_version: `{m.get('schema_version')}`",
        f"- metric_family: `{m.get('metric_family')}`",
        f"- tenant_id: `{m.get('tenant_id')}`",
        f"- task_count: {m.get('task_count')}",
        f"- accepted_reuse: {m.get('accepted_reuse')}",
        f"- candidate_total: {m.get('candidate_total')}",
        f"- insufficient_evidence_count: {m.get('insufficient_evidence_count')}",
        f"- insufficient_evidence_rate: {m.get('insufficient_evidence_rate')}",
        f"- median_time_to_evidence_seconds: {m.get('median_time_to_evidence_seconds')}",
        f"- reviewer_coverage: {m.get('reviewer_coverage')}",
        "",
        "## By status",
        "",
    ]
    by_status = m.get("by_status") or {}
    if by_status:
        for key in sorted(by_status.keys()):
            lines.append(f"- {key}: {by_status[key]}")
    else:
        lines.append("- _(none)_")

    lines.extend(["", "## By decision", ""])
    by_decision = m.get("by_decision") or {}
    if by_decision:
        for key in sorted(by_decision.keys()):
            lines.append(f"- {key}: {by_decision[key]}")
    else:
        lines.append("- _(none)_")

    notes = m.get("notes") or []
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines) + "\n"


def _time_to_evidence(task: ReviewReuseTask) -> Optional[float]:
    start = task.created_at
    ready = None
    for e in task.events:
        if e.event_type == TaskEventType.evidence_pack_ready:
            ready = e.ts
            break
    if ready is None:
        return None
    return max(0.0, float(ready) - float(start))
