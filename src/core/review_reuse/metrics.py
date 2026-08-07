"""Review-workflow metrics from the task ledger (not Track E model metrics)."""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, List, Optional

from .models import CandidateState, ReviewReuseTask, TaskEventType, TaskStatus
from .store import ReviewReuseStore


def compute_review_metrics(
    store: ReviewReuseStore, tenant_id: str
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
