"""Reusable feedback flywheel benchmark helpers."""

from __future__ import annotations

from typing import Any, Dict, List


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_feedback_flywheel_status(
    feedback_summary: Dict[str, Any],
    finetune_summary: Dict[str, Any],
    metric_train_summary: Dict[str, Any],
) -> Dict[str, Any]:
    if not feedback_summary and not finetune_summary and not metric_train_summary:
        return {"status": "missing"}

    total_feedback = _to_int(feedback_summary.get("total"))
    correction_count = _to_int(feedback_summary.get("correction_count"))
    coarse_correction_count = _to_int(feedback_summary.get("coarse_correction_count"))
    average_rating = feedback_summary.get("average_rating")
    finetune_sample_count = _to_int(finetune_summary.get("sample_count"))
    finetune_vector_count = _to_int(finetune_summary.get("vector_count"))
    metric_triplet_count = _to_int(metric_train_summary.get("triplet_count"))
    metric_unique_anchor_count = _to_int(metric_train_summary.get("unique_anchor_count"))

    if total_feedback <= 0:
        status = "missing"
    elif correction_count <= 0 and coarse_correction_count <= 0:
        status = "passive_feedback_only"
    elif finetune_sample_count > 0 and metric_triplet_count > 0:
        status = "closed_loop_ready"
    elif finetune_sample_count > 0 or metric_triplet_count > 0:
        status = "partially_closed_loop"
    else:
        status = "feedback_collected"

    return {
        "status": status,
        "feedback_total": total_feedback,
        "correction_count": correction_count,
        "coarse_correction_count": coarse_correction_count,
        "average_rating": average_rating,
        "finetune_sample_count": finetune_sample_count,
        "finetune_vector_count": finetune_vector_count,
        "metric_triplet_count": metric_triplet_count,
        "metric_unique_anchor_count": metric_unique_anchor_count,
        "review_outcomes": feedback_summary.get("by_review_outcome") or {},
        "review_reasons": feedback_summary.get("by_review_reason") or {},
        "finetune_label_distribution": finetune_summary.get("label_distribution") or {},
        "finetune_coarse_label_distribution": (
            finetune_summary.get("coarse_label_distribution") or {}
        ),
        "metric_anchor_label_distribution": (
            metric_train_summary.get("anchor_label_distribution") or {}
        ),
        "metric_negative_label_distribution": (
            metric_train_summary.get("negative_label_distribution") or {}
        ),
    }


def feedback_flywheel_recommendations(summary: Dict[str, Any]) -> List[str]:
    status = str(summary.get("status") or "").strip().lower()
    if status == "missing":
        return [
            "Produce feedback stats, finetune summaries, and metric-train summaries for the "
            "benchmark flywheel."
        ]
    if status == "passive_feedback_only":
        return [
            "Collect actionable corrections instead of ratings-only feedback before claiming a "
            "closed retraining loop."
        ]
    if status == "feedback_collected":
        return [
            "Run finetune and metric-training summaries so feedback evidence becomes a real "
            "retraining flywheel."
        ]
    if status == "partially_closed_loop":
        return [
            "Close the feedback flywheel by producing both fine-tune and metric-training "
            "artifacts from reviewed samples."
        ]
    return []


def render_feedback_flywheel_markdown(payload: Dict[str, Any], title: str) -> str:
    component = payload.get("feedback_flywheel") or {}
    recommendations = payload.get("recommendations") or []
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        f"- `status`: `{component.get('status', 'missing')}`",
        f"- `feedback_total`: `{component.get('feedback_total', 0)}`",
        f"- `correction_count`: `{component.get('correction_count', 0)}`",
        f"- `coarse_correction_count`: `{component.get('coarse_correction_count', 0)}`",
        f"- `finetune_sample_count`: `{component.get('finetune_sample_count', 0)}`",
        f"- `metric_triplet_count`: `{component.get('metric_triplet_count', 0)}`",
        "",
        "## Label Coverage",
        "",
        f"- `finetune_label_distribution`: `{component.get('finetune_label_distribution', {})}`",
        f"- `finetune_coarse_label_distribution`: "
        f"`{component.get('finetune_coarse_label_distribution', {})}`",
        f"- `metric_anchor_label_distribution`: "
        f"`{component.get('metric_anchor_label_distribution', {})}`",
        f"- `metric_negative_label_distribution`: "
        f"`{component.get('metric_negative_label_distribution', {})}`",
        "",
        "## Recommendations",
        "",
    ]
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- Feedback flywheel benchmark is ready for the next baseline.")
    lines.append("")
    return "\n".join(lines)
