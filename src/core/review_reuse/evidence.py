"""EvidencePack builder — strategy §3.3 minimum fields (+ optional visual)."""

from __future__ import annotations

from typing import Any, Dict, List

from .canonical import canonical_sha256
from .models import CandidateDecision, ReviewReuseTask


def build_evidence_pack(task: ReviewReuseTask) -> Dict[str, Any]:
    """Assemble a machine-readable EvidencePack for export."""
    candidates: List[Dict[str, Any]] = []
    for c in task.candidates:
        scores = {
            "geometric": c.scores.get("geometric"),
            "semantic": c.scores.get("semantic"),
        }
        # Implementation extension (optional): visual score if present.
        if "visual" in c.scores:
            scores["visual"] = c.scores.get("visual")
        candidates.append(
            {
                "candidate_id": c.candidate_id,
                "candidate_source": c.candidate_source,
                "state": c.state.value,
                "scores": scores,
                "score_normalization": c.scores.get("normalization", "dedup2d-v1"),
                "verification": c.verification,
                "rejection_reasons": list(c.rejection_reasons),
                "provenance": c.provenance,
            }
        )

    pack: Dict[str, Any] = {
        "schema_version": "evidence-pack-v1",
        "task_id": task.task_id,
        "task_revision": task.revision,
        "trace_id": task.trace_id,
        "idempotency_key": task.idempotency_key,
        "source_job_id": task.task_id,
        "tenant_id": task.tenant_id,
        "source": {
            "file_name": task.source_file_name,
            "content_sha256": task.source_content_sha256,
        },
        "candidates": candidates,
        "confidence": {
            "score": _top_confidence(task.candidates),
            "band": _confidence_band(_top_confidence(task.candidates)),
        },
        "calibration": {
            "version": task.calibration_version,
            "status": task.calibration_status,
        },
        "evidence": _evidence_items(task.candidates),
        "rejection_reasons": _aggregate_rejections(task.candidates),
        "unsupported_states": [
            c.candidate_id
            for c in task.candidates
            if c.state.value == "insufficient_evidence"
        ],
        "provenance": {
            "model": "dedup2d-workbench-adapter",
            "ruleset": "review-reuse-mvp-0",
            "dataset": "tenant-archive",
            "input": task.source_content_sha256,
        },
        "human_decision": {
            "state": task.human_decision.state.value if task.human_decision else None,
            "allowed_actions": ["reuse", "revise", "new"],
            "submitted": (
                task.human_decision.model_dump(mode="json")
                if task.human_decision
                else None
            ),
        },
    }
    pack["evidence_pack_sha256"] = evidence_pack_digest(pack)
    return pack


def evidence_pack_digest(pack: Dict[str, Any]) -> str:
    digest_payload = dict(pack)
    digest_payload.pop("evidence_pack_sha256", None)
    return canonical_sha256(digest_payload)


def evidence_pack_digest_is_valid(pack: Dict[str, Any]) -> bool:
    stored = pack.get("evidence_pack_sha256")
    return isinstance(stored, str) and stored == evidence_pack_digest(pack)


def evidence_pack_markdown(pack: Dict[str, Any]) -> str:
    lines = [
        f"# EvidencePack — task `{pack.get('task_id')}`",
        "",
        f"- trace_id: `{pack.get('trace_id')}`",
        f"- task_revision: `{pack.get('task_revision')}`",
        f"- evidence_pack_sha256: `{pack.get('evidence_pack_sha256')}`",
        f"- source: `{pack.get('source', {}).get('file_name')}`",
        f"- content_sha256: `{pack.get('source', {}).get('content_sha256')}`",
        f"- calibration: `{pack.get('calibration', {}).get('version')}`",
        "",
        "## Candidates",
        "",
    ]
    for c in pack.get("candidates") or []:
        lines.append(
            f"- **{c.get('candidate_id')}** ({c.get('state')}) "
            f"geo={c.get('scores', {}).get('geometric')} "
            f"sem={c.get('scores', {}).get('semantic')} "
            f"reasons={c.get('rejection_reasons')}"
        )
    lines.extend(["", "## Human decision", ""])
    hd = pack.get("human_decision") or {}
    lines.append(f"- state: `{hd.get('state')}`")
    lines.append(f"- allowed: {hd.get('allowed_actions')}")
    return "\n".join(lines) + "\n"


def _top_confidence(candidates: List[CandidateDecision]) -> float:
    best = 0.0
    for c in candidates:
        for k in ("geometric", "semantic", "visual", "confidence"):
            v = c.scores.get(k)
            if isinstance(v, (int, float)) and float(v) > best:
                best = float(v)
    return best


def _confidence_band(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _evidence_items(candidates: List[CandidateDecision]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in candidates:
        out.append(
            {
                "candidate_id": c.candidate_id,
                "kind": "match",
                "summary": f"{c.state.value} via {c.verification.get('methods', [])}",
            }
        )
    return out


def _aggregate_rejections(candidates: List[CandidateDecision]) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in candidates:
        for r in c.rejection_reasons:
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out
