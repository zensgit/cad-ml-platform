"""Optional live dedup2d adapter for ReviewReuse recall.

Default: OFF. When disabled or unavailable, callers keep the honest
``insufficient_evidence`` / ``tool_unavailable`` offline path.

Does not call training paths, hosted LLMs, or eval_integrity_gate.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Protocol

from .models import CandidateDecision, CandidateState, RejectionReason

ENV_LIVE_DEDUP = "REVIEW_REUSE_LIVE_DEDUP"
_TRUE = frozenset({"1", "true", "yes", "on"})

# Injectable hook for tests / process wiring (sync callable).
LiveRecallFn = Callable[[str, bytes, str], List[Dict[str, Any]]]
_LIVE_RECALL: Optional[LiveRecallFn] = None


class LiveDedupAdapter(Protocol):
    def recall(
        self, file_name: str, file_bytes: bytes, content_sha: str
    ) -> List[CandidateDecision]:
        ...


def live_dedup_enabled() -> bool:
    return os.getenv(ENV_LIVE_DEDUP, "").strip().lower() in _TRUE


def set_live_recall_hook(fn: Optional[LiveRecallFn]) -> None:
    """Register a process-local live recall function (tests / DI)."""
    global _LIVE_RECALL
    _LIVE_RECALL = fn


def get_live_recall_hook() -> Optional[LiveRecallFn]:
    return _LIVE_RECALL


def map_raw_hits_to_candidates(
    raw_hits: List[Dict[str, Any]],
    *,
    content_sha: str,
    file_name: str,
) -> List[CandidateDecision]:
    """Map adapter/tool hit dicts into CandidateDecision rows."""
    out: List[CandidateDecision] = []
    for raw in raw_hits or []:
        state_s = str(raw.get("state") or raw.get("verdict") or "similar")
        try:
            state = CandidateState(state_s)
        except ValueError:
            # Common tool vocab → strategy states
            low = state_s.lower()
            if low in ("duplicate", "dup", "exact"):
                state = CandidateState.duplicate
            elif low in ("different", "diff", "reject"):
                state = CandidateState.different
            elif low in ("insufficient", "insufficient_evidence", "unknown"):
                state = CandidateState.insufficient_evidence
            else:
                state = CandidateState.similar

        scores = dict(raw.get("scores") or {})
        if "geometric" not in scores and raw.get("geometric") is not None:
            scores["geometric"] = raw.get("geometric")
        if "semantic" not in scores and raw.get("semantic") is not None:
            scores["semantic"] = raw.get("semantic")
        # Ensure strategy-minimum keys exist (nullable).
        scores.setdefault("geometric", raw.get("score"))
        scores.setdefault("semantic", None)

        reasons = list(raw.get("rejection_reasons") or [])
        if state == CandidateState.insufficient_evidence and not reasons:
            reasons = [RejectionReason.tool_unavailable.value]

        verification = dict(
            raw.get("verification")
            or {
                "verdict": state.value,
                "level": raw.get("match_level", raw.get("level", 0)),
                "methods": list(raw.get("methods") or ["dedup2d-live-adapter"]),
            }
        )
        out.append(
            CandidateDecision(
                candidate_id=str(
                    raw.get("candidate_id")
                    or raw.get("drawing_id")
                    or raw.get("id")
                    or f"live-{len(out)}"
                ),
                candidate_source=str(raw.get("candidate_source") or "archive"),
                state=state,
                scores=scores,
                verification=verification,
                rejection_reasons=reasons,
                provenance={
                    "input_sha256": content_sha,
                    "query_file": file_name,
                    "model": raw.get("decision_source") or "dedup2d-live",
                },
            )
        )
    return out


def offline_insufficient(
    *, content_sha: str, file_name: str, reason: str = "tool_unavailable"
) -> List[CandidateDecision]:
    return [
        CandidateDecision(
            candidate_id=f"none-{content_sha[:12]}",
            candidate_source="none",
            state=CandidateState.insufficient_evidence,
            scores={"geometric": None, "semantic": None},
            verification={
                "verdict": "insufficient_evidence",
                "level": 0,
                "methods": [],
            },
            rejection_reasons=[reason],
            provenance={"input_sha256": content_sha, "query_file": file_name},
        )
    ]


def recall_candidates(
    *,
    file_name: str,
    file_bytes: bytes,
    content_sha: str,
    seed: Optional[List[Dict[str, Any]]] = None,
) -> List[CandidateDecision]:
    """Primary entry used by ReviewReuseService.

    Priority:
    1. explicit seed (tests / offline archive fixture)
    2. live hook if REVIEW_REUSE_LIVE_DEDUP enabled
    3. honest offline insufficient_evidence
    """
    if seed:
        return map_raw_hits_to_candidates(seed, content_sha=content_sha, file_name=file_name)

    if live_dedup_enabled():
        hook = get_live_recall_hook()
        if hook is None:
            return offline_insufficient(
                content_sha=content_sha,
                file_name=file_name,
                reason=RejectionReason.tool_unavailable.value,
            )
        try:
            hits = hook(file_name, file_bytes, content_sha)
            mapped = map_raw_hits_to_candidates(
                list(hits or []), content_sha=content_sha, file_name=file_name
            )
            if mapped:
                return mapped
            return offline_insufficient(
                content_sha=content_sha,
                file_name=file_name,
                reason=RejectionReason.tool_unavailable.value,
            )
        except Exception:
            return offline_insufficient(
                content_sha=content_sha,
                file_name=file_name,
                reason=RejectionReason.external_service_unavailable.value,
            )

    return offline_insufficient(
        content_sha=content_sha,
        file_name=file_name,
        reason=RejectionReason.tool_unavailable.value,
    )
