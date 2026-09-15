"""Precision pass + filename gate tests (honest geometric / rejection reasons)."""

from __future__ import annotations

from typing import List

import pytest

from src.core.review_reuse.dedup_adapter import map_raw_hits_to_candidates
from src.core.review_reuse.files import is_allowed_review_reuse_filename
from src.core.review_reuse.models import (
    CandidateDecision,
    CandidateState,
    HumanDecisionState,
    RejectionReason,
    TaskEventType,
)
from src.core.review_reuse.precision import apply_precision, set_precision_hook
from src.core.review_reuse.service import (
    ENV_DECISIONS_ENABLED,
    ReviewReuseError,
    ReviewReuseService,
)
from src.core.review_reuse.store import ReviewReuseStore


def _svc() -> ReviewReuseService:
    return ReviewReuseService(ReviewReuseStore())


def test_filename_gate() -> None:
    assert is_allowed_review_reuse_filename("part.dxf")
    assert is_allowed_review_reuse_filename("PART.DWG")
    assert is_allowed_review_reuse_filename("scan.PNG")
    assert not is_allowed_review_reuse_filename("payload.exe")
    assert not is_allowed_review_reuse_filename("upload.bin")
    assert not is_allowed_review_reuse_filename("")


def test_create_rejects_unsupported_file_type() -> None:
    svc = _svc()
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="t-a",
            file_name="payload.exe",
            file_bytes=b"MZ",
        )
    assert ei.value.code == RejectionReason.unsupported_file_type.value


def test_precision_labels_vision_only_without_copying_visual() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "v1",
                "state": "similar",
                "scores": {"semantic": 0.9, "visual": 0.9},
                "methods": ["dedup2d-vision"],
                "decision_source": "dedup2d-vision",
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    assert cands[0].scores.get("geometric") is None
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"not-json")
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.vision_only_unverified.value in out[0].rejection_reasons


def test_precision_missing_geom_json_for_non_vision() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "s1",
                "state": "similar",
                "candidate_source": "archive",
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"dxf")
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_precision_keeps_seeded_l4_geometric() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-001",
                "state": "similar",
                "scores": {"geometric": 0.91, "semantic": 0.72},
                "methods": ["dedup2d-adapter", "precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"dxf")
    assert out[0].scores.get("geometric") == 0.91
    assert RejectionReason.vision_only_unverified.value not in out[0].rejection_reasons


def test_precision_hook_runs_in_create_task() -> None:
    def _hook(
        file_name: str, file_bytes: bytes, cands: List[CandidateDecision]
    ) -> List[CandidateDecision]:
        del file_name, file_bytes
        out = []
        for c in cands:
            copied = c.model_copy(deep=True)
            copied.scores = dict(copied.scores)
            copied.scores["geometric"] = 0.77
            copied.verification = {
                "verdict": copied.state.value,
                "level": 4,
                "methods": ["precision-l4"],
            }
            out.append(copied)
        return out

    set_precision_hook(_hook)
    try:
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-p",
            file_name="a.dxf",
            file_bytes=b"x",
            seed_candidates=[
                {
                    "candidate_id": "h1",
                    "state": "similar",
                    "scores": {"semantic": 0.5},
                    "methods": ["dedup2d-vision"],
                    "decision_source": "dedup2d-vision",
                }
            ],
        )
        assert task.candidates[0].scores.get("geometric") == 0.77
        prec = [
            e
            for e in task.events
            if e.event_type == TaskEventType.precision_completed
        ]
        assert prec
        assert prec[0].detail.get("precision_l4") == 1
    finally:
        set_precision_hook(None)


def test_unknown_candidate_on_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
    svc = _svc()
    task = svc.create_task(
        tenant_id="t-a",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "arch-001",
                "state": "similar",
                "scores": {"geometric": 0.91, "semantic": 0.7},
                "methods": ["precision-l4"],
            }
        ],
    )
    with pytest.raises(ReviewReuseError) as ei:
        svc.submit_decision(
            tenant_id="t-a",
            task_id=task.task_id,
            state=HumanDecisionState.reuse,
            reviewer_id="r1",
            candidate_id="not-on-task",
        )
    assert ei.value.code == "unknown_candidate"


def test_low_precision_reason() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "low",
                "state": "similar",
                "scores": {"geometric": 0.2, "semantic": 0.1},
                "methods": ["precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"x")
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
