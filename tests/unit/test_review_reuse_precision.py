"""Precision pass + filename gate tests (honest geometric / rejection reasons)."""

from __future__ import annotations

import json
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


def _line_geom() -> dict:
    return {
        "layers": {"0": {"color": 7, "linetype": "CONTINUOUS"}},
        "entities": [
            {
                "type": "LINE",
                "layer": "0",
                "start": [0.0, 0.0],
                "end": [100.0, 0.0],
            }
        ],
    }


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


def test_pipeline_failed_persists_failed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.review_reuse.models import TaskEventType, TaskStatus

    def _boom(*_a, **_k):
        raise RuntimeError("precision exploded")

    monkeypatch.setattr("src.core.review_reuse.service.apply_precision", _boom)
    svc = _svc()
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="t-fail",
            file_name="a.dxf",
            file_bytes=b"x",
            seed_candidates=[
                {
                    "candidate_id": "c1",
                    "state": "similar",
                    "scores": {"geometric": 0.9, "semantic": 0.8},
                    "methods": ["precision-l4"],
                }
            ],
        )
    assert ei.value.code == "pipeline_failed"
    listed = svc.list_tasks("t-fail")
    assert len(listed) == 1
    task = listed[0]
    assert task.status == TaskStatus.failed
    assert task.error == "precision exploded"
    assert any(e.event_type == TaskEventType.failed for e in task.events)


def test_low_precision_confidence_ignores_visual() -> None:
    from src.core.review_reuse.models import TaskStatus

    svc = _svc()
    task = svc.create_task(
        tenant_id="t-low-prec",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "duplicate",
                "scores": {"geometric": 0.1, "visual": 0.99, "semantic": 0.99},
                "methods": ["precision-l4"],
            }
        ],
    )
    pack = task.evidence_pack or {}
    reasons = task.candidates[0].rejection_reasons
    assert RejectionReason.low_precision_score.value in reasons
    assert pack["confidence"]["score"] == 0.1
    assert pack["confidence"]["band"] == "low"
    assert task.status == TaskStatus.evidence_ready


def test_vision_only_confidence_stays_low() -> None:
    from src.core.review_reuse.models import TaskStatus

    svc = _svc()
    task = svc.create_task(
        tenant_id="t-conf",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "v1",
                "state": "similar",
                "scores": {"semantic": 0.99, "visual": 0.99},
                "methods": ["dedup2d-vision"],
                "decision_source": "dedup2d-vision",
            }
        ],
    )
    pack = task.evidence_pack or {}
    assert pack["confidence"]["band"] == "low"
    assert pack["confidence"]["score"] == 0.0
    assert task.status == TaskStatus.evidence_ready
    assert RejectionReason.vision_only_unverified.value in task.candidates[0].rejection_reasons


def test_adapter_preserves_candidate_geom_json() -> None:
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    assert cands[0].provenance.get("geom_json") == geom


def test_precision_scores_json_query_and_candidate_geom() -> None:
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 1.0
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value not in out[0].rejection_reasons


def test_precision_extracts_dxf_query_geom(monkeypatch: pytest.MonkeyPatch) -> None:
    geom = _line_geom()
    monkeypatch.setattr(
        "src.core.dedupcad_precision.cad_pipeline.extract_geom_json_from_dxf",
        lambda _path: geom,
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="part.dxf", file_bytes=b"0\nSECTION\n")
    assert out[0].scores.get("geometric") == 1.0
    assert "precision-l4" in (out[0].verification.get("methods") or [])


def test_precision_loads_candidate_geom_from_store(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.core.dedupcad_precision.store import (
        GeomJsonStore,
        GeomJsonStoreConfig,
        create_geom_store,
    )

    geom = _line_geom()
    sha = "a" * 64
    store_dir = tmp_path / "geom"
    GeomJsonStore(GeomJsonStoreConfig(base_dir=store_dir)).save(sha, geom)
    monkeypatch.setenv("DEDUPCAD_GEOM_STORE_DIR", str(store_dir))
    create_geom_store.cache_clear()
    try:
        cands = map_raw_hits_to_candidates(
            [
                {
                    "candidate_id": sha,
                    "state": "similar",
                    "methods": ["seed-adapter"],
                }
            ],
            content_sha="ab",
            file_name="a.dxf",
        )
        out = apply_precision(
            cands,
            file_name="query.json",
            file_bytes=json.dumps(geom).encode("utf-8"),
        )
        assert out[0].scores.get("geometric") == 1.0
        assert "precision-l4" in (out[0].verification.get("methods") or [])
    finally:
        create_geom_store.cache_clear()


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
