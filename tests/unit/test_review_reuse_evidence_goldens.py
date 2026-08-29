"""Golden EvidencePack fixtures for ReviewReuse workbench (strategy §3.3).

Docs-only / fixture tranche: no production service behavior change.
Loads static JSON goldens under tests/golden/review_reuse/ and validates
required field families. Optionally rebuilds packs via build_evidence_pack
from reconstructed domain models (no network, no live dedup).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.core.review_reuse.evidence import build_evidence_pack
from src.core.review_reuse.models import (
    CandidateDecision,
    CandidateState,
    ReviewReuseTask,
    TaskStatus,
)

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "review_reuse"

# PRODUCT_STRATEGY §3.3 / design-lock EvidencePack minimum field families.
REQUIRED_TOP_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "task_revision",
        "evidence_pack_sha256",
        "trace_id",
        "idempotency_key",
        "source_job_id",
        "candidates",
        "confidence",
        "calibration",
        "evidence",
        "rejection_reasons",
        "unsupported_states",
        "provenance",
        "human_decision",
    }
)

REQUIRED_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "candidate_source",
        "state",
        "scores",
        "score_normalization",
        "verification",
        "rejection_reasons",
        "provenance",
    }
)

REQUIRED_SCORE_KEYS = frozenset({"geometric", "semantic"})
REQUIRED_VERIFICATION_KEYS = frozenset({"verdict", "level", "methods"})
REQUIRED_CONFIDENCE_KEYS = frozenset({"score", "band"})
REQUIRED_CALIBRATION_KEYS = frozenset({"version", "status"})
REQUIRED_PROVENANCE_KEYS = frozenset({"model", "ruleset", "dataset", "input"})
REQUIRED_HUMAN_DECISION_KEYS = frozenset({"state", "allowed_actions"})

EXPECTED_CASE_FILES = {
    "duplicate": "evidence_pack_duplicate.json",
    "similar": "evidence_pack_similar.json",
    "insufficient_evidence": "evidence_pack_insufficient_evidence.json",
}


def _load_golden(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} must be a JSON object"
    return data


def _all_golden_paths() -> List[Path]:
    paths = sorted(GOLDEN_DIR.glob("evidence_pack_*.json"))
    assert paths, f"no golden fixtures found under {GOLDEN_DIR}"
    return paths


def _assert_section_33_fields(pack: Dict[str, Any], *, label: str) -> None:
    missing = REQUIRED_TOP_KEYS - set(pack.keys())
    assert not missing, f"{label}: missing top-level keys {sorted(missing)}"

    assert pack["schema_version"] == "evidence-pack-v1", label
    assert (
        isinstance(pack["candidates"], list) and pack["candidates"]
    ), f"{label}: candidates must be a non-empty list"

    conf = pack["confidence"]
    assert REQUIRED_CONFIDENCE_KEYS <= set(conf.keys()), f"{label}: confidence"
    cal = pack["calibration"]
    assert REQUIRED_CALIBRATION_KEYS <= set(cal.keys()), f"{label}: calibration"
    prov = pack["provenance"]
    assert REQUIRED_PROVENANCE_KEYS <= set(prov.keys()), f"{label}: provenance"
    hd = pack["human_decision"]
    assert REQUIRED_HUMAN_DECISION_KEYS <= set(hd.keys()), f"{label}: human_decision"
    assert hd["allowed_actions"] == [
        "reuse",
        "revise",
        "new",
    ], f"{label}: strategy-center allowed_actions"

    for i, c in enumerate(pack["candidates"]):
        clabel = f"{label}.candidates[{i}]"
        missing_c = REQUIRED_CANDIDATE_KEYS - set(c.keys())
        assert not missing_c, f"{clabel}: missing {sorted(missing_c)}"
        assert REQUIRED_SCORE_KEYS <= set(c["scores"].keys()), clabel
        assert REQUIRED_VERIFICATION_KEYS <= set(c["verification"].keys()), clabel
        assert isinstance(c["verification"]["methods"], list), clabel


def _task_from_golden_pack(pack: Dict[str, Any]) -> ReviewReuseTask:
    """Reconstruct a ReviewReuseTask so build_evidence_pack can be re-run offline."""
    candidates: List[CandidateDecision] = []
    for raw in pack["candidates"]:
        candidates.append(
            CandidateDecision(
                candidate_id=raw["candidate_id"],
                candidate_source=raw.get("candidate_source", "archive"),
                state=CandidateState(raw["state"]),
                scores={
                    "geometric": raw["scores"].get("geometric"),
                    "semantic": raw["scores"].get("semantic"),
                },
                verification=dict(raw.get("verification") or {}),
                rejection_reasons=list(raw.get("rejection_reasons") or []),
                provenance=dict(raw.get("provenance") or {}),
            )
        )
    source = pack.get("source") or {}
    return ReviewReuseTask(
        task_id=pack["task_id"],
        tenant_id=pack.get("tenant_id") or "tenant-golden-fixtures",
        status=TaskStatus.evidence_ready,
        created_at=1754600000.0,
        updated_at=1754600001.0,
        source_file_name=source.get("file_name") or "",
        source_content_sha256=source.get("content_sha256") or "",
        idempotency_key=pack.get("idempotency_key"),
        revision=pack.get("task_revision") or 1,
        trace_id=pack["trace_id"],
        candidates=candidates,
        calibration_version=(pack.get("calibration") or {}).get("version")
        or "workbench-mvp-0",
        calibration_status=(pack.get("calibration") or {}).get("status")
        or "uncalibrated",
    )


class TestEvidencePackGoldenFixtures:
    def test_expected_case_files_present(self) -> None:
        for case_id, filename in EXPECTED_CASE_FILES.items():
            path = GOLDEN_DIR / filename
            assert path.is_file(), f"missing golden for {case_id}: {path}"

    @pytest.mark.parametrize(
        "case_id,filename",
        sorted(EXPECTED_CASE_FILES.items()),
        ids=sorted(EXPECTED_CASE_FILES.keys()),
    )
    def test_golden_has_section_33_keys(self, case_id: str, filename: str) -> None:
        payload = _load_golden(GOLDEN_DIR / filename)
        assert payload["case_id"] == case_id
        assert payload["expected_candidate_state"] == case_id
        assert payload["product_strategy_section"] == "3.3"
        pack = payload["evidence_pack"]
        _assert_section_33_fields(pack, label=case_id)
        assert pack["candidates"][0]["state"] == case_id

    @pytest.mark.parametrize(
        "case_id,filename",
        sorted(EXPECTED_CASE_FILES.items()),
        ids=sorted(EXPECTED_CASE_FILES.keys()),
    )
    def test_rebuild_via_build_evidence_pack_matches_golden(
        self, case_id: str, filename: str
    ) -> None:
        """Round-trip: golden → domain task → build_evidence_pack equals golden pack."""
        payload = _load_golden(GOLDEN_DIR / filename)
        expected = payload["evidence_pack"]
        task = _task_from_golden_pack(expected)
        rebuilt = build_evidence_pack(task)
        assert (
            rebuilt == expected
        ), f"{case_id}: build_evidence_pack output drifted from golden fixture"

    def test_all_goldens_load_without_network(self) -> None:
        """Smoke: every evidence_pack_*.json is valid JSON with §3.3 keys."""
        for path in _all_golden_paths():
            payload = _load_golden(path)
            assert "evidence_pack" in payload, path.name
            _assert_section_33_fields(payload["evidence_pack"], label=path.name)

    def test_insufficient_evidence_has_rejection_reason(self) -> None:
        payload = _load_golden(
            GOLDEN_DIR / EXPECTED_CASE_FILES["insufficient_evidence"]
        )
        pack = payload["evidence_pack"]
        assert "tool_unavailable" in pack["rejection_reasons"]
        assert pack["unsupported_states"], "insufficient_evidence ids must be listed"
        assert pack["confidence"]["band"] == "low"

    def test_duplicate_and_similar_have_scores(self) -> None:
        for case_id in ("duplicate", "similar"):
            payload = _load_golden(GOLDEN_DIR / EXPECTED_CASE_FILES[case_id])
            c0 = payload["evidence_pack"]["candidates"][0]
            assert isinstance(c0["scores"]["geometric"], (int, float))
            assert isinstance(c0["scores"]["semantic"], (int, float))
            assert not c0["rejection_reasons"]
