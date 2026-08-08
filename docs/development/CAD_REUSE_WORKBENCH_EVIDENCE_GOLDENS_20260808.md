# ReviewReuse EvidencePack golden fixtures

**Date**: 2026-08-08  
**Scope**: docs + tests/fixtures only (no production service behavior change)  
**Contract**: `docs/PRODUCT_STRATEGY.md` §3.3; `docs/development/L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md` §2.4  
**Builder**: `src/core/review_reuse/evidence.py` → `build_evidence_pack`

## What lives where

| Path | Role |
|------|------|
| `tests/golden/review_reuse/evidence_pack_duplicate.json` | High-confidence **duplicate** EvidencePack |
| `tests/golden/review_reuse/evidence_pack_similar.json` | **similar** candidate EvidencePack |
| `tests/golden/review_reuse/evidence_pack_insufficient_evidence.json` | Honest offline **insufficient_evidence** + `tool_unavailable` |
| `tests/unit/test_review_reuse_evidence_goldens.py` | Loads goldens, asserts §3.3 keys, rebuilds via `build_evidence_pack` |

Each golden file wraps the machine-readable pack:

```json
{
  "case_id": "duplicate|similar|insufficient_evidence",
  "description": "...",
  "product_strategy_section": "3.3",
  "schema_version": "evidence-pack-v1",
  "expected_candidate_state": "<same as case_id>",
  "evidence_pack": { /* build_evidence_pack output */ }
}
```

Field families under `evidence_pack` match strategy §3.3 (candidate id/source, geometric+semantic scores, verification, confidence/calibration, evidence/rejection, provenance, human_decision, trace/idempotency).

## Run the unit test

```bash
# from repo root, with project venv if present
pytest tests/unit/test_review_reuse_evidence_goldens.py -v
```

No network, live dedup, decision enablement, `eval_integrity_gate`, `cost_cap`, or training JSONL is required.

## How to refresh goldens

Refresh only when `build_evidence_pack` intentionally changes its stable contract (or a golden case is wrong). Prefer regenerating from the builder so tests remain a pure round-trip.

### Option A — rebuild from existing golden (recommended)

1. Load the golden JSON.
2. Reconstruct a `ReviewReuseTask` + `CandidateDecision` rows from `evidence_pack` (same logic as `_task_from_golden_pack` in the unit test).
3. Call `build_evidence_pack(task)`.
4. Write the new pack back under `evidence_pack` (keep wrapper metadata).
5. Re-run the unit test; fix any intentional contract deltas in both golden and test expectations.

Minimal sketch:

```python
from pathlib import Path
import json
from src.core.review_reuse.evidence import build_evidence_pack
# reuse _task_from_golden_pack from the unit test or inline the same reconstruction
path = Path("tests/golden/review_reuse/evidence_pack_similar.json")
payload = json.loads(path.read_text())
task = _task_from_golden_pack(payload["evidence_pack"])  # import or copy helper
payload["evidence_pack"] = build_evidence_pack(task)
path.write_text(json.dumps(payload, indent=2) + "\n")
```

### Option B — new case from fixed domain models

Construct `ReviewReuseTask` with **stable** `task_id` / `trace_id` / `idempotency_key` / content sha (do not use random UUIDs), set candidates, call `build_evidence_pack`, wrap as above, add the file name to `EXPECTED_CASE_FILES` in the unit test.

### Do not

- Commit customer drawings or real archive hashes.
- Pipe decision ledger rows into training-readable JSONL (R2 HOLD).
- Treat goldens as a retrain or `eval_integrity_gate` unlock.
- Change production routes/service flags in a fixtures-only PR.

## R2 / product boundaries

- Decision sink remains default-off (`REVIEW_REUSE_DECISIONS_ENABLED`).
- This tranche does not open a second L3 runtime feature PR.
- Goldens are evaluation-style contract anchors for the workbench EvidencePack export surface only.
