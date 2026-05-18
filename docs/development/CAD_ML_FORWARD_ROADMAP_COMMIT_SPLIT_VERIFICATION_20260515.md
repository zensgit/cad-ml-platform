# CAD ML Forward Roadmap — Commit Split & Verification

Date: 2026-05-15
Branch: `phase3-vectors-batch-similarity-router-20260429`
Safety tag: `pre-split-backup-20260515` (pre-split HEAD, for full rollback)

## Scope

Codex executed the entire forward roadmap (Phases 1–6) from
`CAD_ML_FORWARD_DEVELOPMENT_DIRECTION_20260512.md` /
`CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md` as one uncommitted blob
(~263 modified + ~209 untracked files). This document records the
verification and the commit-split that turned it into reviewable units.

## Commit split (5 commits, phase-aligned)

| Commit | Phase | Content |
| --- | --- | --- |
| `17a28676` | Phase 1 | vectors helper-ownership closeout: 14 `src/core/vector_*` modules + `vectors_admin_router.py`; `vectors.py` → compatibility facade. + master roadmap docs. |
| `be48a1e7` | Phase 2 | model readiness registry: real `Path.exists` + SHA256 + fallback_mode per model; `/health/model-readiness`. |
| `ea95776c` | Phase 3 | unified forward scorecard + release gate (fail-closed status). |
| `0deb3a9e` | Phase 4 | strict STEP/IGES eval + B-Rep golden manifest validator (50-sample floor). |
| `c67dcbec` | Phase 5+6 | DecisionService contract + knowledge-grounded manufacturing evidence + review tooling. |

### Why 5 commits, not 6

Phases 5 and 6 are one integrated capability — Phase 6 writes into the
Phase 5 `decision_contract.evidence` and the assistant cites it. Three
files interleave 5+6 in the same hunks (`assistant.py`,
`analysis_manufacturing_summary.py`, `batch_analyze_dxf_local.py`).
A standalone Phase-5 commit would be behaviorally incomplete; merging
5+6 avoids fragile hunk surgery.

### Cross-cutting artifacts

- `config/openapi_schema_snapshot.json` — reflects Phase 2 (ModelReadiness),
  Phase 3 (scorecard fields) and Phase 6 (QueryKnowledgeCitation). The
  generator (`scripts/ci/generate_openapi_schema_snapshot.py`) requires the
  FastAPI app import → Python 3.11, not regenerable per-commit on local 3.9.
  Committed once in the final API-surface commit (Phase 5+6). Commits 2–4
  carry a transiently-stale snapshot (acceptable on a feature branch;
  noted in commit messages).
- `.github/workflows/evaluation-report.yml` — orchestrates Phases 3–6;
  cannot precede what it orchestrates. Folded whole into commit 5.
- `reports/benchmark/` — generated artifact, left untracked (NOT committed;
  not gitignored — candidate for a follow-up `.gitignore` entry).

## Verification (static — read actual control flow, not docstrings)

Four parallel deep-reads, every claim SUPPORTED with file:line evidence:

| Claim | Verdict | Key evidence |
| --- | --- | --- |
| Phase 1 contract-preserving facade | SUPPORTED | real delegations to `src/core/vector_*`; OpenAPI diff is +1 legit endpoint, not a silenced refresh |
| Phase 2 static readiness replaced | SUPPORTED | `loader.models_loaded()` consumes registry; `readiness_registry.py` does real `Path.exists`/SHA256 |
| Phase 3 scorecard fail-closes | SUPPORTED | `forward_scorecard.py` downgrades to blocked/shadow_only; release gate exits non-zero |
| Phase 4 synthetic not counted as success | SUPPORTED | strict-mode `synthetic_geometry_not_allowed`; hard 50-sample release floor |
| Phase 5 unified decision contract | SUPPORTED | 9-field contract; analyze/batch/assistant/benchmark all consume it |
| Phase 6 rule source/version + shared evidence | SUPPORTED | `rule_source`/`rule_version` on recommendations; assistant cites same `decision_contract.evidence` |

## Honest status

- **Infrastructure complete and fail-closed** — the guardrails (no
  synthetic-as-success, no fallback-as-release_ready, B-Rep separate from
  2D) are enforced in code, not prose. No overclaiming observed.
- **Release-evidence population is the remaining work** (the unchecked
  `[ ]` TODO items): 50–100 real STEP/IGES files in the B-Rep golden
  manifest; real reviewed source/payload/detail labels; Graph2D/UV-Net/
  PointNet load-failure fixtures. These are human-in-the-loop, not code.
- Accurate line: **release infrastructure done ≠ release ready** — gates
  exist and wait on real data.

## Validation constraint (transparency)

Local default `python3` is **3.9.6**; the codebase pervasively uses
PEP-604 `X | None` annotations on FastAPI route params (a pre-existing,
21+-file project pattern, CI target = **Python 3.11**). The test suite
cannot be collected on 3.9. Each commit was validated by static review +
`py_compile` (syntax gate, valid on 3.9). **Empirical green is CI's
responsibility (3.11)** and was not observed locally. No commit claims
"tests pass".

## Next

1. Push branch → CI (3.11) is the authoritative empirical gate.
2. Address CI feedback per-phase (each commit independently revertable
   via the hashes above; full rollback via tag `pre-split-backup-20260515`).
3. Begin release-evidence population (B-Rep golden set, reviewed labels).
