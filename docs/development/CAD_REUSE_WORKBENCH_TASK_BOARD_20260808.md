# CAD Reuse Workbench — Task Board

**Date**: 2026-08-08 (board refresh 2026-08-29)  
**Plan**: `CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md`  
**System design**: `CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md`  
**Main baseline**: post-#565 (`origin/main`)  

| Residual class | Status |
|---|---|
| **residual_eng open** | **none** (optional future only — not open codeable DAG) |
| **residual_human** | R11 design-lock ratify · R12 decision enable · Track C C1–C5 |
| **Holds (explicit)** | **R2 HOLD** (no training-path reuse) · **decision default-off** (`REVIEW_REUSE_DECISIONS_ENABLED` unset → 403) |

Legend: **done** · **in_progress** · **residual_eng** · **residual_human** · **blocked** · **optional_future**

---

## Merged PR rollup (Track R + follow-ons)

| PR | Title / scope | Status |
|---|---|---|
| [#547](https://github.com/zensgit/cad-ml-platform/pull/547) | ReviewReuse Workbench **MVP** (domain + API + default-off decision + live adapter stub + metrics) | **MERGED** |
| [#548](https://github.com/zensgit/cad-ml-platform/pull/548) | Board **closeout** after #547 (clear residual_eng for MVP slot) | **MERGED** |
| [#549](https://github.com/zensgit/cad-ml-platform/pull/549) | **R2 HOLD** structural tests + external-gates audit (no false complete) | **MERGED** |
| [#550](https://github.com/zensgit/cad-ml-platform/pull/550) | **Live dedup** vision wiring (default-off) + **filesystem** durable task store | **MERGED** |
| [#551](https://github.com/zensgit/cad-ml-platform/pull/551) | **Audit export**, validated **reviewer** gate, isolated-archive **script** | **MERGED** |
| [#552](https://github.com/zensgit/cad-ml-platform/pull/552) | **Pilot operator checklist** (docs only; Track C not claimed) | **MERGED** |
| [#553](https://github.com/zensgit/cad-ml-platform/pull/553) | EvidencePack **golden fixtures** (duplicate / similar / insufficient) | **MERGED** |
| [#554](https://github.com/zensgit/cad-ml-platform/pull/554) | **Audit-export contract** tests (bundle fields, tenant isolation, R2 HOLD no feedback write) | **MERGED** |
| [#555](https://github.com/zensgit/cad-ml-platform/pull/555) | Board post-#547–#553 + **`make test-review-reuse`** | **MERGED** |
| [#556](https://github.com/zensgit/cad-ml-platform/pull/556) | Review-reuse **metrics markdown** (`format_metrics_markdown` + `?format=markdown`) | **MERGED** |
| [#557](https://github.com/zensgit/cad-ml-platform/pull/557) | **`make review-reuse-isolated-archive`** + pilot checklist pointer | **MERGED** |
| [#558](https://github.com/zensgit/cad-ml-platform/pull/558) | **Validated reviewer** decision-gate API tests (`reviewer_not_validated` 403 + JWT subject path) | **MERGED** |
| [#559](https://github.com/zensgit/cad-ml-platform/pull/559) | Evidence-pack / metrics **markdown API** coverage | **MERGED** |
| [#560](https://github.com/zensgit/cad-ml-platform/pull/560) | Board post-#554–#558 (residual_human open) | **MERGED** |
| [#561](https://github.com/zensgit/cad-ml-platform/pull/561) | Isolated-archive script **CLI** coverage (seed, offline, decisions off) | **MERGED** |
| [#562](https://github.com/zensgit/cad-ml-platform/pull/562) | **JWT pilot runbook** + filesystem **store backup/cleanup** ops | **MERGED** |
| [#563](https://github.com/zensgit/cad-ml-platform/pull/563) | Audit export **CLI** (`audit_bundle.json` + `evidence.md` by `task_id`; R2 quarantine) | **MERGED** |
| [#564](https://github.com/zensgit/cad-ml-platform/pull/564) | Store ops **`list`** tenants (task count + `age_days`; `make review-reuse-store-list`) | **MERGED** |
| [#565](https://github.com/zensgit/cad-ml-platform/pull/565) | Board post-#562 + pilot env **preflight** (`make review-reuse-preflight`) | **MERGED** |

No further L3 runtime PR is required under the 90-day codeable scope unless the owner opens a new design-lock.  
Track C / R11 / R12 remain **residual_human** — do not claim complete. **R2 HOLD** unchanged.

---

## Track R — ReviewReuse Workbench (P0)

| ID | Task | Status | Workflow / PR | Evidence |
|---|---|---|---|---|
| R1 | L3 design-lock §3.3 + default-off decision | done (PROPOSED) | #547 | `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md` |
| R2 | Domain service Task/Event/EvidencePack | done | #547 | `src/core/review_reuse/` |
| R3 | API `/api/v1/review-reuse/*` mounted | done | #547 · #551 | `src/api/v1/review_reuse.py` |
| R4 | Decision sink **default-off** | done | #547 · #558 | `REVIEW_REUSE_DECISIONS_ENABLED` unset → 403; validated-reviewer gate tested when on |
| R5 | Tenant isolation + quarantine (no feedback JSONL) | done | #547 · #549 · #554 | unit+API + R2 HOLD + audit-export contract |
| R6 | Unit/API tests driving shipped code | done | #547–#565 | `tests/unit/test_review_reuse*.py` · `make test-review-reuse` |
| R7 | Isolated-sample runbook + archive script + Make target | done | #547 · #551 · #557 · #561 | runbook + script + `make review-reuse-isolated-archive` |
| R8 | Plan + verification MD | done | #547 | MVP plan/verification MDs |
| R9 | OpenAPI snapshot for new routes | done | #547 · #551 | `config/openapi_schema_snapshot.json` |
| R10 | Live dedup2d adapter + durable store | done | #547 · #550 | default-off live hook; memory/filesystem store |
| R11 | Owner ratify design-lock | **residual_human** | — | owner only |
| R12 | Pilot enable decisions | **residual_human** | — | env flag; **never** self-enable in production |

### R2 HOLD / decision default-off (do not weaken)

| Hold | Rule | Code posture |
|---|---|---|
| **R2 HOLD** | Decision / correction evidence must **not** enter training-readable manifests; `feedback.py` JSONL is not the ledger | No feedback imports; audit export marked `audit_quarantine`; structural tests in `test_review_reuse_r2_hold.py` + audit-export contract (#554) |
| **Decision default-off** | Decisions stay off unless owner enables for a named pilot window | `REVIEW_REUSE_DECISIONS_ENABLED` unset/false → decision POST 403 |
| **Validated reviewer** (when decisions on) | Optional pilot gate: API-key-only subject rejected | `REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER` + API tests (#558); still not a training path |
| Related | No `eval_integrity_gate` replace · no Track E model-release metrics · no `cost_cap` revive · AI has no release authority | Workbench metrics = `review_workflow` only |

---

## Track E — Evaluation Integrity (P0)

| ID | Task | Status | Workflow / PR | Evidence |
|---|---|---|---|---|
| E1 | Preserve E1 dry-run machinery (#542) | done (landed) | audit residual | `scripts/track_e_*` |
| E2 | Post-merge E1 audit at exact head | done | #547 / execute-plan PR 1 | `TRACK_E_E1_POST_MERGE_AUDIT_20260808.md` |
| E3 | No eval_integrity_gate replace | done (boundary) | workflows boundary | no path in workbench PRs |

## Track S — Safety Baselines (P0)

| ID | Task | Status | Workflow / PR | Evidence |
|---|---|---|---|---|
| S1 | Preserve SEAL (#537) | done (landed) | audit residual | assistant opt-in |
| S2 | Preserve identity fail-closed (#538) | done (landed) | audit residual | production_identity |
| S3 | SEAL+identity re-audit + no cost_cap | done | #547 / execute-plan PR 1 | `TRACK_S_SEAL_IDENTITY_BASELINE_AUDIT_20260808.md` |

## Track O — Ops & Evidence (P1)

| ID | Task | Status | Workflow / PR | Evidence |
|---|---|---|---|---|
| O1 | Isolated sample checklist | done | #547 · #552 · #557 | runbook + pilot checklist + Make isolated-archive |
| O2 | Pilot ops package (kill/rollback/export/retention) | done | #547 · #551 · #552 · #554 · #562 · #563 · #564 | Track O + audit export API/CLI + JWT pilot runbook + store backup/cleanup/list |
| O3 | Workbench review metrics export (+ markdown) | done | #547 · #556 · #559 | `metrics.py` + `GET .../metrics` (`json` \| `markdown`; `review_workflow` family) |
| O4 | Kill switch documented | done | runbook §6 + Track O + pilot checklist | done |
| O5 | Live dedup (default-off) + filesystem store docs | done | #550 | `CAD_REUSE_WORKBENCH_LIVE_DEDUP_DURABLE_STORE_20260808.md` |
| O6 | EvidencePack golden fixtures | done | #553 | `tests/golden/review_reuse/` + `CAD_REUSE_WORKBENCH_EVIDENCE_GOLDENS_20260808.md` |
| O7 | Operator Make targets (test + isolated-archive + store + export + preflight) | done | #555 · #557 · #562 · #563 · #564 · #565 | `make test-review-reuse` · `make review-reuse-isolated-archive` · store backup/cleanup/list · `make review-reuse-export-audit` · `make review-reuse-preflight` |
| O8 | JWT pilot runbook + store backup/cleanup/list | done | #562 · #564 | `CAD_REUSE_WORKBENCH_JWT_PILOT_RUNBOOK_20260808.md` · `scripts/review_reuse_store_ops.py` |
| O9 | Pilot env preflight script (advisory; dangerous-combo exit 2) | done | #565 | `scripts/review_reuse_pilot_preflight.py` · `make review-reuse-preflight` |
| O10 | Audit export CLI by `task_id` | done | #563 | `scripts/review_reuse_export_audit.py` · `make review-reuse-export-audit` |

## Track C — Customer Pilot (P1) — **human residual** (not claimed complete)

| ID | Task | Status | Owner | Notes |
|---|---|---|---|---|
| C1 | 10 qualified manufacturer contacts | **residual_human** | Owner | may roll Day 31–60 |
| C2 | 2 lawful sample-data conversations | **residual_human** | Owner | |
| C3 | Named reviewer + baseline metric | **residual_human** | Owner | |
| C4 | Isolated customer archive run | **residual_human** | Owner + eng support | Day 61–90 |
| C5 | Measured pilot + commercial next step | **residual_human** | Owner | Day 90 gate |

Do **not** invent Track C completion evidence in docs or code.

## System / orchestration

| ID | Task | Status | Notes |
|---|---|---|---|
| SYS1 | `cad-reuse-workbench-dev` workflow | done | gap/implement/verify/full |
| SYS2 | `cad-reuse-workbench-90d` workflow | done | multi-track audit |
| SYS3 | `cad-reuse-workbench-system` master workflow | done | `.grok/workflows/cad-reuse-workbench-system.rhai` |
| SYS4 | System execute design + PR Plan DAG | done | `CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md` |
| SYS5 | residual PR1+PR2 (audits + ops) content | done | landed on #547 docs tranche |
| SYS6 | PR3+PR4 content folded into #547 | done | same L3 slot — no second runtime PR |
| SYS7 | Closeout board flip | done | #548 merged |
| SYS8 | R2 HOLD structural tests + external gates audit | done | #549 — `test_review_reuse_r2_hold.py` + EXTERNAL_GATES audit |
| SYS9 | Live dedup + durable store follow-on | done | #550 |
| SYS10 | Audit export / reviewer / archive script | done | #551 |
| SYS11 | Pilot operator checklist | done | #552 (docs; Track C still open) |
| SYS12 | EvidencePack goldens | done | #553 |
| SYS13 | Audit-export contract tests | done | #554 |
| SYS14 | Board refresh + `make test-review-reuse` | done | #555 |
| SYS15 | Metrics markdown report | done | #556 |
| SYS16 | `make review-reuse-isolated-archive` | done | #557 |
| SYS17 | Validated reviewer API tests | done | #558 |
| SYS18 | Evidence/metrics markdown API tests | done | #559 |
| SYS19 | Board post-#554–#558 | done | #560 |
| SYS20 | Isolated-archive CLI tests | done | #561 |
| SYS21 | JWT pilot runbook + store backup/cleanup | done | #562 |
| SYS22 | Board post-#562 + pilot preflight script | done | #565 |
| SYS23 | Audit export CLI by task_id | done | #563 |
| SYS24 | Store ops list tenants | done | #564 |
| SYS25 | Board post-#563–#565 | done (this PR) | task board rollup; residual_human still open |

---

## residual_eng

**Open: none.**

Optional future (not open DAG / not blocking residual_eng):

| Item | Notes |
|---|---|
| Redis multi-node task store | Beyond single-node filesystem pilot store |
| JWT-first reviewer identity always-on | Pilot can require validated reviewer via env; production identity productization is owner-driven |
| PLM write-back | Explicitly out of 90-day workbench scope |
| Decision default-on | **Forbidden** without owner R12 pilot enable + named window |

## residual_human (owner)

1. **R11** — Ratify L3 design-lock (`L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md`).
2. **R12** — Explicit pilot enable of decisions (`REVIEW_REUSE_DECISIONS_ENABLED`) for a named window only.
3. **Track C C1–C5** — Contacts, sample conversations, named reviewer, isolated customer archive, measured pilot / commercial next step.

External audit residual (not eng-closed): Evaluation Hybrid superpass red on main (eval owners; not Track R).

---

## Execution order (post-#547…#565)

1. ~~Land **#547** Track R MVP~~ — **MERGED**.
2. ~~execute-plan PR1–PR4 content~~ — folded into #547 (L3 WIP=1).
3. ~~Closeout board flip~~ — **#548 MERGED**.
4. ~~R2 HOLD regression tests + external-gates audit~~ — **#549 MERGED**.
5. ~~Live dedup + filesystem durable store~~ — **#550 MERGED**.
6. ~~Audit export + validated reviewer + archive script~~ — **#551 MERGED**.
7. ~~Pilot operator checklist~~ — **#552 MERGED**.
8. ~~EvidencePack golden fixtures~~ — **#553 MERGED**.
9. ~~Audit-export contract tests~~ — **#554 MERGED**.
10. ~~Board + `make test-review-reuse`~~ — **#555 MERGED**.
11. ~~Metrics markdown report~~ — **#556 MERGED**.
12. ~~`make review-reuse-isolated-archive`~~ — **#557 MERGED**.
13. ~~Validated reviewer API tests~~ — **#558 MERGED**.
14. ~~Evidence/metrics markdown API tests~~ — **#559 MERGED**.
15. ~~Board post-#554–#558~~ — **#560 MERGED**.
16. ~~Isolated-archive CLI tests~~ — **#561 MERGED**.
17. ~~JWT pilot runbook + store backup/cleanup~~ — **#562 MERGED**.
18. ~~Audit export CLI by task_id~~ — **#563 MERGED**.
19. ~~Store ops list tenants~~ — **#564 MERGED**.
20. ~~Board post-#562 + pilot preflight script~~ — **#565 MERGED**.
21. ~~Board post-#563–#565~~ — **this PR**.
22. **Owner only:** R11 design-lock ratify · R12 pilot decision enable · Track C C1–C5.
23. **External only (audit, not eng-closed):** Evaluation Hybrid superpass red on main.
24. **Boundaries (unchanged):** **R2 HOLD** · no eval_integrity_gate replace · no cost_cap · **decision default-off** · no fake Track C · no production self-enable of decisions.

---

## Local eng regression (ReviewReuse)

```bash
# Preferred Make target — workbench unit tests + EvidencePack goldens
# (docs/Makefile tranche — not part of validate-core-fast)
make test-review-reuse

# Equivalent pytest (glob covers all review_reuse unit modules including goldens)
pytest tests/unit/test_review_reuse*.py -q

# Offline isolated-archive demo (seed-similar; decisions stay disabled)
# Does NOT set REVIEW_REUSE_DECISIONS_ENABLED
make review-reuse-isolated-archive

# Advisory pilot env preflight (does NOT enable decisions; exit 2 on dangerous combos)
make review-reuse-preflight

# Filesystem store backup / cleanup dry-run / list (#562 · #564)
make review-reuse-store-backup
make review-reuse-store-cleanup-dry
make review-reuse-store-list

# Audit export CLI by TENANT + TASK_ID (#563; R2 quarantine; decisions stay off)
# make review-reuse-export-audit TENANT=<tenant> TASK_ID=<uuid> OUT=/tmp/rr_audit
```

Covered modules on main: `test_review_reuse_workbench`, `test_review_reuse_api`, `test_review_reuse_r2_hold`, `test_review_reuse_live_store`, `test_review_reuse_audit_reviewer`, `test_review_reuse_audit_export_contract`, `test_review_reuse_evidence_goldens`, `test_review_reuse_store_ops`, `test_review_reuse_pilot_preflight`, `test_review_reuse_export_audit_script`.

See also: `CAD_REUSE_WORKBENCH_EVIDENCE_GOLDENS_20260808.md`, `CAD_REUSE_WORKBENCH_PILOT_CHECKLIST_20260808.md`, `CAD_REUSE_WORKBENCH_JWT_PILOT_RUNBOOK_20260808.md`, `CAD_REUSE_WORKBENCH_EXTERNAL_GATES_AUDIT_20260808.md`.

---

## Workflow commands

```text
/workflow cad-reuse-workbench-dev     args: {"mode":"verify","root":"."}
/workflow cad-reuse-workbench-dev     args: {"mode":"full","root":"."}
/workflow cad-reuse-workbench-90d     args: {"track":"full","root":"."}
/workflow cad-reuse-workbench-system  args: {"phase":"all","root":"."}

/execute-plan docs/development/CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md --no-graphite --auto-pr --concurrency 2
```

Do **not** run implement agents that invent Track C evidence or flip decision default-on.
