# CAD Reuse Workbench — Task Board

**Date**: 2026-08-08  
**Plan**: `CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md`  
**System design**: `CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md`  
**Track R MVP**: [#547](https://github.com/zensgit/cad-ml-platform/pull/547) **MERGED** to `main` (`db437b8b`, 2026-08-07)  
**Eng residual_eng open**: **none** (R10 live dedup + O3 metrics landed in #547)  
**Human residual**: R11 design-lock ratify · R12 pilot decision enable · Track C C1–C5  

Legend: **done** · **in_progress** · **residual_eng** · **residual_human** · **blocked**

---

## Track R — ReviewReuse Workbench (P0)

| ID | Task | Status | Workflow / execute-plan | Evidence |
|---|---|---|---|---|
| R1 | L3 design-lock §3.3 + default-off decision | done (PROPOSED) | #547 merged | `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md` |
| R2 | Domain service Task/Event/EvidencePack | done | #547 merged | `src/core/review_reuse/` |
| R3 | API `/api/v1/review-reuse/*` mounted | done | #547 merged | `src/api/v1/review_reuse.py` |
| R4 | Decision sink default-off | done | #547 merged | `REVIEW_REUSE_DECISIONS_ENABLED` |
| R5 | Tenant isolation + quarantine (no feedback JSONL) | done | #547 merged | unit+API tests |
| R6 | Unit/API tests driving shipped code | done | #547 merged | 16+ tests green |
| R7 | Isolated-sample runbook | done | #547 merged | `ISOLATED_SAMPLE_ARCHIVE_RUNBOOK_20260808.md` |
| R8 | Plan + verification MD | done | #547 merged | MVP plan/verification MDs |
| R9 | OpenAPI snapshot for new routes | done | #547 merged | `config/openapi_schema_snapshot.json` |
| R10 | Live dedup2d adapter | done | #547 merged `dedup_adapter.py` | default-off live hook; honest offline fallback |
| R11 | Owner ratify design-lock | residual_human | — | owner only |
| R12 | Pilot enable decisions | residual_human | — | env flag; never self-enable in production |

## Track E — Evaluation Integrity (P0)

| ID | Task | Status | Workflow / execute-plan | Evidence |
|---|---|---|---|---|
| E1 | Preserve E1 dry-run machinery (#542) | done (landed) | audit residual | `scripts/track_e_*` |
| E2 | Post-merge E1 audit at exact head | done | #547 / execute-plan PR 1 | `TRACK_E_E1_POST_MERGE_AUDIT_20260808.md` |
| E3 | No eval_integrity_gate replace | done (boundary) | workflows boundary | no path in #547 |

## Track S — Safety Baselines (P0)

| ID | Task | Status | Workflow / execute-plan | Evidence |
|---|---|---|---|---|
| S1 | Preserve SEAL (#537) | done (landed) | audit residual | assistant opt-in |
| S2 | Preserve identity fail-closed (#538) | done (landed) | audit residual | production_identity |
| S3 | SEAL+identity re-audit + no cost_cap | done | #547 / execute-plan PR 1 | `TRACK_S_SEAL_IDENTITY_BASELINE_AUDIT_20260808.md` |

## Track O — Ops & Evidence (P1)

| ID | Task | Status | Workflow / execute-plan | Evidence |
|---|---|---|---|---|
| O1 | Isolated sample checklist | done | #547 | runbook |
| O2 | Pilot ops package (kill/rollback/export/retention) | done | #547 / execute-plan PR 2 | `TRACK_O_PILOT_OPS_PACKAGE_20260808.md` |
| O3 | Workbench review metrics export | done | #547 `metrics.py` + `GET .../metrics` | review_workflow family only |
| O4 | Kill switch documented | done | runbook §6 + Track O package | done |

## Track C — Customer Pilot (P1) — human residual

| ID | Task | Status | Owner | Notes |
|---|---|---|---|---|
| C1 | 10 qualified manufacturer contacts | residual_human | Owner | may roll Day 31–60 |
| C2 | 2 lawful sample-data conversations | residual_human | Owner | |
| C3 | Named reviewer + baseline metric | residual_human | Owner | |
| C4 | Isolated customer archive run | residual_human | Owner + eng support | Day 61–90 |
| C5 | Measured pilot + commercial next step | residual_human | Owner | Day 90 gate |

## System / orchestration

| ID | Task | Status | Notes |
|---|---|---|---|
| SYS1 | `cad-reuse-workbench-dev` workflow | done | gap/implement/verify/full |
| SYS2 | `cad-reuse-workbench-90d` workflow | done | multi-track audit |
| SYS3 | `cad-reuse-workbench-system` master workflow | done | `.grok/workflows/cad-reuse-workbench-system.rhai` |
| SYS4 | System execute design + PR Plan DAG | done | `CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md` |
| SYS5 | residual PR1+PR2 (audits + ops) content | done | landed on #547 docs tranche |
| SYS6 | PR3+PR4 content folded into #547 | done | same L3 slot — no second runtime PR |
| SYS7 | Closeout board flip (PR 5) | done | #548 merged |
| SYS8 | R2 HOLD structural tests + external gates audit | done | `test_review_reuse_r2_hold.py` + EXTERNAL_GATES audit |

---

## Execution order (post-#547 / #548)

1. ~~Land / green CI for **#547** (Track R MVP)~~ — **MERGED**.
2. ~~execute-plan **PR1–PR4** content~~ — folded into #547 (L3 WIP=1).
3. ~~Closeout board flip~~ — **#548 MERGED**.
4. ~~R2 HOLD regression tests + external-gates audit (no false complete)~~ — this eng PR.
5. **Owner only:** R11 design-lock ratify · R12 pilot decision enable · Track C C1–C5.
6. **External only (audit, not eng-closed):** Evaluation Hybrid superpass red on main.
7. Boundaries: R2 HOLD · no eval_integrity_gate replace · no cost_cap · decision default-off · no fake Track C · no production self-enable of decisions.

## Workflow commands

```text
/workflow cad-reuse-workbench-dev     args: {"mode":"verify","root":"."}
/workflow cad-reuse-workbench-dev     args: {"mode":"full","root":"."}
/workflow cad-reuse-workbench-90d     args: {"track":"full","root":"."}
/workflow cad-reuse-workbench-system  args: {"phase":"all","root":"."}

/execute-plan docs/development/CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md --no-graphite --auto-pr --concurrency 2
```
