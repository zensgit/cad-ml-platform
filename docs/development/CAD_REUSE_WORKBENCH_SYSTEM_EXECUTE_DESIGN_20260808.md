# CAD Reuse Workbench — System Execution Design

**Date**: 2026-08-08  
**Status**: Active execution design (maps 90-day plan → workflows + task board + PR Plan DAG)  
**Authority**: `docs/development/CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md`  
**Strategy**: `docs/PRODUCT_STRATEGY.md` §3.3 / §7.1 / §8.2–§8.4  
**Clock**: Day 0 = 2026-07-12 · Day 90 = 2026-10-10 (not re-baselined)

---

## 1. Purpose

Ensure **all engineering content** of the 90-day workbench plan is completed through a closed loop:

```text
workflows (gap/implement/verify)
  → task board (acceptance inventory)
  → design PR Plan DAG
  → execute-plan (worktree implement + review)
  → verification MD + residual human gates
```

**Customer Track C** (10 manufacturer contacts, paid pilot, commercial commitment) is **human residual** and cannot be closed by code alone. It is tracked, not invented.

---

## 2. Current baseline (as of 2026-08-08, post-#547/#548)

| Item | Status | Evidence |
|---|---|---|
| Track R design-lock | On main (**PROPOSED**; owner ratify open) | `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md` |
| Track R runtime MVP | **MERGED** #547 (`db437b8b`) decision default-off | `src/core/review_reuse/*`, `src/api/v1/review_reuse.py` |
| Live dedup adapter | **MERGED** #547 default-off | `dedup_adapter.py` + `REVIEW_REUSE_LIVE_DEDUP` |
| Review metrics | **MERGED** #547 | `metrics.py` + `GET .../metrics` |
| E1 / SEAL / O audits + ops | **MERGED** #547 | Track E/S/O docs |
| Board closeout | **MERGED** #548 | residual_eng cleared on board |
| R2 HOLD structural tests | Eng follow-up PR | `tests/unit/test_review_reuse_r2_hold.py` |
| External gates audit | Docs only | `CAD_REUSE_WORKBENCH_EXTERNAL_GATES_AUDIT_20260808.md` |
| Track C customer | **Human residual** | cannot automate |

**WIP rules:** ≤2 open impl PRs; **L3 runtime WIP = 1**. Runtime workbench line closed on main; no second L3 runtime PR without new design-lock.

---

## 3. Key Decisions

1. **Engineering completeness ≠ full Day-90 product success.** Engineering Tracks R/E/S/O-min are code/docs; Track C is residual-listed.
2. **#547 is the Track R runtime tranche.** Residual L3 runtime (live dedup adapter) waits until #547 merges or is the same stack continuation after #547 is sole L3 slot.
3. **Decision enablement remains default-off** until owner pilot enable; never a retrain-gate change.
4. **No `eval_integrity_gate` replace, no cost_cap revive, no assistant redesign** in this plan.
5. **Orchestration:** use `cad-reuse-workbench-system` workflow for gap→execute handoff; use execute-plan DAG for residual PR stack.

---

## 4. Operating system (workflows × tasks × execute-plan)

### 4.1 Workflows

| Name | Mode / role | When |
|---|---|---|
| `cad-reuse-workbench-dev` | `gap` / `implement` / `verify` / `full` | Close engineering gaps |
| `cad-reuse-workbench-90d` | `r-mvp` / `safety` / `full` | Read-only multi-track audit |
| `cad-reuse-workbench-system` | master | Run gap → decide → point to execute-plan residual |

### 4.2 Task board

Canonical inventory: `docs/development/CAD_REUSE_WORKBENCH_TASK_BOARD_20260808.md`.  
Every acceptance row maps to: status · owner · PR · workflow · residual-human flag.

### 4.3 Scheduled automation (optional)

Daily weekday gap check automation (Grok tasks): run `cad-reuse-workbench-dev` mode=gap and report missing items. Does not merge or enable decisions.

### 4.4 Execute-plan

Design doc path for residual DAG is **this file** (`## PR Plan` below).  
Invoke:

```text
/execute-plan docs/development/CAD_REUSE_WORKBENCH_SYSTEM_EXECUTE_DESIGN_20260808.md --no-graphite --auto-pr --concurrency 2
```

Instructions always injected:

```text
L3 runtime WIP=1. Do not open a second L3 runtime PR while #547 is open.
No retrain unlock. Do not edit scripts/eval_integrity_gate*.
Do not revive cost_cap. Decision default-off. Track C is residual-only.
```

---

## 5. Acceptance inventory (engineering)

| ID | Acceptance | Done when |
|---|---|---|
| A1 | Design-lock binds §3.3 min + default-off decision | Design-lock MD + owner ratify open (not claimed) |
| A2 | Task create → events → EvidencePack JSON+MD | Service + API tests green |
| A3 | Decision default-off / enable reuse\|revise\|new | Unit+API gate tests |
| A4 | Tenant isolation + no feedback JSONL ledger | Tests + code path |
| A5 | Isolated-sample runbook + synthetic e2e | Runbook + tests |
| A6 | Plan + verification MD | Files + pytest evidence |
| A7 | E1 residual audit | Audit MD at exact head |
| A8 | SEAL + identity residual audit | Audit MD |
| A9 | Track O pilot ops package | Ops MD + kill/rollback/export checklist |
| A10 | Live dedup adapter (honest offline fallback remains) | Code + tests (**after #547**) |
| A11 | Workbench review metrics (not Track E model metrics) | Export/API or report scaffold |
| H1 | Track C human residual | Checklist only |

---

## 6. Non-goals

- 10 manufacturer contacts / paid pilot / commercial commitment by code agents
- Owner ratification of design-lock (owner-only)
- Replacing `eval_integrity_gate`, model promotion, clock re-baseline
- Assistant explainer redesign / WorkBuddy clone / PLM write-back

---

## 7. Open Questions

1. Owner: ratify design-lock and keep Day-90 = 2026-10-10?
2. Owner: merge order — land #547 before residual L3 runtime adapter?
3. Owner: enable `REVIEW_REUSE_DECISIONS_ENABLED` only in pilot?

Default if unanswered: keep decision off; merge #547 first; residual L3 waits.

---

## PR Plan

### PR 1: Residual E1 + SEAL/identity audit docs

- **Description:** Docs-only residual audits for Track E Slice E1 (#542) and Track S baselines (#537 SEAL, #538 identity). Confirm dry-run / no retrain unlock / no model-run metrics claim; SEAL hosted opt-in still fail-closed; production identity still fail-closed; #545 cost_cap still absent. Mutation-check load-bearing discriminators where feasible via file evidence at exact head.
- **Files/components affected:** `docs/development/TRACK_E_E1_POST_MERGE_AUDIT_20260808.md`, `docs/development/TRACK_S_SEAL_IDENTITY_BASELINE_AUDIT_20260808.md`, optional pointer update in `CAD_REUSE_WORKBENCH_MVP_VERIFICATION_20260808.md`
- **Dependencies:** None
- **Level notes:** docs-only; may open while #547 is open

### PR 2: Track O pilot operations package

- **Description:** Expand ops package beyond isolated-sample runbook: deployment/kill-switch/backup-rollback/audit-export/retention/provider-egress checklist; pilot metrics field list (task count, top-5 usefulness, accepted reuse, false-dup / missed-reuse human labels, median review time, coverage, insufficient_evidence). No cost_cap module revival — document “if external AI enabled” policy only.
- **Files/components affected:** `docs/development/TRACK_O_PILOT_OPS_PACKAGE_20260808.md`, optional `docs/development/CAD_REUSE_WORKBENCH_TASK_BOARD_20260808.md` status flips
- **Dependencies:** None
- **Level notes:** docs-only; parallel with PR 1

### PR 3: Live dedup2d adapter for ReviewReuse (L3 runtime)

- **Description:** Replace pure offline seed path with optional adapter that calls existing private dedup2d search/precision surfaces when configured, mapping results into `CandidateDecision` + structured rejection reasons. Must keep honest `insufficient_evidence` / `tool_unavailable` when tool absent. Decision remains default-off. Do not touch eval_integrity_gate or training feedback JSONL.
- **Files/components affected:** `src/core/review_reuse/service.py`, new `src/core/review_reuse/dedup_adapter.py`, `src/api/v1/review_reuse.py` (if needed), tests under `tests/unit/test_review_reuse_*`, `.env.example` adapter flags, OpenAPI snapshot only if routes change
- **Dependencies:** PR #547 merged (or sole L3 runtime slot free). **Not started while another L3 runtime PR is open.**
- **Level notes:** L3 runtime WIP=1

### PR 4: Workbench review-workflow metrics export

- **Description:** Export review-workflow metrics from task ledger (counts by status/decision, insufficient_evidence rate, median time-to-evidence if timestamps allow). Explicitly **not** Track E model-release metrics. Prefer read-only report endpoint or export function under review-reuse, default-off or operator-only.
- **Files/components affected:** `src/core/review_reuse/metrics.py` (or similar), API route if needed, tests, docs note in verification MD, OpenAPI snapshot if routes change
- **Dependencies:** PR 3 (or #547 if adapter deferred and metrics only need task store)
- **Level notes:** L3 if new ledger surface; keep behind same workbench boundary

### PR 5: System board closeout + residual human checklist

- **Description:** Final engineering closeout doc: flip task board A1–A11 statuses, list H1 human residuals with owners/dates inside Day 31–90 without moving Day 90, link workflows and execute-plan PLAN_IDs, record pytest evidence pointers.
- **Files/components affected:** `docs/development/CAD_REUSE_WORKBENCH_TASK_BOARD_20260808.md`, `docs/development/CAD_REUSE_WORKBENCH_MVP_VERIFICATION_20260808.md`, optional `docs/development/CAD_REUSE_WORKBENCH_SYSTEM_CLOSEOUT_20260808.md`
- **Dependencies:** PR 1, PR 2; PR 3–4 if executed in same window else residual-list them
- **Level notes:** docs-only

---

## 8. Linearized stack (execute-plan)

```text
PR1 (audits docs) ──┐
                    ├──> PR5 (closeout board)
PR2 (ops package) ──┘
PR3 (dedup adapter) ──> PR4 (metrics) ──> PR5   [only when L3 slot free]
```

Default first execute-plan batch (while #547 open): **PR1 + PR2 only**, then **PR5** if 3/4 deferred.

---

## 9. Verification

For each residual PR:

1. Docs PRs: file exists; required sections present; no runtime unlock claims.
2. Runtime PRs: pytest workbench suites + openapi if routes change; decision still default-off; boundary grep clean.
3. Workflow: `cad-reuse-workbench-dev` mode=`verify` after stack lands.
4. Do not self-merge; required CI green.

---

## 10. Resume / babysit

- execute-plan: `/execute-plan --resume <PLAN_ID>`
- PR babysit Graphite not available → plain-git + `gh pr create`
- System workflow: `/workflow cad-reuse-workbench-system`
