# External Gates Audit — CAD Reuse Workbench (no false completion)

**Date**: 2026-08-08  
**Authority**: 90-day plan; design-lock R1–R8 (R2 HOLD = no training-path decision reuse)  
**Main baseline**: post-#547 / #548 (`origin/main`)  
**Status**: **AUDIT ONLY** — external/human gates remain open

---

## 1. What “R2 HOLD” means here

Design-lock invariant **R2**: decision / correction evidence **must not** enter training-readable manifests; `feedback.py` JSONL is **not** the ledger store.

Related holds still in force:

| ID | Hold | Code posture |
|---|---|---|
| R1 | Decision default-off | `REVIEW_REUSE_DECISIONS_ENABLED` unset → 403 |
| R2 | No training-path reuse | no feedback imports; decision does not write JSONL |
| R5 | No retrain / eval_integrity replace | workbench does not touch gate |
| R6 | No Track E model-release metrics | metrics family = `review_workflow` |
| R7 | No cost_cap revive | `cost_cap.py` absent |
| R8 | AI no release authority | human POST only |

---

## 2. Codeable DAG status (engineering)

| Node | Status | Evidence |
|---|---|---|
| Track R MVP runtime | **DONE** | #547 merged `db437b8b` |
| Live dedup adapter (default-off) | **DONE** | `dedup_adapter.py` on main |
| Review metrics export | **DONE** | `metrics.py` + `GET /metrics` |
| E1 / SEAL / O audits + ops | **DONE** | Track E/S/O docs on main |
| Board closeout after merge | **DONE** | #548 merged |
| R2 HOLD structural tests | **this PR** | `tests/unit/test_review_reuse_r2_hold.py` |

**No further L3 runtime PR required** under 90-day codeable scope unless owner opens a new design-lock.

---

## 3. External / human gates — **not claimed complete**

| Gate | Status | Owner | Why not auto-closed |
|---|---|---|---|
| Design-lock owner ratify | **OPEN** | Owner | Product authority |
| Pilot enable decisions | **OPEN (default-off)** | Owner | Explicit pilot enable only |
| Track C: 10 contacts | **OPEN** | Owner | Real-world outreach |
| Track C: 2 sample conversations | **OPEN** | Owner | Lawful data talks |
| Track C: named reviewer + baseline | **OPEN** | Owner | Customer workflow |
| Track C: isolated customer archive | **OPEN** | Owner + eng support | Needs customer data |
| Track C: measured pilot / commercial | **OPEN** | Owner | Day 90 business gate |
| Evaluation Hybrid superpass red on main | **AUDIT residual** | Eval owners | Non-required; not Track R |
| Day-90 clock change | **NOT proposed** | Owner | Would need strategy amendment |

---

## 4. Workflow / execute-plan posture

| Tool | Role now |
|---|---|
| `cad-reuse-workbench-dev` mode=`verify` | Re-verify engineering on main |
| `cad-reuse-workbench-system` | Inventory + residual_human handoff |
| execute-plan on system design | **No open codeable PR nodes**; residual is human |

Do **not** run implement agents that invent Track C evidence or flip decision default-on.

---

## 5. Sign-off

| Role | Claim |
|---|---|
| Engineering (codeable 90d DAG) | Closed on main within R2 HOLD |
| External gates | Audited open — **not** falsely completed |
| Owner | Ratify lock; pilot enable; Track C |
