# CAD Reuse Workbench 90-Day Development Plan

**Date**: 2026-08-07  
**Status**: Proposed execution plan; does not reset the ratified 90-day clock  
**Authority**: `docs/PRODUCT_STRATEGY.md` section 3.3, section 7.1, section 8.2, and section 8.4  
**Scope**: implement the ratified CAD reuse / revise / new decision slice, not a generic CAD chatbot, WorkBuddy clone, or multi-agent platform
**Clock**: Day 0 = 2026-07-12, Day 30 = 2026-08-11, Day 60 = 2026-09-10, Day 90 = 2026-10-10

## 1. Executive Summary

This plan implements the strategy that is already ratified in `PRODUCT_STRATEGY.md`:

- section 3.3 defines the stable decision contract;
- section 8.2 defines the vertical slice: ingest archive, find candidates, review evidence, decide reuse / revise / new, then export or write back through the customer's authority;
- section 7.1 makes audit ledgers, auth, customer-data handling, model-release gates, and AI output sinks L3 work.

The WorkBuddy comparison is useful as product language: task workbench, deliverables,
process visibility, and skill boundaries. It is not the source of authority. The
authority is the repo strategy itself.

The product spine is:

```text
Upload drawing or ingest archive
  -> create ReviewReuseTask
  -> retrieve candidate drawings
  -> run deterministic geometric verification
  -> produce EvidencePack from the section 3.3 contract
  -> human reviewer decides reuse / revise / new
  -> audit and quarantined review evidence
```

The day-90 gate remains the strategy gate anchored to the strategy review / ratification
calendar, not to this document date.

| Strategy window | Calendar dates | Required interpretation in this plan |
|---|---|---|
| Day 0-30 | 2026-07-12 through 2026-08-11 | Engineering safety baselines are mostly landed; remaining risk is customer evidence and independent audit, not rebuilding the gates. |
| Day 31-60 | 2026-08-12 through 2026-09-10 | Build and demonstrate the reuse / revise / new workbench MVP without canonical write-back, including one isolated-archive end-to-end task → EvidencePack export. |
| Day 61-90 | 2026-09-11 through 2026-10-10 | Run isolated archive / named-reviewer evidence and secure a measured pilot commitment. |

As of the current review date, 2026-08-08, the repo is still in the Day 0-30 window
(ends 2026-08-11). The workbench MVP belongs to the Day 31-60 window starting
2026-08-12. This document does not silently move the day-90 gate to November. If the
owner wants a new calendar, `PRODUCT_STRATEGY.md` must be explicitly amended.

**Day 0-30 residual rule (does not move Day 90):** engineering closeout for Day 0-30
may be marked complete when the L3 workbench design-lock is submitted for owner
ratification and the baseline safety / E1 audits in §7 are done or explicitly residual-
listed. Customer-side Day 0-30 items (10 qualified manufacturer contacts; 2 lawful
sample-data conversations) may roll into Day 31-60 without resetting the strategy
clock, **but must not delay the design-lock**. Customer discovery must still start in
parallel; it is not postponed behind engineering polish.

## 2. Current State Assumptions

This document is grounded on the current strategic direction plus a targeted read of
the real code paths. Before implementation, re-check the exact target branch, PR head,
and CI state.

Important current-state assumptions from `origin/main@16bbba3f` (post-#545 revert of
the erroneous residual tranche; re-pin head before implementation):

- Assistant provider SEAL has landed through #537. Treat assistant safe-park as a
  preserved baseline to audit, not new work to rebuild in this plan.
- Production identity fail-closed has landed through #538. Treat it as a preserved
  baseline to audit, not new work to rebuild here.
- Track E Slice E1 has landed through #542. It provides dry-run split / manifest
  integrity machinery, not the full model-run metrics lane.
- #543 day-90 residual work was reverted by #545. Do not reintroduce a parallel
  residual board, a silent clock re-baseline, or a silent revival of
  `src/core/assistant/cost_cap.py` unless the owner separately ratifies that line.
- The local checkout may lag `origin/main`; exact implementation must not rely on a
  stale worktree.

## 3. Product Boundary

The product remains a private-deployment CAD drawing reuse and release decision engine.
It helps engineers answer:

- can this drawing or part be reused?
- should it be revised?
- should a new drawing or part be created?
- what evidence supports or rejects the recommendation?

AI has no release authority. A human reviewer or the owning PLM workflow remains the
authority for release, write-back, merge, delete, or canonical product-data mutation.

## 4. Stable Contract Mapping

`EvidencePack` must be drafted directly from `PRODUCT_STRATEGY.md` section 3.3. It is
not a parallel invention.

| Section 3.3 field | EvidencePack field family | Notes |
|---|---|---|
| candidate identifier and source | `candidate_id`, `candidate_source`, redacted/hashable display identifiers | Strategy minimum |
| normalized geometric and semantic scores | `scores.geometric`, `scores.semantic`, `score_normalization` | Strategy minimum |
| deterministic verification result | `verification.verdict`, `verification.level`, `verification.methods` | Strategy minimum |
| confidence and calibration version | `confidence.score`, `confidence.band`, `calibration.version` | Strategy minimum |
| evidence and rejection reasons | `evidence[]`, `rejection_reasons[]`, `unsupported_states[]` | Strategy minimum |
| model, ruleset, and dataset provenance | `provenance.model`, `provenance.ruleset`, `provenance.dataset`, `provenance.input` | Strategy minimum |
| human decision state | `human_decision.state`, `human_decision.allowed_actions` | Strategy minimum; states center on reuse / revise / new |
| trace and idempotency identifiers | `trace_id`, `task_id`, `idempotency_key`, `source_job_id` | Strategy minimum |
| *(implementation extension)* | `scores.visual` | Optional; does **not** change the §3.3 minimum contract; must not block design-lock ratification |

The first design-lock must bind the **strategy-minimum** fields before runtime code
writes an audit ledger or exposes a human decision sink. Implementation extensions
(e.g. `scores.visual`) may be listed in the design-lock as optional and deferred.

## 5. Re-Ordered Tracks

| Priority | Track | Goal | 90-day deliverable |
|---|---|---|---|
| P0 | Track R: ReviewReuseTask Workbench | Implement the section 3.3 / 8.2 review slice | L3 design-lock, then task API, step state, events, EvidencePack, human decision ledger |
| P0 | Track E: Evaluation Integrity | Preserve and audit model-promotion safety | E1 post-merge audit, dry-run split verification, promotion remains fail-closed |
| P0 | Track S: Safety Baselines | Preserve already-landed SEAL and identity gates | Post-merge audit and regression coverage for #537 and #538 invariants |
| P1 | Track C: Customer Pilot | Validate with real workflows | 10 qualified contacts, 2 lawful sample conversations, 1 archive run, 1 measured pilot |
| P1 | Track O: Ops and Evidence | Make pilot operation auditable | Report/export first, runbook, kill switch, rollback, audit export, isolated-sample checklist |

### PR and L3 slot discipline

This plan inherits repository WIP rules (same as recent L3 lines):

- **≤2 open implementation PRs** at a time unless the owner raises the cap in writing.
- **L3 runtime WIP = 1**: only one L3 runtime implementation PR may be open on the
  workbench line at a time.
- **Design-lock is docs-only** and does **not** consume the L3 runtime slot; it must be
  owner-ratified before any workbench runtime implementation PR opens.
- **No runtime implementation PR** (and no ledger-writing code) before the L3
  design-lock is ratified.
- Closed #510/#511-style branches are not revived; rebuild from current `main` against
  the ratified lock.

## 6. Immediate Owner Decisions

These decisions are required before runtime development starts:

1. Confirm that "workbench" means implementing `PRODUCT_STRATEGY.md` section 3.3 and
   section 8.2, not expanding into a WorkBuddy-like generic assistant.
2. Confirm the day-90 clock. Default is the ratified strategy clock ending around
   2026-10-10. Any different calendar requires a strategy amendment.
3. Confirm that the ReviewReuseTask / EvidencePack / TaskEvent / human decision
   ledger design-lock is the next product design document.
4. Confirm that assistant evidence explanation remains out of the 90-day runtime
   scope. The already-landed SEAL is preserved; redesign is later.
5. Confirm whether #542 E1 gets a post-merge audit now, because this plan depends on
   it as a hard safety baseline.
6. Resolve the portfolio-level #507 questions enough to avoid two October customer
   tracks competing for the same owner attention without priority. **#507 need not be
   merged** to start the Track R design-lock, provided the owner states in writing that
   the workbench is the single product track for owner attention through Day 90.

## 7. Day 0-30 Closeout: Design-Lock, Audit, Customer Evidence

### Goals

Close the true Day 0-30 window without creating another unsafe runtime surface. The
strategy-required safety baselines are mostly landed; do not rebuild them. Human
decision state and audit ledger work are L3 by strategy section 7.1, so the new
workbench line starts with a docs-only design-lock that the owner ratifies before
implementation.

**Calendar pressure:** as of 2026-08-08 the Day 0-30 window ends 2026-08-11. Treat
customer-side §8.4 Day 0-30 outcomes as **start-now / may-roll residual** (see §1),
not as a reason to skip or defer the design-lock.

### Required Deliverables

#### Engineering closeout (Day 0-30 gate for this plan)

- L3 design-lock docs-only PR for:
  - `ReviewReuseTask`
  - `TaskEvent`
  - `EvidencePack`
  - `HumanDecision`
  - `CandidateDecision`
  - tenant isolation and identity attribution
  - audit export and deletion/retention boundaries
  - trace and idempotency identifiers

- EvidencePack contract sourced from section 3.3, including:
  - candidate identity and source
  - normalized score fields (**geometric + semantic** are the strategy minimum;
    `scores.visual` is an optional implementation extension only)
  - deterministic verification result
  - confidence and calibration version
  - evidence and rejection reasons
  - provenance
  - human decision state (strategy center: reuse / revise / new)
  - trace and idempotency IDs

- TaskEvent contract:
  - `submitted`
  - `input_validated`
  - `recall_started`
  - `recall_completed`
  - `precision_started`
  - `precision_completed`
  - `evidence_pack_ready`
  - `decision_submitted`
  - `failed`
  - `canceled`

- E1 post-merge audit:
  - confirm E1 remains dry-run and cannot unlock retraining;
  - confirm it does not claim model-run metrics;
  - mutation-check the load-bearing discriminators where feasible.

- Baseline safety audit:
  - confirm #537 assistant SEAL invariants still hold at exact head;
  - confirm #538 production identity invariants still hold at exact head;
  - confirm #545 revert did not remove needed safety gates.

- **Isolated sample handling — minimum definition and verification** (strategy
  §8.3 / §8.4 Day 0-30; may ship as runbook + checklist, not as a revived cost-cap
  module):
  - no shared multi-tenant processing of customer sample drawings;
  - no hosted LLM provider egress for sample processing (rely on existing SEAL /
    offline posture unless the owner separately enables hosted AI under §8.3);
  - sample storage location, retention period, and deletion procedure documented;
  - explicit ban: before pilot gates pass, customer drawings are not processed in a
    production or shared tenant environment.

#### Customer action (start in Day 0-30; may roll residual into Day 31-60)

- inventory and contact at least 10 qualified manufacturers;
- obtain 2 lawful sample-data conversations;
- identify actual reviewer workflow and baseline metric.

These customer items **must not** block design-lock ratification. If incomplete by
2026-08-11, list them as residual with owner owner and target dates inside Day 31-60
without amending the Day 90 date.

### Acceptance

- **Engineering Day 0-30 closeout:** design-lock PR is open and ready for owner
  ratification (or already ratified); E1 and safety baselines are independently
  audited or explicitly residual-listed; isolated-sample minimum runbook/checklist
  exists.
- **Customer Day 0-30:** contacts and sample conversations either done or listed as
  residual with owners — discovery has started, not postponed behind polish.
- Day-90 date is not silently re-baselined.
- Owner-ratified design-lock exists before runtime ledger implementation.

## 8. Day 31-60: Workbench MVP

### Goals

Build the review workbench around existing assets instead of rewriting the dedup
pipeline.

Align to strategy §8.4 Days 31-60: **demonstrate the reuse / revise / new review
flow without canonical write-back**, and support the strategy wording "reproducible
evaluator on the first real archive" **as interpreted for this slice**:

- **In scope:** one **isolated / offline-approved archive** end-to-end path —
  `ReviewReuseTask` → candidate recall/precision → **EvidencePack** export that a
  fresh operator can re-open and re-verify from stored task/evidence IDs (task +
  evidence reproducibility, not model-promotion metrics).
- **Out of scope:** replacing `scripts/eval_integrity_gate.check()`, claiming full
  §8.1 exit condition, or emitting Track E model-run metrics (invariant H).

The current assets already include:

- tenant-isolated async dedup jobs;
- sync/async search responses;
- queue depth, callback, and Prometheus metrics;
- `duplicates` / `similar` grouping;
- `verdict`, `match_level`, `levels`, and `level_stats`;
- local L4 precision scoring via `dedupcad_precision`;
- partial rejection information in warning strings;
- per-tenant threshold, version-gate, and precision-weight configuration.

The MVP gap is mainly:

- step-level task events;
- stable EvidencePack export;
- human decision ledger, implemented only after L3 design-lock ratification;
- trace and idempotency IDs;
- calibration version fields where available;
- structured rejection reason normalization;
- isolated-archive end-to-end exercise of the above.

### Required Deliverables

- Runtime implementation of the owner-ratified design-lock (API prefix matches repo
  convention — **full paths under `/api/v1/review-reuse/...`**, not bare `/v1/...`):
  - `POST /api/v1/review-reuse/tasks`
  - `GET /api/v1/review-reuse/tasks/{task_id}`
  - `GET /api/v1/review-reuse/tasks`
  - `POST /api/v1/review-reuse/tasks/{task_id}/cancel`
  - `GET /api/v1/review-reuse/tasks/{task_id}/events`
  - `GET /api/v1/review-reuse/tasks/{task_id}/evidence-pack`
  - `POST /api/v1/review-reuse/tasks/{task_id}/decision`

- Candidate states:
  - `duplicate`
  - `similar`
  - `different`
  - `insufficient_evidence`

- Structured rejection reasons:
  - `missing_geom_json`
  - `version_gate_filtered`
  - `low_precision_score`
  - `vision_only_unverified`
  - `unsupported_file_type`
  - `external_service_unavailable`
  - `tool_unavailable`

- EvidencePack export:
  - JSON for machines;
  - Markdown or HTML report for reviewers.

- Human decision ledger:
  - **Strategy-center states:** `reuse`, `revise`, `new`
  - **Implementation extensions** (optional; design-lock must separate them so they
    do not blur CandidateDecision vs HumanDecision): `reject_candidate`,
    `need_more_info`
  - reviewer identity from validated identity only;
  - reason codes;
  - timestamp;
  - source task and candidate references;
  - idempotency behavior;
  - **enablement:** decision submission stays default-off / gated until L3 acceptance
    tests pass; Day 61-90 pilot use requires an **explicit owner enable step**
    (config/flag documented in the design-lock), not a retrain-gate change.

- Review evidence quarantine:
  - decision and correction evidence does not enter a training-readable manifest;
  - existing `feedback.py` JSONL training-oriented path is not reused as the ledger store.

- Isolated-archive MVP exercise (strategy §8.4 Days 31-60 interpretation):
  - one lawful sample or synthetic archive under the isolated-sample rules from §7;
  - create task → complete pipeline → export EvidencePack → optional human decision
    if enablement is on;
  - document that this satisfies workbench "reproducible evaluator" for the **task /
    evidence** family, not Track E model-release metrics.

### Acceptance

- A reviewer can complete one reuse / revise / new decision from an EvidencePack
  (when decision enablement is on).
- The task workflow can be exercised without reading raw dedup API output.
- Task, event, evidence, and decision access are tenant-isolated.
- Decision submission is default-off or gated until the L3 acceptance tests pass.
- Metrics are review-workflow metrics, not model-release metrics.
- **Hard MVP bar:** at least one isolated-archive end-to-end `ReviewReuseTask` with
  reproducible EvidencePack export is demonstrated and recorded.

## 9. Day 61-90: Customer Evidence and Measured Pilot

### Goals

Validate that the workbench reduces search and review cost on a real customer workflow
before the ratified 2026-10-10 day-90 gate.

### Required Deliverables

- At least one isolated customer archive run.
- Named reviewers who perform real review decisions.
- Pilot metrics report/export:
  - task count;
  - top-5 candidate usefulness;
  - accepted reuse;
  - human-labeled false duplicate cases;
  - human-labeled missed reuse cases;
  - median review time;
  - reviewer coverage;
  - unsupported or insufficient-evidence states.

- Pilot operations package:
  - deployment runbook;
  - kill switch;
  - backup and rollback;
  - audit export;
  - data retention and deletion procedure;
  - provider-egress policy;
  - cost cap if external AI is explicitly enabled.

- Commercial next step:
  - paid pilot, or
  - contractual commitment, or
  - written decision to pause independent product work and fold the engine into another CAD/PLM shell.

### Acceptance

- At least one partner agrees to a measured pilot with data access, named reviewers,
  metrics, data policy, and a commercial next step.
- If this condition is not met by the ratified day-90 date, pause feature work and
  reassess target customer, workflow pain, and product wedge.

## 10. Track E Boundary

Do not mix two metric families:

- Workbench metrics come from the ReviewReuseTask ledger and named reviewer decisions.
  These include top-5 usefulness, accepted reuse, review time, human-labeled false
  duplicates, and human-labeled missed reuse.
- Model-release metrics belong to the deferred Track E model-run lane. These include
  per-class, macro, calibration, false-duplicate, and missed-reuse metrics bound to a
  candidate model, split digest, evaluator version, and thresholds.

E1 dry-run split / manifest work is useful and load-bearing, but it does not satisfy
the full section 8.1 exit condition and must not unlock retraining or model promotion.

Replacing `scripts/eval_integrity_gate.check()` with a real two-phase gate is not in
this 90-day workbench plan. That future replacement is its own owner-gated L3 runtime
decision because it reopens the model-promotion seam.

## 11. Assistant Boundary

The 90-day runtime scope does not include assistant-as-explainer work.

Preserve and audit the already-landed assistant SEAL. Do not expand assistant behavior
into the review path during this window unless the owner separately ratifies a redesign
lock. This avoids reopening the explicit "seal now, redesign later" boundary in the
assistant design-lock.

Allowed during this plan:

- verify no unopted hosted-provider egress;
- verify tool failure status remains structured;
- verify assistant deployment posture stays honest.

Not allowed during this plan:

- new assistant evidence-explanation runtime;
- generic CAD chat;
- new remote dispatch or IM-bot integrations;
- hosted provider expansion.

## 12. Skill Boundaries

### Allowed in the 90-day scope

- Retrieve candidates from private drawing archives.
- Run deterministic geometric checks and explain rejection reasons.
- Produce auditable EvidencePacks.
- Let humans submit governed decisions after L3 design-lock ratification.
- Export decisions through a customer-owned authority boundary.

### Not allowed in the 90-day scope

- Generic CAD ChatGPT positioning.
- General desktop agent or WorkBuddy clone.
- Autonomous CAD, PLM, or ERP write-back.
- Generative CAD design.
- Hosted LLM provider expansion.
- Cross-customer training by default.
- Model promotion before Track E exits.
- Treating AI text as release authority.

## 13. Engineering TODO

### Track R: ReviewReuseTask Workbench

- Draft and ratify the L3 design-lock before runtime implementation.
- Define stable Pydantic/domain schemas for `ReviewReuseTask`, `TaskEvent`,
  `EvidencePack`, `HumanDecision`, and `CandidateDecision`.
- Add a service layer that wraps the existing dedup 2D pipeline without duplicating
  search logic.
- Add API endpoints under `/v1/review-reuse` only after the design-lock is ratified.
- Add storage abstraction for task state, events, evidence packs, and decisions.
- Add JSON EvidencePack export.
- Add Markdown or HTML EvidencePack export.
- Add behavioral tests for task creation, event progression, evidence export, decision
  submission, tenant isolation, idempotency, and cancellation.

### Track E: Evaluation Integrity

- Run a post-merge audit of #542 / E1.
- Preserve the unconditional fail-closed retraining / model-promotion seam until the
  later two-phase gate exists.
- Do not replace `scripts/eval_integrity_gate.check()` under this plan.
- Do not add model-run metrics to E1 retroactively.
- Keep active-learning and feedback corrections out of training-readable manifests
  until Track E exit.

### Track S: Safety Baselines

- Treat #537 assistant provider SEAL and #538 production identity fail-closed as
  completed baselines, not implementation TODOs to repeat.
- Regression-check assistant provider SEAL invariants at exact head.
- Regression-check production identity fail-closed invariants at exact head.
- Ensure #545's revert did not reopen cost-cap or residual-board behavior in a way that
  changes pilot gates silently.
- Keep assistant redesign out of this plan unless separately ratified.

### Track C: Customer Pilot

- Build a qualified account list of at least 10 manufacturers with legacy 2D archive pain.
- Secure at least 2 lawful sample-data conversations.
- Define baseline measurement: search time, duplicate creation, accepted reuse, or review time.
- Identify named reviewers and their release/reuse workflow.
- Run one isolated archive evaluation before any shared or production tenant exposure.

### Track O: Ops and Evidence

- Document unsupported formats and failure states.
- Add audit export for tasks, evidence packs, decisions, and provider-egress settings.
- Add kill switch and rollback runbook.
- Prefer report/export first; only build dashboard UI once a named operator exists.
- Maintain the **isolated sample handling** minimum runbook/checklist from §7
  (no shared tenant, no hosted egress for samples, retention/deletion, pilot-gate ban).
- Treat provider spend, budget alerts, and fail-closed external-AI cost cap as a pilot
  release gate from `PRODUCT_STRATEGY.md` section 8.3. Do not reintroduce the reverted
  #543 `src/core/assistant/cost_cap.py` implementation unless the owner explicitly
  decides to revive or redesign that line after #545.

## 14. Verification Plan

Required verification before claiming the workbench implemented:

- Design-lock ratification evidence for the L3 ledger and decision sink.
- Unit tests for every new schema normalizer and state transition.
- Contract tests for `/api/v1/review-reuse` APIs.
- Tenant-isolation tests for task, event, evidence, and decision access.
- Idempotency tests for task creation and decision submission.
- Golden task fixture that produces a deterministic EvidencePack.
- Fail-first tests for forged identity and unauthorized tenant access on the new APIs.
- Fail-first tests that review decisions cannot enter a training-readable manifest.
- Regression tests preserving assistant SEAL and production identity baselines where
  the workbench touches adjacent paths.
- Documentation check that EvidencePack fields map to `PRODUCT_STRATEGY.md` section 3.3.

## 15. Success Metrics

Product metrics:

- top-5 candidate usefulness.
- accepted reuse count.
- human-labeled false duplicate cases.
- human-labeled missed reuse cases.
- median task review time.
- reviewer decision coverage.
- unsupported or insufficient-evidence rate.
- archive onboarding time.
- pilot conversion and commercial next step.

Non-goal metrics:

- model count.
- provider count.
- endpoint count.
- PR count.
- raw accuracy without evaluation-integrity proof.
- chatbot demo quality.

## 16. Non-Authorization

This document does not authorize:

- runtime implementation before the L3 design-lock is ratified;
- replacing `scripts/eval_integrity_gate.check()`;
- model promotion or retraining;
- hosted provider egress;
- assistant redesign;
- customer-data processing in a production or shared tenant;
- pilot deployment;
- merge.

`merged != enabled != safe to enable`.

## 17. PR Hygiene

This document must land from an **isolated worktree branch**, not from a dirty
canonical checkout mixed with unrelated files. Prefer a docs-only PR titled with
`[for-review]` before any design-lock or runtime PR.

Recommended sequence after this plan is accepted:

1. Docs-only PR: this workbench 90-day plan on `main`.
2. Docs-only L3 design-lock PR for ReviewReuseTask / EvidencePack / ledger (does not
   consume the L3 runtime slot).
3. Owner ratify design-lock.
4. Single L3 runtime implementation PR for `/api/v1/review-reuse` (WIP=1).
