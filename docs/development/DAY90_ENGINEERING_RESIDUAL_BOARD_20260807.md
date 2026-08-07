# Day-90 Engineering Residual Board (2026-08-07)

- **Status**: living engineering map of `PRODUCT_STRATEGY.md` §8.1–§8.4 for the
  implementable remainder of the 90-day program. **Not** a commercial pilot commitment.
- **Grounding**: `origin/main@b5780aad` (post #542 Track E E1) unless a row pins a newer merge.
- **Authority**: strategy wins on conflict. Owner-only rows cannot be closed by code alone.
- **This document does not** amend strategy end states, re-open retraining, or claim Day-90 pilot success.

## How to read

| Label | Meaning |
|---|---|
| **done** | On `main` with pinned merge/SHA; residual risk called out |
| **open-engineering** | Unblocked or partially blocked engineering work with a next PR action |
| **owner-only** | Requires owner commercial/legal/calendar action; engineering cannot complete |

---

## §8.1 Track E — evaluation-integrity-v2

| Item | Status | Evidence | Next action |
|---|---|---|---|
| Design-lock contract | **done** | #531 → `c26bfe49` | — |
| E1 dry-run split (A–C, F split) | **done** | #542 → `b5780aad`; `scripts/track_e_eval_integrity.py` | Operator entry (this tranche) |
| E1 versioned manifest + provenance (D–E, F verify) | **done** | #542; `scripts/track_e_manifest.py` | Operator entry (this tranche) |
| E1 discriminators 1–14 + gate floor | **done** | unit tests in #542; gate still raises | — |
| Invariant H real metrics / two-phase release | **open-engineering** (supply-gated) | deferred by design-lock | Wait model-run + data; **not** this tranche |
| Retrain re-enable | **owner-only** / fail-closed | `eval_integrity_gate.check()` always raises | Code change to `check()` body only after H |

## §8.2 Track C — customer discovery / paid pilot

| Item | Status | Evidence | Next action |
|---|---|---|---|
| Two discovery or paid-pilot attempts | **owner-only** | strategy §8.2 | Owner contacts / commercial |
| Day-90 measured pilot commitment | **owner-only** | calendar ~mid-Oct 2026 | Owner; engineering supplies materials only |

## §8.3 Pilot release gates

| Item | Status | Evidence | Next action |
|---|---|---|---|
| Production auth fail-closed | **done** | #538 → `191ce7ef`; `src/api/production_identity.py` | Maintain harness opt-in only |
| Authenticated tenant/user identity (JWT sub) | **done** | #538 integration_auth | — |
| DWG conversion opt-in default | **done** | #534 → `665e9408` | — |
| Hosted provider SEAL + tool_status | **done** | #537 / #535 (docs) | — |
| Fail-closed **cost cap** before external AI | **open-engineering → this tranche** | was missing as enforce seam | Ship `cost_cap` + wire `get_provider` |
| Isolated sample mode (no provider egress) | **open-engineering → this tranche** | strategy §8.3 offline sample | Ship `ISOLATED_SAMPLE_MODE` refuse |
| Independent customer holdout **metrics** | **open-engineering** (H) | E1 reproduces **split** only | After model-run supply |
| Customer-data retention/deletion policy | **open-engineering** (ops docs) | partial product docs | Follow-on policy/runbook PR |

## §8.4 Thirty / sixty / ninety day calendar

| Window | Engineering-completable | Owner-only |
|---|---|---|
| Days 0–30 | Eval-integrity contract freeze (**done** #531); production-auth defaults (**done** #538); isolated sample **definition+enforcement** (this tranche partial) | Ten manufacturers; legal sample conversations |
| Days 31–60 | Reproducible **split** dry-run on legal sample offline (**E1**); review UX without write-back (partial elsewhere) | Real archive access; pilot environment |
| Days 61–90 | Keep fail-closed floors; no feature sprawl | Measured pilot commitment |

## Hygiene board (not §8, but blocks CI)

| Item | Status | Next |
|---|---|---|
| #541 pin-policy sync | **done** on main `46b52eab` | Prove on a Dependabot head |
| Open Dependabot actions PRs | **open** | Sync policy → merge green / close stale |
| #476 | **open** until post-sync green proof | Close when proof attached |
| #507 portfolio | **owner-only** Layer 4 | Do not merge-drive |

## Highest-priority non-owner residual (selected)

1. **§8.3 external-AI cost cap + isolated sample refuse** (enforce seam before hosted providers).
2. **E1 operator dry-run entry** (`scripts/run_track_e_e1_dry_run.py` + Make target) so Day-30/60 engineering can run offline without archaeology.
3. **Dependabot pin-sync proof** for #476.

Not selected (blocked or owner-only): H metrics, retrain unlock, pilot signature, #507.
