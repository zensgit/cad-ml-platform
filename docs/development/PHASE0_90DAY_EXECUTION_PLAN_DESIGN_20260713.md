# Phase-0 90-Day Execution Plan — Development Order, Parallelism & Model Tiering

**Date:** 2026-07-13
**Baseline:** `origin/main` @ `e2facd99`
**Companion:** `PHASE0_90DAY_PLAN_CLAIMS_VERIFICATION_20260713.md` (the evidence this plan is built on)
**Status:** for-review design. **Nothing here is merged, ratified, armed, or enabled by the author.**

This document turns the owner's 2026-07-13 conclusion into an executable order: what runs in parallel, which tier of model does each piece by difficulty, and — critically — which work the author may build vs which is owner-gated.

---

## 0. Governance guardrails (invariant across all tracks)

The author operates under the single-maintainer L3 protocol. On every track:

- **Build for-review only.** The author does **not** merge, does **not** ratify designs, does **not** arm gates, does **not** enable `require_code_owner_reviews`.
- **L3 runtime lands `default-off` / dry-run**, fail-first (observed-RED banked before green), and passes an adversarial independent review before it is a merge candidate.
- **Owner-gated items are proposed, not executed:** merges/merge-windows, design ratification, arming Hard Gate, enabling CODEOWNERS review, branch-protection changes, and the entire customer line.
- **No new L3 runtime PRs during the current hold** beyond what the ratified order permits; Track E/F runtime waits for its governance gate (below).

---

## 1. Model-tiering rule (difficulty → model)

| Activity | Model | Rationale |
|---|---|---|
| Design, design-locks, audit, adversarial review | **Fable 5** (→ Opus 4.8 on daily cap) | highest-judgment work |
| Spec-complete mechanical implementation | **Sonnet 5** | the design fully determines the code |
| Hardest exhaustive enumeration / verification, and Fable fallback | **Opus 4.8** | e.g. the activation-surface enumeration (Opus produced the authoritative 13-point list) |

This turn's verification already exercised the rule: Fable ×3 (semantic claims), Opus ×1 (activation surface), Sonnet ×1 (mechanical grep-level confirm). Implementation turns shift toward Sonnet once each design-lock is ratified.

---

## 2. Track register (status · buildable · model · gate)

| Track | Scope | Now | Author may build? | Model path | Gated on |
|---|---|---|---|---|---|
| **G** Governance / close-out | refresh #498; #512 as path-list (no CODEOWNERS enforce); split #513 → Phase A/B; Hard Gate observation | — | design ✅ · **merge/ratify/enable = owner** | Fable | — |
| **I** Production identity fail-closed | prod refuses default creds + `disabled`; bind identity to `auth_subject`; drop `x-user-id` override; negative tests | 10% | ✅ for-review, default-off | Fable → Sonnet → Fable | — (independent of E) |
| **F** Activation freeze (**#513 Phase A**) | prod-disable `/model/reload`; fail-closed unproven startup loads; CI activation-surface enumerator over the 8-point set | design/0% | ✅ for-review, default-off | Fable → Sonnet(enumerator) → Opus review | design-lock ratify |
| **E** Track E eval-integrity-v2 | 7× §8.1: manifest+provenance, content/family/label/side digests, quarantine, holdout, real/synth/aug metrics, two-phase candidate gate | **5–10%** (only #509 safety-brake) | design ✅ · **runtime after gate** | Fable → Sonnet → Fable | **#512 landed + design-lock ratify** |
| **M** Full proof membrane (**#513 Phase B**) | signed proof store, revocation, LKG re-validation | 0% | design ✅ | Fable | **conditional** — only when a pilot needs dynamic model-swap |
| **P** Offline pilot/eval pack | ingest → candidate → deterministic evidence → reuse/revise/new export | 0% | ✅ after a real sample | Fable → Sonnet | first lawful customer archive |
| **C** Customer discovery / data agreement / pilot | named customer, sample rights, pilot commitment | no repo evidence | ❌ **owner-only** | — | owner |

---

## 3. Parallelism model — design parallel, runtime **WIP=1** (owner-ratified 2026-07-13)

Single maintainer ⇒ **runtime L3 is WIP=1**: exactly one runtime track is implemented at a time. Design-locks, audits, and model-generated *evidence* run in parallel; the human remains the sole ratifier. Running two runtime L3 tracks at once would collapse the isolated-critic, observed-RED, and final-head binding together. (This corrects the earlier draft's parallel-runtime DAG.)

**Ratified runtime order (serial):**

```
①#513 Phase A freeze → ②Production identity → ③Track E v2 → ④Phase A rest (activation-surface freeze) → ⑥#513 Phase B (DEFERRED: pilot-pull only)
                                                    ⑤Pilot pack — only with a real lawful sample
```

- **Design in parallel now (no runtime):** the #513 Phase A/B split (#513), the production-identity design-lock (#517), and the Track E design-lock may be authored + critiqued concurrently. **No runtime is written until the relevant design-lock is ratified.**
- **First runtime L3 is NOT Track E** — it is freezing the live `/model/reload` boundary + production identity. #509 already blocks contaminated-eval retrain; the open live boundary is identity + caller-path reload.
- **Customer line (C) runs in parallel from Week 1**, owner-driven. **No lawful sample by Week 4 → cancel the Week 7–8 pilot build** rather than fill it with simulated data.

---

## 4. Per-track deliverable cadence (each ends in design + verification MD)

Every track produces, per the owner's standing instruction:

1. **Design / design-lock MD** (Fable) — grounded on a pinned `origin/main` SHA, with the completeness principle enforced by a mechanism (e.g. the CI enumerator for F), not a hand-count.
2. **Implementation** (Sonnet) — fail-first: the failing/RED test is written and observed-RED **before** the passing implementation.
3. **Adversarial review** (Fable, default-refute) — the independent-critic pass; findings reproduced-or-refuted.
4. **Verification MD** — observed-RED evidence, positive controls, and the adversarial review's confirmed set.

### Track I — production identity, the first runtime L3 (design-lock #517)
- **No default credential authenticates:** `get_api_key` must compare to a configured key set (today it accepts ANY non-empty key — `dependencies.py:8`); `ADMIN_TOKEN` must not authenticate at its `test` default.
- **Production explicit + fail-closed:** an `ENVIRONMENT`/`APP_ENV` guard boot-refuses on default/`test` creds or `INTEGRATION_AUTH_MODE=disabled`; dev/test permissiveness is an explicit opt-in.
- **JWT verifies issuer/audience/expiry** (today `jwt.decode` passes none — `integration_auth.py:88`); **actor derives ONLY from the validated `sub`** — the `x-user-id` override is **fix-now, not latent** (live raw-header readers: `audit/logger.py:614`, `feature_flags/decorators.py:136`, `request_context:213`), and a test **locks in** the bad contract (`test_integration_auth_middleware.py:110`) — fail-first flips it.
- **Forged identity headers rejected; audit reads the validated identity, not the raw header.**

### Track F — activation freeze / #513 Phase A (design highlights)
- Production-disable dynamic `/model/reload` (verification §2); unproven startup model loads fail closed.
- **CI activation-surface enumerator** asserting the 8-point arbitrary-deserializer set (verification §5) is fully covered — the mechanism that replaces the repeatedly-miscounted hand-count. New prod deserializer without coverage → CI red.
- Explicitly **not** in Phase A: signed proof store / HSM / revocation / LKG — those are Phase B (M), deferred until a pilot needs dynamic swap, per the owner's split.

### Track E — after the governance gate (design highlights)
- Portable, versioned manifest + provenance schema (add hash/family/split/source/license/label_authority columns the current `golden_*_set.csv` lack).
- Content/family/label/side digests; conflict quarantine; family+time holdout; real/synth/augmented separated metrics; macro-F1/calibration/false-duplicate/missed-reuse; two-phase candidate-binding gate; fresh-clone reproduction + observed-RED. `auto_retrain` stays **off** throughout.

---

## 5. What blocks what (owner actions that unblock author work)

| Owner action | Unblocks |
|---|---|
| Merge #512 (as path-list) + verify CODEOWNERS chain | Track E runtime governance gate |
| Ratify #513 **Phase A** design-lock | Track F runtime |
| Ratify Track E design-lock | Track E runtime |
| Decide branch-protection posture (req=0 CI-gated vs req=1 + real reviewer) | merge windows for all tracks |
| First lawful customer archive | Track P (pilot pack) |

Author proceeds now on **design-locks only** (E/I/F design + isolated critic). **No runtime — including fail-first tests — until the relevant design-lock is ratified** (owner: 暂不启动任何 runtime).

---

## 6. Ratified two-month schedule (owner, 2026-07-13)

Owner-ratified; supersedes the earlier "front-load Track I" open question (resolved: identity/reload-freeze **is** the first runtime L3, as W2–3). Estimate: governance + identity + Track E + Phase A ≈ **26–34 eng-days**; +offline pilot pack 5–7 days *with a real sample*; full Phase B +10–15 days, **not this round**.

| Week | Main deliverable | Required exit |
|---|---|---|
| W1 | close #498; #512 as path-list only; fix #513 live facts + split A/B; draft production-identity design-lock | **two design-locks ratified; NO runtime** |
| W2 | Phase A0: production unconditionally disable external `/model/reload`; start identity impl | caller path/force can't reach the loader; observed-RED |
| W3 | complete production identity membrane | any key, default token, missing/no-expiry JWT, forged user header — all fail-closed |
| W4 | Track E: portable manifest, provenance, content/family/label/side digest, conflict quarantine | byte-change, label-change, duplicate-content all RED |
| W5 | family/time holdout, non-empty both sides, max-component check, real/synth/augmented layering | fresh clone can recompute split |
| W6 | metrics + candidate proof: macro/per-class, calibration, false-duplicate, missed-reuse; CI dry-run | split-change must red; #509 stays closed |
| W7 | activation-surface enumerator, startup-load freeze, deploy safety checks | new unclassified loader reds CI |
| W8 | with lawful samples: offline pilot pack; without: ops/security close-out only (no features) | ingest→candidate→evidence→export; no training, no write-back |

**Customer line, parallel from W1:** 10 target manufacturers, 2 lawful-sample conversations, one named reviewer. **No lawful sample by W4 → cancel W7–8 pilot dev** (do not simulate).

**W1 status:** #513 A/B split done (`c65f952c`); #517 production-identity design-lock open; isolated critic running for ratification evidence. #512 already reduced to path-list; #498 refresh + owner ratifications pending. No runtime started.
