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
| **E** Track E eval-integrity-v2 | 7× §8.1: manifest+provenance, content/family/label/side digests, quarantine, holdout, real/synth/aug metrics, two-phase candidate gate | ~15% | design ✅ · **runtime after gate** | Fable → Sonnet → Fable | **#512 landed + design-lock ratify** |
| **M** Full proof membrane (**#513 Phase B**) | signed proof store, revocation, LKG re-validation | 0% | design ✅ | Fable | **conditional** — only when a pilot needs dynamic model-swap |
| **P** Offline pilot/eval pack | ingest → candidate → deterministic evidence → reuse/revise/new export | 0% | ✅ after a real sample | Fable → Sonnet | first lawful customer archive |
| **C** Customer discovery / data agreement / pilot | named customer, sample rights, pilot commitment | no repo evidence | ❌ **owner-only** | — | owner |

---

## 3. Parallel DAG

```
Week 1        Weeks 2–4 (parallel)              Week 5      Week 6       Weeks 7–8
─────────     ────────────────────────          ───────     ────────     ──────────
G (design) ─┬─► E.design ─(gate)─► E.runtime ───────────────────────────► (E done)
            │
            ├─► I.design ─► I.runtime ─────────► I hardening ───────────► (I done)
            │
            └─► F.design (#513 Phase A) ──────────────────► F.runtime ──► (F done)

C (customer discovery) ═══════════ runs the whole time, owner-driven ═══════════►
                                                             P (pilot pack) ─► after first sample
M (#513 Phase B) … deferred until a pilot requires dynamic model-swap
```

- **Critical path:** `G → E.runtime` (E's runtime is the only track blocked on a governance gate).
- **Three independent design fronts start immediately and in parallel:** E.design, I.design, F.design. None blocks another.
- **Recommended deviation from the owner's week-numbering (flagged, default-on unless vetoed):** because the verification confirmed **default-open auth as the true P0** and it is fully independent of Track E, start **I (production identity) design + fail-first negative tests now, in parallel with Track E design** — rather than waiting for Week 5. This does **not** reorder any merge; it only front-loads the P0's for-review evidence.

---

## 4. Per-track deliverable cadence (each ends in design + verification MD)

Every track produces, per the owner's standing instruction:

1. **Design / design-lock MD** (Fable) — grounded on a pinned `origin/main` SHA, with the completeness principle enforced by a mechanism (e.g. the CI enumerator for F), not a hand-count.
2. **Implementation** (Sonnet) — fail-first: the failing/RED test is written and observed-RED **before** the passing implementation.
3. **Adversarial review** (Fable, default-refute) — the independent-critic pass; findings reproduced-or-refuted.
4. **Verification MD** — observed-RED evidence, positive controls, and the adversarial review's confirmed set.

### Track I — the immediate P0 (design highlights)
- **Boot-refuse in production:** when `ENVIRONMENT`/`APP_ENV` indicates production, refuse to start (or reject all requests) if `ADMIN_TOKEN`/`X-API-Key` are at the literal `test` default or `INTEGRATION_AUTH_MODE=disabled`. This is the missing guard (verification §1). Fail-first test: prod + default creds → boot/refuse; dev + defaults → unaffected.
- **Identity binding:** downstream identity derives from `auth_subject` (authentic), not the `x-user-id` header; delete or correctly wire the dead `create_api_actor_from_request` sink (verification §3).
- **Negative suite:** anonymous, forged header, wrong tenant, default-cred-in-prod — all must fail closed; dev/test opt-in stays explicit.

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

Author proceeds now on all **design** fronts (E/I/F) and Track I fail-first tests without waiting; runtime lands only as each gate opens.

---

## 6. Open decision for the owner

**Ordering of Track I (production identity).** Verification confirms it is the P0 and it is parallelizable. Default action (unless vetoed): begin I.design + fail-first negative tests now, alongside E/F design, landing for-review/default-off — without changing any merge sequence. If you prefer to hold I to Week 5 as originally scheduled, say so and it moves back.
