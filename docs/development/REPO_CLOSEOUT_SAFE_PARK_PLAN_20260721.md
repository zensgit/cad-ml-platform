# Repository Closeout — Safe-Park Plan (2026-07-21)

- **Status**: FOR REVIEW. Nothing in this document is self-ratifying; every gate below is
  owner-decided. This document **implements** the ratified strategy and design locks — it does
  **not** amend them. If anything here conflicts with `docs/PRODUCT_STRATEGY.md`, the strategy
  wins and this document is wrong.
- **Owner**: repository owner (sole ratifier/merger).
- **Code grounding**: `origin/main@7160694d` (post #526). PR states, branch protection, and
  seal states below were re-verified live on 2026-07-21, not restated from prior documents.
- **Authorship & routing disclosure**: assembled by the session agent (Fable) as a
  *mechanical transcription* of already-ratified decisions (#513 design lock,
  `PRODUCT_STRATEGY.md`, `L3_SAFETY_DESIGN_AND_VERIFICATION_20260715.md` Part 3) plus
  live-verified repository facts. Per the locked model-routing rule, Fable authors **no**
  security semantics and makes **no** completion judgment: all security verdicts in the plan
  below are routed to opus gate lenses, and only the owner declares any phase "done".

---

## 1. What "closeout" means for this repository

The ratified `PRODUCT_STRATEGY.md` already fixes the only three possible end states. They are
selected by **calendar and customer evidence, not by engineering progress**:

| End state | Trigger | Nature |
|---|---|---|
| **A — product continues** | Day-90 gate (~mid-Oct 2026) produces a measured pilot commitment (§8.4) | Not a closeout; entry into Year-1 (§9) |
| **B — fold into component** | Month-6 gate (~mid-Jan 2027) has no payment/contract → ratified kill criterion fires (§0, §9 Year-1) | Engine becomes an internal component of the chosen CAD/PLM product |
| **C — mothball** | Owner decision | Archive posture |

All three end states share one prerequisite intermediate state, which this plan names the
**engineering safe-park**:

> **Safe-park** = Phase A static fixed-hash containment complete + every dangerous path sealed
> + the honest posture documented, at a pinned SHA with CI evidence — indefinitely parkable,
> restartable without archaeology.

Safe-park is deliberately achievable **without** real data, signing keys, or a customer.
Therefore:

> **Closeout plan = drive the repo to safe-park (countable engineering, ~4–6 weeks)
> + execute the decision ladder (owner calendar actions, zero engineering).**

The only completion statement permitted at safe-park is the owner-locked sentence:

> *"Phase A static fixed-hash containment complete; external reload remains sealed; retraining
> remains disabled; Track E, signed proofs, and enablement are not complete."*

---

## 2. Layer 0 — clear the in-flight queue (target: week of 2026-07-21)

All 13 open PRs, dispositioned one by one. States verified live 2026-07-21.

| PR | Verified state (2026-07-21) | Disposition |
|---|---|---|
| **#527** Wave-1 reachability audit | head `7a16a057`, rebased onto `7160694d`, 62 pass / 0 fail, CLEAN, 0 threads | Merge. It is the C2 input (23 LIVE sites / 11 logical activations / denominator rule). |
| **#528** Phase-A C1 core | head `17ebaee2`; a local **un-pushed authority-fix** exists (`/private/tmp/cadml-c1-authority-fix`, +3857/−509 across 7 files, includes the fix for the gate-blocking vacuous partial-freeze test) | Builder pushes the authority-fix → **full fresh opus gate review** (the 2026-07-20 CONCERNS verdict does not transfer to the new delta; the openat2-seccomp-fallback P2 from codex-connector gets an explicit re-check) → owner ratifies exact SHA → merge. |
| **#529** finetune train-and-reload seal | head `eaa32158` unchanged since the GO verdict; 2 unresolved gemini nit threads are the only blocker (`required_conversation_resolution=true`) | Reply + resolve threads → owner merges. GO verdict remains valid while the head is unchanged. |
| **#525** fail-closed raw DWG analyze input | open; pairs with issue #524 (fail-open silent degradation) | L3 review, then merge. This is a **seal item** — it belongs to safe-park, not to backlog. |
| **#523** keep public PR jobs off self-hosted runners | CLEAN/MERGEABLE | Review, merge; closes the runner-routing environment debt. |
| **#507** portfolio strategy (cross-repo) | open for-review; carries 3 owner-ratify questions | Answer during Layer 4 (see §6) — it decides end-state B's fold-in target. Not a merge race. |
| **#497, #481, #394, #393, #392, #391, #389** dependabot ×7 | oldest open since 2026-05-18 | Batch-disposition: merge the CI-green ones, close the rest; fix issue **#476** (dependabot SHA-pin config) so future dependabot PRs pass Action Pin Guard instead of accumulating. |

Open issues: **#524** is addressed by #525; **#476** is addressed in the dependabot batch.
Layer-0 exit criterion: open-PR list contains only #507 (parked deliberately for Layer 4).

---

## 3. Layer 1 — finish Phase A → safe-park (3–5 weeks; target mid-Aug 2026)

### 3.1 Owner decisions that unblock the build (cheapest, highest leverage)

1. **Two-distinct-files KIND** for `part/v16-v6pt` (`PartClassifierV16._load_models` loads
   `cad_classifier_v6.pt` AND `cad_classifier_v14_ensemble.pt` in one activation): two
   tuples/one-id vs per-file pins vs bundle KIND. Blocks 4 of the 23 LIVE sites in C2 and 2 of
   the 11 C3 tuples. The same shape must be **recorded** for the UNMOUNTED
   `classifier_api.py::V16Classifier.load` (out of Phase-A scope).
2. **Reload-pathway id modeling**: one id (`pickle-classifier/reload`) vs two pins for the two
   call points of the sealed reload/rollback activation.
3. **"Complete" caliber for artifact-less LIVE families**: several LIVE activations have no
   artifact in the shipped image (`classifier_v1.pkl` absent; sentence-transformers not
   installed; `part` family default-disabled via `PART_CLASSIFIER_PROVIDER_ENABLED=false`;
   DeepSeek gated on `DEEPSEEK_HF_REVISION`). Recommended caliber (owner confirms): Phase-A
   complete = **wiring complete + no-artifact → degraded/503**; no manufactured fixtures
   pretending to be production artifacts. This decision shapes the C6 acceptance matrix.

### 3.2 Build sequence (builder implements; opus gates; owner ratifies per PR)

Per the ratified Part-3 decomposition and the Wave-1 denominator (23 LIVE sites → 11 logical
activations, non-LIVE sites recorded gate-before-wired and **not** wired):

| Block | Content | Shape |
|---|---|---|
| **C2** per-family wiring | Route the 11 LIVE logical activations through `load_pinned_file` / `load_pinned_bundle`. Risk-split: normal single-file families (graph2d, history, pointnet, vision3d-uvnet, part/v6) in ~2 PRs; pickle-classifier in its own opus-reviewed PR; OCR (3 ids) + embedding in 1–2 PRs. Each PR ships a remove-wrapper→RED discriminator. | 4–5 PRs |
| **C3** baseline manifest | 11 `(logical_activation_id, artifact_id, kind, digest)` tuples; not runtime-repointable; pre-Track-E tuple-field change = refused promotion. | 1 PR |
| **C4** degraded/503 | Missing/mismatched pin → explicit degraded capability, never a silent stub. | 1 PR (may ride with C3) |
| **C5** enumerator structural assertion | Raw loader outside the canonical wrapper (and not marked latent/unmounted/offline) → CI RED. | 1 PR |
| **C6** golden matrix + closeout MD | Full design-lock §5 Phase-A matrix as executed evidence; final Dev&V. | 1 PR |

Estimated volume: ~1,500–2,500 production lines + ~5,000–7,000 test lines + 4–6 Dev&V docs.
The binding constraint is **gate rounds, not typing** (observed history: #513 took ~10
NO-GO/fix rounds over 6 days; C1 took one full gate cycle plus an authority-fix pass).

Verification constraints carried from C1: no local Docker on the dev box → Linux-root /
uid-65534 / openat2 suites are CI-only; local green is never claimed as verification
(local ≠ CI).

---

## 4. Layer 2 — honest-posture inventory (parallel with Layer 1; docs only)

Safe-park's deliverable is a **SAFE-PARK CLOSEOUT MD** freezing the parked state of every known
§5 gap. Current states, re-verified in source on `origin/main@7160694d`:

| Strategy §5 gap | Parked state |
|---|---|
| §5.2 evaluation contamination (28.7% val/train byte-identical) | **Sealed**: `scripts/auto_retrain.sh` Step-0 runs the unconditional L3 gate (`scripts/eval_integrity_gate.py`) before any mutation; no pass path, no env toggle; re-enablement is a code change (Track E two-phase gate). Verified in source. Keep sealed. |
| §5.4 flywheel not closed (`src/api/v1/feedback.py` still JSONL placeholder) | **Do not build** (strategy §5.4 forbids resurrecting it outside a real reviewer workflow). Parked as-is. |
| §5.5 production auth | Production posture fail-closes the `test` defaults (`src/api/dependencies.py::_production_posture`, landed with the bleeding-control PR). The **full §8.3 pilot-release gates are pilot preconditions, not safe-park preconditions** — the closeout MD states this distinction explicitly so closeout does not silently double in scope. |
| §5.3 B-Rep | Untouched per strategy (sourcing start, not a moat). |
| External reload | `/model/reload` sealed 403; retraining disabled. |

The closeout MD must contain: exact SHAs, CI evidence links, residual risks (including the C1
honest residual: non-atomic mkdir+fd binding can leave a safe empty directory shell, zero model
bytes), and a **restart manual** (how Track E / Phase B resume when their supplies appear).

---

## 5. Layer 3 — supply-gated tracks: explicitly parked, never scheduled

These never enter the engineering schedule until their external supply exists. Pre-building any
of them is forbidden by strategy §6 and the design lock (the literal fake-green).

| Track | Missing supply | Consequence of faking it |
|---|---|---|
| **Track E** (evaluation-integrity-v2) | Real data + model-run environment | Metrics without them = fabricated metrics |
| **Phase B** (signed proofs) | Signing-key custody (HSM / human-gated signer outside CI) | Proofs without them = forged signatures |
| **Enablement gate 1** (Phase-A baseline-pin activation) | Owner-supplied §7.2 evidence: named target environment, named owner AND user, date, staging replay, observed-RED, rollback, kill switch, user-outcome telemetry, no paths in logs | Owner-only decision; rides on Phase A + Wave-1 |
| **Enablement gate 2** (dynamic swap / retraining) | Phases A–E complete + separate owner decision | Last gate; re-enable = replacing a body, never a flag |

Safe-park requires **none** of these — that is what makes it indefinitely parkable.

---

## 6. Layer 4 — product decision ladder (owner calendar; zero engineering)

Anchored on the strategy's ratification (last reviewed 2026-07-12):

| Date | Gate | Action |
|---|---|---|
| ~mid-Oct 2026 | **Day-90** (§8.4) | No measured pilot commitment → pause feature work + wedge review. |
| ~mid-Jan 2027 | **Month-6** (§0/§9) | No payment/contract → **end state B fires automatically** (already-ratified kill criterion; no new decision needed). |

**Recommendation**: do not let end-state B's fold-in target be decided under time pressure in
January. After safe-park lands (mid-Aug), answer #507's three owner-ratify questions — in
particular the system-of-record choice (Yuantus vs PLM-standalone). If end state B fires, the
remaining closeout work is then only: point the engine's stable decision contract (§3.3) at the
chosen shell + archive the commercial docs. No further engine code.

---

## 7. Governance

| Role | Holder | Scope |
|---|---|---|
| Ratify / merge / enablement / KIND & caliber decisions / #507 | **Owner** | Every gate above |
| C2–C6 implementation | Builder (codex) | Per Layer-1 sequence |
| Design-lock conformance gate per PR | Gate reviewer (session agent orchestrating **opus** lenses) | Verdicts authored by opus; a new head always invalidates the previous verdict |
| Mechanical transcription / doc sync / status reporting | Fable | Never security semantics, never completion judgment |

Worktree discipline: all work in isolated `/private/tmp/cadml-*` worktrees, never the canonical
checkout. CI truth: `gh pr checks` / `gh run view` conclusions on the exact SHA; never local
runs, never `gh run watch` exit codes.

## 8. What this plan explicitly does NOT do

- No Track E / Phase B pre-building (supplies absent — §5 above).
- Nothing from strategy §6's stop-building list (no new providers, no B-Rep breadth, no
  dashboards, no speculative adapters).
- No promotion of §8.3 pilot gates into safe-park scope (they are pilot preconditions).
- No completion claim beyond the locked sentence in §1.
- The session agent merges nothing, resolves no review threads, and ratifies nothing.

## 9. Timeline summary

| When | Milestone | Who |
|---|---|---|
| Week of 2026-07-21 | Layer 0 complete: #527/#529/#523/#525 merged; authority-fix pushed + gated; dependabot batch + #476 | Builder / opus gate / **owner merges** |
| By end of July | KIND ×2 + artifact-less caliber decisions | **Owner** |
| Late July → mid-Aug | C2 → C3/C4 → C5 → C6, each gated + ratified | Builder / opus / owner |
| **Mid-Aug 2026** | **Safe-park: closeout MD on `main` at a pinned SHA** | Owner ratifies |
| Late Aug | #507 three questions answered (fold-in target fixed) | **Owner** |
| ~Mid-Oct 2026 | Day-90 gate | Owner |
| ~Mid-Jan 2027 | Month-6 gate → end state A or B | Owner |
