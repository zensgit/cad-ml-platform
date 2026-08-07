# Repository Closeout — Safe-Park Plan (2026-07-21; Layer-0 refresh 2026-08-07)

- **Status**: FOR REVIEW (Layer-0 refreshed after Track-1 batch). Nothing in this document is
  self-ratifying; every gate below is owner-decided. This document **implements** the ratified
  strategy and design locks — it does **not** amend them. If anything here conflicts with
  `docs/PRODUCT_STRATEGY.md`, the strategy wins and this document is wrong.
- **Owner**: repository owner (sole ratifier of product/strategy gates). Engineering merges
  happen only under explicit owner authorization for the named PRs.
- **Code grounding**: `origin/main@0ffe6ce5` (post #531/#527/#523 batch on 2026-08-07; also
  includes #525/#539/#538/#537/#536/#535/#532/#529/#528). Layer-0 table below was rewritten
  live against that tip — **do not trust the 2026-07-21 open-queue snapshot**.
- **Authorship & routing disclosure**: original 2026-07-21 draft assembled by the session agent
  (Fable) as a *mechanical transcription* of already-ratified decisions (#513 design lock,
  `PRODUCT_STRATEGY.md`, `L3_SAFETY_DESIGN_AND_VERIFICATION_20260715.md` Part 3). 2026-08-07
  Layer-0 refresh is a fact update only (merged SHAs / still-open queue); it does **not** claim
  safe-park complete and does **not** amend strategy end states.

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

## 2. Layer 0 — clear the in-flight queue

### 2.1 Historical target (week of 2026-07-21) — **SUPERSEDED**

The original Layer-0 table assumed a 13-PR queue at `main@7160694d` and an exit criterion of
“open list contains only #507”. That snapshot is **false as of 2026-08-07**. Do not use it for
merge planning. Facts below replace it.

### 2.2 Merged since the original Layer-0 draft (verified on `main@0ffe6ce5`)

| PR | What landed | Merge on main (approx) |
|---|---|---|
| **#528 / #532** | Phase-A model activation membrane (C1 + C2–C6 wiring / golden path) | 2026-07 → early Aug |
| **#529** | finetune train-and-reload seal | post-#528 |
| **#536** | assistant hosted-provider dependency lock | 2026-08-06 |
| **#535** | assistant SEAL **design / docs** only (commit: “Docs-only. Does not implement runtime SEAL”) | 2026-08-06 |
| **#537** | assistant provider **runtime SEAL** on live path (incl. §2.C `TOOL_REGISTRY` canonical status / `tool_status` contract) | 2026-08-06 |
| **#538** | production identity fail-closed (#517 design-lock runtime) | 2026-08-07 `191ce7ef` |
| **#525** | fail-closed raw DWG analyze input (SECTION structure + empty-stub guard) | 2026-08-07 `e98f7e12` |
| **#539** | OCR/PointNet honesty (no fabricated OCR; refuse random PointNet weights) | 2026-08-07 `2dd6225b` |
| **#531** | Track E evaluation-integrity-v2 **design-lock** (docs only; unlocks E1) | 2026-08-07 `c26bfe49` |
| **#527** | Wave-1 reachability audit (docs; gated=38 enumerator alignment) | 2026-08-07 `63f1f928` |
| **#523** | keep public PR jobs off self-hosted runners (CI) | 2026-08-07 `0ffe6ce5` |

### 2.3 Still open (live 2026-08-07) — disposition

| PR | Verified state | Disposition |
|---|---|---|
| **#530** (this plan) | open; Layer-0 rewritten this commit | Re-review after Layer-0 refresh → owner merges when accurate. |
| **#507** portfolio strategy | open for-review; 3 owner-ratify questions | **Park at Layer 4** (§6). Not a merge race. |
| **#534** pilot DWG default-disabled | **draft** | Finish draft → L2 review; not Layer-0 blocking. |
| **#497, #481, #394, #393, #392, #391, #389** dependabot ×7 | oldest open since 2026-05-18 | Still valid batch debt: merge CI-green ones, close the rest; fix issue **#476** (dependabot SHA-pin) so Action Pin Guard stops the pile-up. |

Open issues: **#524** addressed by #525 (merged); **#476** remains for the dependabot batch.

### 2.4 Layer-0 exit criterion (updated)

Layer-0 is **cleared for the original 2026-07-21 implementation queue** (the PRs that plan once
listed as in-flight seals/audits). Remaining open items are intentionally **not** Layer-0:

- **#507** — Layer 4 product decision (owner only).
- **#530** — this document (merge after re-review of the refresh).
- **dependabot ×7 + #476** — maintenance batch (parallel, not strategy-blocking).
- **#534 draft** — pilot safety follow-up.

False claim **removed**: “open-PR list contains only #507” was never re-verified after
Phase-A / identity / SEAL landings and is not a current exit criterion.

---

## 3. Layer 1 — finish Phase A → safe-park (3–5 weeks; target mid-Aug 2026)

> **Status note (2026-08-07):** Phase-A static fixed-hash containment and related seals listed
> in §2.2 are **on `main`**. Layer 1 is no longer “start C1 from zero”; remaining safe-park work
> is residual inventory (Layer 2 closeout MD), owner KIND/caliber decisions still open in §3.1
> if not already recorded in the Phase-A Dev&V docs, and **not** reopening sealed reload paths.
> Track E **design-lock #531 is ratified**. Per that lock, **Slice E1** (torch-free dry-run /
> reporting only) is **not** blocked on real data — its start conditions were (1) design-lock
> ratify and (2) a free L3 PR slot; both are now true. Real data / model-run supply remains the
> gate only for **invariant H and the two-phase release path** (explicitly out of E1). See §5.

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
§5 gap. Current states, re-grounded on `origin/main@0ffe6ce5` (Layer-0 refresh 2026-08-07):

| Strategy §5 gap | Parked state |
|---|---|
| §5.2 evaluation contamination (28.7% val/train byte-identical) | **Sealed**: `scripts/auto_retrain.sh` Step-0 runs the unconditional L3 gate (`scripts/eval_integrity_gate.py`) before any mutation; no pass path, no env toggle; re-enablement is a code change (Track E two-phase gate). Verified in source. Keep sealed. |
| §5.4 flywheel not closed (`src/api/v1/feedback.py` still JSONL placeholder) | **Do not build** (strategy §5.4 forbids resurrecting it outside a real reviewer workflow). Parked as-is. |
| §5.5 production auth | **Hardened further on main via #538**: production-identity fail-closed (`src/api/production_identity.py`, boot refuse, JWT aud/iss/exp, identity from `sub` only; harness opt-in only under `ENVIRONMENT=development|test`). Bleeding-control remains the earlier partial. The **full §8.3 pilot-release gates are pilot preconditions, not safe-park preconditions** — the closeout MD states this distinction explicitly so closeout does not silently double in scope. |
| §5.3 B-Rep | Untouched per strategy (sourcing start, not a moat). |
| External reload | `/model/reload` sealed 403; retraining disabled. |

The closeout MD must contain: exact SHAs, CI evidence links, residual risks (including the C1
honest residual: non-atomic mkdir+fd binding can leave a safe empty directory shell, zero model
bytes), and a **restart manual** (how Phase B / enablement / post-E1 model-run resume when their
supplies appear; how E1 resumes as ordinary L3 engineering once a slot is free).

---

## 5. Layer 3 — what is still parked vs what is now unblocked

This section must **not contradict** the ratified Track E design-lock
(`docs/development/L3_TRACK_E_EVALUATION_INTEGRITY_V2_DESIGNLOCK_20260721.md`, merged via #531).
This closeout plan **implements** design locks; it does not amend them.

### 5.1 Track E — split by design-lock (not a single “wait for data” gate)

| Slice | Status (2026-08-07) | What blocks / does not block | Fake-green risk |
|---|---|---|---|
| **E1** dry-run / reporting only (torch-free; invariants A–G; discriminators 1–14; **no** import of `eval_integrity_gate`; cannot mint unlock; `release_eligible:false` / `unlocks_retraining:false`) | **Unblocked to author** | Design-lock start conditions: **(1) #531 ratify** ✅ **(2) L3 PR slot free** ✅. **Does not require real data or a model-run environment.** | Claiming full §8.1 exit (“fresh clone reproduces the **evaluation result**”) or any retrain unlock from E1 outputs |
| **Invariant H + two-phase release / real metrics** (§8.1.4; explicitly **NOT in E1**) | Still parked | Real holdout metrics over model-run (torch + data) + later owner decision on release gate | Fabricated metrics / forged release eligibility |

E1 is **one** implementation PR and the line’s **single** L3 runtime PR while open. Closed #510/#511
are **not** revived; rebuild from `main` against the ratified lock.

### 5.2 Still fully supply-gated (do not schedule until supply exists)

| Track | Missing supply | Consequence of faking it |
|---|---|---|
| **Phase B** (signed proofs) | Signing-key custody (HSM / human-gated signer outside CI) | Proofs without them = forged signatures |
| **Enablement gate 1** (Phase-A baseline-pin activation) | Owner-supplied §7.2 evidence: named target environment, named owner AND user, date, staging replay, observed-RED, rollback, kill switch, user-outcome telemetry, no paths in logs | Owner-only decision; rides on Phase A + Wave-1 |
| **Enablement gate 2** (dynamic swap / retraining) | Phases A–E complete + separate owner decision | Last gate; re-enable = replacing a body, never a flag |

Safe-park requires **none** of Phase B / enablement — that is what makes it indefinitely parkable.
E1 is **orthogonal** to safe-park completion: it may proceed under L3 discipline without claiming
safe-park or retrain enablement.

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

- No Phase B pre-building (signing supply absent — §5.2). No Track E **fake metrics / release
  unlock** (invariant H + two-phase gate stay deferred — §5.1). **E1 dry-run is not “supply-blocked”**;
  once a free L3 slot is used, it is ordinary gated engineering against #531, not an enablement claim.
- Nothing from strategy §6's stop-building list (no new providers, no B-Rep breadth, no
  dashboards, no speculative adapters).
- No promotion of §8.3 pilot gates into safe-park scope (they are pilot preconditions).
- No completion claim beyond the locked sentence in §1.
- No self-ratify of product gates; merges of strategy/design-lock docs and L3 runtimes only under
  explicit owner authorization for the named PR (session agent does not invent ratify).

## 9. Timeline summary

| When | Milestone | Who |
|---|---|---|
| Week of 2026-07-21 | Original Layer-0 target (historical) | — |
| **2026-08-06…07** | Layer-0 implementation queue cleared: Phase-A membrane, seals, #538 identity, #525/#539 honesty, #531 design-lock, #527 audit, #523 runners — on `main@0ffe6ce5` | Builder / gate / **owner-authorized merges** |
| Next | Re-review + merge **#530** (this refreshed plan); dependabot batch + #476; finish or close **#534** draft | Owner / builder |
| Residual | Safe-park closeout MD (Layer 2 inventory) at a pinned SHA if any §5 gaps remain undocumented | Owner ratifies |
| **Now (slot free)** | **Track E E1** may start: torch-free dry-run from `main` per #531; single L3 runtime WIP; never unlocks retrain; H/real metrics still deferred | Builder / gate / owner |
| Late Aug+ | #507 three questions answered (fold-in target fixed) — **not** merge-driven | **Owner** |
| ~Mid-Oct 2026 | Day-90 gate | Owner |
| ~Mid-Jan 2027 | Month-6 gate → end state A or B | Owner |
