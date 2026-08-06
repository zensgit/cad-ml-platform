# Phase-A C2–C6 build handoff (2026-07-21) — self-contained brief for an unattended continuation

This branch (`l3-phasea-c2c6-wiring-20260721`) is **stacked on PR #528's head `6e645bc2`** (the C1
model-activation core, which passed a fresh 6-lens opus conformance gate: GO, 0 blocking, 0 runtime
holes). It is NOT on `main`. **Rebase onto `main` only after the owner merges #528.**

You (a continuation agent, possibly running headless with zero prior context) must finish Phase-A
C2–C6 on THIS branch. Read this whole file first; it is your only context.

## Hard governance rules (never violate)
- **You merge NOTHING. You ratify NOTHING. You resolve NO review threads.** Those are the repository
  owner's, always. Open a PR; leave it for the owner.
- **Do not touch** PR #527/#528/#529/#530, their branches, or `main`. Only commit to
  `l3-phasea-c2c6-wiring-20260721` (or a child branch of it) and open ONE PR targeting `main`,
  clearly marked "stacked on #528 — do not merge before #528."
- **Security-critical wiring must be authored/reviewed by an opus subagent**, not by a base model.
  Specifically the pickle-classifier family and the C5 enumerator structural proof and the C6 attack
  matrix. Spawn `Task`/Agent subagents with the opus model for those. Normal single-file family
  wiring may use sonnet.
- Local verification is not CI. State honestly what ran and what didn't. This box/cloud has no
  Docker; the repo's global `tests/conftest.py` fails to collect under some Python versions (PEP-604
  in FastAPI routers) — run C1/gateway tests with `--noconftest`. CI-on-Linux is the authority.

## Ratified scope decisions (owner, 2026-07-21) — do not re-litigate
1. `part/v16-v6pt` two files = **two SINGLE_FILE pins sharing one `logical_activation_id`**
   (`artifact_id` = `v6pt` and `v14ens`). Two `load_pinned_file` calls.
2. reload pathway = **one id** (`pickle-classifier/reload`), two call sites — LATENT, sealed 403,
   **NOT wired**, register-only. Not in the denominator.
3. Artifact-less LIVE families = **wiring complete + no-artifact ⇒ degraded/503**, never a raw-load
   fallback. Prod default posture = NO-PIN (service starts degraded).
4. **OCR/embedding (4 activations) = gate-before-wired, NOT wired in Phase-A.** They call third-party
   loaders (`from_pretrained` / `SentenceTransformer` / `PaddleOCR(det_model_dir=...)`) that need a
   PATH string, but C1's `FrozenBundle` is fd-only by design (the TOCTOU guarantee). Record them as
   "wireable pending C1 bundle-path support" — a different mechanism = a later Phase. **Phase-A
   completion denominator = 7 logical activations / 8 pins.**

## DONE on this branch (foundation, C3+C4)
- `src/core/model_activation/baseline_manifest.py` — `load_baseline_pins(source=None) -> tuple[PinRecord,...]`.
  Env `MODEL_ACTIVATION_BASELINE_MANIFEST`; unset/absent ⇒ `()` (NO-PIN); malformed/unreadable ⇒
  `ValueError` (fail-loud, never silent-empty); rejects dup keys + traversal relpath.
- `src/core/model_activation/activation_gateway.py` — process-wide lazy `ControlledStore`
  (`MODEL_ACTIVATION_STORE_ROOT`, pins, optional `MODEL_ACTIVATION_FREEZE_PARENT`).
  `activate_file(id, artifact_id) -> Optional[bytes]` / `activate_bundle(...) -> Optional[FrozenBundle]`:
  verified value on success, **`None` on ANY refusal or unconfigured store** (path-safe log), which is
  the universal "degrade this family" signal. Malformed manifest fails LOUD at bootstrap.
  `reset_gateway_for_tests()` provided.
- `tests/unit/test_activation_gateway.py` — 16 passing discriminating tests
  (`PYTHONPATH=. python3.11 -m pytest --noconftest tests/unit/test_activation_gateway.py -q`).

## C1 public API you will call (verified)
`from src.core.model_activation.activation_gateway import activate_file, activate_bundle`
- `activate_file(logical_activation_id, artifact_id) -> Optional[bytes]` — SINGLE_FILE.
- returns verified bytes, or `None` ⇒ the family must degrade (never raw-load).

## C2 — the 7 wireable activations (wire each; degrade on None)
Pattern: replace the raw `torch.load(path)` / `pickle.load(f)` with:
`data = activate_file(id, artifact_id)` → if `data is None`: take the family's EXISTING degraded
branch (return None / model_unavailable / fallback) — never raw-load. Else reconstruct from bytes:
`torch.load(io.BytesIO(data), map_location=..., weights_only=<as today>)` or `pickle.loads(data)`.

| id | artifact_id | file : load line(s) | model to use |
|---|---|---|---|
| pickle-classifier/main | main | src/ml/classifier.py:85 (pickle.load) | **opus** (high-risk) |
| graph2d/main | main | src/ml/vision_2d.py:136 (torch.load), :163 load_state_dict | sonnet |
| history/sequence | main | src/ml/history_sequence_classifier.py:162, :203 | sonnet |
| vision3d-uvnet/main | main | src/ml/vision_3d.py:196, :220 | sonnet |
| pointnet/main | main | src/ml/pointnet/inference.py:108, :116-130 | sonnet |
| part/v6 | main | src/ml/part_classifier.py:62, :79 (gate src/core/analyzer.py:99) | sonnet |
| part/v16-v6pt | v6pt | src/ml/part_classifier.py:655, :683 (gate analyzer.py:70) | sonnet |
| part/v16-v6pt | v14ens | src/ml/part_classifier.py:695, :698 | sonnet |

Notes: part/v6 honors `CAD_CLASSIFIER_MODEL` env; part/v16 v6 component is hardcoded — keep them as
SEPARATE pins (they can diverge). pointnet/history default env paths are empty (family off unless
configured) — that is fine, they degrade. Each family already has a degraded branch today; route the
`None` case into it and REMOVE any raw-load fallback so a missing/tampered artifact cannot load
unverified. Each wired family needs a discriminating test (pin-absent ⇒ degrade; a fixture pin ⇒
success bytes; digest-tamper ⇒ degrade).

Correctness findings to fix while wiring (do NOT let a pin paper over them, and do NOT expand scope
beyond noting/fixing the fail-closed behavior): `src/core/ocr/providers/paddle.py:206` fabricates OCR
text when the model is absent (confidently-wrong, violates decision #3 — but OCR is out of C2 scope
per decision #4, so only record it); `pointnet/inference.py:128` silently random-inits the feature
extractor if `extractor_state_dict` absent (record).

## C5 — enumerator structural assertion (opus)
Extend the activation-surface enumerator so a raw loader (`torch.load`/`pickle.load[s]`/`joblib.load`)
that is NOT inside the canonical `activate_file`/`activate_bundle` path (or a wrapper that calls it)
AND not marked latent/unmounted/offline ⇒ CI RED. Ship a remove-the-wrapper→RED discriminator. Keep
the manifest counts consistent (currently 129 sites / 38 gated / 11 families per the last enumerator
run; adding the gateway wrapper must not silently drop a gated site). The enumerator becomes
**blocking only after all 7 in-scope live activations are wired** (per the ratified W4).

## C6 — golden matrix + Dev&V MD (opus)
Full Phase-A §5 golden matrix as EXECUTED evidence (a small legit fixture proves the fixed-pin SUCCESS
path; the RED cases: pin-absent⇒degrade, digest/tree tamper⇒degrade, symlink/`..` escape, single-file
& bundle TOCTOU, caller-path/env-swap/hot-swap reject, count/size over-limit, new un-annotated loader
CI RED). Then a Dev&V MD `docs/development/L3_PHASE_A_C2_C6_DEV_AND_VERIFICATION_20260721.md` with:
exact base/head SHAs, per-family wiring diff summary, observed-RED evidence, the 7-activation
denominator + the 4 gate-before-wired OCR/embedding activations with the fd-vs-path rationale,
residual risks, and the honest completion statement (verbatim):
> "Phase A static fixed-hash containment complete; external reload remains sealed; retraining remains
> disabled; Track E, signed proofs, and enablement are not complete."

## When you finish
Open ONE PR from this branch to `main`, titled "feat(l3): Phase-A C2–C6 model-activation wiring
[stacked on #528 — do not merge before #528]", body summarizing the above + explicitly stating the
owner must (a) merge #528 first, (b) rebase this onto main, (c) ratify. **Do not merge it.** Post the
C6 verdict as the PR body; leave ratification to the owner.
