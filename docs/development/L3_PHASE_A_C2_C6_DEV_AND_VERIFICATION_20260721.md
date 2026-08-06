# L3 Phase-A C2–C6 — Development & Verification record (2026-07-21)

> **CURRENT-STATE HEADER (2026-07-22, supersedes conflicting statements below).**
> Latest head on this branch is **`4d9ee6d6`** (round-4: C5 scope-shadowing
> fail-closed + `vector_pipeline` missing-marker fail-closed — the work §16
> below describes as "uncommitted on top of `ae332c5e`" **is now committed**
> as this head). Live CI on `4d9ee6d6` (PR #532), using the same de-duped
> check-run methodology as §0/§15.2 (the raw `statusCheckRollup` shows 83
> items, but 4 of those are `Unit Tests (Shard 1-4)` re-run duplicates within
> the same workflow — the same double-count §0 already flags — and collapse
> to 79 distinct check-runs): **64 pass / 15 skip / 0 fail**, 79 check-run
> items total, unchanged from the `ae332c5e` rollup because round-4 touched
> no workflow YAML. Every statement below that says round-4 is "not yet
> committed" or that head is `ae332c5e` is a **superseded, point-in-time
> record** — accurate for what had landed when it was written, retained below
> as history, and not rewritten. Round-5 (C5 lexical-scope import env +
> buffer-canonical resolution) is the current in-flight round on top of
> `4d9ee6d6`; a follow-up commit will carry it and the head will advance
> again. Treat every CI number in this document as
> **head-accurate-as-of-`4d9ee6d6`** unless a later current-state note says
> otherwise.
>
> **R5 CLOSURE ADDENDUM (2026-07-22, supersedes the header above where it
> conflicts).** Round-5 (C5 lexical-scope import env + buffer-canonical
> resolution), described as "in-flight" immediately above, **is now
> committed** as head **`3c33dc0a`** (`fix(l3): round-5 NO-GO remediation on
> 4d9ee6d6 — C5 lexical-scope import env + buffer canonical`). Live CI on
> `3c33dc0a` (PR #532, same head the round-5 commit landed on): raw
> `statusCheckRollup` shows **83** items / **68** passing raw entries; deduping
> the 4 `Unit Tests (Shard 1-4)` re-run duplicates the same way §0/§15.2/§16.2
> already do yields the authoritative headline **64 pass / 15 skip / 0 fail,
> 79 distinct check-runs** — identical to the `4d9ee6d6` rollup above, because
> round-5 touched no workflow YAML. Its findings (gateway-name lexical
> scoping, buffer-wrapper canonical resolution, doc-R4-status) were accepted
> by the reviewer on this live head. The "in-flight" wording above and in §16
> is a superseded, point-in-time record and is **not rewritten**, only
> superseded. See §16.3 and **§16.4** for the full account. Round-6
> (loader-alias lexical discovery) is **no longer in-flight** — it is committed
> as head **`405d141d`** with live CI terminal (68 pass / 15 skip / 0 fail); see
> §16.4. This document's inherent one-commit lag (it is committed as part of
> advancing the head) is flagged in §16.3/§16.4.

## 0. Scope & honesty frame

This record documents the C2 family wiring, the C3 baseline manifest, the C4
degraded contract, the C5 activation-surface enumerator structural check, and
the **C6 golden matrix** for the cad-ml-platform model-activation membrane, on
branch `l3-phasea-c2c6-wiring-20260721`.

Honesty frame — read this before trusting any "green" below:

- **Local is not CI.** This box has **no Docker**, so the Linux-root / `openat2`
  suites in the C1 core are **CI-only** and were not (could not be) run here.
- The repo-global `tests/conftest.py` fails to collect under the local Python
  (PEP-604 `X | Y` annotations in FastAPI routers), so every run below uses
  `--noconftest`. That is a local harness workaround, **not** how CI runs.
- **torch is installed only under the box's `/usr/bin/python3` (3.9.6)**; the
  sandbox `python3.11` has **no torch**. Torch-dependent tests therefore run
  under 3.9.6 and *skip* (not fail) under 3.11 via `pytest.importorskip`.
- **CI-on-Linux is the authority.** The local results in §9 and §15.1 are
  **pre-push evidence** — what ran locally, on which interpreter, with what
  result — for the commits that produced them. They are not a substitute for
  the live CI rollup (§15.2), which is the authoritative outcome for the
  pushed head.

This branch is **stacked on PR #528** and is **not on `main`**. It merges
nothing, ratifies nothing, resolves no review threads — those are the owner's.

**Post-push update (this record).** The round-1/round-2/round-3 work described
in §§1–15 is committed and pushed; head is `ae332c5e` (see §1). Live CI ran on
that head: **64 pass / 15 skip / 0 fail** (the reviewer's authoritative rollup
counts **79** check-run items total — not 83; an earlier 83 double-counted
re-run check-runs and is corrected here). §15's "not committed" / "working
tree" language below describes the state *at the time those sections were
written*, i.e. before this push — it is not rewritten, only annotated, so the
record does not contradict itself. A further, separate round of fixes (§16,
"round-4") was made **on top of** `ae332c5e` in the working tree after this
push and is **not yet committed**; see §16 for its own, explicitly-labelled
pre-commit status.

## 1. Base / head SHAs and governance

| | value |
|---|---|
| Base (C1 core) | `6e645bc2d23b9ea54ae3f44504006511b6dc0525` |
| Base provenance | PR #528 head — C1 model-activation core; passed a fresh 6-lens opus conformance gate (**GO**, 0 blocking, 0 runtime holes) |
| Branch | `l3-phasea-c2c6-wiring-20260721` |
| Foundation commit (C3+C4) | `c9696f1160ad395f35cb0d6066e6cd64c6c43798` |
| C2/C5/C6 delivery | committed and pushed through round-3 remediation, head `ae332c5e13408c3e01da739065cb8e8e0de135f2` (family sources, C2 unit tests, enumerator, this record) |
| Live CI on `ae332c5e` | **64 pass / 15 skip / 0 fail** (reviewer's authoritative rollup, 79 check-run items total) — this is the **round-4 pre-fix baseline**; see §16 |

Landing requires the owner to: **(a) merge #528 first, (b) rebase this onto
`main`, (c) ratify.** Do not merge before #528.

## 2. C1 public API used (verified)

```python
from src.core.model_activation.activation_gateway import activate_file, activate_bundle
# activate_file(logical_activation_id, artifact_id) -> Optional[bytes]
#   verified bytes on success, or None on ANY refusal → the family MUST degrade,
#   NEVER raw-load. A malformed-manifest bootstrap ValueError propagates (loud).
```

Every wired family replaces its raw `torch.load(path)` / `pickle.load(f)` with
`data = activate_file(id, artifact_id)`; on `None` it takes the family's
existing degraded branch; otherwise it reconstructs from bytes
(`torch.load(io.BytesIO(data), …)` / `pickle.loads(data)`). No family reads a
model path directly any more (Phase-A decision #3).

## 3. Per-family C2 wiring table (the 7 in-scope activations / 9 pins)

Denominator = **7 logical activations / 9 pins** wired. Degrade = the `None`
path routes into the family's pre-existing unavailable branch; a
missing/tampered/unpinned artifact can never load unverified.

| logical id | artifact | source : load site | degraded behavior | C2 unit test |
|---|---|---|---|---|
| `pickle-classifier/main` | `main` | `src/ml/classifier.py` (`pickle.loads` of gateway bytes) | model unavailable / fallback | `test_c2_pickle_classifier.py` (opus-authored) |
| `graph2d/main` | `main` | `src/ml/vision_2d.py:144` `_load_model` | `model=None`, `_loaded=False`, `_load_error=None` (benign degrade, like uvnet/pointnet) | `test_c2_graph2d.py` |
| `history/sequence` | `main` | `src/ml/history_sequence_classifier.py:165` `_load_model` | degrade to prototype scorer (`source != history_sequence_model`) | `test_c2_history_sequence.py` |
| `vision3d-uvnet/main` | `main` | `src/ml/vision_3d.py:196` `_load_model` | mock encoder; `_load_error` stays `None` on benign degrade | `test_c2_vision3d_uvnet.py` |
| `pointnet/main` | `main` | `src/ml/pointnet/inference.py:102` `_try_load_model` | fallback mode; `_load_error` stays `None` on benign degrade | `test_c2_pointnet.py` |
| `part/v6` | `main` | `src/ml/part_classifier.py:60` `_load_model` (gate `analyzer.py`) | `raise RuntimeError("part/v6 … unavailable")` → analyzer falls back to V6/rules | `test_c2_part.py` |
| `part/v16-v6pt` | `v6pt` | `src/ml/part_classifier.py` `_load_models` | all-or-nothing `RuntimeError` (owner decision 1) | `test_c2_part.py` |
| `part/v16-v6pt` | `v14ens` | `src/ml/part_classifier.py` `_load_models` | all-or-nothing `RuntimeError` | `test_c2_part.py` |
| `part/v16-v6pt` | `v16config` (F5) | `src/ml/part_classifier.py` `_load_models` (config json → ensemble weights; `store_relpath` `models/cad_classifier_v16_config.json`) | all-or-nothing `RuntimeError` | `test_c2_part.py` |

Notes: `part/v16-v6pt` is **three SINGLE_FILE pins sharing one logical id**
— `v6pt` + `v14ens` (owner decision 1) **plus the F5 `v16config`** ensemble-config
pin. The config json sets `self.v6_weight`/`self.v14_weight` (it combines the V6
and V14 predictions), so even with both checkpoints pinned, editing it changes
outputs — it is itself a weight-bearing artifact and is pinned as a third
SINGLE_FILE artifact. **Ratified this review round:** decision-1's two-pins-one-id
shape is extended to three (`v6pt`, `v14ens`, `v16config`) under the same
logical activation id, all-or-nothing, NOT a bundle — this is the owner ruling
that also moves the family/pin denominator from 8 to **9 pins** (§3). Absent/tampered
config degrades the WHOLE V16 family — never a silent proceed with the default weights
(consistent with F4, design-lock line 403). The `pickle-classifier/reload`
pathway is **latent, sealed 403, register-only, not wired, not in the
denominator** (owner decision 2).

## 4. C3 baseline manifest mechanism

`src/core/model_activation/baseline_manifest.py` — `load_baseline_pins(source=None)`:

- **Default NO-PIN.** Env `MODEL_ACTIVATION_BASELINE_MANIFEST` unset, or the
  manifest file absent → `()` (empty). The production default posture: the
  store has zero pins and every activation degrades. It **never** falls back to
  an unverified raw load.
- **Fail LOUD on malformed.** A present-but-unreadable / non-JSON / non-list /
  bad-schema / bad-digest / duplicate-key / traversal-`store_relpath` manifest
  raises `ValueError`. A corrupt manifest must never masquerade as "no pins."
- Schema locks the full `(logical_activation_id, artifact_id, kind, digest,
  store_relpath)` tuple; unknown fields rejected; `store_relpath` re-validated
  against the raw-pin domain (defense-in-depth for attacker-influenceable config).
- **Path-safe:** errors identify the offending entry by **list index only** —
  no filesystem path in any message or log.

## 5. C4 degraded contract

`src/core/model_activation/activation_gateway.py` owns a process-wide, lazily
built `ControlledStore`:

- **UNCONFIGURED store** (`MODEL_ACTIVATION_STORE_ROOT` unset) → store never
  built; `activate_*` returns `None` (degraded); the manifest is not even
  parsed. This is the default production posture; it does **not** raise.
- **CONFIGURED store** → the baseline manifest loads; a malformed manifest
  raises `ValueError` at bootstrap and that error **propagates through**
  `activate_*` (fail LOUD, never swallowed to `None`).
- On **any refusal** (pin absent, artifact missing, digest mismatch, kind
  mismatch, symlink/escape, bounds, store unconfigured) `activate_*` returns
  `None` — the **universal "degrade this family" signal**.
- Degraded logs emit a **path-safe reason** (the `RefusalReason` value, or the
  `store_unconfigured` sentinel) plus the logical/artifact ids; never a
  filesystem path.

## 6. C5 enumerator structural check

`scripts/ci/activation_surface_enumerator.py` + `activation_surface.json`:

- **Completeness by construction:** a new un-annotated model-load site
  (`torch.load` / `pickle.load[s]` / `joblib.load` / import-aware
  `from_pretrained` / `SentenceTransformer` / `PaddleOCR` / `onnx.load` /
  `InferenceSession`) that is not classified in the manifest → **CI RED** (exit
  1); manifest/parse malfunctions fail closed at **exit 2**, distinct from a
  finding.
- **Structural wiring check (ratified W4):** a `wired` raw loader must
  reconstruct from `activate_file`/`activate_bundle` bytes. If the gateway
  wrapper is removed and it reads straight off a path again, the check goes RED
  — the **remove-the-wrapper discriminator**
  (`test_remove_the_wrapper_is_observed_RED_under_enforce`).
- **Advisory-by-default; blocking under `ACTIVATION_ENFORCE_WIRING`.** Per W4 it
  is present-but-advisory (printed, exit unchanged) until the owner flips the
  env flag after all in-scope live activations are wired; the inverse lies
  (gate-before-wired-but-wrapped; gated-without-a-wiring-field) also RED under
  enforce.
- **The real tree passes enforce mode green today**
  (`test_real_tree_is_structurally_consistent_under_enforce`): every `wired`
  raw loader routes through the gateway and every `gate-before-wired`/`latent`
  loader is still a raw load — so the owner can flip enforce ON safely.
- Current manifest inventory: **129 sites** — 38 `gated` (**18 wired**, 18
  `gate-before-wired`, 2 `latent`), 44 `producer`, 40 `offline`, 4 `infra`, 3
  `unmounted` — across **11 families**.
- **Round-3 strengthening (the enumerator's own false-green): a gateway call
  now must RESOLVE to the canonical gateway module, not merely match a bare
  name.** Two holes existed in the "wired" check itself: (1) `_is_activation_call`
  matched on the bare spelling `activate_file`/`activate_bundle` regardless of
  where that name came from — a project-local FAKE or a shadowing
  `def activate_file(...)` / `activate_file = lambda ...` counted as gateway-derived
  and would have scored `wired`/`wrapped=True` even though it never touched
  `src.core.model_activation.activation_gateway`. This is now resolved through
  the file's own import table (`gw_names` for the bare/aliased
  `from ...activation_gateway import activate_file [as x]` form, `gw_modules`
  for the `import ...activation_gateway as gw` / attribute-call form) — only a
  name that traces back to the one canonical gateway module counts.
  (2) Cross-scope dominance was unsound: an outer-scope gateway binding was
  relaxed to "always dominates" and inherited into every nested closure, so a
  nested function/lambda that calls a raw loader BEFORE (or independent of) an
  outer `x = activate_file(...)` binding could still score `wrapped=True` —
  the check cannot actually prove call order across scopes. Each function scope
  now carries only its OWN gateway bindings (fail-closed: a raw loader inside a
  nested scope is `wrapped` only if a gateway binding within that SAME scope
  dominates it). Both fixes are enumerator/structural-check-only — no
  `activation_surface.json` classification or C1/C2 runtime code changed.
  Covered by the expanded cases in `tests/unit/test_activation_surface_enumerator.py`.

## 7. C6 golden matrix results

`tests/unit/test_c2_golden_matrix.py` — the Phase-A §5 golden matrix **executed
at the WIRING level** through a real wired family (`PartClassifier` /
`part/v6`), so it proves the *family* activates verified bytes and degrades on
refusal, not merely that the store layer refuses. It does **not** duplicate the
C1-internal REDs; it references them (see §7.2).

### 7.1 Executed rows (all pass under `/usr/bin/python3`)

| row | mechanism | asserted outcome |
|---|---|---|
| **GREEN** fixed-pin | valid checkpoint pinned + digest-locked | loads the exact verified bytes through `activate_file` **and produces real inference output** (deterministic argmax → `cat_b`, conf > 0.9, valid probability distribution) — proven end-to-end through the family `__init__`, not the store layer |
| **RED** pin-absent | store configured, no pin for `part/v6` | `activate_file → None` (PIN_ABSENT); family raises; on-disk decoy never read |
| **RED** store-unconfigured | no `MODEL_ACTIVATION_STORE_ROOT` | `activate_file → None` (store_unconfigured); family raises; decoy never read |
| **RED** digest-tamper | bytes swapped on disk after pinning | `activate_file → None` (DIGEST_MISMATCH); tampered bytes never returned; family raises; decoy never read |
| **RED** wrong-kind | pinned as BUNDLE, activated as file | `activate_file → None` (KIND_MISMATCH, refused before a byte is read); family raises; decoy never read |

Every RED places a **real, loadable** checkpoint at `model_path` and proves the
family still **raises** rather than reading it — i.e. the refusal came from the
gateway, never a raw path load (decision #3). The four REDs degrade
**identically** despite four different refusal reasons.

### 7.2 Referenced (not re-executed) coverage — with executable pointers

Two pointer tests parse the sibling suites' ASTs and fail if a referenced test
is renamed/removed, so the §5 coverage claim is anchored to real tests:

- **C1 core** (`test_model_activation_c1_core.py`) — escape/symlink/path-swap
  (`test_intermediate_symlink_refused`, `…_leaf_symlink_refused…`,
  `…_parent_swap_to_symlink_mid_walk_refused`,
  `…_bundle_symlink_member_red_pass1_no_freeze_created`); same-fd TOCTOU
  (`test_same_fd_toctou_returns_hashed_bytes_not_reread`,
  `…_growing_file_refused_on_same_fd`, `…_inode_swap_after_open…`,
  `…_freeze_inplace_mutation_red`, `…_freeze_path_redirect_red`); bounds/bombs
  (`…_oversized_single_file_red`, `…_bundle_{directory,dirent,depth,relpath}_bomb_red`,
  `…_bundle_file_count_red`, `…_bundle_per_file_bytes_red`); wrong-kind both
  API directions + digest (`test_wrong_kind_single_file_api_on_bundle_pin`,
  `…_bundle_api_on_single_file_pin`, `…_single_file_digest_mismatch_red`,
  `…_bundle_digest_mismatch_red`) → `test_referenced_c1_reds_are_documented`.
- **C5 enumerator** (`test_activation_surface_enumerator.py`) — new
  un-annotated loader → CI RED (`test_new_unclassified_load_site_reds`), the
  remove-the-wrapper structural RED under enforce
  (`test_remove_the_wrapper_is_observed_RED_under_enforce`) and its
  advisory-by-default control, and the real-tree-passes-enforce control
  (`test_real_tree_is_structurally_consistent_under_enforce`) →
  `test_referenced_c5_discriminators_are_documented`.

## 8. The updated pre-existing tests (decision #3 fallout)

Three pre-existing tests asserted the OLD raw-load-by-path behavior that
decision #3 deliberately removed. Each was updated to assert the NEW contract —
loading flows through `activate_file`; a `model_path`-based corrupt/absent file
is **no longer read** (degrade via the gateway `None` path, never a raw
`torch.load(path)`) — while keeping the original intent and remaining a real
guard (each still fails if a raw path load is re-introduced):

| test | old (removed) assertion | new (gateway) contract |
|---|---|---|
| `test_history_sequence_classifier.py::test_history_sequence_classifier_loads_checkpoint_model` | checkpoint at `model_path` loads → `source=history_sequence_model` | (A) real checkpoint at `model_path`, **unpinned** → NOT read, `_loaded_model=False`, `source≠model`; (B) same bytes **pinned** → loads, `source=history_sequence_model`, `label=beta` |
| `test_model_readiness_registry.py::test_pointnet_load_error_captured_on_corrupt_checkpoint` | corrupt file at `model_path` → `_load_error` set | (A) corrupt file at `model_path`, unconfigured gateway → NOT read → `_load_error=None` (cold); (B) same corrupt bytes **pinned** (digest matches garbage) → gateway returns them → `torch.load` raises → `_load_error` captured |
| `test_model_readiness_registry.py::test_uvnet_load_error_captured_on_corrupt_checkpoint` | corrupt file at `model_path` → `_load_error` set | same shape as pointnet (UVNet degrades to the mock encoder with `_load_error=None`; the load error now comes only from gateway-delivered verified-but-undecodable bytes) |

The **third** test (`test_uvnet_…`) was not named in the brief's original list
of two; it was discovered failing for the **same** raw-load-removed reason
(verified failing in isolation on the original file) and fixed the same way, as
the brief instructed. (The graph2d sibling
`test_graph2d_load_error_captured_on_corrupt_checkpoint` was **not** touched: it
still passes because graph2d degrades benignly with `_load_error=None` — cold,
exactly like uvnet/pointnet — and only sets `_load_error` on gateway-delivered
verified-but-undecodable bytes. Half (A) asserts `cold._load_error is None`.)

## 9. VERIFICATION — exact commands + verbatim pass lines

All runs: `cd /private/tmp/cadml-c2c6`, `PYTHONPATH=.`, `--noconftest`,
`-p no:cacheprovider`. Interpreter noted per block.

### 9.1 The three fixed pre-existing tests (`/usr/bin/python3`, torch 3.9.6)

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -v \
  tests/unit/test_history_sequence_classifier.py::test_history_sequence_classifier_loads_checkpoint_model \
  tests/unit/test_model_readiness_registry.py::test_pointnet_load_error_captured_on_corrupt_checkpoint \
  tests/unit/test_model_readiness_registry.py::test_uvnet_load_error_captured_on_corrupt_checkpoint
```
```
tests/unit/test_history_sequence_classifier.py::test_history_sequence_classifier_loads_checkpoint_model PASSED [ 33%]
tests/unit/test_model_readiness_registry.py::test_pointnet_load_error_captured_on_corrupt_checkpoint PASSED [ 66%]
tests/unit/test_model_readiness_registry.py::test_uvnet_load_error_captured_on_corrupt_checkpoint PASSED [100%]
======================== 3 passed, 8 warnings in 2.79s =========================
```

### 9.2 The C6 golden matrix (`/usr/bin/python3`, torch 3.9.6)

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -v tests/unit/test_c2_golden_matrix.py
```
```
tests/unit/test_c2_golden_matrix.py::test_green_fixed_pin_loads_and_produces_real_output PASSED [ 14%]
tests/unit/test_c2_golden_matrix.py::test_red_pin_absent_degrades_never_raw_loads PASSED [ 28%]
tests/unit/test_c2_golden_matrix.py::test_red_store_unconfigured_degrades_never_raw_loads PASSED [ 42%]
tests/unit/test_c2_golden_matrix.py::test_red_digest_tamper_degrades_never_raw_loads PASSED [ 57%]
tests/unit/test_c2_golden_matrix.py::test_red_wrong_kind_bundle_asked_as_file_degrades PASSED [ 71%]
tests/unit/test_c2_golden_matrix.py::test_referenced_c1_reds_are_documented PASSED [ 85%]
tests/unit/test_c2_golden_matrix.py::test_referenced_c5_discriminators_are_documented PASSED [100%]
============================== 7 passed in 1.67s ===============================
```

### 9.3 Golden matrix + the six C2 family units together (`/usr/bin/python3`)

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -q \
  tests/unit/test_c2_pickle_classifier.py tests/unit/test_c2_graph2d.py \
  tests/unit/test_c2_history_sequence.py tests/unit/test_c2_vision3d_uvnet.py \
  tests/unit/test_c2_pointnet.py tests/unit/test_c2_part.py \
  tests/unit/test_c2_golden_matrix.py
```
```
29 passed, 7 warnings in 1.91s
```

### 9.4 The two edited test files in isolation (`/usr/bin/python3`)

```
tests/unit/test_history_sequence_classifier.py  →  9 passed in 0.67s
tests/unit/test_model_readiness_registry.py     → 13 passed, 8 warnings in 3.40s
```

### 9.5 C3/C4/C5 foundation (`python3.11`, no torch — the verified-state suite)

```
PYTHONPATH=. python3.11 -m pytest --noconftest -q \
  tests/unit/test_activation_gateway.py \
  tests/unit/test_activation_surface_enumerator.py \
  tests/unit/test_activation_manifest_truth.py
```
```
58 passed in 31.21s
```

### 9.6 Interpreter nuance (no false CI-green claim)

- Under `python3.11` (no torch) the golden matrix and the torch-dependent fixed
  tests **skip cleanly** (`1 skipped`), not error — so CI on either interpreter
  is safe.
- The C1 Linux-root / `openat2` suites are **CI-only** (no local Docker) and are
  **not** part of the local evidence above.
- **CI-on-Linux is the authority.** These are local, single-box, pre-push
  results for the commits that produced them. The head this record now cites
  (`ae332c5e`) has since been pushed and run on live CI: **64 pass / 15 skip /
  0 fail** (79 check-run items, reviewer's authoritative rollup — not 83). That
  number was **observed by the reviewer on CI**, not reproduced in this
  sandbox (no Docker here); it is cited, not re-derived.

## 10. Residual risks & correctness findings (surfaced, out of scope)

1. **Test-isolation note (findings 1 & 2 below are now SUPERSEDED by the F4
   remediation — see §13).** The original C6 record described a cross-file leak
   where `importlib.reload(vision_2d)` left the `_graph2d` singleton's
   `_load_error` non-`None`, flipping four readiness tests to `status="error"`.
   That premise no longer holds. What actually neutralizes the "error" leak is
   finding-2's correction — graph2d degrades benignly with `_load_error=None`
   (confirmed by the current `src/ml/vision_2d.py:_load_model` and by
   `test_c2_graph2d.py` asserting `_load_error is None`) — combined with
   `test_c2_graph2d.py` not reloading the `_graph2d` singleton. (F4 does NOT stop
   this leak on its own: `_status_from_evidence` checks `if error: return "error"`
   *before* the `activation_based` branch, so a leaked non-`None` `_load_error`
   would still surface as `"error"`.) F4 is a **separate** improvement: it
   reshapes readiness to be **activation-based**, so a wired family's status is
   `fallback` (explicit degraded) whenever it is not `loaded`, never `"available"`
   on the strength of a checkpoint file existing. The empirical proof both hold
   is a single-process cross-file run (test_c2_graph2d.py before
   test_model_readiness_registry.py, the feared alphabetical order): 155 passed.
   The four readiness tests
   (`test_missing_local_checkpoints_are_degraded_fallbacks`,
   `test_checkpoint_presence_does_not_confer_available_without_activation`
   [renamed from `…reports_available_and_checksum`],
   `test_required_missing_model_blocks_readiness`,
   `test_model_readiness_health_endpoint`) pass under the corrected contract.
2. **~~graph2d degrade sets `_load_error`~~ — CORRECTED (stale claim).** The
   earlier record claimed `src/ml/vision_2d.py:_load_model` set
   `_load_error="model_activation_degraded"` on a benign degrade; the actual code
   sets `_load_error=None` on the gateway `None` (degrade) path, identical to
   pointnet/UVNet. There is no graph2d-specific degrade-vs-error divergence.
3. **`src/core/ocr/providers/paddle.py:206` fabricates OCR text when the model
   is absent** (confidently-wrong; violates the spirit of decision #3). OCR is a
   **gate-before-wired** activation (decision #4), so this is recorded, not
   fixed, in Phase-A.
4. **`pointnet/inference.py` silently random-inits the feature extractor when
   `extractor_state_dict` is absent** from an otherwise-valid checkpoint —
   recorded, not fixed (in-family behavior, not an activation-membrane hole).
5. **The 4 OCR/embedding activations are gate-before-wired, not wired**
   (decision #4). They call third-party loaders (`from_pretrained` /
   `SentenceTransformer` / `PaddleOCR(det_model_dir=…)`) that need a **path
   string**, but C1's `FrozenBundle` is **fd-only by design** (the TOCTOU
   guarantee). Bridging fd → path is a different mechanism = a **later Phase**;
   the enumerator marks them `gate-before-wired` so they stay a raw load today
   and pass the structural check consistently.

## 11. Owner-gated next steps

1. Merge PR #528 (C1 core) first.
2. Rebase `l3-phasea-c2c6-wiring-20260721` onto `main`.
3. Ratify the C2–C6 wiring PR (owner-only).
4. **Enablement is a separate owner gate** (Phase-A §7.2): real production
   digests are supplied per activation at enablement, out of Phase-A scope. The
   default posture ships **NO-PIN / degraded**.
5. After all in-scope live activations are wired and ratified, flip
   `ACTIVATION_ENFORCE_WIRING` to make the C5 structural check blocking (W4).
6. Consider the finding-1/2 test-isolation + graph2d degrade-vs-error cleanup in
   a follow-up (out of this record's scope).

## 12. Completion statement

> Phase A static fixed-hash containment complete; external reload remains
> sealed; retraining remains disabled; Track E, signed proofs, and enablement
> are not complete.

Framed honestly: this holds **for the 7 in-scope activations**; **landing
requires owner merge of #528 + rebase onto `main` + owner ratification.** This
step merges nothing and ratifies nothing.

**Scope of "containment complete" (corrected after the ratified NO-GO gate
review).** The word *complete* here is NOT "every family serves a verified
model." It means the **containment membrane** is what these deliverables
actually establish, and only after the §13 remediation:

1. the **explicit-degraded C4 contract** — every wired family that cannot
   activate its pin enters a *defined, caller/health-visible degraded state*
   and never silently serves a stand-in (design-lock line 403);
2. **live-gate wiring** — activation is decided by the C1 gateway at the load
   site, not by a legacy `os.path.exists` pre-gate that could bypass the pin;
3. the **strengthened C5 structural check** — every wired raw-load site is
   enumerated and asserted.

Containment says *"an unpinned/tampered family degrades explicitly,"* NOT
*"the family is loaded."* Default posture ships **NO-PIN / degraded**; that is
containment working, not a gap.

**Consumption of the degrade marker (added after round-3; do not read the
three points above as covering this).** A family entering its defined degraded
state is necessary but not sufficient — a *consumer* of that family's output
that ignores the degrade marker and treats a mock/fallback result as if it
were verified reopens the same hole one layer downstream. Round-2 established
the marker fields (`used_for_fusion`/`degraded`/`model_available` for history,
`embedding_degraded`/`last_encode_degraded` for UV-Net) but, as the gate review
found in round-3, did not prove every reader of those fields honored them.
**As of round-3, three specific, named consumer paths are verified to honor
their marker:** (a) `src/ml/hybrid_classifier.py` excludes a degraded history
prediction from fusion (round-2); (b) `src/ml/hybrid/explainer.py` independently
excludes the same degraded history prediction from its own model-contribution /
candidate-label / disagreement-count treatment (round-3, §13 F4-history); (c)
`src/core/vector_pipeline.py` excludes a degraded UV-Net embedding from L3
registration/similarity (round-3, §13 F4-UVNet). This is **not** a claim that
every reader of `embedding_degraded`/`used_for_fusion`/`last_encode_degraded`
in the tree has been audited — it is a claim about these three, now-fixed,
reviewer-identified paths.

## 13. F3 + F4 + F1 remediation (post-NO-GO gate fix)

This unit closed the containment holes the gate review flagged. All source
changes are additive to the C2–C6 wiring and change no C1 core.

**F3 — legacy path pre-gates removed (a valid pin must be able to activate).**
- `src/core/analyzer.py::_get_v16_classifier` no longer gates V16 construction
  on `os.path.exists(v6_path) and os.path.exists(v14_path)`. `PartClassifierV16`
  is constructed unconditionally (its `_load_models` is lazy); the C1 gateway
  (`activate_file("part/v16-v6pt", …)`) decides activation-vs-degrade at predict
  time. A valid pin with the old convenience files absent now activates; a
  degrade raises inside `_load_models`, is caught by `_classify_with_v16` → V6 →
  rules, and — by design — does **not** populate `_v16_classifier_load_error`
  (missing pin = *degraded*, not *error*). The `DISABLE_V16_CLASSIFIER` feature
  flag is preserved.
- `src/ml/vision_2d.py::EnsembleGraph2DClassifier._load_models` no longer gates
  each `model_path` on `os.path.exists` and no longer re-instantiates a
  `Graph2DClassifier` per path. **Decision:** the Phase-A baseline manifest pins
  exactly ONE graph2d activation (`graph2d/main`); the ensemble is **not** a
  distinct pinned family, so it is driven by that single gateway-routed
  classifier (the already-activated module singleton) **once** — no re-pull of
  the same pin, no legacy path bypass. Multi-model ensembling needs distinct
  pins and is **out of scope until Track E**; `GRAPH2D_ENSEMBLE_MODELS` is kept
  only for config-echo visibility.

**F4 — no silent stand-in (design-lock line 403); rule-based results are
EXPLICITLY labeled degraded.**
- `src/ml/history_sequence_classifier.py::predict_from_tokens` — when the pinned
  model is unavailable and the prototype (rule-based) scorer answers, the payload
  is stamped `model_available=False`, `degraded=True`,
  `degraded_reason="pinned_model_unavailable"`. **Ratified this review round:**
  the functional `status="ok"` / `source="history_sequence_prototype"` is kept
  only for legacy compat (the pre-existing prototype contract reads it), but a
  `degraded=True` history prediction now **EXITS model fusion** —
  `src/ml/hybrid_classifier.py` detects `degraded`/`model_available=False` and
  excludes it from the model-vote/fusion path
  (`decision_path` gets `history_degraded_excluded_from_fusion`; the retained
  record is stamped `used_for_fusion=False`, `fusion_excluded_reason=
  "degraded_model_unavailable"`, `auxiliary_role="rule_fallback"`). It survives
  only as an explicit rule/fallback auxiliary result, never as a model signal.
  Covered by the new `tests/unit/test_hybrid_history_degraded.py`. **As of this
  review round, `hybrid_classifier` was the only consumer this held for** —
  `src/ml/hybrid/explainer.py` independently re-derived model-vote treatment
  from the raw `history_prediction` dict and still scored a
  `used_for_fusion=False`/`degraded=True` history as a model signal
  (`history_sequence_label` contribution, a candidate-label alternative, and a
  disagreeing-source count). **Round-3 closes this**: a new
  `_history_excluded_from_fusion()` helper (mirroring the same
  `used_for_fusion`/`degraded`/`model_available` marker checks) gates every one
  of those four explainer paths — feature contributions, the model-statistics
  summary, alternative-label candidates, and the model-disagreement count — so
  an excluded history prediction surfaces only as a zero-contribution
  `rule_fallback` auxiliary entry, never as a `history_sequence` model source.
  Covered by the new `tests/unit/test_hybrid_explainer_history_degraded.py`.
- `src/ml/vision_3d.py` — the mock B-Rep embedding path is now non-silent: it
  logs a WARNING and sets `last_encode_degraded=True`, and the encoder exposes a
  `model_available` property (mirrors gateway activation) for callers/health.
  `src/core/feature_pipeline.py` propagates that marker with the 3D embedding
  result as `embedding_degraded`/`embedding_provenance` (`"mock_heuristic"` vs
  `"uvnet_model"`), fail-closed (an encoder with no marker defaults to
  degraded). A degraded embedding IS still cached, but TAGGED degraded — the
  cache key bumped `l4_v1` -> `l4_v2` to invalidate any pre-fix entries that
  predate the marker, so a later cache HIT can never re-surface an untagged
  mock as if it were verified. Covered by the new cases in
  `tests/unit/test_feature_pipeline.py`. **This round-2 fix stopped at the
  marker/cache layer — it did not stop the marked-degraded embedding from being
  CONSUMED.** A reviewer reproduced `l3_dim=3`: `src/core/vector_pipeline.py`
  registered any `embedding_vector` present in the features mapping under
  `VECTOR_LAYOUT_L3`/`base_sem_ext_v1+l3` regardless of `embedding_degraded`,
  so a mock heuristic embedding still entered L3 registration and participated
  in similarity. **Round-3 closes this**: `_build_feature_vector` now reads the
  co-written `embedding_degraded` marker and skips L3 registration (falls back
  to the pre-L3 vector layout) whenever it is truthy — a degraded/mock
  embedding is tagged-degraded in the cache (round-2) AND excluded from L3
  registration/similarity (round-3); only a verified UV-Net embedding (marker
  `False`, or a legacy caller that never sets it) still registers as L3.
  Root-caused in `src/ml/vision_3d.py::encode()`: `last_encode_degraded` is now
  defaulted `True` at the very top of every `encode()` call and flipped `False`
  in exactly one place — immediately after a genuine model inference has
  materialized a real (non-zeros-guard) embedding list — so the mock path, every
  zeros-guard fallback, and the exception fallback all inherit the degraded
  default; previously the flag was cleared at the *start* of the inference
  branch, so a zeros-fallback from a schema/dim mismatch or a mid-inference
  exception could still read as a verified embedding. Covered by the new
  `tests/unit/test_vector_pipeline_degraded.py` and the added cases in
  `tests/unit/test_c2_vision3d_uvnet.py`.
- `src/models/readiness_registry.py::_status_from_evidence` — for gateway-wired
  families (v16, graph2d, uvnet, pointnet) readiness is judged on **ACTIVATION**,
  not legacy checkpoint-file presence: an enabled family that is not `loaded` is
  explicit `fallback`/`missing`, **never** `"available"` on the strength of a
  file existing. Non-wired families (ocr provider-managed, embedding) keep their
  existing signal. The `to_dict` no longer emits `checkpoint_paths` and the
  path-keyed per-file checksums are dropped (design-lock: no paths in telemetry);
  the boolean `checkpoint_exists` and combined `checksum` (a hash) remain.
- `src/api/health_utils.py` — degraded reasons and the readiness telemetry no
  longer emit resolved filesystem paths (`graph2d_model_missing:{path}` →
  `graph2d_model_missing`; `graph2d_model_path` / `v6_model_path` /
  `v14_model_path` keys dropped). Path-free `*_present` booleans are retained.

**F1 — full-suite miss fixed.**
`tests/unit/test_part_classifier_coverage.py::TestV16ModelLoadingErrors` — the
old `test_v6_not_found_raises_error` expected `FileNotFoundError("V6模型不存在")`;
the wired impl raises the gateway degrade
`RuntimeError("part/v16-v6pt v6pt component activation unavailable")`. Renamed to
`test_v6_not_found_degrades_via_activation_gateway` and re-pointed at the new
safety contract.

**Completed this review round (was residual): config-echo path scrub.**
`config.ml.classification` in the health payload previously echoed operator-configured
*config* paths (`hybrid_config_path`, `graph2d_model_path`,
`graph2d_temperature_calibration_path`). These are now folded into the same
path-free C4/telemetry contract as the readiness/health-state surface (design-lock
line 403): `src/api/health_models.py` renames the fields to
`hybrid_config_name` / `graph2d_model_name` / `graph2d_temperature_calibration_name`,
and `src/api/health_utils.py` emits `os.path.basename(...)` (never the resolved
path), collapsing any calibration-path-bearing `graph2d_temperature_source` to
the fixed token `"calibration"`. Covered by the new
`tests/unit/test_p2_path_scrub.py`. `history/sequence` is a wired family with
**no `readiness_registry` item** yet; adding one is an owner-scoped follow-up,
not an omission in this unit.

**Completed this review round (was residual): test coverage of the new marker
FIELDS.** The degrade *behavior* was already tested — `test_c2_history_sequence`
proves the pinned-model-unavailable path answers with
`source != "history_sequence_model"`, and the readiness suite proves
`uvnet:fallback` — so design-lock line 410 ("degraded contract is tested") was
already behaviorally satisfied. The specific new marker fields now have positive
assertions too:
- `history_sequence` `model_available`/`degraded`/`degraded_reason` **and** its
  fusion-exit consequence (`used_for_fusion=False`,
  `fusion_excluded_reason="degraded_model_unavailable"`,
  `auxiliary_role="rule_fallback"`) are asserted in the new
  `tests/unit/test_hybrid_history_degraded.py`.
- `vision_3d` `last_encode_degraded`/`model_available` and its
  `embedding_degraded`/`embedding_provenance` propagation + cache-tagging
  through `src/core/feature_pipeline.py` are asserted in the new cases in
  `tests/unit/test_feature_pipeline.py`
  (`test_degraded_embedding_is_marked_and_not_cached_as_verified`,
  `test_absent_marker_defaults_to_degraded_fail_closed`,
  `test_cache_hit_preserves_degraded_marker`).

**Residual (F3(a) success side) — NOW FILLED by F5.** The degrade side of the
analyzer gate removal was already proven (F1 test → gateway `RuntimeError` →
V6/rules). The positive claim — a valid V16 pin activates with the old
convenience files absent — is now exercised: F5 added a real `part/v16-v6pt`
**three-component** pin fixture. `test_v16_valid_config_pin_loads_weights` and
`test_v16_fixture_pins_success_uses_exact_bytes_for_both` activate the full V16
family from pinned bytes (both checkpoints reconstructed from their exact pinned
bytes, and the ensemble weights read from the pinned config).

## 14. F5 — pin the V16 ensemble config (`v16config`, third single-file pin)

`PartClassifierV16._load_models` (`src/ml/part_classifier.py`) previously read
`cad_classifier_v16_config.json` directly from the mutable `model_dir` and, if
present, set `self.v6_weight`/`self.v14_weight` from it (silently defaulting to
0.3/0.7 if absent). That json is a **weight-bearing artifact**: even with both
checkpoints pinned, editing it changes the ensemble output. F5 routes its read
through `activate_file("part/v16-v6pt", "v16config")` — a THIRD SINGLE_FILE pin
under the same logical id, `store_relpath` `models/cad_classifier_v16_config.json`.
On `None` (pin absent / digest mismatch) the WHOLE V16 activation degrades with
`RuntimeError("part/v16-v6pt v16config activation unavailable")` — no silent
proceed with default weights (owner F4 no-silent-stand-in; design-lock line 403).

Discriminating tests (`test_c2_part.py`):
- `test_v16_config_pin_absent_degrades_whole_family` — both checkpoints pinned,
  config pin absent → V16 degrades; weights stay the untouched `__init__` defaults.
- `test_v16_config_digest_tamper_degrades` — config swapped on disk after pinning
  → gateway digest-miss → degrade; tampered (attacker-skewed) weights never applied.
- `test_v16_valid_config_pin_loads_weights` — valid config pin → weights applied
  (0.4/0.6, distinct from defaults) and the full family activates.

**Owner-ratified this review round:** adding `v16config` extends decision-1's
*two-pins-one-id* shape to **three** pins under `part/v16-v6pt`. Mechanically
identical (another SINGLE_FILE pin, same logical id), and the artifact **count**
for this logical activation (2→3) — and the resulting 9-pin denominator (§3) —
is now ratified, not pending.

## 15. Round-3 remediation (post-round-2 ratified NO-GO, this record)

Round-2 (§13/§14, committed at `d8ebd971`) fixed the containment membrane
itself. The round-2 gate review found the membrane's own structural check had
a name-only false-green, and that the C4 degrade contract stopped at the
family/marker boundary without proof its *consumers* honored the marker.
Round-3 closes both. No C1 core file was touched (`store.py`/`types.py`/
`resolver.py`/`pin_domain.py`/`digest.py`/`fd_dir.py` untouched);
`activation_gateway.py` was read, not edited.

**Committed-status update.** At the time this section was originally written,
round-3 was working-tree-only and uncommitted; it has since been committed as
`ae332c5e13408c3e01da739065cb8e8e0de135f2` and pushed (see §1). §15.1's
"uncommitted" / "working tree" framing below is left as the accurate
description of that point in time and is not rewritten — §16 records the
subsequent (round-4) working-tree state on top of this commit.

**C5 — canonical-gateway resolve + no cross-scope inheritance**
(`scripts/ci/activation_surface_enumerator.py`, §6 above). A gateway call now
counts only if the file's own import table resolves the call target to
`src.core.model_activation.activation_gateway` (bare/aliased name-import or
module-import + attribute form) — a same-named local fake/shadow no longer
passes as wired. A nested function/lambda scope no longer inherits an outer
scope's gateway bindings as always-dominant; each scope is judged on its own
bindings only (fail-closed, since the enumerator has no real call-order
analysis across scopes).

**UVNet non-consumption** (`src/ml/vision_3d.py` + `src/core/vector_pipeline.py`,
§13 F4-UVNet above). `last_encode_degraded` now defaults `True` at the top of
every `encode()` and is cleared in exactly one place, right after a genuine
model inference materializes a real embedding list — so every non-inference
exit (mock, zeros-guard, exception) is degraded by construction, not by an
early flag flip that a mid-inference failure could outrun. `vector_pipeline.py`
now reads the co-written `embedding_degraded` marker and excludes a degraded
embedding from L3 vector registration and similarity (previously any
`embedding_vector` present registered as L3 regardless of the marker — the
reviewer-reproduced `l3_dim=3` false-registration).

**Explainer non-consumption** (`src/ml/hybrid/explainer.py`, §13 F4-history
above). A new `_history_excluded_from_fusion()` helper applies the same
`used_for_fusion=False` / `degraded=True` / `model_available=False` test
`hybrid_classifier` already used, independently, inside the explainer's own
feature-contribution, model-statistics-summary, alternative-label, and
model-disagreement-count logic — all four previously re-derived model-vote
treatment straight from `history_prediction.get("label")` and would still
score a fusion-excluded history prediction as a live model source. An excluded
history prediction now emits a zero-contribution `rule_fallback` auxiliary
entry instead.

**Scope, explicitly:** these three fixes are the C4/C5 contract *record* for
this round — they close the specific reviewer-identified gaps in the
structural check and in two named consumers (UV-Net→L3 pipeline,
history→explainer). They do not constitute, and this section does not claim,
an audit of every reader of every degrade marker in the tree (see §12
"Consumption of the degrade marker").

### 15.1 Verbatim verification (`/usr/bin/python3`, `--noconftest -p no:cacheprovider`)

```
cd /private/tmp/cadml-c2c6
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -p no:cacheprovider -q \
  tests/unit/test_activation_surface_enumerator.py
```
```
...............................................                          [100%]
47 passed in 50.91s
```

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -p no:cacheprovider -q \
  tests/unit/test_c2_vision3d_uvnet.py
```
```
.......                                                                  [100%]
7 passed in 1.41s
```

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -p no:cacheprovider -q \
  tests/unit/test_hybrid_explainer_history_degraded.py
```
```
.....                                                                    [100%]
5 passed in 0.13s
```

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -p no:cacheprovider -q \
  tests/unit/test_vector_pipeline_degraded.py
```
```
..                                                                       [100%]
2 passed in 0.35s
```

Combined (same four full files, one invocation):

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -p no:cacheprovider -q \
  tests/unit/test_activation_surface_enumerator.py \
  tests/unit/test_c2_vision3d_uvnet.py \
  tests/unit/test_hybrid_explainer_history_degraded.py \
  tests/unit/test_vector_pipeline_degraded.py
```
```
.............................................................            [100%]
61 passed in 56.12s
```

**Residual — same honesty frame as §0/§9.6:** these four files are the full
extent of what this unit verified locally; a full-suite / CI run to check for
unrelated regressions elsewhere in the tree was **not** run as part of this
unit (round-3 touches four files plus two new test files; the change surface
is narrow, but "narrow" is not the same claim as "full-suite green," and this
record does not make that claim). At the time this paragraph was originally
written the working tree was intentionally uncommitted (per that unit's
instructions) and these results described the working tree, not a commit SHA.

**Superseded by push (see §15.2/§1):** round-3 is no longer working-tree-only.
It is committed at `ae332c5e13408c3e01da739065cb8e8e0de135f2` and pushed, and
live CI has run on that head. The local numbers immediately above remain
accurate as **pre-push evidence** for that commit; they are not restated as a
CI result.

### 15.2 Live CI rollup on the pushed head (`ae332c5e`)

This is the reviewer-reported, authoritative CI outcome for
`ae332c5e13408c3e01da739065cb8e8e0de135f2` — not a local reproduction (no
Docker in this sandbox):

- **64 pass / 15 skip / 0 fail**, 79 check-run items total.
- This 79/64/15/0 figure is the corrected rollup. An earlier report of 83
  items was wrong — it double-counted re-run check-runs — and is superseded
  here; do not cite 83.
- This is the **round-4 pre-fix baseline**: it reflects the tree as committed
  through round-3 (§13–§15), **before** the round-4 fixes in §16 below, which
  exist only as uncommitted working-tree changes on top of this head at the
  time of this record and have not themselves been through CI.

## 16. Round-4 fixes (uncommitted, on top of `ae332c5e`) — C4/C5 contract record

Two further, reviewer-identified gaps on top of round-3 (§13–§15). Both are
present in the working tree at the time of this record; **neither is
committed**, per this unit's own instructions (docs-only; no commit made
here) — a follow-up commit is expected to carry them. They are folded into the
C4/C5 contract record here (extending §6 and §13's "consumption of the
degrade marker" framing), not into §15, so that §15 continues to describe
exactly what `ae332c5e` contains.

**C5 — scope-shadowing fail-closed** (`scripts/ci/activation_surface_enumerator.py`).
Round-3 (§6, §15) stopped an outer-scope gateway binding from being inherited
as always-dominant into a nested scope, and required a call to resolve to the
canonical gateway module rather than match by bare name. It did not yet handle
the inverse: a nested scope that **shadows** a canonical gateway name — a
function parameter named `activate_file`, a local `def activate_file(...)` /
`activate_file = lambda ...`, or any local rebinding — was still resolved
against the file-level import table and could score as a genuine gateway call
even though, within that scope, the name no longer refers to the canonical
gateway. The fix collects, per function scope, the set of canonical-gateway
names (`gw_names`/`gw_modules`) that scope locally shadows (via parameters and
a store-context AST scan that does not descend into nested `def`/`lambda`/
`class` bodies), and `_is_activation_call` now requires the resolved name to
be **both** import-traceable to the canonical module **and** unshadowed in the
current scope's cumulative shadow set. This is fail-closed by construction: it
only ever adds a reason to reject a call as non-gateway, so it cannot
false-green a shadowed call, and it cannot false-RED any real wired family
(none binds `activate_file`/`activate_bundle` as a local). New discriminator
tests: `test_param_shadow_of_activate_file_is_not_the_gateway`,
`test_param_shadow_of_activate_file_is_observed_RED_under_enforce`,
`test_local_def_shadow_of_activate_file_is_not_the_gateway`,
`test_local_def_shadow_of_activate_file_is_observed_RED_under_enforce`.

**vector_pipeline missing-marker fail-closed** (`src/core/vector_pipeline.py`).
Round-3 (§13 F4-UVNet, §15) excluded a degraded UV-Net embedding from L3
registration/similarity when `embedding_degraded` was **truthy**
(`bool(features_3d.get("embedding_degraded", False))`) — but that form
defaults a **missing** key, a `None` value, or any falsy-but-present value to
"not degraded," i.e. fail-**open**: an untagged payload (e.g. a pre-fix
cache-hit entry, or any producer that never co-wrote the marker) carrying an
`embedding_vector` still registered as `base_sem_ext_v1+l3` and entered
similarity. The fix admits an embedding to L3 **only** when the marker is
explicitly the boolean `False` (`features_3d.get("embedding_degraded") is
False`); every other state — missing key, `None`, truthy, or a non-bool value
— is treated as degraded and excluded, falling back to the pre-L3 vector
layout. New discriminator tests in `tests/unit/test_vector_pipeline_degraded.py`:
`test_missing_marker_key_stays_base_fail_closed`,
`test_none_marker_stays_base_fail_closed`,
`test_untagged_cache_hit_payload_stays_base_fail_closed` (the last is the
exact untagged-cache-hit shape the reviewer identified). The existing
`test_vector_pipeline.py::test_run_vector_pipeline_uses_qdrant_registration_and_similarity`
positive-path fixture was updated to co-write `embedding_degraded: False`,
since it now needs to be explicitly tagged verified to keep exercising the L3
registration/similarity path it was written for.

**Scope, explicitly:** as with §13/§15's own scope notes, these two fixes
close the specific reviewer-identified gaps in the structural check and in
the vector_pipeline consumer; they are not a claim that every reader of every
degrade marker in the tree has been re-audited.

### 16.1 Verbatim verification (`/usr/bin/python3`, `--noconftest -p no:cacheprovider`)

These are **local, pre-commit, pre-CI** results for the uncommitted working
tree described above — not a CI result, and not claimed as one.

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -p no:cacheprovider -v \
  tests/unit/test_activation_surface_enumerator.py \
  tests/unit/test_vector_pipeline.py \
  tests/unit/test_vector_pipeline_degraded.py
```
```
tests/unit/test_activation_surface_enumerator.py::test_every_gated_site_carries_a_family PASSED [  8%]
tests/unit/test_activation_surface_enumerator.py::test_syntaxerror_in_scope_is_malfunction_exit_2 PASSED [  9%]
tests/unit/test_activation_surface_enumerator.py::test_unicodedecodeerror_in_scope_is_malfunction_exit_2 PASSED [ 11%]
tests/unit/test_activation_surface_enumerator.py::test_valueerror_from_ast_parse_is_malfunction_exit_2 PASSED [ 12%]
tests/unit/test_activation_surface_enumerator.py::test_unreadable_in_scope_file_is_malfunction_exit_2 PASSED [ 14%]
tests/unit/test_activation_surface_enumerator.py::test_wellformed_file_is_not_a_malfunction PASSED [ 16%]
tests/unit/test_activation_surface_enumerator.py::test_missing_manifest_is_malfunction_exit_2 PASSED [ 17%]
tests/unit/test_activation_surface_enumerator.py::test_corrupt_json_manifest_is_malfunction_exit_2 PASSED [ 19%]
tests/unit/test_activation_surface_enumerator.py::test_manifest_schema_violation_is_malfunction_exit_2 PASSED [ 20%]
tests/unit/test_activation_surface_enumerator.py::test_manifest_without_a_sites_object_is_malfunction_exit_2[{}] PASSED [ 22%]
tests/unit/test_activation_surface_enumerator.py::test_manifest_without_a_sites_object_is_malfunction_exit_2[{"notsites": {}}] PASSED [ 24%]
tests/unit/test_activation_surface_enumerator.py::test_manifest_without_a_sites_object_is_malfunction_exit_2[{"sites": []}] PASSED [ 25%]
tests/unit/test_activation_surface_enumerator.py::test_manifest_without_a_sites_object_is_malfunction_exit_2["a string"] PASSED [ 27%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[import torch as t\ndef f(): return t.load('x')-torch.load] PASSED [ 29%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[from torch import load\ndef f(): return load('x')-torch.load] PASSED [ 30%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[from torch import load as L\ndef f(): return L('x')-torch.load] PASSED [ 32%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[import pickle as p\ndef f(): return p.loads(b'x')-pickle.loads] PASSED [ 33%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[import joblib as jl\ndef f(): return jl.load('x')-joblib.load] PASSED [ 35%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[from transformers import AutoModel\ndef f(): return AutoModel.from_pretrained('x')-from_pretrained] PASSED [ 37%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[from sentence_transformers import SentenceTransformer\ndef f(): return SentenceTransformer('x')-ctor:SentenceTransformer] PASSED [ 38%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[import sentence_transformers as st\ndef f(): return st.SentenceTransformer('x')-ctor:SentenceTransformer] PASSED [ 40%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[from paddleocr import PaddleOCR\ndef f(): return PaddleOCR()-ctor:PaddleOCR] PASSED [ 41%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[import onnx\ndef f(): return onnx.load('m.onnx')-onnx.load] PASSED [ 43%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[import onnxruntime as ort\ndef f(): return ort.InferenceSession('m')-ctor:InferenceSession] PASSED [ 45%]
tests/unit/test_activation_surface_enumerator.py::test_import_aware_detection_no_blind_spots[from onnxruntime import InferenceSession\ndef f(): return InferenceSession('m')-ctor:InferenceSession] PASSED [ 46%]
tests/unit/test_activation_surface_enumerator.py::test_real_hf_and_embedding_loaders_are_gated PASSED [ 48%]
tests/unit/test_activation_surface_enumerator.py::test_wrapped_wired_site_detected_as_wrapped PASSED [ 50%]
tests/unit/test_activation_surface_enumerator.py::test_wired_site_with_wrapper_present_is_green_even_enforced PASSED [ 51%]
tests/unit/test_activation_surface_enumerator.py::test_remove_the_wrapper_is_observed_RED_under_enforce PASSED [ 53%]
tests/unit/test_activation_surface_enumerator.py::test_f2_gateway_called_but_result_discarded_is_unwrapped PASSED [ 54%]
tests/unit/test_activation_surface_enumerator.py::test_f2_discard_case_is_observed_RED_under_enforce PASSED [ 56%]
tests/unit/test_activation_surface_enumerator.py::test_gateway_bound_after_loader_is_unwrapped PASSED [ 58%]
tests/unit/test_activation_surface_enumerator.py::test_gateway_bound_after_loader_is_observed_RED_under_enforce PASSED [ 59%]
tests/unit/test_activation_surface_enumerator.py::test_remove_the_wrapper_is_advisory_only_by_default PASSED [ 61%]
tests/unit/test_activation_surface_enumerator.py::test_gate_before_wired_unwrapped_raw_loader_is_consistent PASSED [ 62%]
tests/unit/test_activation_surface_enumerator.py::test_gate_before_wired_but_actually_wrapped_reds_under_enforce PASSED [ 64%]
tests/unit/test_activation_surface_enumerator.py::test_gated_raw_loader_missing_wiring_reds_under_enforce PASSED [ 66%]
tests/unit/test_activation_surface_enumerator.py::test_real_tree_is_structurally_consistent_under_enforce PASSED [ 67%]
tests/unit/test_activation_surface_enumerator.py::test_every_gated_site_carries_a_valid_wiring PASSED [ 69%]
tests/unit/test_activation_surface_enumerator.py::test_closure_load_not_wrapped_by_outer_gateway_binding PASSED [ 70%]
tests/unit/test_activation_surface_enumerator.py::test_closure_load_before_outer_gateway_is_observed_RED_under_enforce PASSED [ 72%]
tests/unit/test_activation_surface_enumerator.py::test_local_fake_activate_file_is_not_the_gateway PASSED [ 74%]
tests/unit/test_activation_surface_enumerator.py::test_local_fake_activate_file_is_observed_RED_under_enforce PASSED [ 75%]
tests/unit/test_activation_surface_enumerator.py::test_param_shadow_of_activate_file_is_not_the_gateway PASSED [ 77%]
tests/unit/test_activation_surface_enumerator.py::test_param_shadow_of_activate_file_is_observed_RED_under_enforce PASSED [ 79%]
tests/unit/test_activation_surface_enumerator.py::test_local_def_shadow_of_activate_file_is_not_the_gateway PASSED [ 80%]
tests/unit/test_activation_surface_enumerator.py::test_local_def_shadow_of_activate_file_is_observed_RED_under_enforce PASSED [ 82%]
tests/unit/test_vector_pipeline.py::test_run_vector_pipeline_registers_local_vector_and_updates_memory_meta PASSED [ 83%]
tests/unit/test_vector_pipeline.py::test_run_vector_pipeline_uses_qdrant_registration_and_similarity PASSED [ 85%]
tests/unit/test_vector_pipeline.py::test_run_vector_pipeline_registration_failure_does_not_block_similarity PASSED [ 87%]
tests/unit/test_vector_pipeline.py::test_run_vector_pipeline_reports_reference_not_found_without_similarity_compute PASSED [ 88%]
tests/unit/test_vector_pipeline.py::test_run_vector_pipeline_adds_faiss_entry_when_backend_enabled PASSED [ 90%]
tests/unit/test_vector_pipeline.py::test_run_vector_pipeline_reports_qdrant_reference_not_found_without_similarity PASSED [ 91%]
tests/unit/test_vector_pipeline_degraded.py::test_degraded_embedding_stays_base_and_is_not_appended PASSED [ 93%]
tests/unit/test_vector_pipeline_degraded.py::test_missing_marker_key_stays_base_fail_closed PASSED [ 95%]
tests/unit/test_vector_pipeline_degraded.py::test_none_marker_stays_base_fail_closed PASSED [ 96%]
tests/unit/test_vector_pipeline_degraded.py::test_untagged_cache_hit_payload_stays_base_fail_closed PASSED [ 98%]
tests/unit/test_vector_pipeline_degraded.py::test_verified_embedding_is_appended_as_l3 PASSED [100%]

============================= 62 passed in 53.07s ==============================
```

Combined with the round-3 referenced files (§15.1), same four files plus
`test_vector_pipeline.py`:

```
PYTHONPATH=. /usr/bin/python3 -m pytest --noconftest -p no:cacheprovider -q \
  tests/unit/test_activation_surface_enumerator.py \
  tests/unit/test_c2_vision3d_uvnet.py \
  tests/unit/test_hybrid_explainer_history_degraded.py \
  tests/unit/test_vector_pipeline_degraded.py
```
```
....................................................................     [100%]
68 passed in 55.67s
```

**Residual — same honesty frame as §0/§9.6/§15.1:** these are the full extent
of what this unit verified locally for the round-4 change surface (two source
files, one existing test file's fixture updated, one new-cases test file
extended); a full-suite / CI run to check for unrelated regressions elsewhere
in the tree was **not** run as part of this unit. The working tree is
**uncommitted** at the time of this record — these results describe the
working tree on top of `ae332c5e`, not any commit SHA, and are not the same
claim as the §15.2 live-CI rollup (which covers `ae332c5e` only, before these
round-4 changes).

### 16.2 CURRENT-STATE / superseded note (2026-07-22)

**Everything above in §16, including its title's "(uncommitted, on top of
`ae332c5e`)" and the "Residual" paragraph's "working tree is uncommitted" /
"not any commit SHA" language, is a superseded, point-in-time record.** It is
retained unedited as the historical account of what was true immediately
after those local runs, before the next push. It is **no longer current**:

- The two round-4 fixes described above (C5 scope-shadowing fail-closed;
  `vector_pipeline` missing-marker fail-closed) **are now committed**, as head
  `4d9ee6d6` (`fix(l3): round-4 NO-GO remediation on ae332c5e — C5
  scope-shadowing + vector_pipeline fail-closed`).
- Live CI **completed** on that head (PR #532): **64 pass / 15 skip / 0 fail**,
  79 check-run items total (same de-duped-by-workflow methodology as §0/§15.2;
  the raw rollup shows 83, which double-counts the 4 `Unit Tests (Shard 1-4)`
  re-run entries, exactly the double-count §0 already corrected for
  `ae332c5e`) — zero failing checks, and the same 79-item total as
  `ae332c5e` since round-4 added no workflow. This supersedes §16.1's local,
  pre-commit numbers as the authoritative outcome for `4d9ee6d6`, the same
  relationship §15.2 has to §15.1 for the prior head.
- The §16.1 local commands were independently re-run against the committed
  working tree at `4d9ee6d6` and remain green: the 4-file set (62 passed) and
  the round-3-combined 4-file set (68 passed) both reproduce verbatim.
- **Round-5** (C5 lexical-scope import env + buffer-canonical resolution) is
  a separate, currently in-flight round of work on top of `4d9ee6d6`, not
  covered by the `4d9ee6d6` CI rollup above. A follow-up commit is expected to
  carry it, at which point the head — and the authoritative CI number — will
  advance again. Do not read the 68/15/0/83 figures above as covering
  round-5.

### 16.3 CURRENT-STATE / R5 closure + R6 in-flight note (2026-07-22)

**§16.2's characterization of round-5 as "a separate, currently in-flight
round of work on top of `4d9ee6d6`" is itself now a superseded, point-in-time
record, retained unedited above as history.** It is no longer current:

- Round-5 (C5 lexical-scope import env + buffer-canonical resolution) **is
  now committed**, as head **`3c33dc0a`** (`fix(l3): round-5 NO-GO
  remediation on 4d9ee6d6 — C5 lexical-scope import env + buffer
  canonical`), on top of `4d9ee6d6`.
- Live CI **completed** on `3c33dc0a` (PR #532). Using the raw
  `statusCheckRollup`: **83** total items, of which **68** report a passing
  conclusion and **15** report skipped, 0 failing. Applying the same
  de-duplication methodology as §0/§15.2/§16.2 (the 4 `Unit Tests (Shard
  1-4)` entries are re-run duplicates within the same workflow, not distinct
  checks) collapses this to the authoritative headline: **64 pass / 15 skip /
  0 fail**, **79** distinct check-runs — the same 64/15/0/79 figures as the
  `4d9ee6d6` rollup in §16.2, unchanged because round-5 added no workflow
  YAML. Both the raw (68/83) and deduped (64/79) countings describe the same
  all-green outcome; this document reports both so neither is mistaken for a
  regression against the other.
- Round-5's 3 findings — gateway-name lexical scoping, buffer-wrapper
  canonical resolution, and the doc's R4-status wording — were **ACCEPTED by
  the reviewer** on this live head (`3c33dc0a`).
- **Round-6** (loader-alias lexical discovery) is the current in-flight round
  of work on top of `3c33dc0a`, not covered by the `3c33dc0a` CI rollup above.
  A further commit is expected to carry it, at which point the head — and the
  authoritative CI number — will advance again.
- **One-commit-lag note:** this document, in describing head `3c33dc0a` as
  current, is itself committed as part of advancing the branch to a new head
  beyond `3c33dc0a`. That means by the time this paragraph is read, the true
  branch head may already be one commit ahead of the head this closure names
  — an inherent lag of a doc that must be committed to exist, describing the
  state as of the commit just before its own. This is flagged here so that
  gap is read as expected structure, not as an error or an untracked drift.

### 16.4 CURRENT-STATE / R6 closure (2026-07-22)

**§16.3's characterization of round-6 as "the current in-flight round" is now a
superseded, point-in-time record, retained unedited above as history.** Round-6
is no longer in-flight — it is **carried by the current implementation commit**:

- Round-6 (loader-alias lexical discovery — the file-level single-value
  module-alias table let a later/sibling same-name import erase a load site;
  fixed by extending the round-5 lexical `_Scope` env to loader receivers,
  fail-closed **for discovery**) **is committed** as head **`405d141d`**
  (`fix(l3): round-6 NO-GO remediation on 3c33dc0a — loader-alias lexical
  discovery (fail-closed)`), on top of `3c33dc0a`.
- Live CI **completed** on `405d141d` (PR #532): raw `statusCheckRollup` = 68
  passing / 15 skipped / **0 failing**, all checks terminal (`tests (3.10)` and
  `tests (3.11)` both passed; no `#528` C1 concurrency-flake recurrence).
  De-duped per §0/§15.2 convention (4 `Unit Tests (Shard 1-4)` re-run
  duplicates): 64 pass / 15 skip / 0 fail / 79 distinct check-runs — unchanged
  from prior heads, as round-6 added no workflow YAML. Both countings describe
  the same all-green outcome.
- This follow-up commit adds the four reviewer-checklist positive/negative
  controls (normal module alias, function-local alias, ambiguous cross-loader
  rebind fail-closed-discover, and a non-loader receiver that must **not** be
  invented as a load site) as permanent tests, plus this §16.4 closure. It
  advances the branch head beyond `405d141d`; the exact SHA and CI-terminal
  state of **this** head are confirmed post-push (the same inherent one-commit
  doc-lag noted in §16.3).
- **This "non-blocking residual" framing was WRONG and the reviewer rejected it in round-7/8 review.**
  §16.4 as originally written described the kind-**label** ambiguity (a sorted-first pick among multiple
  loader candidates for one alias) as a cosmetic, non-blocking loose end because "the site is still
  discovered." That undersold the actual risk: the discovered **kind** is not just a label — it is the
  input `structural_findings` (the C5 raw-wiring / wrap check) branches on. `structural_findings` skips
  any site whose kind is not in `RAW_LOADER_KINDS` (`torch.load` / `pickle.load` / `pickle.loads` /
  `joblib.load`) — non-raw kinds like `onnx.load`, `load_state_dict`, `from_pretrained`, and `ctor:*` are
  not subject to the wrap requirement. So an alias ambiguous between a RAW loader module (`torch`) and a
  declared-but-non-raw one (`onnx` — declared in `_MODULE_LOADERS` but deliberately excluded from
  `RAW_LOADER_KINDS`), whose kind resolution picked the non-raw candidate (a plain sorted-first pick does
  exactly this, since `"onnx"` sorts before `"torch"`), would classify the site `onnx.load` — and
  `structural_findings` would then `continue` straight past it. A real, unwrapped, `wiring=wired`
  `torch.load(path)` behind that alias would **never be checked for the wrap it is required to have**: the
  C5 gate would be silently bypassed for that site, with no finding and no RED. That is a REAL bypass, not
  a cosmetic residual — the kind label is a security-relevant decision, not just a display string.
  **This is CLOSED by round-8's fail-closed raw-kind preference:** kind resolution (both the module-alias
  path in `_classify`'s `Attribute` branch, and `_resolve_load_from` for the `Name`/from-import path) now
  prefers any candidate that resolves to a `RAW_LOADER_KINDS` kind over any non-raw candidate, before
  falling back to the old sorted-first tie-break only when no candidate is raw. Observed-RED coverage:
  three new tests (`test_raw_vs_nonraw_ambiguity_classifies_raw_not_onnx`,
  `test_raw_vs_nonraw_ambiguity_c5_wrap_check_actually_runs`,
  `test_raw_vs_nonraw_ambiguity_reds_under_enforce`) all **fail against the pre-round-8 code** (confirmed
  by reverting only the source fix and re-running: the site classifies as `onnx.load`, `structural_findings`
  returns no `wired-but-unwrapped` finding for it, and `ACTIVATION_ENFORCE_WIRING=1` stays green) and **pass**
  once the fix lands — i.e. the bypass is demonstrated, not merely asserted.
- Round-8 additionally closes a second, independently-found discovery gap — **class-body scope**: a load
  call made directly in a class body (not inside a method), using an alias imported directly in that same
  class body, was previously **invisible to discovery entirely** (not even surfaced as "unclassified" —
  `enumerate_sites()` returned no site at all for it), because no scope was ever pushed for a class body.
  That omission was deliberately correct for **methods** (a nested `def` must not inherit its class's
  namespace — real Python does not look up class-scope names from inside a method), but it over-applied
  the same exclusion to the class body's own direct statements, which DO run in that namespace. Fixed by
  giving the class body its own `_Scope` marked **`is_class=True`**, pushed once in `visit_ClassDef` and
  left on the chain for the whole body (a **depth-uniform** design — NOT a pop-around-each-nested-def).
  `_LoadVisitor._visible_scopes()` then consults a class scope **only when it is the innermost scope** — so
  a class-body-direct statement sees the class-body bindings, while any nested `def`/`class` (including one
  nested inside an `if`/`try`/`for` in the class body) does NOT inherit them (matching Python: a class scope
  is not a lexical parent of its methods). The earlier "pop-around-each-def" sketch was rejected: it
  fail-opened on a `def` nested inside a compound statement (the class scope leaked into it), which could
  launder an unverified raw load to `wrapped=True` and invent NameError load sites. Observed-RED /
  regression coverage: `test_class_body_direct_load_call_is_discovered` fails against pre-round-8 code
  (found=={}) and passes post-fix; the negative control `test_class_body_scope_is_not_inherited_by_methods`
  and the nested-in-compound discriminators pass on the shipped `is_class` design but RED against the
  rejected pop-around variant (regression guards).

### 16.5 CURRENT-STATE / R7 closure + R8 in-flight note (2026-07-22)

**CI-count correction (applies wherever this document cites a "de-duped" 64-pass headline as
authoritative, §0/§15.1-§15.2/§16.2-§16.4 included): that de-dup methodology is itself corrected here.**
The reviewer's authoritative live figure for this branch is the **raw `statusCheckRollup`: 68 success /
15 skipped / 0 failing**. The prior "64 pass / 15 skip / 0 fail, 79 distinct check-runs" headline collapsed
4 `Unit Tests (Shard 1-4)` entries as same-workflow re-run duplicates — that collapse is **not** how the
reviewer counts the gate: **68/15/0 is the number to cite going forward**; do not re-derive a 64-figure by
subtracting the shard entries again. Every earlier section's 64-pass wording above is retained unedited as
the historical, point-in-time record of what this document asserted at the time (per this document's own
established convention of appending a superseding CURRENT-STATE section rather than rewriting history) —
this section is the correction, not a silent edit of those sections.

- **Round-6-follow-up** (the reviewer-checklist positive/negative controls for round-6's loader-alias
  lexical discovery, plus the §16.4 closure text) is committed as head **`d1beb7e5`** (`test(l3): round-6
  follow-up — reviewer-checklist controls + Dev&V R6 closure`), on top of `405d141d`. This is the head this
  worktree started round-8 from.
- **Round-7** (reviewer's NO-GO on `d1beb7e5`) identified the kind-ambiguity bypass corrected in this
  section's rewrite of the former §16.4 "non-blocking residual" bullet above: the sorted-first kind pick
  could resolve a raw-vs-non-raw-ambiguous alias to a non-raw kind, silently exempting the site from the C5
  wrap check. There is no separate round-7 commit — its finding is what round-8 (this round) fixes.
- **Round-8 (this round, in-flight)** — two fixes, both in `scripts/ci/activation_surface_enumerator.py`,
  no C1 core changes:
  1. Fail-closed raw-kind preference in kind resolution (`_classify`'s `Attribute`-branch loop over
     `_resolve_load_mods`, and `_resolve_load_from`): any candidate resolving to a `RAW_LOADER_KINDS` kind
     now wins over a non-raw candidate, closing the bypass described above.
  2. Class-body scope: a class body gets its own `_Scope` marked `is_class=True`, pushed once and left on
     the chain for the whole body (**depth-uniform**; `_visible_scopes()` consults it only when innermost).
     A class-body-direct load call using a class-body-local alias is discovered (previously invisible to
     `enumerate_sites()` entirely), while methods — including a `def` nested inside an `if`/`try`/`for` in
     the class body — correctly do NOT inherit it. (The pop-around-each-def sketch was rejected: it
     fail-opened on a def nested in a compound statement — see the §16.5-referenced note above.)
  Both fixes ship with new permanent tests in `tests/unit/test_activation_surface_enumerator.py`: the
  raw-vs-non-raw ambiguity trio for the module-alias form (`test_raw_vs_nonraw_ambiguity_*` ×3) **and** the
  from-import twin (`test_raw_vs_nonraw_fromimport_ambiguity_*` ×3), `test_class_body_direct_load_call_is_discovered`,
  the negative control `test_class_body_scope_is_not_inherited_by_methods`, and the nested-in-compound
  discriminators (regression guards vs the rejected pop-around). Full local run `/usr/bin/python3 -m pytest
  tests/unit/test_activation_surface_enumerator.py --noconftest -p no:cacheprovider`: **75 passed**. The
  live tree itself (`ACTIVATION_ENFORCE_WIRING=1 /usr/bin/python3 scripts/ci/activation_surface_enumerator.py`)
  remains **129 load sites, all classified, exit 0** — neither fix changes any real-tree classification,
  only the fail-closed handling of the pathological alias shapes the new tests construct.
- This round-8 work is **uncommitted** at the time of this record (per this worktree's task instructions,
  not committed as part of this pass). The head at the top of this section (`d1beb7e5`) is therefore still
  the true branch head; the same one-commit-lag caveat noted in §16.3/§16.4 applies once round-8 is
  committed and pushed — the exact SHA and CI-terminal state of that future head are confirmed post-push,
  not asserted here.

### 16.6 CURRENT-STATE / R8 closure (2026-07-22)

**§16.5's "Round-8 (this round, in-flight) … uncommitted" wording is now a superseded,
point-in-time record, retained unedited above as history.** Round-8 is no longer in-flight:

- Round-8 (class-body `is_class` depth-uniform scope + fail-closed raw-kind preference) **is
  committed** as head **`af39e261`** (`fix(l3): round-8 NO-GO remediation on d1beb7e5 — class-body
  scope + fail-closed raw-kind preference`), on top of `d1beb7e5`.
- Live CI **completed** on `af39e261` (PR #532): **68 pass / 15 skip / 0 fail**, all checks
  terminal — `tests (3.10)` and `tests (3.11)` both passed, no `#528` C1 concurrency-flake
  recurrence this run. This is the reviewer's authoritative raw-rollup convention (§16.5), not a
  re-derived de-duped figure.
- The full enumerator suite is **75 passed** (py3.9 + py3.11); `ACTIVATION_ENFORCE_WIRING=1` CLI =
  129 sites all classified, exit 0; manifest set-equality holds; full base-vs-branch wired diff = 0
  branch-only failures.
- This closure is a docs-only follow-up commit; it advances the branch head beyond `af39e261`
  (the same inherent one-commit doc-lag noted in §16.3–§16.5). The exact SHA of **this** head is
  confirmed post-push.
- Separately: the `#528` concurrency-test flake (`test_dup_dir_fd_caller_owned_and_cleanup_concurrent`)
  is addressed by a distinct #528-targeted PR (deterministic cleanup-entered hook, no security-semantic
  change) — not part of #532.

### 16.7 CURRENT-STATE / post-#528 rebase acceptance (2026-07-23)

This section supersedes the earlier statements that #532 is still stacked on
an unmerged #528:

- PR #528, including the #533 deterministic concurrency hook, was squash-merged
  to `main` as `c2ecfe558a2a027ab9ee63c07788c19203df6926`.
- The 11 C2-C6 commits were rebased from the former C1 boundary `6e645bc2` onto
  that `main`. `git range-diff` reported all 11 patches as equivalent, and the
  resulting `origin/main...HEAD` diff contains only the 41 C2-C6 source, test,
  manifest, and documentation files. It contains no duplicate C1 core or #533
  changes.
- The first rebased head was `398a26f6`. This section and the test-only
  acceptance correction below advance the branch again, so the final SHA and
  authoritative CI result must be taken from PR #532 after this commit is
  pushed; no green result for that future head is claimed here.

The post-rebase local acceptance run found a test-contract gap hidden by the
fresh-clone fixture shape. Six legacy tests in `test_part_classifier.py`
treated the mere existence of `models/cad_classifier_v2.pt` as authorization
to load it. Fresh-clone CI does not contain that local model and therefore
skipped those branches, while a developer checkout with the model correctly
failed because Phase A now refuses every unpinned raw-path load.

The tests now inject those local bytes through the mocked
`activate_file("part/v6", "main")` success boundary, and the stub asserts that
exact logical/artifact tuple before returning bytes. This preserves the useful
real-checkpoint loading assertions while proving that success comes from
gateway-verified bytes, never from path existence. Product code is unchanged
by this follow-up.

Local evidence on the rebased tree:

```text
# The six previously hidden failures
6 passed

# Gateway/manifest/C5, all C2 families, degraded consumers, readiness,
# path-scrub, and part/vector integration (root conftest excluded because the
# local Python 3.9 FastAPI stack cannot evaluate existing PEP-604 annotations)
226 passed, 4 skipped, 0 failed
```

The complete Linux CI matrix on the final pushed SHA remains the authoritative
landing gate. Until it is terminal green, this branch is not ratified or
merge-ready.
