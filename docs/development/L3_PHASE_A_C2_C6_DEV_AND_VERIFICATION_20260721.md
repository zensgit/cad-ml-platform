# L3 Phase-A C2–C6 — Development & Verification record (2026-07-21)

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
- **CI-on-Linux is the authority.** Nothing here claims CI-green; it claims
  what ran locally, on which interpreter, with what result.

This branch is **stacked on PR #528** and is **not on `main`**. It merges
nothing, ratifies nothing, resolves no review threads — those are the owner's.

## 1. Base / head SHAs and governance

| | value |
|---|---|
| Base (C1 core) | `6e645bc2d23b9ea54ae3f44504006511b6dc0525` |
| Base provenance | PR #528 head — C1 model-activation core; passed a fresh 6-lens opus conformance gate (**GO**, 0 blocking, 0 runtime holes) |
| Branch | `l3-phasea-c2c6-wiring-20260721` |
| Foundation commit (C3+C4) | `c9696f1160ad395f35cb0d6066e6cd64c6c43798` |
| C2/C5/C6 delivery | working-tree changes on the branch (family sources, C2 unit tests, enumerator, this record) |

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

## 3. Per-family C2 wiring table (the 7 in-scope activations / 8 pins)

Denominator = **7 logical activations / 8 pins** wired. Degrade = the `None`
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
SINGLE_FILE artifact (owner two-pins-one-id shape extended to three; **needs an
owner nod** since it extends decision-1's two-pin count). Absent/tampered config
degrades the WHOLE V16 family — never a silent proceed with the default weights
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
- **CI-on-Linux is the authority.** These are local, single-box results.

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
  `degraded_reason="pinned_model_unavailable"`. The functional `status="ok"` /
  `source="history_sequence_prototype"` is retained (the hybrid consumer and the
  prototype contract read it), so this is the owner-approved "rule-based result,
  explicitly labeled" shape — *not a bare silent ok*.
- `src/ml/vision_3d.py` — the mock B-Rep embedding path is now non-silent: it
  logs a WARNING and sets `last_encode_degraded=True`, and the encoder exposes a
  `model_available` property (mirrors gateway activation) for callers/health.
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

**Residual (scoped, not fixed here):** `config.ml.classification` in the health
payload still echoes operator-configured *config* paths (`hybrid_config_path`,
`graph2d_model_path`, `graph2d_temperature_calibration_path`) and basename model
filenames. These are configuration echo, distinct from readiness/health-state
telemetry (the design-lock line-403 target, now path-free). A broader config-echo
path scrub is a separate change with wider (ops-dashboard) blast radius.
`history/sequence` is a wired family with **no `readiness_registry` item** yet;
adding one is an owner-scoped follow-up, not an omission in this unit.

**Residual (test coverage of the new marker FIELDS).** The degrade *behavior* is
tested — `test_c2_history_sequence` proves the pinned-model-unavailable path
answers with `source != "history_sequence_model"`, and the readiness suite proves
`uvnet:fallback` — so design-lock line 410 ("degraded contract is tested") is
behaviorally satisfied. But the specific new marker fields
(`history_sequence` payload `model_available`/`degraded`/`degraded_reason`;
`vision_3d` `last_encode_degraded`/`model_available`) have **no positive
assertion**, and none of this unit's three owned test files import those modules,
so there is no clean home to add one here. The fields are **additive and
unlocked** — a candidate follow-up (a `test_c2_history_sequence` positive-marker
case and a `vision_3d` mock-path flag case), not existing coverage.

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

**Owner nod needed:** adding `v16config` extends decision-1's *two-pins-one-id*
shape to **three** pins under `part/v16-v6pt`. Mechanically identical (another
SINGLE_FILE pin, same logical id), but the artifact **count** for this logical
activation changes 2→3, so it should get an explicit owner acknowledgement.
