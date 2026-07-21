# L3 Phase A — C1 model-activation core — Dev & Verification (2026-07-19)

> **Scope & honesty frame.** This documents **only the C1 reusable core** delivered in this worktree:
> controlled store, immutable pin records, raw-pin domain, store-root-anchored resolver, single-file
> same-fd SHA-256 activation, and bundle `tree-digest-v1` over a service-private **sealed** freeze
> handled by a held directory **fd** (not a public mutable path), plus the **authority-fix** pass for
> path-safe refusals, freeze lease/identity ledger, descriptor-relative cleanup, and P1 residual
> windows. It does **NOT** wire any model family, create production pins, enable reload/retraining,
> implement C2–C6, Track E, or Phase B. `merged != enabled != safe`. Grounded on
> `origin/main@7160694d` (#513 design lock + #526) with PR #528 head
> `17ebaee255a7450d5c3a06c7f4baf9e4922ba230` as the pre-fix base. Wave-1 audit evidence is
> read-only elsewhere and is **not** edited here.

Authority: `docs/development/L3_MODEL_ACTIVATION_MEMBRANE_DESIGNLOCK_20260712.md` (ratified #513) and
`docs/development/L3_SAFETY_DESIGN_AND_VERIFICATION_20260715.md` (Phase A item C1).

---

## 1. What C1 delivers

| Surface | Location | Contract |
|---|---|---|
| Package | `src/core/model_activation/` | Narrow public API; no family imports |
| Types | `types.py` | `ArtifactKind`, `PinRecord`, `BoundPolicy`, path-safe `RefusalReason`; **`FrozenBundle`** = dir-fd handle via **`dup_dir_fd()`** (no borrowed public `dir_fd`, no public `path`) |
| Raw pin domain | `pin_domain.py` | Split-on-`/` **before** any `Path`/`os.path` normalization |
| Resolver | `resolver.py` | Linux **`syscall(SYS_openat2, …)`** with `RESOLVE_BENEATH\|RESOLVE_NO_SYMLINKS` when available (ENOSYS / probe EPERM/EACCES → component walk); runtime ENOSYS after successful probe is INTERNAL (no same-request fallback); never `resolve()+open`; shared raw pre-gate; `ENOTDIR` → `NOT_DIRECTORY` |
| Digests | `digest.py` | Single-file SHA-256; ratified `tree-digest-v1`; OSError mapped path-safe (raise outside `except`) |
| Fd readdir | `fd_dir.py` | **`os.listdir(dir_fd)`** (Python 3.10+ Unix); OSError → fail-closed; no silent partial list |
| Store | `store.py` | Immutable pins; two-pass bundle; **`FreezeResourceLease`** (empty start, trusted freeze parent, creation-time identity ledger, reserve→create→adopt→finalize); digest from frozen snapshot; O(depth) source fds; ControlledStore close drains in-flight then pending leases |

### Required behavior — implemented (post authority-fix)

1. Server-owned store root + immutable pins; absent pin fail-closed; no re-pin / path-swap / network / hot-swap.
2. Shared raw POSIX pin pre-gate for both resolver implementations.
3. Store-root-anchored containment (`O_NOFOLLOW` / openat2 resolve flags); `O_NONBLOCK` on file leaves **and** freeze member / create opens (`_MEMBER_FLAGS`, `_FILE_FLAGS`, `_FREEZE_LEAF_FLAGS`).
4. Bounded policy; refuse special/symlink/non-UTF-8/malformed — never skip.
5. Single-file: same fd, bounded read once, hash those bytes, return those bytes.
6. Bundle **(b-ii)**:
   - **Pass 1 — metadata preflight** (descriptor-relative DFS): refuse metadata-detectable limits **before any freeze copy** (zero destination writes).
   - **Pass 2 — freeze**: same-fd bound recheck + bounded copy from the **already-open source member fd** (no path re-open between fstat and copy); destroy partial freeze on failure; live source dir fds are **O(depth)**.
   - **Digest from freeze**: after seal, recompute `tree-digest-v1` by reading the frozen tree via its dir fd (not source-side copy buffers).
7. Typed path-safe refusals: map inside `except`, **raise outside** so `__cause__` and `__context__` are `None` (no path-bearing OSError leak in formatted tracebacks).
8. KIND-narrow API; does not decide part/v16 composite KIND.

### Freeze lease & cleanup contract (authority-fix)

- **No default temp directory.** Bundle activation requires an explicit trusted service-private freeze parent: absolute, non-symlink, owned by euid, `mode & 0o077 == 0` before `mkdtemp`.
- Lease starts **empty**. Protocol for every create: **reserve** ledger capacity → mkdir/O_CREAT → **adopt** fd into lease **before** fstat/finalize → finalize identity into the **creation-time** ledger. On fstat failure the pending fd remains owned; release retries fstat, records identity when possible, closes once, and scrubs.
- **Pending parent-dir fd ownership:** every `_PendingNode` owns a **CLOEXEC dup** of the parent directory fd at adopt time (not a borrow of the caller's nested dir fd). This survives freeze-walk recursive unwind that closes the caller's `mkdir_owned` / parent fd after a nested create leaves a pending node. Successful finalize or successful scrub closes the owned parent dup; mismatch / fstat failure retains both child and parent dups for honest retry. Parent-fd dup failure: best-effort destroy the just-created child via the still-live caller parent; if destroy fails, retain the child on pending without claiming `cleanup_complete` (never orphan).
- Dest-dir open failure after mkdir: **cancel reservation only** — never name-stat/commit whatever now sits at the name (may be foreign after a rename race).
- Cleanup is **descriptor-relative** (`unlinkat` / `rmdir` via dir fds + lazy `os.scandir` with cap+1 discrimination). **Never** `shutil.rmtree(path)`. Cleanup **never adopts** cleanup-time objects into the ledger.
- `reconcile_observed_against_owned_ledger` requires **exact equality**: observed−ledger (foreign) refuses without delete; ledger−observed (owned missing/moved) refuses.
- `FrozenBundle.dup_dir_fd()` returns a **caller-owned** duplicate; concurrent cleanup closes only the bundle-owned fd.
- Ordinary owned partial model bytes are destroyed on construction failure; incomplete cleanup retains a pending lease on the store for retry at `close()`.

### Honest residual (non-atomic mkdir+fd binding)

Portable POSIX cannot atomically mkdir and bind a directory fd to the newly created inode. Therefore:

- If `mkdir` succeeds and the subsequent `open(O_DIRECTORY)` fails (including a rename race that swaps a foreign empty directory into the name), the implementation **cancels the reservation only** and does not name-stat/commit the object at that name.
- A **safe empty directory shell** may remain (the original owned empty dir, possibly renamed away by an adversary). **No model bytes** are left, and nothing is digested or loaded.
- The stat/rmdir race test exercises identity mismatch and post-stat/pre-rmdir replacement: foreign replacements survive; owned empty shells may remain.

### Explicit non-goals

No family wiring (C2), production pin manifest (C3), degraded/503 mapping (C4), enumerator guard-assertion (C5), full golden matrix packaging (C6), Track E, Phase B, reload re-open, retraining, enablement. No part/v16 KIND decision. No production workflow edits in this authority-fix pass.

---

## 2. Local verification evidence

**Environment:** macOS Darwin, Python 3.11 (`/opt/homebrew/bin/python3.11`). `openat2` unavailable → component-walk exercised; openat2 cases skipped (not false-green). `last_open_impl()` distinguishes actual openat2 use when present (Linux CI).

**Commands:**

```sh
PYTHONPATH=. python3.11 -m pytest -q tests/unit/test_model_activation_c1_core.py
python3.11 -m flake8 \
  src/core/model_activation/__init__.py \
  src/core/model_activation/digest.py \
  src/core/model_activation/fd_dir.py \
  src/core/model_activation/resolver.py \
  src/core/model_activation/store.py \
  src/core/model_activation/types.py \
  tests/unit/test_model_activation_c1_core.py \
  --max-line-length=100
python3.11 -m mypy src/core/model_activation --config-file mypy.ini
git diff --check
```

**Result (local, this worktree, post authority-fix + audit fixes + nested-pending parent-fd ownership):** **109 passed, 11 skipped, 0 failed.** flake8 clean on the seven source files + test; mypy clean (7 source files); `git diff --check` clean.

**Docker:** daemon unavailable on the verification host (`Cannot connect to the Docker daemon at unix:///Users/chouhua/.docker/run/docker.sock`). Linux root / uid `65534` suite **not** re-run here.

**Discriminating tests added (authority-fix + audit follow-ups):**

| Test | Proves |
|---|---|
| `test_finalize_fstat_failure_after_ocreat_leaves_no_model_byte` | P1: fstat fail after O_CREAT → no `cadml-freeze-*/m.bin` residual; path-safe refusal |
| `test_inventory_max_1_leaves_no_model_byte` | P1: inventory max=1 reserves before create → no model-byte residual |
| `test_dest_dir_open_failure_race_preserves_foreign_no_mbin` | mkdir nested dest + open race: foreign survives, zero `m.bin`, owned empty shell may remain |
| `test_ordinary_dest_dir_open_failure_zero_model_bytes` | Ordinary dest-dir open failure → zero model bytes |
| `test_reconcile_foreign_survives_and_refuses` | observed−ledger foreign not deleted |
| `test_reconcile_foreign_dir_refuses_before_descent_no_recursion` | P1: foreign dir (80-deep under `setrecursionlimit(50)`) → typed `FREEZE_MUTATED` before descent; no `RecursionError` |
| `test_reconcile_owned_missing_refuses` | ledger−observed missing/moved refuses |
| `test_retained_lease_release_serialized_two_threads` | P1: concurrent `release()` max body concurrency 1; both results honest |
| `test_scrub_pending_success_pops_ledger_reaches_complete` | P1: pending remove pops ledger; closes child + owned parent dup; `cleanup_complete=True`; zero model residual |
| `test_scrub_pending_mismatch_retains_pending_incomplete` | pending name mismatch keeps same child + owned parent fd; foreign untouched; incomplete |
| `test_nested_pending_file_survives_parent_dir_fd_close` | P1: nested `sub/m.bin` pending after closing `sub_fd` still scrubs; sibling `already.bin` MODEL-BYTES destroyed; pending/ledger empty; all owned fds closed |
| `test_nested_pending_dir_survives_parent_dir_fd_close` | P1: nested pending directory under closed mid parent still scrubs; sibling model bytes destroyed; complete |
| `test_nested_pending_mismatch_retains_ownership_and_foreign` | nested pending mismatch retains child+parent dups and sibling ownership; foreign at name untouched; honest incomplete |
| `test_adopt_pending_parent_dup_failure_no_orphan_no_claim` | parent-fd dup failure: best-effort destroy via live parent; no orphan; no cleanup claim |
| `test_adopt_pending_parent_dup_failure_retains_when_destroy_fails` | parent-fd dup + destroy both fail: child retained on pending (`parent_fd=-1`); `cleanup_complete=False` |
| `test_pending_leases_retained_by_identity_not_value_eq` | store pending list uses identity (`eq=False` leases) |
| `test_ast_refusal_mapping_completeness_guard` | no `raise … from` inside OSError handlers; mapper surfaces present |
| `test_refusal_no_oserror_context_*` / `test_list_dir_fd_oserror_no_context` / `test_read_fd_oserror_no_context` | `__cause__`/`__context__` None; no hostile path in traceback |
| `test_dup_dir_fd_caller_owned_and_cleanup_concurrent` | caller-owned dup; cleanup then refuse further dup |
| `test_stat_rmdir_race_identity_mismatch` | identity mismatch at cleanup; foreign survives |
| `test_openat2_enosys_runtime_is_internal_no_same_request_fallback` | runtime ENOSYS → INTERNAL, no component fallback same request |
| `test_enotdir_maps_to_not_directory` | ENOTDIR → NOT_DIRECTORY (not SYMLINK_REJECTED) |

**Review-blocker closure (#528 review 2026-07-20, "vacuous destroy-partial-freeze observed-RED"):**

- `test_bundle_digest_mismatch_red` now asserts `cadml-freeze-* == []` after the refusal — this
  path genuinely creates+seals a freeze before the mismatch, so the assertion is load-bearing.
- `test_partial_freeze_cleanup_on_mid_walk_failure` rewritten to a **genuine mid-Pass-2 failure**:
  `os.write` raises `EIO` on the second member; mkdir/write spies prove the freeze root existed and
  was partly populated before the destroy (`mkdir ≥ 1`, `write ≥ 2`, leftovers `== []`).
- The old vacuous symlink variant is renamed `test_bundle_symlink_member_red_pass1_no_freeze_created`
  and now asserts the honest claim: zero freeze `mkdir` calls — Pass-1 refusal precedes any freeze.
- `test_bundle_per_file_bytes_red` closes the one uncovered cap (`BUNDLE_PER_FILE_BYTES`).
- **Mutation evidence** (review's empirical bar, now inverted): neutering
  `FreezeResourceLease.release()` → **10 tests fail**; skipping only the `assert_bundle_digest`
  failure-path `lease.release()` call → **4 tests fail** (including both tests named above). The
  review's original proof — "deleting the cleanup keeps the whole suite green" — no longer holds.

Would-fail-against-old notes (practical):

- Old path used `raise … from exc` → `__context__` held path-bearing OSError (AST + traceback tests).
- Old path used `shutil.rmtree` + default `tempfile.gettempdir()` (parent validation / no-default-temp tests).
- Old path closed O_CREAT fd without lease adopt-before-fstat (fstat-failure residual test targets that window).
- Old path had no inventory reserve-before-create (max=1 residual test).
- Old path had no dest-dir open cancel-only race handling (foreign could be name-stat committed or deleted).
- Old path borrowed the caller's parent dir fd on `_PendingNode` (nested freeze-walk close of `sub_fd` after create fstat failure → `release()` forever False; sibling `already.bin` MODEL-BYTES residual — nested-pending parent-fd ownership tests target that window).

| Area | Positive (GREEN) | Observed-RED / discriminators |
|---|---|---|
| Exact file hash | single-file exact match | mismatch, pin-absent, missing artifact |
| KIND | — | wrong KIND both directions |
| Raw pin domain | valid relative pins | `//`, `/./`, `/../`, NUL, absolute, trailing `/` |
| Resolver | component green; openat2 green when available + `last_open_impl` (incl. BUNDLE root) | both impls reject same raw set; openat2 BUNDLE intermediate-symlink RED via OPENAT2 spy; ENOTDIR; runtime ENOSYS INTERNAL |
| Same-fd TOCTOU | returned bytes; inode swap | growing file |
| Preflight | — | dir/file/aggregate caps → **zero `os.write` counts** |
| DFS / RLIMIT | 200 sibling empty dirs with `RLIMIT_NOFILE=128` | — (would UNREADABLE under O(n) stack) |
| Bundle digest | exact + determinism; source mutation after freeze | digest mismatch; **freeze-write corruption → DIGEST_MISMATCH** |
| Freeze handle | `read_member` / `dup_dir_fd`; no public `path` / borrowed `dir_fd` | sealed mode **0400** always; in-place write OSError when DAC applies (`euid!=0`); path-redirect still reads good |
| Lease / ledger | trusted parent; reserve/adopt/finalize; pending owns parent-dir dup | fstat-after-O_CREAT residual closed; inventory max=1; dest-dir race foreign survives; nested pending survives parent close + scrubs sibling model bytes; parent-dup failure no-orphan; reconcile exact equality |
| fd_dir | `os.listdir(dir_fd)` happy | OSError → `UNREADABLE` fail-closed; no context leak |
| Bounds bombs | — | dirent/dir/depth/relpath/file-count |
| Path-safe refusals | — | AST guard; traceback contains no hostile path |

**Skips (genuine platform limits only):**

- `openat2_*` (single-file green, BUNDLE green, BUNDLE intermediate-symlink RED, raw-domain parity matrix, component/openat2 green parity, runtime ENOSYS) — kernel/syscall not present on Darwin; not false-green. Runtime ENOSYS test skips when openat2 unavailable.
- `test_bundle_non_utf8_entry_red` — APFS rejects illegal byte sequence names at create.
- **Privilege note (not a local skip under non-root):** `test_freeze_inplace_mutation_red` always asserts sealed file mode `0400`; the write-refusal portion is omitted only when `geteuid()==0` because root bypasses 0400 DAC.

**CI-not-yet-run:** GitHub Actions (Linux + real `SYS_openat2` + non-UTF-8 dirents) has **not** been executed from this worktree. Curated L3 steps list this suite; this document does **not** claim CI green.

### Linux Docker verification

When Docker is available:

```sh
docker run --rm -v "$PWD:/workspace:ro" -w /workspace python:3.11-slim \
  sh -lc 'pip install -q pytest && PYTHONPATH=. python -m pytest -q tests/unit/test_model_activation_c1_core.py'
docker run --rm --user 65534:65534 -e HOME=/tmp -v "$PWD:/workspace:ro" \
  -w /workspace python:3.11-slim sh -lc \
  'python -m venv /tmp/venv && /tmp/venv/bin/pip install -q pytest && \
   PYTHONPATH=. /tmp/venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_model_activation_c1_core.py'
```

**Local Docker status:** recorded at verification time in §2b (pass/blocker).

---

## 2b. Docker verification

**Blocker (this worktree host, re-checked):** Docker daemon not running —

`Cannot connect to the Docker daemon at unix:///Users/chouhua/.docker/run/docker.sock. Is the docker daemon running?`

Cannot start `python:3.11-slim` as root or as uid/gid `65534:65534`. Linux/`SYS_openat2` /
non-UTF-8 dirent / root-vs-65534 freeze-mode matrix is therefore **not** re-run here; Darwin local
results in §2 stand. When Docker is available, use the commands in §2.

---

## 3. CI routing (minimum)

`.github/workflows/ci.yml` and `.github/workflows/ci-tiered-tests.yml` L3 step includes
`tests/unit/test_model_activation_c1_core.py`. No production pins; no family wiring job.
**This authority-fix pass does not edit workflows** unless a genuine routing gap requires it (none required).

---

## 4. Unresolved risks / design notes

1. **openat2** uses raw `syscall(SYS_openat2)` for known Linux arches; probe treats ENOSYS / EPERM / EACCES as unavailable. Local Darwin cannot execute the openat2 body — Linux CI must.
2. **Sealed freeze modes** stop casual owner writes; a privileged actor can still `chmod`+mutate the backing tree if they learn the private path. The activation contract is: digest was over freeze bytes, and **reads go through the held dir fd** (path-redirect does not rebind the fd). Callers must not re-open a path string; use `dup_dir_fd()` if a dir fd is required.
3. **Default `BoundPolicy` numbers** are core defaults; per-family caps are C2/C3.
4. **`load_pinned_*`** return verified bytes / `FrozenBundle` fd-handle — framework load remains C2.
5. Wave-1 owner questions (part/v16 KIND; reload id shape) still block later family modeling, not C1.
6. **Empty-shell residual** under non-atomic mkdir+open failure / dest-dir rename race is documented and accepted; model bytes must not remain.
7. **Reconcile** refuses foreign identities **before** directory descent (no recursive foreign walk; global inventory bound; typed `FREEZE_MUTATED`, not `RecursionError`).
8. **`FreezeResourceLease.release()`** is serialized under a per-lease lock (safe for concurrent store drain / bundle cleanup).

---

## 5. Changed files (this slice + authority-fix)

- `src/core/model_activation/` — C1 package (authority-fix: lease, path-safe raises, resolver, dup_dir_fd)
- `tests/unit/test_model_activation_c1_core.py` — positive + RED suite + authority-fix discriminators
- `docs/development/L3_PHASE_A_C1_MODEL_ACTIVATION_CORE_DEV_AND_VERIFICATION_20260719.md` — this document

**Not changed:** `pin_domain.py` (unless a proven defect — none), model families, `/model/reload`, enumerator, Track E, Phase B, production pins, workflows, PR #527 / wave1 audit worktree.
