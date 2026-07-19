# L3 Phase A — C1 model-activation core — Dev & Verification (2026-07-19)

> **Scope & honesty frame.** This documents **only the C1 reusable core** delivered in this worktree:
> controlled store, immutable pin records, raw-pin domain, store-root-anchored resolver, single-file
> same-fd SHA-256 activation, and bundle `tree-digest-v1` over a service-private **sealed** freeze
> handled by a held directory **fd** (not a public mutable path). It does **NOT** wire any model
> family, create production pins, enable reload/retraining, implement C2–C6, Track E, or Phase B.
> `merged != enabled != safe`. Grounded on `origin/main@7160694d` (#513 design lock + #526). Wave-1
> audit evidence is read-only elsewhere and is **not** edited here.

Authority: `docs/development/L3_MODEL_ACTIVATION_MEMBRANE_DESIGNLOCK_20260712.md` (ratified #513) and
`docs/development/L3_SAFETY_DESIGN_AND_VERIFICATION_20260715.md` (Phase A item C1).

---

## 1. What C1 delivers

| Surface | Location | Contract |
|---|---|---|
| Package | `src/core/model_activation/` | Narrow public API; no family imports |
| Types | `types.py` | `ArtifactKind`, `PinRecord`, `BoundPolicy`, path-safe `RefusalReason`; **`FrozenBundle` = dir-fd handle** (no public `path`) |
| Raw pin domain | `pin_domain.py` | Split-on-`/` **before** any `Path`/`os.path` normalization |
| Resolver | `resolver.py` | Linux **`syscall(SYS_openat2, …)`** with `RESOLVE_BENEATH\|RESOLVE_NO_SYMLINKS` when available (ENOSYS → component walk); never `resolve()+open`; shared raw pre-gate |
| Digests | `digest.py` | Single-file SHA-256; ratified `tree-digest-v1` |
| Fd readdir | `fd_dir.py` | **`os.listdir(dir_fd)`** (Python 3.10+ Unix); OSError → fail-closed; no silent partial list |
| Store | `store.py` | Immutable pins; two-pass bundle (metadata preflight → DFS freeze); **digest from frozen snapshot**; O(depth) source fds |

### Required behavior — implemented (post review-blocker fixes)

1. Server-owned store root + immutable pins; absent pin fail-closed; no re-pin / path-swap / network / hot-swap.
2. Shared raw POSIX pin pre-gate for both resolver implementations.
3. Store-root-anchored containment (`O_NOFOLLOW` / openat2 resolve flags); `O_NONBLOCK` on file leaves.
4. Bounded policy; refuse special/symlink/non-UTF-8/malformed — never skip.
5. Single-file: same fd, bounded read once, hash those bytes, return those bytes.
6. Bundle **(b-ii)**:
   - **Pass 1 — metadata preflight** (descriptor-relative DFS): refuse metadata-detectable limits **before any freeze copy** (zero destination writes).
   - **Pass 2 — freeze**: same-fd bounded copy, full bound recheck, destroy partial freeze on failure; live source dir fds are **O(depth)**, never O(directory count).
   - **Digest from freeze**: after seal, recompute `tree-digest-v1` by reading the frozen tree via its dir fd (not source-side copy buffers).
7. Typed path-safe refusals.
8. KIND-narrow API; does not decide part/v16 composite KIND.

### FrozenBundle contract (immutability)

- Preferred handle: held freeze **directory fd**.
- **No public mutable absolute path** (a path is renamable/repointable and is not the immutability mechanism).
- Reads: `open_member` / `read_member` relative to the held fd.
- Freeze tree sealed (files `0400`, dirs `0500`); `cleanup()` owns private backing path + fd.
- Observed-RED: in-place post-freeze write fails; path-redirect of backing does not change fd reads.
- GREEN: source mutation after freeze does not change fd reads.

### Explicit non-goals

No family wiring (C2), production pin manifest (C3), degraded/503 mapping (C4), enumerator guard-assertion (C5), full golden matrix packaging (C6), Track E, Phase B, reload re-open, retraining, enablement. No part/v16 KIND decision.

---

## 2. Local verification evidence

**Environment:** macOS Darwin, Python 3.11.15. `openat2` unavailable → component-walk exercised; openat2 cases skipped (not false-green). `last_open_impl()` distinguishes actual openat2 use when present (Linux CI).

**Commands:**

```sh
python3.11 -m pytest tests/unit/test_model_activation_c1_core.py -v --tb=short
python3.11 -m flake8 src/core/model_activation tests/unit/test_model_activation_c1_core.py --max-line-length=100
python3.11 -m mypy src/core/model_activation --config-file mypy.ini
docker run --rm -v "$PWD:/workspace:ro" -w /workspace python:3.11-slim \
  sh -lc 'pip install -q pytest && python -m pytest -q tests/unit/test_model_activation_c1_core.py'
docker run --rm --user 65534:65534 -e HOME=/tmp -v "$PWD:/workspace:ro" \
  -w /workspace python:3.11-slim sh -lc \
  'python -m venv /tmp/venv && /tmp/venv/bin/pip install -q pytest && \
   /tmp/venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_model_activation_c1_core.py'
```

**Result (local, this worktree, post-blocker fixes):** **65 passed, 11 skipped, 0 failed.** flake8 clean; mypy clean (7 source files).

**Linux `python:3.11-slim` verification:** **76 passed, 0 skipped, 0 failed** as root and again as UID/GID `65534:65534`. This exercises the real Linux `SYS_openat2` path, non-UTF-8 directory-entry refusal, and both privilege variants of the freeze-mode test.

| Area | Positive (GREEN) | Observed-RED / discriminators |
|---|---|---|
| Exact file hash | single-file exact match | mismatch, pin-absent, missing artifact |
| KIND | — | wrong KIND both directions |
| Raw pin domain | valid relative pins | `//`, `/./`, `/../`, NUL, absolute, trailing `/` |
| Resolver | component green; openat2 green when available + `last_open_impl` (incl. BUNDLE root) | both impls reject same raw set; openat2 BUNDLE intermediate-symlink RED via OPENAT2 spy |
| Same-fd TOCTOU | returned bytes; inode swap | growing file |
| Preflight | — | dir/file/aggregate caps → **zero `os.write` counts** |
| DFS / RLIMIT | 200 sibling empty dirs with `RLIMIT_NOFILE=128` | — (would UNREADABLE under O(n) stack) |
| Bundle digest | exact + determinism; source mutation after freeze | digest mismatch; **freeze-write corruption → DIGEST_MISMATCH** |
| Freeze handle | `read_member` via dir fd; no public `path` | sealed mode **0400** always; in-place write OSError when DAC applies (`euid!=0`); path-redirect still reads good |
| fd_dir | `os.listdir(dir_fd)` happy | OSError → `UNREADABLE` fail-closed |
| Bounds bombs | — | dirent/dir/depth/relpath/file-count |

**Skips (genuine platform limits only):**

- `openat2_*` (single-file green, BUNDLE green, BUNDLE intermediate-symlink RED, raw-domain parity matrix, component/openat2 green parity) — kernel/syscall not present on Darwin; not false-green.
- `test_bundle_non_utf8_entry_red` — APFS rejects illegal byte sequence names at create.
- **Privilege note (not a local skip under non-root):** `test_freeze_inplace_mutation_red` always asserts sealed file mode `0400`; the write-refusal portion is omitted only when `geteuid()==0` because root bypasses 0400 DAC.
**CI-not-yet-run:** GitHub Actions (Linux + real `SYS_openat2` + non-UTF-8 dirents) has **not** been executed from this worktree. Curated L3 steps list this suite; this document does **not** claim CI green.

---

## 3. CI routing (minimum)

`.github/workflows/ci.yml` and `.github/workflows/ci-tiered-tests.yml` L3 step includes
`tests/unit/test_model_activation_c1_core.py`. No production pins; no family wiring job.

---

## 4. Unresolved risks / design notes

1. **openat2** uses raw `syscall(SYS_openat2)` for known Linux arches; probe treats ENOSYS as unavailable. Local Darwin cannot execute the openat2 body — Linux CI must.
2. **Sealed freeze modes** stop casual owner writes; a privileged actor can still `chmod`+mutate the backing tree if they learn the private path. The activation contract is: digest was over freeze bytes, and **reads go through the held dir fd** (path-redirect does not rebind the fd). Callers must not re-open a path string.
3. **Default `BoundPolicy` numbers** are core defaults; per-family caps are C2/C3.
4. **`load_pinned_*`** return verified bytes / `FrozenBundle` fd-handle — framework load remains C2.
5. Wave-1 owner questions (part/v16 KIND; reload id shape) still block later family modeling, not C1.

---

## 5. Changed files (this slice)

- `src/core/model_activation/` — C1 package
- `tests/unit/test_model_activation_c1_core.py` — positive + RED suite (65 pass / 11 skip local)
- `docs/development/L3_PHASE_A_C1_MODEL_ACTIVATION_CORE_DEV_AND_VERIFICATION_20260719.md` — this document
- `.github/workflows/ci.yml`, `.github/workflows/ci-tiered-tests.yml` — curated L3 test list only

**Not changed:** model families, `/model/reload`, enumerator, Track E, Phase B, production pins, PR #527 / wave1 audit worktree.
