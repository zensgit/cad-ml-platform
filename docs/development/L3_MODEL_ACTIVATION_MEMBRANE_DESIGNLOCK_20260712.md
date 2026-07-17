# L3 Design-Lock — Model-Release & Activation Proof Membrane

**Date**: 2026-07-12 (rev 2026-07-15 — canonical-strategy alignment) · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1) · **Grounded on**: `origin/main@c6625831`; activation-map count on `origin/main@f2ebe2fa` (2026-07-14; #516 enumerator + #519 hardening merged). The import-aware CI enumerator — the count authority, NOT a hand-count — executes on current `main` as **128 sites / 38 marked `gated` / 11 families**. Its summary was corrected to the conservative-AST / (b) wording by #521 (merged `a0e517e8`), so the earlier "production-reachable" / rejected-(a) blanket-hard-refuse labels are no longer emitted. This lock uses the executed count while treating `gated` as a conservative AST classification: per-site logical reachability is a **Wave-1 audit**, NOT asserted here, and several sites are latent or not-yet-proven-reachable (§1.B(cont)); a hand-count has been wrong ≥4 times.
**Authority**: `PRODUCT_STRATEGY.md` §4 (AI safety), §5.2 (evaluation integrity not release-grade),
§8.1 (Track E). Scheduled deliverable for the 7/20–7/26 week; pulled forward because the runtime
work is P0-blocked and a design-lock is a doc, not runtime.

> **This is a proposal, not an implementation.** It changes no runtime, touches no model-activation
> code, and does not itself close any surface. It defines the contract the future implementation must
> satisfy so it can be ratified before any code is written — precisely the "propose, don't build"
> mode L3 requires. Unattended routines may not author or merge this runtime surface.

### Solo-maintainer L3 review protocol

This repository currently has one human developer. That makes a second human GitHub approval
unavailable; it does **not** justify fabricating one or silently lowering L3 rigor. For this repository,
`PRODUCT_STRATEGY.md` §4.7 and §7.1 are satisfied by the following compensating protocol:

1. An isolated critic that did not author the patch first derives the load-bearing discriminators
   from this ratified design lock, then reviews the implementation. Use a different model when
   available; at minimum use a fresh context with no inherited implementation rationale.
2. The PR records the critic findings, fail-first golden, observed-RED run, positive controls for the
   verifier itself, and the exact final head reviewed. A material post-review change invalidates that
   evidence and requires another pass.
3. The critic provides **evidence, not approval**. The sole human owner explicitly ratifies the design
   and separately authorizes the pinned implementation head. A second account controlled by the same
   person is not independent review.
4. Merge, deployment, and enablement are separate decisions. The implementation lands default-off.
   **Dynamic** activation and any **re-enablement** of `/model/reload` remain blocked on the proof
   membrane, the production-identity gate, and the Track E evidence. Ratifying this lock authorizes
   building **Phase-A baseline containment**, not enabling it and not promoting a model: before Track E,
   Phase A may load only the exact already-in-service
   `(logical_activation_id, artifact_id, kind, digest)` tuple (§0.5/§5).

`.github/CODEOWNERS` may inventory this surface, but `require_code_owner_reviews` remains false while
the repository has only one developer; enabling it would create an impossible approval gate, not an
independent critic. If an independent human reviewer becomes available later, their review strengthens
this protocol but is not simulated in the meantime.

---

## 0. Why this exists (and why the first attempt was insufficient)

#509 made `scripts/auto_retrain.sh` unconditionally fail-closed. That was correct **but narrow**, and
an earlier claim that it "closed retraining on main" was **overstated**. Corrected here (second
review): `auto_retrain.sh` is a **producer** (it prints a deploy command, it does not activate a
running service), so #509 closes **none of the runtime activation points** (the import-aware CI
enumerator executes as 38 entries marked `gated` across 11 model families; this lock does not inherit
the **pre-#521** "production-reachable" label (REMOVED when #521 merged `a0e517e8` — the current
     enumerator summary emits the conservative-AST framing, not "production-reachable") — reachability is a **Wave-1 audit** (§1.B(cont)) —
see §1.B; a hand-count has been wrong ≥4 times, hence the CI-enumerator contract). It is
bleeding-control one layer upstream; the runtime membrane is unbuilt.

Corrections from review, all load-bearing:

1. **Model families are distinct release paths, not one path's bypasses.** `POST /api/v1/model/reload`
   hot-reloads the **pickle classifier** (`CLASSIFICATION_MODEL_PATH` at `src/ml/classifier.py:22`, loaded at `:85`; the hot-reload path is `reload_model` at `:227` → `pickle.loads` at `:535`).
   `auto_retrain.sh`/#509 promotes the **Graph2D checkpoint** (`GRAPH2D_MODEL_PATH`, resolved in
   `Graph2DClassifier.__init__` (`src/ml/vision_2d.py:40-41`) and loaded via `torch.load` at
   `src/ml/vision_2d.py:136`). These are **different models with
   different activation surfaces**. The membrane must be **cross-family**.

2. **A file-reference count is a discovery list, not a boundary.** The 15 `*_MODEL_PATH` / `load` hits
   are not 15 production entry points. The acceptance goal is *"every production-reachable activation
   point passes the **phase-appropriate activation gate**"* — Phase A: a fixed-`SHA-256` pin; Phase B:
   the signed proof — **not** mechanically wrapping every `load()`.

---

## 0.5 Phasing — pin first (Phase A: static fixed-hash), prove later (Phase B: signed proof)

**PROPOSED phasing — no ratification has occurred.** This doc is for-review; the owner alone
ratifies it (no owner review/comment/pinned-head ratification exists as of this writing). The
proposal is that the membrane ships in **two separately-ratifiable phases**.
Phase A is cheap containment that needs **no** cryptographic proof store; Phase B is the full proof
membrane and is **deferred until a real pilot needs dynamic model-swap**. Building the signed store now
(Phase B) would spend ~2–3 weeks and add **no customer evidence**. The external `/model/reload` boundary is now **sealed (#516, 403)**; the remaining gap is
the **internal `gated` loaders** still unpinned (of the 38 conservatively-`gated` sites; per-site reachability is a **Wave-1 audit** and several are latent/unproven — §1.B(cont)). Pin the reachable ones first (owner decision (b)).

### Phase A — Static-artifact activation (fixed-`SHA-256` pin; build next, after **this** lock ratifies — the internal fixed-hash loaders do NOT depend on caller identity; the production-identity gate only gates a future re-open of `/model/reload`)

**Goal:** no arbitrary, caller-path, or hot-swapped model activation can occur in production — with
**no signing keys / no signed proof store** (a plain content-hash comparison only; the signed proof
is Phase B). Per the owner's decision (b), models still load from pinned server-owned artifacts, but
before Track E that means **only the exact already-in-service
`(logical_activation_id, artifact_id, kind, digest)` tuple**. Phase A is a containment migration, not
a model-promotion path.
1. **Production-disable the external `POST /api/v1/model/reload` route — no environment-variable bypass.**
   It was LIVE (§1.A/§3.2); **#516 has already sealed it (403)**. It refuses unconditionally; **no flag or env var re-opens it** (re-opening is a Phase B + identity-gate
   decision, §3.2). A request carrying a valid `path`+`force`+default `test` creds must not reach
   `reload_model`. (This route is DONE — #516.)
2. **Static fixed-hash activation at every §1.A/1.B `gated` site (owner decision (b)).** Each `gated`
   loader may activate a model **only** from a *server-owned* artifact resolved from a controlled store:
   **no caller-influenced path, no env-var path-swap, no dynamic replacement, and no network fetch at
   load** (offline-only). An unclassified/unpinned load cannot activate. This is a subset of §3's
   `verify_and_load` **without** the signed proof store (Phase B).

   **Shared bounded pre-read (Phase A AND Phase B).** Before a full file read, bundle copy, digest, or
   framework loader runs, reject an artifact whose declared KIND/type is wrong, whose single-file size
   exceeds its family limit, or whose bundle exceeds a bounded **file-count, per-file-size, aggregate-byte,
   total-entry-count (ALL dirents — files, directories, AND any other entry type), directory-count,
   traversal-depth, or per-entry relpath byte-length** limit (the relpath byte-length cap also bounds the
   `tree-digest-v1` record encoding — §0.5 step-2 — since each record embeds the UTF-8 relpath). Validate file type / a bounded magic prefix where the family has one; for an unpacked
   bundle, use `lstat` metadata plus only bounded prefixes, never read the full tree to decide whether it
   is safe to copy — so metadata-detectable violations are refused **before any copy**. A
   malformed/unreadable entry or metadata overflow is a refusal, not a best-effort
   skip; a per-file/aggregate cap that can only be crossed **mid-freeze** under (b-ii) (e.g. a source file
   that grows after `lstat`) is likewise not silently accepted — the copy refuses and the **partial freeze
   is DESTROYED** (§0.5 step-2 (b-ii); nothing partially-frozen is ever digested or loaded). These limits
   are server policy, not caller or environment input.

   Two artifact KINDS — because several `gated` families are NOT single files (review 7c). Both kinds
   open their pinned path through ONE shared resolver:

   **Store-root-anchored path resolver — shared by BOTH kinds; `resolve()` is FORBIDDEN.** POSIX
   `O_NOFOLLOW` protects ONLY the final path component, so a leaf-only `O_NOFOLLOW` is insufficient — a
   symlink at an **intermediate** directory (e.g. `store/family` → outside the store) is silently
   followed and the external file is read (owner-verified). **Pinned-relpath precondition (enforced
   in-process on the RAW POSIX pin string, BEFORE any `openat` AND BEFORE any
   `Path()`/`PurePosixPath`/`os.path` normalization):** the check operates on the **raw pin bytes split on
   `/`** — running ANY path normalization first is **FORBIDDEN**, because normalization silently swallows
   the illegal components this check exists to catch (`Path("a//b")` and `Path("a/./b")` both collapse to
   `a/b`, so an `//`, `.`, or `..` component would never be seen). On that raw split the pin MUST be a
   **relative** path with **no absolute prefix** and **no `.`, `..`, or empty component** (an empty
   component is exactly what a `//`, a leading `/`, or a trailing `/` produces in the raw split), and the
   raw pin MUST contain **no NUL (`0x00`) byte anywhere** — the NUL rejection applies to the whole pin
   string here, not only to readdir entry names (which §0.5-below covers separately). ANY absolute pin,
   ANY `.`/`..`/empty component, or ANY NUL byte is REJECTED fail-closed (`degraded`/503) before the walk
   starts. This precondition is REQUIRED, not decorative,
   because **`O_NOFOLLOW` does NOT stop `..`**: a `..` component is not a symlink, so
   `openat(dir_fd, "..", O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)` **succeeds** and walks ABOVE the store root
   (owner-verified escape) — the per-component walk cannot rely on `O_NOFOLLOW` alone to stay contained,
   so the `..`/`.`/absolute/empty/NUL components must be rejected up front on the raw split. Both the single-file leaf and the
   bundle root are then opened by walking the validated pinned **relative** path
   **component-by-component from a trusted, pre-opened store-root directory fd**: each **intermediate
   directory** via `openat(dir_fd, name, O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)` (a symlink or non-directory
   intermediate → open fails / `fstat` rejects → `degraded`/503), and the **terminal node opened per
   KIND** — because the shared resolver serves BOTH a single-file leaf (a **regular file**) AND a bundle
   root (a **directory**), it MUST NOT blanket-require `S_ISREG` (that would reject EVERY legitimate
   bundle, whose terminal node is a directory): for a **single-file** pin the final component is opened
   `openat(dir_fd, leaf, O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)` then `fstat` requiring **`S_ISREG`**
   (rejecting symlink / FIFO / socket / block- or char-device); for a **bundle** pin the final component
   is the bundle **root directory**, opened `openat(dir_fd, root, O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)` then
   `fstat` requiring **`S_ISDIR`** (its member files are each opened later
   `openat(root_fd, name, O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)` + `fstat`(`S_ISREG`) per entry,
   §0.5 step-2 (b-ii)). **`O_NONBLOCK` on the single-file leaf / each bundle-entry-file open is
   REQUIRED**: opening a FIFO WITHOUT it blocks waiting for a writer BEFORE `fstat` can run, so the
   special-file rejection would be unreachable. **The raw-pin precondition above is a SHARED pre-gate that
   runs FIRST on the raw pin for BOTH resolver implementations — it is NOT a property of the fallback
   alone.** On Linux prefer a single `openat2(store_root_fd, relpath, RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS)`
   for the walk; where `openat2` is unavailable, fall back to the component-by-component `openat` walk.
   **`RESOLVE_BENEATH`/`RESOLVE_NO_SYMLINKS` is NOT a substitute for the raw-pin precondition:** the kernel
   *collapses* `//` and `.` during `openat2` path resolution, and `RESOLVE_BENEATH` only blocks an
   *escape* — so a bare `openat2` would silently ACCEPT `a//b` / `a/./b` and never see an empty component.
   Therefore the raw split-on-`/` rejection of `.`/`..`/empty/NUL/absolute MUST run on the raw pin BEFORE
   either walk, so the two implementations reject the IDENTICAL input set (`RESOLVE_BENEATH` additionally
   rejects the `..`/absolute escape at resolution time, but the raw precondition — not `openat2` alone — is
   what makes the openat2 path and the component-walk fallback agree; a divergence is itself a RED, §5).
   **Falling back to `resolve()` + `open(path)` is FORBIDDEN** — it follows intermediate symlinks AND
   re-introduces a resolve→open TOCTOU. This resolver IS the containment mechanism (not a leaf flag); it
   supersedes every earlier leaf-only-`O_NOFOLLOW` / `resolve()`-containment phrasing below.

   - **single-file** (`torch.load`/`pickle.load`/`joblib.load`/`onnx.load` of one file — graph2d, part,
     hybrid, history, vision3d, pickle-classifier, anomaly-monitor): **the leaf MUST be a regular file**
     (same input-domain lock as bundle) — open it **via the store-root-anchored resolver above**, whose
     `fstat`(`S_ISREG`) rejects ANY non-regular entry (symlink / FIFO / socket / block- or char-device)
     and whose per-component walk rejects an intermediate-directory symlink → `degraded`/503, **even when
     the leaf does NOT escape the store root**; then **read the bytes ONCE from THAT SAME leaf fd**
     (bounded by the size cap), `SHA-256(bytes) ==` the pinned value, and load **THOSE** bytes. **Never
     re-open by path** between the `fstat` check and the read/load — that single open+fstat+read on one fd
     closes BOTH the hash→load race AND the **inode-swap-after-check** TOCTOU (a path re-opened after the
     check could resolve to a different inode). This is the original contract, input-domain-locked.
   - **bundle / tree** (`from_pretrained` / `SentenceTransformer` / `PaddleOCR` — ocr, embedding —
     which load a **directory of many files**, and may otherwise fetch from a network hub): the pin is
     a **deterministic, versioned tree digest** (`tree-digest-v1`) = `SHA-256` over a **canonical
     encoding** of the **sorted** list of `(posix-relpath, SHA-256(file-bytes))` for **every** file under
     the artifact root — **`tree-digest-v1` canonical encoding (fully pinned, no implementation freedom):**
     the digest covers **only regular files** (zero-byte files included); the directory is **traversal-only**.
     **Input-domain lock (no implementation freedom): on ANY symlink, ANY other non-regular entry
     (FIFO / socket / block- or char-device), ANY over-limit entry, OR ANY path whose POSIX relpath fails
     UTF-8 encoding, the activation is REJECTED fail-closed (`degraded`/503) — never silently
     "excluded"/"skipped" (an "exclude" would let two implementations diverge: one skips, one rejects).
     For the (b-i) atomic snapshot the whole input-domain check runs on the snapshot before any digest; for
     the (b-ii) per-entry freeze the rejection may occur mid-walk after earlier entries were already
     copied — in that case the service-private partial freeze is DESTROYED and NOTHING is digested or
     handed to the framework loader. The security property does NOT depend on "zero bytes were ever
     copied", only that no partially-frozen tree is ever digested or loaded. (Deliberate, owner-locked:
     an implementation MUST NOT try to restore a "reject before ANY copy" property by first opening and
     HOLDING a descriptor for EVERY entry before copying — the held-descriptor count would scale with
     bundle size and can exhaust the process file-descriptor limit; the locked shape is per-entry
     same-fd bounded-copy with destroy-partial-freeze on any failure.)** For
     each surviving regular file, the record's
     path is its **POSIX relpath from the artifact root, UTF-8-encoded**; `len` = the **byte length of that
     UTF-8 relpath, written as ASCII decimal**; each record = `len` · byte `0x1F` · relpath-bytes · byte
     `0x1F` · **lowercase-hex** `SHA-256(file-bytes)`; records are **sorted bytewise by the UTF-8 relpath**
     and **joined by byte `0x00`**; the tree digest is `SHA-256` of that whole byte string. Two independent
     implementations (the Phase-A fixed-hash body and Phase-B `verify_and_load`) therefore compute the
     **identical** digest for the same tree; a different encoding is a new version id (`tree-digest-v2`),
     never a silent change. The site (a) resolves a
     server-owned artifact id to a controlled, **read-only** (access-control only — NOT a load-duration
     immutability guarantee, which is why (b) copies), **already-unpacked** local directory that has
     passed the file-count / per-file / aggregate-byte / total-entries / directory-count / depth / relpath-length pre-read bounds —
     **never** a hub id, and with `HF_HUB_OFFLINE=1` / `local_files_only=True` so no network load can
     occur; the bundle root and every entry are opened through the **store-root-anchored resolver above**
     (per-component from the store-root fd; `resolve()` forbidden), so an intermediate-directory symlink
     cannot redirect the root outside the store; (b) **freezes an immutable snapshot** of that directory by ONE of two pinned implementations
     (no freedom to path-copy): **(b-i)** an **atomic read-only FS snapshot** taken BEFORE any scan, so the
     input-domain check, the file-count / per-file / aggregate-byte / total-entries / directory-count / depth / relpath-length bounds, the digest, AND the framework
     load ALL read that one snapshot; or **(b-ii) descriptor-relative freeze** — walk from the store-root fd
     via the **store-root-anchored resolver above** (intermediate dirs `openat(…, O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC)`;
     each file entry `openat(dir_fd, name, O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC)` — `O_NONBLOCK` so a FIFO
     entry does NOT block before `fstat`) + `fstat`(`S_ISREG`) per entry (rejecting ANY symlink / non-regular /
     intermediate-symlink entry BEFORE it is copied). **Traversal input-domain lock (no implementation freedom):**
     the recursive `readdir` enumeration MUST NEVER descend into the `.` or `..` dirents — they are NOT bundle
     members, and `..` is a **real directory** that `openat(dir_fd, "..", O_DIRECTORY|O_NOFOLLOW)` would follow
     ABOVE the artifact root (`O_NOFOLLOW` does NOT stop `..`, and it is a legitimate directory so neither
     `fstat`(`S_ISDIR`) nor the symlink check catches it); any entry name that is empty or contains a `/` or a
     NUL byte is likewise REJECTED fail-closed (`degraded`/503). **Resource-bound lock (total-entry-count /
     directory-count / traversal-depth / relpath byte-length; no implementation freedom):** the walk also
     enforces the total-dirent, directory-count, traversal-depth, and per-entry relpath-length caps DURING
     descent. A cap that is metadata-detectable in the pre-scan is refused there before any copy; a cap
     crossable only mid-walk/mid-freeze — a directory added between the pre-scan and the walk, or a depth
     threshold crossed as descent proceeds — refuses IMMEDIATELY and **DESTROYS the partial freeze** (same
     contract as the per-file/aggregate caps; nothing partially-frozen is digested or handed to the loader).
     Only the surviving in-tree regular-file entries
     are copied — each via **bounded-copy of its bytes from THAT SAME opened
     fd** into a **service-private freeze dir**. **Invariant (design-level; the concrete filesystem
     mechanism — creation mode, snapshot vs copy, how a path re-open is anchored — is Phase-A's, and
     Phase-A ships the observed-REDs):** once frozen, the snapshot is **immutable by construction** — a
     service-private copy that no other principal can **modify in place**, nor **redirect a re-open to** by
     swapping the snapshot tree or ANY ancestor of its path. **Phase-A ships fail-first goldens** for an
     in-place post-freeze content swap AND for a path-redirect (rename of the frozen dir or an ancestor) —
     both MUST RED. (The **(b-i)** atomic RO snapshot is immutable by definition; a **(b-ii)** service-owned
     freeze tree achieves the same — either way a re-open cannot resolve to attacker-controlled bytes; this
     is what makes returning the frozen path safe in the **guard-verification-contract escape hatch**,
     below.) On ANY rejected /
     over-limit / errored entry, immediately refuse
     and **destroy the partial freeze** (nothing partially-frozen is digested or loaded). **A path-based copy
     is FORBIDDEN**: a `shutil.copytree` / re-open-by-path
     re-reads the *still-mutable* source and re-exposes the **pre-scan → copy TOCTOU** — an entry swapped
     from a regular file to a symlink / FIFO / oversized file AFTER the pre-scan but BEFORE the copy would
     be followed, block, or over-read; a **bind-mount** re-exposes the source the same way. So the entry
     that is input-domain- and bounds-checked IS the entry that is copied and digested — a swap between
     pre-scan and copy is impossible, closing BOTH the **pre-scan→copy race** AND the **digest→framework-read
     race** (the tree analog of the single-file "one fd, never re-open by path");
     (c) recomputes the tree digest over that **frozen**, service-private directory (an immutable
     service-owned copy — source containment was ALREADY enforced by the store-root-anchored resolver during
     (a)/(b): the pinned-relpath precondition (on the RAW pin split-on-`/`, before any normalization) rejects an absolute pin, a NUL byte, or any `.`/`..`/empty component before the
     walk, the traversal never descends the `.`/`..` dirents, per-component `O_NOFOLLOW` from the store-root fd
     rejects symlinks, and `resolve()` is forbidden — so symlink / non-regular / `..`-or-absolute root-escaping
     entries are all refused up front, never post-hoc); (d) `== the pinned tree
     digest`, then hands the framework loader the **frozen snapshot's local path** (never the mutable
     original); else refuses fail-closed. A controlled unpack (if the release ships an archive) verifies
     the archive digest first and unpacks into the store with path-traversal rejection, before the
     frozen snapshot and its tree digest are taken.

   Either KIND refuses fail-closed on mismatch / unknown-id / missing / any containment escape or
   shared bounded-pre-read failure.
3. **CI activation-surface enumerator (§1/§3) — the completeness authority FOR DECLARED loader idioms** (import-aware torch/pickle/joblib/onnx `.load`, `load_state_dict`, `from_pretrained`, curated constructors, `reload_model(`; NOT a proof of every possible Python load — a novel framework escapes until its pattern is added). Marks every
   `torch.load`/`pickle.load(s)`/`joblib.load` (import-alias-aware) / `load_state_dict` / `from_pretrained` (HF) / model constructors (`SentenceTransformer`/`PaddleOCR`/…) / `reload_model(` site `gated|producer|offline|unmounted|infra` and
   REDS when a new un-annotated load MATCHING A DECLARED IDIOM appears, or a `gated` site is neither fixed-hash-checked (Phase A, owner decision (b)) nor routed through
   `verify_and_load` (Phase B). This replaces the hand-count (wrong ≥4×). Seed = the §1 map; authority = the
   IMPORT-AWARE enumeration (f2ebe2fa): **128 load sites total, 38 `gated`** across 11 families
   (pickle-classifier, graph2d, pointnet, part, part-v16, hybrid, history, vision3d-uvnet, **ocr** —
   DeepSeek HF `from_pretrained`+PaddleOCR via mounted /ocr — **embedding** — SentenceTransformer — and
   **anomaly-monitor** — the conservatively-gated production metrics model). The 38 is a **conservative
   count of AST load sites**, NOT a proven-live count: several are latent or not-yet-proven-reachable
   (the `auto_remediation` rollback; the `_reload_model_impl` hot-reload deserialization now that
   `/model/reload` is sealed; the two `MetricsAnomalyDetector.load_models` sites with no `src/` caller).
   **Per-site logical reachability is a Wave-1 audit**, not hand-asserted here (§1.B(cont)). The
   enumerator summary was corrected to the conservative-AST / (b) wording by #521 (merged
   `a0e517e8`); this document relies on current `main` for the executable count/classification data.
   A name-only matcher (review 5) missed the ocr/embedding families and
   import aliases entirely — the enumerator is now import-aware (+onnx/ort, review 6) and reds on any new un-annotated load matching a declared idiom.

4. **Guard-verification contract — the enumerator must PROVE each guard is wired, not just classify
   (review 7c).** #516's enumerator only classifies a site (`class`/`family`/`reason`); it is GREEN
   even with **zero** guards wired — that is correct for **Phase A0** (discovery), but full Phase A
   must make "the guard is present" a machine-checked fact. The manifest schema therefore gains, per
   **logical activation** (NOT per AST call site — see below):

   - `logical_activation_id` — one stable id per *logical* model activation. The 38 AST call sites are
     **not** 38 activations: a single `_load_model()` often has one `torch.load` **plus** several
     `load_state_dict` calls that together are **one** activation; the contract binds the id to the
     activation (its entry function), and the enumerator groups the member AST sites under it.
   - **Every `gated` load must be a raw `torch.load`/`pickle.load`/`from_pretrained`/… that appears
     ONLY inside one of the canonical wrappers** `load_pinned_file(artifact_id, family, env)` /
     `load_pinned_bundle(artifact_id, family, env)` (Phase B: `verify_and_load`). The wrappers are the
     **single sanctioned home** of every raw loader idiom; the enumerator whitelists the raw load lines
     *inside* those wrapper bodies and REDS on a raw `gated` load **anywhere else**. A call site
     activates a model only by calling the wrapper — deleting/renaming/inlining the wrapper makes a raw
     `torch.load`/`from_pretrained` reappear at a non-wrapper site → **observed-RED** (a test ships that
     deletes the wrapper and asserts CI reds). This is a structure the AST scanner CAN verify; an
     arbitrary inter-procedural "guard dominates the load" claim is one it CANNOT, so the contract does
     not depend on a manifest-self-reported `guard_symbol` string.
   - **Escape hatch, same-function only — and it MUST consume the verified immutable value, never re-open
     OR re-derive a mutable path:** where a raw load genuinely cannot move into a wrapper, it is accepted
     ONLY if an `assert_fixed_hash`/`assert_bundle_digest` call on the **same artifact** precedes it
     **lexically in the same function body** — a local dominance the AST can check with no inter-procedural
     reasoning — **AND** the raw load **consumes the exact immutable value RETURNED by that assertion, not
     a path it re-opens or re-derives.** A bare `assert_fixed_hash(path)` followed by a load that re-opens
     `path` is **REJECTED**: it re-introduces the hash→load TOCTOU (owner-verified — replacing the file
     AFTER the passing hash guard makes the loader read the swapped bytes). **Both what the assertion
     RETURNS and how the load site CONSUMES it are KIND-restricted to closed forms, because
     binding-identity alone is NOT enough — a returned *mutable path*, OR a path RE-DERIVED at the load
     site from a returned fd, is still swappable even though the load "consumes the same local":**
       - **single-file `assert_fixed_hash` MUST return the already-read in-memory `bytes` (the single
         ONCE-read copy) — NEVER a path, and NOT an `fd`.** A returned path is a re-openable swappable
         handle; a returned `fd` is also unsafe — the assertion must read the inode to hash it, and a second
         read at load re-reads the SAME inode, so an in-place content swap of the (not asserted-immutable)
         server-owned leaf BETWEEN the hash-read and the load serves un-hashed bytes. Only the **in-memory
         `bytes`** satisfy §0.5 step-2's "read the bytes ONCE … and load **THOSE** bytes" — one physical
         read, hash and load the same in-memory copy. The load MUST consume those in-memory bytes; the AST
         **REJECTS any load-site expression that re-opens or re-derives a path/fd** (an `open(...)`,
         `open(fd.name)`, or an fd→path recovery), and **Phase-A ships a fail-first golden for a post-hash
         in-place swap**. (A single-file loader that would side-load **external data by path** — e.g. an
         ONNX model referencing external tensors — is NOT a self-contained single file; it is classified and
         handled as a **bundle/tree** KIND, so the bytes-only rule is not circumvented.)
       - **bundle `assert_bundle_digest` MUST return the §0.5 step-2(b) frozen-snapshot handle — the
         directory fd (preferred) or its immutable-by-construction path — NEVER the mutable original
         directory.** The §0.5(b) freeze invariant (immutable by construction: no in-place swap and no
         path-redirect re-open, under (b-i) atomic snapshot or (b-ii) service-owned freeze tree; Phase-A
         ships the observed-REDs) is what makes returning the frozen path safe — not the "same local" alone;
         re-opening the original source is REJECTED.
     The AST check requires the raw load's source argument to be **the assertion's returned immutable value
     (the same local binding — resolved by reaching-definitions, so an intervening reassignment to a
     re-opened path/fd is caught)**, of the KIND-restricted form above (single-file: the in-memory `bytes`,
     never an `fd` and never an `open(...)`; bundle: the §0.5(b) frozen-snapshot fd or immutable path, never
     the source). Anything relying on a guard in a *different* function, or a load whose source is a
     re-opened/re-derived mutable path (or a re-read `fd`) rather than the assertion's immutable
     in-memory/frozen return value, is rejected.
   - `guard_mode` ∈ `{ sealed | fixed-hash | bundle-digest | verify_and_load | unbuilt }` — what
     protects this activation. `sealed` = the route is 403 (`/model/reload`, #516); `fixed-hash` /
     `bundle-digest` = the Phase-A (b) single-file / tree wrapper; `verify_and_load` = Phase B;
     `unbuilt` = no guard yet (all 38 conservatively-`gated` sites today).

   **The enumerator's Phase-A assertion:** for every `gated` `logical_activation_id`, `guard_mode` must
   be a real guard (not `unbuilt`) **and** the load must be structurally inside a canonical wrapper (or
   same-function-lexically guarded, per above) — verified from the AST, **never** from a self-reported
   symbol string; **deleting or bypassing the wrapper reds CI (an observed-RED test ships with it).**
   Until full Phase A wires the wrappers, the internal loaders are honestly recorded `guard_mode:
   unbuilt` and the assertion is **advisory** (A0) → **blocking** (full Phase A). This closes both the
   "38 sites all unguarded but CI green" gap and the "guard_symbol domination is unverifiable" gap.

**Phase A exit criteria (observed-RED, REQUIRED — NOT claimed executed here):** external
`/model/reload` refuses in prod with no env bypass; a new un-annotated prod loader REDS CI; **every**
`gated` §1.A/1.B site activates ONLY via the §0.5 step-2 per-KIND check (single-file fixed-`SHA-256` / bundle tree-digest) over a server-owned artifact (owner
decision (b)) — no caller path, no env path-swap, no dynamic replacement; before Track E, only the
exact already-in-service `(logical_activation_id, artifact_id, kind, digest)` tuple may load; any
tuple-field change, mismatch, unknown-id, or bounded-pre-read failure refuses fail-closed; and the
enumerator asserts every `gated` site is either fixed-hash-checked
(Phase A) or routed through `verify_and_load` (Phase B).

> **What #516 actually delivers = Phase A0 only, not full Phase A.** #516 seals the external
> `/model/reload` route (403), fail-closes the default `test` creds in a production posture, and ships
> the enumerator as a *discovery/classification* gate. It does **NOT** yet **pin** (fixed-hash /
> bundle-digest-check) the internal `gated` loaders (of the 38 conservatively-`gated` sites; they still load), and the enumerator does
> **NOT** yet assert each `gated` site is fixed-hash/bundle-digest-checked (Phase A, owner (b)) or
> routed through `verify_and_load` (Phase B). So the middle exit criterion above (every gated
> site fixed-hash/bundle-digest-checked, owner decision (b)) and the enumerator guard-assertion are **still unbuilt**. Honest #516 closeout: *external
> reload sealed; producer disabled; internal runtime activation remains proof-unbound.* Full Phase A
> (fixed-hash-check every gated site per (b) + enumerator asserts hashed-or-verified) is the next build
> after this lock ratifies.

**Phase A does NOT build:** the proof schema (§2.2), the signed proof store / issuer / key-custody (§2.3),
`verify_and_load`'s proof lookup (§3 steps 4–5), revocation/expiry, the LKG re-validation readiness probe,
or the append-only activation audit (§3.3). All of those are Phase B.

**Phase A — default-off, pin authority, and failure product-semantics (review 7c).**

- **Default-off at land.** When the Phase-A code merges it ships with **no production pin configured**;
  the guard defaults to *refuse* (`degraded`, §below) until a pin manifest is supplied. **Enabling a
  pin is a separate owner/deployment decision — a controlled release asset, not a code flag** (there
  is no `ENABLE_PIN=1`; a pin exists or it does not) — this is the **Phase-A baseline-pin activation gate
  (1)** (§3 build order): environment-owner-reviewed, no `/model/reload` re-open, go-evidence (canonical
  `PRODUCT_STRATEGY.md` §7.2 "Definition of done") = **the enablement date** + **the named product owner
  AND intended user** + staging replay + observed-RED matrix + rollback + kill switch + **user-outcome
  telemetry** (§7.2: "telemetry measures user outcome, not only service health" — the non-sensitive
  family/reason activation counters are *service-health* telemetry, necessary but NOT sufficient), no
  filesystem paths in logs or telemetry. `merged != enabled != safe`.
- **Pin authority & immutability.** The pin manifest (`logical_activation_id / artifact_id → {kind,
  SHA-256 | tree-digest-v1}`) lives in a **controlled release asset that the running service cannot
  modify** (read-only mount / signed release bundle / deploy-time-baked config). It is **not**
  runtime-writable and **not** env-swappable. The service reads it once at startup into immutable state.
- **Pre-Track-E baseline lock — model promotion stays fail-closed.** Phase A accepts only the exact
  `(logical_activation_id, artifact_id, kind, digest)` tuple captured from the already-in-service target
  deployment and explicitly owner-reviewed as the migration baseline. The baseline record contains no
  filesystem path. **Any change to `logical_activation_id`, `artifact_id`, `kind`, or `digest` is a
  model promotion (or contract migration) and is REFUSED before Track E**; there is no "re-pin the same
  baseline" concept and no generic "new pin = deploy" permission. If
  the target environment cannot prove the exact baseline tuple, no pin is issued and the family remains
  `degraded`/503. After Track E exits, changing the tuple requires a separately ratified model-promotion
  contract bound to its versioned reproducible evaluation artifact **and satisfying `PRODUCT_STRATEGY.md`
  §4.5's governed-model-change requirements (customer-holdout shadow evaluation, calibrated thresholds,
  canary evidence, rollback)**; this design-lock does not authorize that promotion merely because a
  deployment can replace a file.
- **Failure = an explicit, defined product state — NO silent stub.** On missing pin / hash-miss /
  bundle-digest-miss / containment-escape, each family MUST enter a **defined `degraded` state with a
  `503` (or family-appropriate) health/readiness contract** — the endpoint tells the client the model
  is unavailable; it does **not** silently serve a stand-in. This is a **hard change from today**:
  `src/core/ocr/providers/deepseek_hf.py` currently **falls back to a stub** on load failure
  (`deepseek_hf.py:93`), which is *fail-open* (a caller silently gets a stub answer). Phase A replaces
  every such silent-stub/best-effort fallback with the explicit `degraded`/503 contract; a family's
  Phase-A shard is not accepted until its silent fallbacks are gone and its degraded contract is tested.

### Phase B — Proof membrane (DEFERRED — build **only** when a real pilot needs dynamic model-swap)

The full contract in §2–§3: content-bound split digest (§2.1), the signed proof envelope (§2.2), the trust
source + key-custody + revocation + expiry + LKG re-validation (§2.3), the `verify_and_load` choke-point with
**server-owned artifact IDs** and TOCTOU-safe hash-and-load (§3), and the append-only audit (§3.3). Phase B
**replaces** each Phase-A fixed-hash body with real signed-proof-gated activation — re-enablement is replacing the body,
not adding a flag (`PRODUCT_STRATEGY.md` §7.2). Phase B depends on **Track E** (§8.1) existing (the split/manifest/metrics the proof
binds to) and on a **HSM / human-gated signer outside CI** (§2.3 key-custody).

**Do not start Phase B while no pilot requires dynamic model-swap.** Sections §2, §2.3, and §3 below
define Phase B; §1 and the enumerator define Phase A.

> **RESOLVED — owner selected (b) Static-artifact-only startup (2026-07-14).** (This records the
> owner's design choice; the owner still ratifies the final pinned head — see the protocol above.)
> Phase A is therefore **static fixed-hash artifact activation**, NOT a blanket hard-refuse of all
> ML: a `gated` loader may load **only** from a *server-owned* artifact resolved from a controlled
> store — **no caller path, no env-var-swap of the path, and dynamic replacement is forbidden**. Models
> work; hot-swap does not. **Before Track E, "models work" is limited to the exact owner-reviewed
> already-in-service baseline tuple; any changed tuple field is a refused promotion/contract migration.** Each
> `gated` site first applies the shared bounded pre-read and then the static check for its §0.5-step-2 **KIND**
> (a subset of §3's `verify_and_load`, with **no signed proof store and no signing keys** — that is
> Phase B): **single-file** — resolve a server-owned artifact id → read once → `SHA-256(bytes)` == the
> pinned value → load THOSE bytes, or refuse; **bundle/tree** — recompute the deterministic tree digest
> over an offline, **frozen** snapshot (source containment enforced at open by the store-root-anchored resolver, §0.5) → == the pinned tree-digest →
> load from the frozen snapshot, or refuse. The rejected alternative (a) — every `gated` loader hard-refuses, ML
> fully off with a defined `degraded`/health contract — is recorded for the audit trail but NOT the
> chosen path.
>
> #516 (Phase A0) prejudged neither: it sealed only the external `/model/reload` and left the internal
> `gated` loaders loading as-is (the 38 conservatively-`gated` sites; the sealed external `/model/reload`
> route is separate — NOT among the 38, §3 shard; several of the 38 latent/unproven-reachable — §1.B(cont)).
> Full Phase A implements (b) at each `gated` site once its
> reachability is confirmed (Wave-1 audit).

---

## 1. Activation map (verified `file:line` — the boundary this membrane must cover)

Classified by reachability, per the review's taxonomy.

### 1.A External-reachable activation (highest risk)
| Point | Family | Evidence | Current guard |
|---|---|---|---|
| `POST /api/v1/model/reload` (was `reload_model(payload.path, force=…)`) | pickle classifier | `src/api/v1/model.py` | **SEALED 403 by #516.** The externally reachable arbitrary-deserialization / code-execution risk (arbitrary caller `path`; `api_key`+`admin_token` both default `"test"`; no proof binding) is CLOSED. No execution PoC is claimed. Re-open only under Phase B + the identity gate (§3.2). |

### 1.B Startup / runtime-config activation (mutates a RUNNING service)
| Point | Family | Evidence | Current guard |
|---|---|---|---|
| `CLASSIFICATION_MODEL_PATH` → `pickle.load` on **first `predict()`** (lazy, not at startup: `load_model()` is the loader (`classifier.py:47`); `load_models()`-style readiness only builds a snapshot) | pickle classifier | `src/ml/classifier.py:22,85` (load fires via `classifier.py:124`) | **none — no magic-number check and no proof binding.** (The magic-number check is only in the *hot-reload* path `reload_model`, `classifier.py:313`, NOT this lazy-first-predict load — corrected from the first draft.) |
| `GRAPH2D_MODEL_PATH` → `torch.load` in `Graph2DClassifier` | Graph2D | path `src/ml/vision_2d.py:41`, load `src/ml/vision_2d.py:136`; `src/main.py:61` only *reads the flag*, does not load | **no proof binding** |

### 1.C NOT production-reachable — reclassified (corrected from the first draft)
The first draft over-counted these as live "activation points". They are not:
| Point | Real class | Evidence |
|---|---|---|
| `auto_retrain.sh` | **PRODUCER, not activator** | prints `SUCCESS: Ready for deployment` + `export GRAPH2D_MODEL_PATH=…` (`auto_retrain.sh:206,212`) — it generates+quantizes a candidate and prints a deploy command; it does **not** activate a running service. (Still #509 fail-closed, correctly, as a producer of promotable artifacts.) |
| `finetune_from_feedback.py` → `reload_model(force=True)` | **offline CLI** | `scripts/finetune_from_feedback.py:302`; **no `src/` importer** (verified) — the reload happens inside a CLI process that then exits, mutating no running production service. |
| `auto_remediation._action_rollback_model` → `reload_model(prev_path)` | **LATENT (future surface)** | `src/ml/monitoring/auto_remediation.py:301`; `AutoRemediation` is defined/exported but **no live scheduler calls `evaluate_and_act`** (verified). Must gain a proof check *before* it is ever wired to fire. |

### 1.D Explicitly OUT of the membrane
- **Offline tools** — `scripts/quantize_*.py`, `scripts/finetune_*.py`, training scripts: they
  *produce* artifacts; a produced artifact only becomes live by passing 1.A/1.B.
- **Unmounted serving scaffold** — `src/ml/serving/worker.py`, `grpc_service.py`: imported by **0**
  mounted routes (verified). Not a production boundary today; mark `inert` (or delete). If ever
  mounted it is promoted into 1.A and must gain a proof check first.

### 1.B (cont.) MORE `gated` loads — a hand-count kept missing these (reachability = Wave-1 audit)
An earlier draft said "**exactly 3**". That was wrong (the fourth such miscount), because the model zoo
is larger than two families. Additional `gated`, proof-unbound loads (conservatively classified — per-site reachability is a **Wave-1 audit**; the ones flagged below are latent or not-yet-proven-reachable):
| Point | Family | Evidence |
|---|---|---|
| PointNet via the **mounted** pointcloud router | pointnet | router imported+mounted `src/api/__init__.py:269,522`; the endpoint loads the point-cloud model |
| V16Classifier ensemble | part-v16 | `torch.load`+`load_state_dict` `src/inference/classifier_api.py::V16Classifier.load` (`:613,615,627,630`); manifest family `part-v16` (4 sites), distinct file from `part`. **Reachability = Wave-1 audit** — `.load()` is lazy (on `predict()`) / standalone-app / CLI; the mounted app imports only `result_cache` (`__init__`, no load) and no mounted route calls `predict()` → not proven live |
| HybridClassifier branch checkpoints | hybrid(stat/text) | `torch.load` `src/ml/hybrid_classifier.py:448,476` |
| PartClassifier / V16 / V14 | part | `torch.load` `src/ml/part_classifier.py:62,655,695` (via `/analyze`, `/health`) |
| HistorySequence | history | `torch.load` `src/ml/history_sequence_classifier.py:162` (via `/analyze`) |
| Vision3D encoder (`UVNET_MODEL_PATH`) | vision3d-uvnet | `torch.load` `src/ml/vision_3d.py:196` (via `/analyze` on 3D/STEP/IGES inputs; format+cache-miss gated but real) |
| DeepSeek OCR (HF) + PaddleOCR — **bundle/tree** | ocr | `from_pretrained` `src/core/ocr/providers/deepseek_hf.py:128,132` + `PaddleOCR` `:86,268`; **mounted** `/ocr` (a directory artifact — bundle-digest KIND) |
| SentenceTransformer embedding — **bundle/tree** | embedding | `SentenceTransformer` `src/core/assistant/embedding_retriever.py:59` (also `semantic_retrieval.py`, `ml/embeddings/model.py`); via the assistant (a directory artifact — bundle-digest KIND) |
| MetricsAnomalyDetector production metrics model | anomaly-monitor | `joblib.load` + `pickle.load` `src/ml/monitoring/anomaly_detector.py` (`load_models`); conservatively `gated` (production monitoring) — **not-yet-proven-reachable**: no `.load_models(` caller exists in `src/` (only class def + exports); **single-file KIND** — `load_models(path)` reads ONE file via a `joblib.load`-or-`pickle.load` **fallback** (verified: both idioms open the same `src`), so the two AST sites are ONE logical activation → one single-file pin |

**Reachability caveat (owner review, 2026-07-15).** The evidence requires `gated` to be interpreted
**conservatively**, not as the **pre-#521** "production-reachable" summary (removed when #521 merged `a0e517e8`): the 38 entries
are **AST load sites, not proven-live loaders** (the enumerator summary was corrected to this framing by #521, merged `a0e517e8`).
Known latent / not-yet-proven-reachable among the 38: the
`auto_remediation` rollback (§1.C — no live scheduler); the **part-v16 `V16Classifier.load`** (lazy on
`predict()` / standalone-app / CLI — no mounted route calls `predict()`); the pickle-classifier `_reload_model_impl`
hot-reload deserialization (`classifier.py:535`), reachable only via the now-sealed `/model/reload`,
the latent `auto_remediation`, and offline `finetune_from_feedback`; and the two
`MetricsAnomalyDetector.load_models` sites (no `.load_models(` caller in `src/`). **Per-site logical
reachability is a Wave-1 audit** — the safe contract is *38 conservatively-`gated` AST entries / 11
families, several latent/unproven, reachability to be confirmed in Wave 1* — never a hand-asserted live count.

**The recurring lesson — a hand-enumerated count is the wrong contract.** It has been wrong ≥4
times. The membrane's completeness must be enforced **by construction, not by a list**: ship a CI
**activation-surface enumerator** that AST-parses (import-aware) every `torch.load` / `pickle.load(s)` /
`joblib.load` / `onnx.load` / `load_state_dict` / `from_pretrained` / model-constructor / `reload_model(`
call site, marks each `gated | producer | offline | unmounted | infra`, and **fails if any `gated` site
is neither fixed-hash-checked (Phase A) nor routed through `verify_and_load` (Phase B, §3), or if a new
un-annotated load appears.**
That inverts the burden: a new activation surface reds CI until it is gated or explicitly classified
out. This §1 map is the *seed* of that enumerator, not the authority.

**Coverage gap today:** **#509 closes NONE of the runtime activation points** — it closes the
`auto_retrain` *producer* (necessary, but upstream of activation). Every `gated` load (**38 across 11 families** per the enumerator — a conservative classification; per-site reachability is a **Wave-1 audit**, seeded in §1.B/§1.B(cont), the enumerator not the map being the authority) is proof-unbound. The runtime activation membrane is **entirely unbuilt**.

---

## 2. The proof (what "may activate" means) — **Phase B**

An activation is authorized **iff** the model being activated is bound to a *reproducible evaluation
that a fresh clone can re-derive*. The proof has two phases (Track E, `PRODUCT_STRATEGY.md` §8.1).

### 2.1 Pre-training proof — the data is sound
Bind the training/evaluation split to **content**, not paths:
- **portable, versioned manifest** — no workstation absolute paths (today's manifests carry
  `/Users/.../…`); source drawings tracked or content-addressed so provenance is independent.
- **canonical split digest over `content_hash + family + label + side`** — NOT `(file_path → side)`.
  The split digest on the (unmerged) Track E branch `claude/track-e-eval-integrity-splitter-...`
  hashed only `(file_path → side)` — blind to *same-path-changed-bytes* / *same-path-changed-label*,
  the 262/914 content-overlap class it must catch. (That file is NOT on current `origin/main@c6625831` — re-verified; this
  requirement is what the Track E digest MUST satisfy, not a description of current main.)
- **non-empty both sides** + **largest-component share** reported and bounded (real data: `file:syn`
  is a single dominant component (~50%+ of assignable rows — a PR #510-review figure, not a
  reproducible current metric; re-run gives ~53%) — so a 20% "family holdout" over that is not a family
  sample). Release-quality requires authoritative family/source fields, not the filename heuristic.

### 2.2 Post-training proof — the model is the one that was evaluated
The evaluation result is bound, cryptographically, to the exact artifact being activated:

```
proof = {
  artifact_kind     : "single-file" | "bundle-tree"       # §0.5 step-2 KIND — Phase B binds BOTH (a bundle is not one file)
  artifact_digest   : sha256(file bytes) | tree-digest-v1  # THE candidate — single-file byte-hash OR the versioned tree digest (§0.5 step 2)
  model_family      : "pickle-classifier" | "graph2d"     # families do not share a proof
  split_digest      : content+family+label+side digest    # §2.1
  manifest_digest   : sha256 over source + license + provenance + label-authority (§2.1)
  evaluator_version : pinned                               # which metric code produced the numbers
  thresholds        : the gate values applied              # per-class / macro-F1 / ECE / dup / miss
  environment       : deployment env id                    # a proof for staging ≠ prod
  policy_version    : proof-schema/policy version          # for revocation + forward compat
  not_after         : expiry                               # a stale proof is not a valid proof
  metrics           : per-class, macro-F1, ECE, false-duplicate, missed-reuse, source-stratified
  issuer, signature : who signed it + a signature over all of the above  # see 2.3
}
```

### 2.3 A proof needs a TRUST SOURCE, not just the right fields (review gap)
Binding the fields is necessary but **not sufficient** — the fields alone prove *format*, not
*authority to issue*. The lock therefore also requires:
- **Trusted issuer + signature (or a server-owned, unforgeable record).** A proof is valid only if
  signed by a trusted evaluator identity, or recorded server-side where the requester cannot write
  it. Otherwise anyone who can write a well-formed file has "a proof".
  **Key custody (review gap):** the signing key must **not** be a CI-accessible repo secret — if it
  were, the code-generating routine could mint valid proofs from a branch, collapsing the two-actor
  separation §3.1 relies on. The key lives outside CI (an HSM / a human-gated signer); signing is
  invoked only by the evaluator identity, never by a PR job; and the verifying service **pins** the
  issuer public key (it does not trust whatever key a proof names).
- **`manifest_digest` over provenance**, since the current `split_digest` covers only the split
  assignment — not source, license, or label-authority. Provenance must be in the signed envelope.
- **Policy version + revocation + expiry (`not_after`)** so a compromised or superseded proof stops
  being accepted.
- **Server-owned `family` and `environment`.** These are determined by the activation site and the
  deployment, **never supplied by the requester** (a caller must not label its own model's family or
  claim `environment=staging` to dodge the prod policy).
- **Readiness / fallback on proof-miss.** A failed check does not just raise — it defines the service
  state: fall back to the last-known-good model, or refuse to serve (fail-closed), never silently
  serve the unverified one.
- **The last-known-good must STILL have a valid proof, and needs a re-validation MECHANISM (review
  gap).** "Was good once" is not a licence to keep running. But `verify_and_load` (§3) runs only at
  *activation* time — so revoking the currently-serving model's proof would do nothing without a
  post-activation check. **Mechanism (required, not just asserted):** a readiness probe re-validates
  the *currently-serving* `artifact_digest`'s proof against the store's revocation/expiry list on a bounded
  interval (and on a revocation-push if the store supports it); a serving model whose proof is now
  revoked/expired flips readiness to **red** (drain / refuse to serve). Without this loop, revocation
  only affects the NEXT activation, and the toothless case §2.3 exists to fix persists.

An activation point resolves a **server-owned artifact ID** to bytes from a controlled store (never a
caller path — §3), requires a **signed** proof whose `artifact_digest` (single-file SHA or tree digest) + server-owned `family` +
server-owned `environment` match, is unexpired and unrevoked. **Any miss → fail-closed** per the
readiness rule. This makes the token *bound and authorized* — the defect that sank the first gate (an
artifact for dataset A green-lighting a model on dataset B, and "a passing-format artifact anyone can
emit") cannot recur.

---

## 3. The membrane (one choke-point at every production-reachable activation) — **Phase B** (Phase A fixed-hash-checks these same sites; see §0.5)

A single function — `verify_and_load(artifact_id, family, env)` — where **`artifact_id` is a
server-owned identifier (or model hash), NOT a caller-supplied filesystem path (review gap).** The
first draft let the caller pass `payload.path`, which — even with later verification — first *opens
that path*, creating a path-probe / arbitrary-file-read / memory-IO-DoS surface (a caller could point
it at `/etc/shadow` or a 50 GB file). Instead:
1. **Resolve `artifact_id` in a controlled, read-only, content-addressed store.** The caller names
   *which approved artifact*, the server owns *where its bytes are*. Reject an unknown id.
2. **Apply the shared bounded pre-read from Phase A (§0.5 step 2):** KIND/type and bounded magic-prefix,
   single-file size, and bundle file-count / per-file / aggregate-byte / total-entries / directory-count / depth / relpath-length caps. A hostile, corrupt, or
   oversized object is rejected before a full read/copy or framework load.
3. **One immutable read (per KIND) — input-domain-locked, per §0.5 step 2:** for a **single-file**
   artifact, open the leaf **via §0.5's store-root-anchored resolver** (its **pinned-relpath precondition**
   runs on the RAW pin split-on-`/` BEFORE any `Path()`/`PurePosixPath`/`os.path` normalization and first
   rejects an absolute pin, a NUL byte, or any `.`/`..`/empty component — `O_NOFOLLOW` does NOT stop `..`,
   and `openat2`'s `RESOLVE_BENEATH` is NOT a substitute since the kernel collapses `//`/`.` — then a
   per-component `O_NOFOLLOW` walk from
   the store-root fd — intermediate dirs `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, leaf
   `O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC`, or one `openat2(RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS)` that
   covers both; `resolve()`+open forbidden) + `fstat`(`S_ISREG`) (reject any symlink at any component / FIFO / socket /
   device — `O_NONBLOCK` so a FIFO does not block before `fstat`), then read the bytes ONCE from THAT fd and
   **hash + load from THE SAME bytes**; for a **bundle/tree** artifact, **freeze an immutable snapshot** by the
   atomic-snapshot **or** descriptor-relative method of §0.5 (root and entries opened through the SAME
   store-root-anchored resolver — `openat(...O_NOFOLLOW|O_NONBLOCK)`+`fstat`, same-fd bounded-copy, partial
   freeze destroyed on any rejected entry — **never a path-copy / bind-mount**), take the tree digest over the
   frozen snapshot, and hand the framework **that frozen path**. Either way the
   entry that is checked IS the entry that is digested and loaded — **no re-open-by-path between check and
   load** — so "verified A, loaded B", the **pre-scan→copy** race, and the **inode-swap-after-check** race are
   all impossible (closes the single-file and the bundle TOCTOUs).
4. Look up a **signed** proof for `(artifact_digest, server-owned family, server-owned env)` in a **read-only,
   out-of-band proof store** (a model may not carry its own passing proof — self-attestation again).
   **Store UNREACHABLE ≠ proof ABSENT:** a store timeout/outage is NOT a transient "keep serving" —
   it is an unverifiable state → fail-closed (refuse to activate; hold LKG only while LKG's proof is
   independently cached-valid). A builder must not implement store-down as "skip the check";
   verify the signature/issuer and that the proof is **unexpired and unrevoked** with a current
   `policy_version`/`evaluator_version`/`thresholds`.
5. On **any miss**, apply the readiness rule (§2.3): fall back to LKG **only if LKG's own proof is
   still valid**, else refuse to serve — never load the unverified bytes. Emit an audit record (§3.3).

The set of call sites is **NOT a hand-list** (a hand count has been wrong repeatedly — §1.B(cont)).
The **CI activation-surface enumerator is the completeness authority for the DECLARED loader idioms** (bounded — not a proof of every possible load): the membrane is accepted only
when the enumerator confirms **every `gated` load site is fixed-hash-checked (Phase A) or routed
through `verify_and_load` (Phase B)**. The §1 map is that enumerator's *seed*, not the boundary.
Implementation is **sharded per model family**, each shard wiring the Phase-A fixed-hash check (in
Phase B, `verify_and_load`) **before** the load and shipping its own enumerator entry + golden:

- **pickle-classifier** — the external `/model/reload` route (`model.py:48`) stays **sealed 403** (#516, not fixed-hash-wired; not one of the 38 `gated` load sites); the Phase-A body wires `classifier.py:85` (the lazy
  first-`predict()` `pickle.load`, which today has NO magic/hash/opcode check). The current hot-reload
  path deserializes **before** it checks (`classifier.py:535` `pickle.loads` runs ahead of the
  whitelist/hash check, and the hash is truncated to 16 hex) — `verify_and_load`'s "cheap guards +
  one immutable read, hash-and-load the same bytes" (above) replaces that ordering.
- **graph2d** — `vision_2d.py:136`.
- **hybrid** (its own container auto-enables on file presence, same footing as graph2d) —
  `hybrid_classifier.py:448` (stat branch) and `:476` (text branch).
- **pointnet** — the **mounted** `/pointcloud` router → `pointnet/inference.py:108`.
- **part** (`PartClassifier` + `PartClassifierV16` + V14, all in `part_classifier.py`, manifest family `part`) — `:62,655,695` (reached via `/analyze`, `/health`).
- **part-v16** (`V16Classifier` in `src/inference/classifier_api.py`, manifest family `part-v16`) — `torch.load` `:613,627` + `load_state_dict` `:615,630`; **reachability is a Wave-1 audit** (load is lazy-on-`predict()` / standalone-app / CLI; no mounted route calls `predict()` — not proven live via the mounted app).
- **history** — `history_sequence_classifier.py:162` (reached via `/analyze`).
- **vision3d-uvnet** — `vision_3d.py:196` (`UVNET_MODEL_PATH`, reached via `/analyze` on 3D inputs).
- **ocr** (bundle-digest KIND) — DeepSeek HF `from_pretrained` (`deepseek_hf.py:128,132`) + PaddleOCR (`:86,268`), **mounted** `/ocr`; a directory artifact → tree-digest, offline-only, no silent stub (§0.5 step 2 + failure-semantics).
- **embedding** (bundle-digest KIND) — SentenceTransformer (`embedding_retriever.py:59`, `semantic_retrieval.py`, `ml/embeddings/model.py`), via the assistant; directory artifact → tree-digest.
- **anomaly-monitor** — `anomaly_detector.py` `load_models(path)`; the conservatively-gated production metrics model. **single-file KIND** — one `path`, read via a joblib-or-pickle **fallback** (both open the same file; one logical activation → one fixed-hash pin).
- any surface the enumerator later discovers → its own shard before it can go live.

The §1.C **latent activator** `auto_remediation.py:301` gains the same activation guard before it is
ever scheduled. `auto_retrain.sh` is different: it is a **producer, never an activator**, and it MUST
NOT call the runtime activation membrane. #509's unconditional block remains until Track E's exit
condition is satisfied; only then may a separately reviewed **Track E model-promotion gate** produce a
candidate plus its versioned evaluation-integrity artifact for an explicit owner promotion decision.
Producing that evidence does not load or activate the candidate; every later runtime load still passes
the phase-appropriate activation guard. The **unmounted serving scaffold** (`src/ml/serving/*`) is
promoted into a shard automatically if it is ever mounted (the enumerator reds until it is).

Per owner decision (b), the **Phase-A body at each `gated` site is the per-KIND static check of §0.5
step 2** — **single-file** (read once → `SHA-256(bytes)` == the pinned value → load THOSE bytes) or
**bundle/tree** (recompute the deterministic tree-digest over the offline **frozen** unpacked snapshot —
source containment enforced at open by the store-root-anchored resolver (§0.5), not `resolve()` — → == the pinned tree-digest → hand the framework the **frozen snapshot's** local path), else refuse to a
defined `degraded`/503 — models load, but **only** from pinned server-owned artifacts, and **before
Track E only the exact owner-reviewed already-in-service `(logical_activation_id, artifact_id, kind,
digest)` baseline tuple** (§0.5 pin-authority; any tuple-field change is a refused promotion, not a
deploy-time re-pin) — with no caller path, no env path-swap, no runtime hot-swap. The external `/model/reload`
route is the exception: it stays **sealed (403, done by #516)**, re-opened only under Phase B + the
identity gate (§3.2). Once **Track E** and the signed proof store exist, Phase B **replaces** the
fixed-hash body with the signed-proof `verify_and_load` — re-enablement is replacing the body, not
adding a flag (`merged != enabled != safe to enable`, `PRODUCT_STRATEGY.md` §7.2). Build order (reviewer-locked): **Phase A →
Track E → Phase B**. The single word "enablement" is overloaded, so two DISTINCT owner gates ride on
this order — do NOT conflate them:
- **(1) Phase-A baseline-pin activation** — after Wave-1 + Phase A, a **separate owner gate** that supplies
  the **environment-owner-reviewed** baseline pin: the exact already-in-service `(logical_activation_id,
  artifact_id, kind, digest)` tuple (§0.5 — an activation of the in-service tuple, **never** a promotion). It
  does **NOT** re-open `/model/reload` (that stays sealed 403 until Phase B), and its go-evidence
  (canonical `PRODUCT_STRATEGY.md` §7.2 "Definition of done") is **the enablement date + the named product
  owner AND intended user + staging replay + the observed-RED attack matrix + rollback + a kill switch +
  user-outcome telemetry** (§7.2: "telemetry measures user outcome, not only service health" — the
  non-sensitive family/reason activation counters are *service-health* telemetry, necessary but NOT
  sufficient), with **no filesystem paths in logs or telemetry**.
- **(2) Phase-B dynamic-swap / retraining enablement** — only after **Track E + Phase B**, replacing the
  fixed-hash body with the signed-proof `verify_and_load`. This is what re-opens dynamic activation.

The Phase-A gate activates a **static already-in-service baseline**; it does NOT enable dynamic swap or
retraining, which remain gated on (2).

---

## 3.1 Two threat actors — do NOT conflate them (review gap)

The first draft blurred two distinct threats; the membrane addresses one, and the other is covered
elsewhere. Keeping them separate keeps the design honest.

- **Runtime API caller** (someone hitting `POST /model/reload`, or influencing a startup config).
  *This* is what the activation membrane defends: an authenticated-but-wrong or malicious caller
  cannot activate a model without passing the **phase-appropriate activation gate** — Phase A: the
  static fixed-hash / bundle-digest check over a server-owned pinned artifact (owner (b)); Phase B: the
  signed proof. The unattended routine is **not** this actor —
  it has no runtime path to `/model/reload`.
- **Code-generating routine** (the unattended loop): it can modify branches and open PRs, but it
  **cannot thereby reach a runtime activation**. It is governed by *different* controls. Current
  branch protection (live facts: **0 required approvals**, **11 required status checks (strict:false — live-verified via the branch-protection API)**,
  required conversation-resolution, `enforce_admins`, no force-push/deletion) blocks a direct push and
  a checks-failing PR, but **requires NO human approval** — so it is not independent review and must
  not substitute for disabling an unsafe routine (a checks-passing PR is not review-gated).
  `CODEOWNERS` is only a path inventory in this solo-maintainer repository;
  `require_code_owner_reviews=false` and must remain so while the owner cannot approve its own PR.
  The isolated-critic protocol above supplies review evidence, while the human owner alone ratifies
  and authorizes a pinned head. Do not claim CODEOWNERS containment. Conflating
  the two would mis-scope the membrane (it is not a defense against branch edits) and under-protect the
  runtime (which needs the proof regardless of who wrote the code).

## 3.2 The external `/model/reload` HARD-DEPENDS on the production-identity gate (review gap)

The proof membrane authorizes *which model* may activate; it does **not** authenticate *who* is
asking. `POST /model/reload` is **now SEALED (403, #516)**; before the seal it was guarded only by `api_key` +
`admin_token` **both defaulting to `test`** (`dependencies.py`: the `X-API-Key` default `"test"` at `:29`
and the admin-token default fallback at `:68-81` — #516 also fail-closes both defaults in a production
posture, evidence at `:35-39` (API-key refusal) and `:66-84` (admin-token refusal)). Completing the proof membrane must **not** re-open the route while identity is
fail-open — a trusted-but-defaulted caller passing a valid proof would still be an anonymous activation.

**Ordering lock:** the external `/model/reload` route may be enabled **only when BOTH gates hold**:
1. the production-identity gate (separate design-lock): no default `test` credentials, unambiguous
   authenticated tenant/user, `x-user-id` cannot override the token subject (`src/api/middleware/integration_auth.py:104`);
2. the proof membrane (this doc).
The route **was LIVE (the original vulnerability, §1.A/§6); #516 has now SEALED it (403)** and it
stays sealed until BOTH gates hold — re-opening it is a Phase B + identity-gate decision, never a
flag. So for `/model/reload`, "disabled" is now a fact (#516); for the §1.B internal loads it is the
Phase-A fixed-hash body that must still be built. (The §1.B loads are internal and gated by
the proof membrane alone, but they must not read a caller-influenced path either). Neither gate alone
is sufficient; do not ship one and call the surface safe.

## 3.3 Append-only activation audit (review suggestion — adopted)

Every activation *attempt* (success or refusal) writes one **append-only** record:
`{ timestamp, actor (authenticated identity), decision (activated|refused|fell-back-to-LKG),
proof_id, artifact_id, model_family, environment, artifact_kind, candidate_artifact_digest,
previous_artifact_digest, new_artifact_digest (== candidate on 'activated', null on 'refused'),
failure_reason }` (drop the ambiguous bare `model_hash`; the digest is the single-file SHA or the tree
digest per `artifact_kind`; `candidate_artifact_digest` is what was attempted, `new_artifact_digest`
what is now serving).
**No filesystem paths or other sensitive strings** (per `PRODUCT_STRATEGY.md` §4.4's
redacted-where-possible customer-data discipline). The ledger is append-only so a bad activation cannot be erased, and it is what makes
revocation and incident review possible after the fact.

## 4. Non-goals / explicit exclusions

- Not building the proof store or Track E here — this locks their **contract**.
- Not touching auth defaults here — that is a **separate** L3 design-lock (production-identity model:
  `dependencies.py:8,29` default `test` (the `:8` constant definition + its `:29` `X-API-Key` usage), `src/api/middleware/integration_auth.py:104` header-overrides-subject). Cross-ref,
  don't merge the two.
- Not covering offline training/quantization (§1.D) — they don't activate.
- Customer corrections are **isolated until Track E's exit condition is satisfied** (`PRODUCT_STRATEGY.md`
  §8.2: pre-Track-E corrections stay **quarantined** — not appended to a training manifest, not used to
  promote a model). After it, **admission to a training-readable store is a separate owner decision**
  under the model-promotion contract — and only under explicit customer authorization, single-customer
  isolation, and with cross-customer training default-off; this design-lock does not itself grant it.
  (The earlier "never enter a training-readable store" was too absolute, but the permission is the
  owner's to grant, not this doc's.)

## 5. Golden matrix the implementation must ship (observed-RED, REQUIRED — not yet executed; the membrane is unbuilt)

**Phasing of this matrix (owner decision (b)).** **Phase A** is *baseline-only static fixed-`SHA-256`
activation* — before Track E, a `gated` site loads ONLY the exact owner-reviewed already-in-service
server-owned artifact tuple, **no proof store, no signing keys**; its
golden cases are the first table. **Phase B** adds the signed-proof binding (the second table); any row
needing a proof store / signature / evaluator / revocation is Phase B **by definition**. Phase B depends
on **Track E** existing, so the build order is **Phase A → Track E → Phase B**, with the **Phase-B
dynamic-swap/retraining enablement** last — distinct from the separate **Phase-A baseline-pin activation**
gate, which rides on Phase A (§3 build order; two distinct owner gates, do not conflate).

### Phase A (b) — baseline-only static fixed-hash activation (no proof store, no signing keys)
| Case | Required result |
|---|---|
| a `gated` site loads a **server-owned** artifact whose bytes' `SHA-256` **== the pinned value** | **GREEN — the Phase-A green** (this is the fixed-hash success case; models work) |
| **pre-Track-E:** any configured tuple field differs from the owner-reviewed already-in-service baseline | RED → REFUSED / `degraded`/503; any difference is a model promotion/contract migration, not a deploy-time re-pin |
| bytes' `SHA-256` **≠** the pinned value, or the pinned artifact is missing/unknown-id | RED → the site enters a defined **`degraded`/503** state (never loads the mismatched bytes) |
| caller supplies a filesystem `path`, or an env var swaps the artifact path | REJECTED — no caller-influenced path is ever opened |
| attempt to **hot-swap / re-point the pinned manifest at runtime** | REJECTED — the baseline manifest is **immutable at runtime**; before Track E, any changed tuple field is refused even across a deploy |
| the resolved artifact **escapes the store root** via an intermediate-directory symlink, `..`, or an absolute path (e.g. `store/family` → outside) | RED — the **store-root-anchored resolver** rejects it up front: the **pinned-relpath precondition** (on the RAW pin split-on-`/`, BEFORE any normalization) refuses an absolute pin, a NUL byte, or any `.`/`..`/empty component BEFORE the walk (`O_NOFOLLOW` does NOT stop `..` — it is not a symlink), and per-component `openat(O_DIRECTORY\|O_NOFOLLOW)` from the store-root fd refuses a symlink at ANY component (or one `openat2(RESOLVE_BENEATH\|RESOLVE_NO_SYMLINKS)`, which rejects the escape but does NOT catch `//`/`.`/empty — the raw precondition does); a `resolve()`+`open(path)` fallback is FORBIDDEN |
| single-file exceeds its family size cap or has the wrong KIND/type/magic | REJECTED before full read or framework load |
| bundle exceeds file-count, per-file-size, aggregate-byte, **total-entry-count, directory-count, depth, or relpath-length** cap, or contains a malformed/unreadable entry | REJECTED — metadata-detectable violations (file-count, per-file-size via `lstat`, aggregate-byte, total-entry-count, directory-count, relpath-length) are refused during the bounded metadata/prefix pre-scan **before any copy**; a violation that can only surface mid-walk/mid-freeze under (b-ii) per-entry copy (a read error, a byte-cap crossed only as bytes are copied, a directory added between pre-scan and walk, or a depth threshold crossed during descent) refuses immediately and **DESTROYS the partial freeze** — nothing partially-frozen is EVER digested or handed to the framework loader (the invariant is "no partially-frozen tree is digested/loaded", NOT "zero bytes were copied"; §0.5 step-2 (b-ii)). The (b-i) atomic snapshot checks all bounds on the snapshot before any digest |
| **[resource-bound, P1-2]** a **directory-bomb** bundle — a huge number of (empty) directories / dirents at near-zero file bytes | RED — the total-entry-count and directory-count caps refuse it (the aggregate FILE-byte and file-count caps alone do NOT catch a near-zero-byte directory/dirent explosion); metadata-detectable in the pre-scan → refused before any copy, or if only crossable mid-walk → immediate refuse + DESTROY partial freeze → `degraded`/503 |
| **[resource-bound, P1-2]** a **depth-bomb** bundle — pathologically deep nesting | RED — the traversal-depth cap refuses it; a depth threshold crossed during descent refuses immediately and DESTROYS the partial freeze → `degraded`/503 |
| **[resource-bound, P1-2]** a bundle entry with an **over-long relpath** (exceeds the per-entry relpath byte-length cap) | RED — the relpath byte-length cap refuses it (this cap also bounds the `tree-digest-v1` record encoding, §0.5 step-2) → `degraded`/503 |
| **bundle/tree** family (HF/SentenceTransformer/PaddleOCR): the deterministic tree digest over the **frozen** unpacked snapshot **== the pinned tree-digest** | **GREEN** (bundle Phase-A green; loads from the **frozen snapshot's** local path) |
| a file is added/removed/changed inside the bundle dir | RED — the tree digest changes → `degraded`/503 |
| **[pre-scan→copy TOCTOU]** a bundle entry is **swapped from a regular file to a symlink / FIFO / oversized file AFTER the pre-scan, BEFORE the copy** | RED — the freeze reads the SAME `openat(O_NOFOLLOW\|O_NONBLOCK)` fd (or an atomic snapshot taken before the scan); a path-based copy that would follow / block / over-read is FORBIDDEN; on any such rejection the partial freeze is destroyed and nothing is digested or loaded |
| the loader would **fetch from a network hub** (hub id / online) | REJECTED — offline-only (`HF_HUB_OFFLINE`/`local_files_only`); a hub id is never a valid artifact id |
| any file in the bundle **escapes the store root** via a symlink or `..`/absolute at ANY path component | RED — refused up front by the **store-root-anchored resolver**: per-component `O_NOFOLLOW` from the store-root fd rejects a symlink, and the **pinned-relpath precondition** (RAW pin split-on-`/`, before normalization) plus the traversal's `.`/`..`-dirent rejection refuse a `..`/absolute/empty component or a NUL byte (`O_NOFOLLOW` does NOT stop `..`, and the `readdir` walk never descends `.`/`..`), not post-hoc; the frozen snapshot is a service-private copy, so its own root cannot escape (§0.5 step-2 (b)/(c)) |
| **[input-domain lock]** a **symlink** anywhere under the bundle root — even one that does NOT escape | RED — the violating entry is itself never copied (its per-entry `fstat`/name check precedes its own copy); any previously-copied partial output is DESTROYED; nothing partially-frozen is ever digested or loaded → `degraded`/503 (ALL symlinks refused, not only escaping ones; under (b-i) the atomic snapshot is checked before any digest) |
| **[input-domain lock]** a **non-regular entry** (FIFO / socket / block- or char-**device**) under the bundle root | RED — the violating entry is itself never copied (its per-entry `fstat`/name check precedes its own copy); any previously-copied partial output is DESTROYED; nothing partially-frozen is ever digested or loaded → `degraded`/503 |
| **[input-domain lock]** a bundle path whose **POSIX relpath fails UTF-8 encoding** | RED — the violating entry is itself never copied (its per-entry `fstat`/name check precedes its own copy); any previously-copied partial output is DESTROYED; nothing partially-frozen is ever digested or loaded → `degraded`/503 (positive control: an all-regular-file, UTF-8-clean bundle → GREEN, above) |
| load fails and the provider tries a **silent stub / best-effort fallback** (e.g. deepseek_hf.py:93) | FORBIDDEN — must enter the explicit `degraded`/503, never serve a stub |
| **no provable exact baseline pin** at land (default-off) | the guard refuses → `degraded`/503 (baseline capture + owner review is separate from merge; no code flag opens it) |
| **swap a single-file artifact between hash and load** (TOCTOU) | RED — read once, hash and load THE SAME bytes |
| **[input-domain, single-file]** the leaf is a **symlink** inside the store root (even non-escaping) | RED — the resolver's leaf `openat(O_NOFOLLOW)` fails / `fstat` rejects → `degraded`/503 |
| **[input-domain, single-file]** the leaf is a **non-regular** entry (FIFO / socket / block- or char-device) | RED — `fstat` rejects non-`S_ISREG` → `degraded`/503 |
| **[input-domain, single-file]** the leaf's **inode is swapped AFTER the `fstat` check, before the read** | RED — the read is from the SAME already-open fd; re-open-by-path is forbidden (positive control: a stable regular-file leaf → GREEN) |
| **[input-domain, resolver]** an **intermediate directory** on the pinned path is a **symlink** (e.g. `store/family` → outside), single-file OR bundle | RED — leaf-only `O_NOFOLLOW` would follow it; the store-root-anchored per-component resolver refuses a symlink at ANY component → `degraded`/503 (owner-verified escape) |
| **[input-domain, resolver]** a **parent directory** of the leaf is **swapped to a symlink mid-walk** (parent-dir race) | RED — each component is opened `O_NOFOLLOW` relative to the previous fd (or one `openat2(RESOLVE_BENEATH\|RESOLVE_NO_SYMLINKS)`), so a mid-walk symlink swap is refused, never followed |
| **[raw input-domain, P1-1]** a pin with an empty component from a **`//`**, e.g. `family/model//weights.pt` | RED — the RAW split-on-`/` precondition sees the empty component and refuses (`degraded`/503) BEFORE any `Path()`/`PurePosixPath`/`os.path` normalization could collapse the `//`; normalization before the check is FORBIDDEN |
| **[raw input-domain, P1-1]** a pin with a **`/./`** component, e.g. `family/./model.pt` | RED — the RAW split refuses the `.` component up front, BEFORE normalization (which would silently collapse `a/./b` → `a/b` and hide it) → `degraded`/503 |
| **[raw input-domain, P1-1]** a pin with a **`/../`** component, e.g. `family/../secret.pt` | RED — the RAW split refuses the `..` component up front (`O_NOFOLLOW` does NOT stop `..`); rejected before any walk → `degraded`/503 |
| **[raw input-domain, P1-1]** a pin containing a **NUL (`0x00`) byte** anywhere | RED — the raw-pin NUL rejection fires on the whole pin string (not only readdir entry names) → `degraded`/503 |
| **[raw input-domain, P1-1]** an **absolute** pin (leading `/`), e.g. `/etc/model.pt` | RED — the precondition refuses any absolute prefix (the leading `/` is also an empty first component in the raw split) → `degraded`/503 |
| **[raw input-domain, P1-1 CONSISTENCY]** each raw-invalid pin above (`a//b`, `a/./b`, `a/../b`, a NUL pin, an absolute pin) is fed to BOTH the `openat2(RESOLVE_BENEATH\|RESOLVE_NO_SYMLINKS)` implementation AND the component-walk fallback | RED — BOTH MUST reject IDENTICALLY to the same fail-closed `degraded`/503; because `openat2` collapses `//`/`.` and `RESOLVE_BENEATH` only blocks escapes, the shared RAW split-on-`/` precondition (NOT `openat2` alone) is what enforces identical rejection — an implementation divergence (one accepts, one rejects) is ITSELF a RED |
| **[input-domain, FIFO-block]** the leaf/entry is a **FIFO** (would a plain `O_NOFOLLOW` open block before `fstat`?) | handled — the open carries `O_NONBLOCK`, so it returns and `fstat` rejects the non-`S_ISREG` FIFO instead of hanging for a writer → `degraded`/503 |
| **change a bundle file AFTER the tree digest, before/during the framework's read** (bundle TOCTOU) | RED — the digested tree is a **frozen immutable snapshot**; the framework reads the digested files, never the mutable original (§0.5 step 2) |
| a new **un-annotated** prod loader appears | CI RED (the enumerator, §1/§3) |

### Phase B — signed-proof binding (needs Track E + the signed store; replaces the Phase-A body)
| Case | Required result |
|---|---|
| activate a model whose bytes have **no** proof | RED at every §1.A/1.B point |
| **swap the file (single-file) or a bundle file (tree) between digest and load** (TOCTOU) | RED — the loaded bytes/tree are the digested ones; the bundle loads from a frozen snapshot (§3) |
| **unsigned / wrongly-signed** proof (well-formed but not from a trusted issuer) | RED (no authority — §2.3) |
| **expired / revoked** proof (`not_after` passed, or policy revoked) | RED (stale proof is not valid) |
| requester supplies `family` or `environment=staging` to dodge policy | IGNORED — both are server-owned (§2.3) |
| caller supplies a filesystem `path` (not a server-owned artifact id) | REJECTED — no caller path is opened (§3) |
| shared bounded pre-read fails (wrong KIND/type/magic, oversized file/bundle, malformed entry) | REJECTED before full read/copy or framework load (§0.5/§3) |
| fall back to a last-known-good model whose proof is REVOKED/EXPIRED | RED — readiness refuses; LKG needs a valid proof (§2.3) |
| complete the proof membrane but leave `ADMIN_TOKEN=test` on `/model/reload` | route stays DISABLED — needs the identity gate too (§3.2) |
| proof for family A, activate a family-B model of same bytes-len | RED (family mismatch) |
| proof for `staging`, activate in `prod` | RED (environment mismatch) |
| change one byte of the model file after the proof was issued | RED (`artifact_digest` mismatch) |
| add/remove/change a file in a **bundle** artifact after the proof was issued | RED (tree-digest → `artifact_digest` mismatch) |
| change a **label** in the manifest, re-derive split_digest | RED (split digest changes — §2.1) |
| rewrite an existing file's **bytes** to duplicate another's content | RED (content in digest — §2.1) |
| bump `evaluator_version` / a threshold | RED (stale proof) |
| a genuine, signed, unexpired, fresh-clone-reproducible proof matching (`artifact_digest`, family, env) — single-file OR bundle | GREEN — the Phase-B green (the Phase-A green is the fixed-hash match above) |

## 6. What this changes about the current risk statement

Accurate post-#509 status, by threat actor (§3.1 — do not conflate them):

- **The unattended routine** is constrained by branch protection (live: **0 required approvals**,
  **11 required checks (strict:false)**, conversation-resolution, `enforce_admins`, no force-push/deletion) —
  which stops a direct push and a checks-failing PR but does **not require a human approval**, so it
  does not by itself prevent a checks-passing self-merge; cannot retrain via `auto_retrain.sh` (#509);
  and — being a code-generating actor — **has no runtime path to any activation point** (it cannot call
  `/model/reload`). It is **not** contained by CODEOWNERS (#512 is an unarmed ownership map), and branch
  protection is not a substitute for stopping it. No unattended routine may author or merge this L3
  runtime.
- **A runtime API caller / operator** can still reach the `gated` AST load sites (**38 across 11 model
  families** — a conservative classification; per-site reachability is a **Wave-1 audit** and several are
  latent/unproven (§1.B(cont)); these AST sites group into fewer *logical* activations — §0.5 step 4 — the guard-coverage denominator) (the CI activation-surface enumerator, not a hand count, is the
  authority — §1.B(cont)/§3). The external `/model/reload` is **now SEALED (403, #516)** and is no
  longer reachable; the remaining gap is the **internal `gated` loaders** (of the 38 conservatively-`gated`
  sites (the sealed external `/model/reload` route is separate — NOT among the 38, §3 shard); several latent/unproven — §1.B(cont)) — the pickle-classifier lazy-first-predict load
  and the graph2d / hybrid-branch (stat, text) / pointnet / part / part-v16 / history / vision3d-uvnet /
  ocr / embedding / anomaly-monitor surfaces (per-site reachability **to be** confirmed in the Wave-1 audit — not yet run) — which still load **unpinned** (no Phase-A fixed-hash check
  yet). *That* is what full Phase A must close, and it is **unbuilt**.

**So: strong bleeding-control on the code-gen actor; the runtime activation membrane is not built.**
This design-lock defines that membrane. Stopping the routine and owner-ratifying this design under the
solo-maintainer protocol gate *building* it. The isolated critic is mandatory evidence but cannot
ratify or merge. The runtime membrane remains a separate, additional need.

---

## Appendix A — reproduction

```sh
# activation map (run at repo root)
grep -rn 'reload_model(' src scripts --include='*.py' | grep -v 'def reload_model'
grep -rn 'CLASSIFICATION_MODEL_PATH\|GRAPH2D_MODEL_PATH' src --include='*.py' | grep -v test
sed -n '46,72p' src/api/v1/model.py            # external /reload + auth deps
sed -n '38,42p;136,136p' src/ml/vision_2d.py   # __init__ path + real torch.load
sed -n '60,72p' src/main.py                    # flag-check only, NOT a load
# serving scaffold is unmounted:
grep -rln 'ml.serving' src/api --include='*.py' | wc -l     # -> 0
```
