# Week 3 · B-Rep data readiness — provenance audit + labeling worklist

- **Status**: WORKLIST (data readiness). Branch only, **no PR**, **no code, no model**. Human/owner-gated data track — this enumerates the work, it does not fabricate golden results.
- **Grounded on** `origin/main @ 8337ea6e`, observed live.
- **Extends** `docs/development/CAD_ML_REAL_EVIDENCE_SOURCING_WORKLIST_20260622.md` (the code-gates-in-place worklist) — this focuses on the *data* half.

---

## 1. Verified state: the harness is ready; the data isn't curated

| element | state | evidence |
|---|---|---|
| eval harness | **built, waiting** | `brep-golden-eval.yml`: *"does NOT add new eval logic… Until the 50–100 real release-eligible STEP/IGES files are curated into a real manifest, runs operate on [the example manifest]… informational (does not fail)."* |
| real files on disk | **60 present** | `data/brep_golden/public_cad/nist/` — 60 real NIST STEP/IGES files (corrects the #499 "single placeholder" claim) |
| release-eligible manifest | **MISSING** | no manifest / `expected_groups` under `data/brep_golden` |
| provenance markers | **MISSING** | no `README`/`LICENSE`/provenance file alongside the 60 files |

So the bottleneck (the memory's *"瓶颈是数据"*) is concretely: **the 60 files are raw, un-audited, and un-curated into the manifest the ready harness needs.** This is human-review + curation work, not modeling.

## 2. The worklist (in order)

**W-B1 — provenance audit (blocks everything).** For each of the 60 files: record source (NIST which dataset?), **license / redistribution rights**, and a content hash. Produce `data/brep_golden/PROVENANCE.md` (or `.csv`). **A file with no clear redistribution right must not enter a release-eligible golden set** — flag and quarantine it. This is the first gap: nothing today records where these came from or whether they can ship.

**W-B2 — dedup-relevance triage.** The golden is for L4 dedup precision; a file is only useful if it participates in a *known* same/different relationship. Triage the 60 into candidate groups (variants, rotations, near-duplicates) vs singletons. Human-review gap list: which files lack a known pair/group.

**W-B3 — labeling → expected_groups.** From W-B2, hand-author `expected_groups.json` (the harness's missing input): the ground-truth grouping the eval scores against. This is the labeling worklist — per-file group assignment, human-verified.

**W-B4 — manifest curation.** Assemble the release-eligible manifest (`min_samples` per `brep-golden-eval.yml:25`) from the audited (W-B1) + labeled (W-B3) subset. Only license-clean, labeled files qualify. If fewer than the `min_samples` (50–100) survive audit, **source more** rather than pad — the worklist must `log()` the shortfall, not silently ship an under-sized golden.

**W-B5 — flip the harness to real.** Once the manifest exists, point `brep-golden-eval.yml` at it and change the run from informational to gated (an **owner** action — making a check blocking + required is branch-protection-adjacent, same class as A4/prune-safety arming).

## 3. Anti-fake-green rules (this is where data tracks rot)
- **No synthetic "golden" pairs.** Every expected group is human-verified from real files, or it's not in the manifest.
- **Provenance is a hard precondition**, not a nice-to-have — an un-licensed file in a shipped golden is a legal + integrity failure.
- **Shortfall is reported, not hidden** — if audit leaves < min_samples, the worklist says so (per the "no silent caps" discipline); it does not down-scope the target to whatever survived.
- The eval stays **informational until W-B1..W-B4 are done and owner-ratified**; flipping it to blocking is W-B5, owner-only.

## 4. Why worklist-only now
- It's **human curation + a licensing/rights judgment** — not an unattended agent task; fabricating labels would poison the moat's only ground truth.
- W-B5 (flip to gated/required) is a **branch-protection-class owner action**.
- The right agent contribution is exactly this: enumerate the audit/label/curate steps + the anti-fake-green rules, so the human work is scoped and the harness flip is one step once data is ready.

## 5. Follow-ups
- Assign W-B1 (provenance audit of the 60 files) — the unblocker.
- Owner-ratify the labeled manifest before W-B5.
- **Model:** authored in-session on Opus 4.8 (Fable 5 at daily cap). The mechanical parts (hashing, manifest assembly from a labeled CSV) are Sonnet-5-class once labels exist; the labeling/licensing judgment is human.
