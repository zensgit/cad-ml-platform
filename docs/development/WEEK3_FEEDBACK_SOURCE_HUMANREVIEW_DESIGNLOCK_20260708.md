# Week 3 · feedback-source + human-review — DESIGN-LOCK (PROPOSED, owner-ratify)

- **Status**: DESIGN-LOCK PROPOSAL. Branch only, **no PR**, **no code**. Owner-gated (the flywheel/moat track); this is the "propose, don't build" deliverable.
- **Grounded on** `origin/main @ 8337ea6e`, evidence observed live.
- **Do NOT resurrect `FeedbackLearningPipeline`** — it is deleted by #502 (orphan, write-only EMA weights). This design-lock replaces the *idea* of it with the real missing pieces.

---

## 1. Verified state: the spine works; two pieces are missing

The #499 audit's core correction holds against current code:

| element | state | evidence |
|---|---|---|
| inference → queue | **works** | `hybrid_classifier` calls `low_conf_queue.maybe_enqueue` → `data/review_queue/low_conf.csv` (`src/ml/low_conf_queue.py:83`) |
| queue → retrain, hard-gated | **works** | `auto_retrain` reads `low_conf.csv`, gated on `MIN_REVIEWED=200` human-verified rows behind a real hard governance gate |
| the human-review **consumer** | **exists** | `LowConfidenceQueue.human_verified_entries()` (`low_conf_queue.py:177`) returns rows where `reviewed_label` **and** `human_verified` are set |
| the human-review **producer** | **MISSING** | `reviewed_label` / `notes` / `human_verified` / `eligible_for_training` are `"left blank; filled by human annotator"` (`low_conf_queue.py:46-51`) — nothing writes them |
| the feedback **source (volume)** | **MISSING** | `feedback.py` exists ("Collects user corrections… Data Flywheel") but DedupCAD is outbound-only (`dedupcad_vision_requests_total`, `analysis_metrics.py:816`); the queue only fills from low-confidence inference, not from real corrections |

So the moat isn't "reconnect a pipeline" — the pipe and its gate are built. It's: **(a) a human-review action that fills the review columns, and (b) a feedback source that gives the queue volume.** Both are genuinely new, small, and un-invented.

## 2. Schema-lock (already implicit — make it canonical)

The `low_conf.csv` header **is** the correction schema; lock it as the single canonical one so nobody invents a parallel store (the mistake `FeedbackLearningPipeline` + `feedback_log.jsonl` made):
```
id, file_ref/file_hash, predicted_class, confidence, sample_source,
reviewed_label, notes, human_verified, eligible_for_training, ts
```
Rule: **one store, one schema.** New feedback lands here, not in a new JSONL.

## 3. Locked design — two thin slices, both default-off

**Slice F1 — human-review action (the producer).** A minimal review surface that lists un-reviewed `low_conf.csv` rows and writes `reviewed_label` / `human_verified=true` / `eligible_for_training` back through `LowConfidenceQueue` (reusing the existing read side). Form: a gated `/review` endpoint or a CLI; **no ML.** Default-off flag. This is what turns `MIN_REVIEWED=200` from "never satisfied" into reachable.

**Slice F2 — feedback source (the volume).** Route real corrections into the same queue: a correction channel from the DedupCAD dedup workflow (a human marking a dedup result wrong) or from golden-labeling, writing canonical-schema rows. Precondition for F1 to have anything to review at scale.

**Ordering:** F2 (source) is the true precondition — without volume, F1 reviews an empty queue and `auto_retrain` stays dry. Lock F2 → F1 → (existing) gate → retrain.

## 4. Verification contract for the eventual build (observed-RED required)
- **F1**: fail-first golden — an un-reviewed row is NOT in `human_verified_entries()`; after the review action writes it, it IS. Observed-RED: `auto_retrain`'s `HV_VERIFIED>=200` gate stays red at 199, flips at 200.
- **F2**: a correction event produces exactly one canonical-schema row; observed-RED that a malformed correction is rejected, not silently dropped.
- **Security membrane**: the review action is an untrusted-write surface — writes must be actor-attributed and gated (same principle as the metasheet2 personal-views work); a review label is a *proposal* feeding a hard eval gate, never a direct model write.

## 5. Why design-lock only now
- It's the **moat** — highest-value, so it deserves ratification before build, not an unattended slice.
- F1/F2 are **owner/product decisions** (what the correction UX is; which DedupCAD signal counts as a correction).
- The build is **gated on Week-2 stabilizing** (per the owner's sequencing) and on #502 landing (so the deleted pipeline can't be half-resurrected).

## 6. Follow-ups
- Owner-ratify F2's correction source + F1's review surface shape.
- Then build F2 → F1 as default-off slices with the §4 observed-RED.
- **Model:** grounded/authored in-session on Opus 4.8 (Fable 5 at daily cap). Build slices are Sonnet-5-class once the schema/UX are ratified.
