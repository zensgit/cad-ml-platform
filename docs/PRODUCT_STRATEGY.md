# CAD Reuse Decision Engine - Product Strategy

- **Status**: FOR REVIEW. Merging this document ratifies the repo-level product direction and authorizes only the 90-day work in section 8.
- **Owner**: repository owner
- **Last reviewed**: 2026-07-12
- **Code grounding**: `origin/main@0af2fa80`
- **Canonicality**: this is the current, stable-name product strategy for `cad-ml-platform`. Dated roadmap and verification documents remain evidence, not competing current strategies.

## 0. Decision

Continue developing the system, but stop developing it as a generic "CAD AI platform".

The product is:

> **A private-deployment CAD drawing reuse and release decision engine for manufacturers with large, inconsistent legacy drawing archives.**

It helps an engineer answer one expensive question before creating or releasing a drawing or part:

> Reuse an existing design, revise an existing design, or create a genuinely new one?

The system must return candidates, deterministic geometric evidence, classification context, confidence, and an auditable reason. A human or the owning PLM workflow remains the authority for release and write-back.

This is a **go with a narrow wedge**, not a go for the existing broad platform story. If the wedge does not produce a paid pilot within six months, the independent-product attempt stops and the engine becomes an internal component of the chosen CAD/PLM product.

## 1. Why this can survive AI commoditization

AI will reduce the value of standalone OCR, embeddings, image similarity, classification, chat, and provider integration. Large CAD and PLM vendors already combine similar-part search, classification, review, and lifecycle workflows. Those features are baseline, not a moat.

The defensible system is the layer that models do not provide by themselves:

1. **Customer-specific data onboarding**: turn a legacy drawing archive into a calibrated, evaluated deployment quickly.
2. **Deterministic decision evidence**: combine model recall with geometric verification and explicit rejection reasons.
3. **Workflow placement**: run before part creation or drawing release, where avoiding duplication has measurable value.
4. **Human decisions and provenance**: capture accepted reuse, rejected matches, version-family relations, and reviewer rationale.
5. **Private deployment and governance**: keep drawings private by default, make model swaps measurable, and keep write authority outside AI.

Models are replaceable suppliers. The product owns the decision contract, evaluation corpus, calibration, audit trail, and workflow integration.

Customer drawings, labels, and decisions remain customer-owned. Cross-customer training is prohibited by default. The defensible asset is the repeatable onboarding, calibration, evaluation, and governance process, not silent ownership of customer data.

## 2. Market position

### 2.1 Initial customer profile

Start with manufacturers that have:

- thousands to hundreds of thousands of old 2D DXF/PDF drawings;
- inconsistent file names, part numbers, and folder structures;
- repeated custom or non-standard design work;
- expensive duplicate parts, duplicate drawings, or avoidable re-engineering;
- a requirement for on-premise or private-network deployment;
- engineers who can review proposed matches inside a real release workflow.

The current taxonomy and data distribution suggest a first vertical hypothesis around non-standard process equipment and machinery: flanges, shafts, housings, vessels, heat exchangers, heads, and manholes. This is a hypothesis to test with customers, not a proven market claim.

### 2.2 Outcome sold

Do not sell "AI accuracy" or "a CAD model". Sell an operational outcome:

- less time spent searching historical drawings;
- fewer duplicate drawings and part numbers;
- more accepted design reuse;
- faster review before release;
- traceable reasons for reuse, revision, or new-part decisions.

### 2.3 Competitive reality

Similarity search, OCR, classification, and human review are already available from established vendors:

- [CADDi](https://us.caddi.com/) combines legacy drawing ingestion, OCR, similarity search, drawing comparison, and links to purchasing, supplier, quality, ERP, and PLM data.
- [Siemens Geolus Shape Search](https://blogs.sw.siemens.com/teamcenter/geolus-shape-search-in-teamcenter/) integrates shape search and visual comparison into Teamcenter workflows.
- [Siemens Classification AI](https://blogs.sw.siemens.com/teamcenter/classification-ai/) includes confidence thresholds, batch classification, and subject-matter-expert review.
- [PTC Windchill AI Parts Rationalization](https://www.ptc.com/en/news/2026/ptc-launches-windchill-ai-parts-rationalization) identifies similar and duplicate parts inside PLM and change workflows.

The product cannot win by matching those feature lists. It can win only with a narrow deployment wedge: private legacy 2D archives, explainable geometric evidence, fast customer calibration, and integration into the customer's existing release authority.

## 3. Product boundary

### 3.1 What this repository owns

`cad-ml-platform` owns a headless evidence and decision engine:

```text
query drawing
  -> visual / OCR / metadata recall
  -> deterministic geometry verification
  -> classification and confidence context
  -> ranked evidence bundle
  -> human or PLM decision
  -> audited outcome and feedback
```

The current code already contains useful pieces of this boundary:

- `src/api/v1/dedup.py` exposes the 2D dedup flow;
- `src/core/dedupcad_vision.py` calls the visual-recall service with retries, circuit breaking, and metrics;
- `src/core/dedupcad_precision/` performs local deterministic precision checks;
- the classifier surfaces confidence and decision evidence;
- Redis-backed jobs, tenant configuration, health signals, and rollback-oriented evaluation machinery exist.

These are assets. They are not yet proof of a customer-ready product.

### 3.2 What this repository does not own

This repository must not become another system of record, PLM, CAD editor, procurement suite, or generic agent platform.

The upstream CAD/PLM product owns:

- authenticated user and tenant authority;
- drawing and part lifecycle state;
- permissions and release approval;
- canonical version, BOM, supplier, cost, and quality records;
- final write-back.

The dated 2026-07-06 record's claim that DedupCAD is the sole real consumer is no longer current. The cross-repository audit in PR #507 found a live Yuantus client calling this service, while this repository calls `dedupcad-vision` for visual recall. That is an integration fact, not a ratified decision about the permanent product shell.

The permanent system-of-record choice is a separate portfolio decision. [PR #507](https://github.com/zensgit/cad-ml-platform/pull/507) currently proposes that cross-repository decision and must be ratified independently. This document does not claim that Yuantus, DedupCAD, or another PLM is already the permanent product shell.

### 3.3 Stable decision contract

Every analysis response intended for product use must converge on a model-independent contract:

- candidate identifier and source;
- normalized geometric and semantic scores;
- deterministic verification result;
- confidence and calibration version;
- evidence and rejection reasons;
- model, ruleset, and dataset provenance;
- human decision state;
- trace and idempotency identifiers.

Provider-specific raw output stays behind adapters. No downstream workflow should depend on one model's prose or JSON shape.

## 4. AI safety and replacement rules

1. **AI carries no authority.** Authorization comes only from the authenticated user and owning workflow.
2. **Domain validation fails closed.** Invalid AI output is rejected, not silently coerced into a valid class, part, or action.
3. **High-impact actions require deterministic checks and human confirmation.** AI may propose; it may not release, merge, delete, or rewrite canonical product data by itself.
4. **Customer drawings stay private by default.** External-provider transmission is explicit opt-in, documented, redacted where possible, and covered by a customer data policy.
5. **Model changes are governed.** A provider or model change requires customer-holdout shadow evaluation, calibrated thresholds, canary evidence, and rollback.
6. **No sandbox assumption.** Model output and plugin output are untrusted inputs even if produced inside this stack.
7. **Load-bearing goldens need independent criticism.** The author of a guard must not be the only author or reviewer of the discriminator that claims to prove it.

## 5. Evidence status: what is real and what is not yet trustworthy

### 5.1 Real engineering assets

- deterministic geometry scoring and diff machinery;
- deployed-service concerns such as jobs, retries, circuit breaking, metrics, and tenant configuration;
- classification models and evaluation/governance scaffolding;
- human-verification fields and training eligibility gates;
- active cleanup work that is shrinking agent-generated scaffolding and making CI checks meaningful.

### 5.2 Evaluation integrity is not release-grade today

A content-hash audit on 2026-07-12 resolved the files referenced by the current manifests and found:

- training rows: `3,660`; validation rows: `914`;
- `262/914` validation rows (`28.7%`) have exact SHA-256 content matches in training;
- 23 overlapping hashes have different train and validation label sets;
- synthetic plus augmented paths account for `2,471/3,660` training rows (`67.5%`) and `616/914` validation rows (`67.4%`);
- the three largest validation classes account for `830/914` rows (`90.8%`).

The current governance check in `scripts/ci/check_training_data_governance.py` compares only `file_path` and `cache_path` strings. The corresponding unit test also checks only path strings. It therefore reports a clean split even when different paths contain identical bytes.

The source drawings are not reproducible from a fresh clone, so the audit above is workstation evidence, not a release artifact. Until section 8.1 lands, accuracy claims such as `91.5%` are advisory and must not be used as a customer promise.

### 5.3 B-Rep evidence is not a shipped moat

The current workstation contains 62 files under an ignored `data/brep_golden/` directory, including public NIST STEP/STP files and source metadata. `origin/main` tracks no corpus files there and has no release-eligible real manifest; only examples, builders, validators, and an informational workflow are reproducible.

Treat this as a useful sourcing start, not as proprietary data, a production golden, or a market moat. B-Rep work proceeds only when a customer use case requires it and the corpus has tracked provenance, labels, licenses, and release criteria.

### 5.4 The feedback flywheel is not closed

`src/api/v1/feedback.py` still writes a JSONL placeholder. Low-confidence queue and training gates exist, but there is no proven product workflow in which customer reviewers consistently produce governed corrections and model changes are measured against customer holdouts.

Do not resurrect deleted orphan learning abstractions. Build a feedback source only as part of a real reviewer workflow.

### 5.5 Production authentication is not safe by default

`src/api/dependencies.py` accepts a default API key value of `test` and defaults `ADMIN_TOKEN` to `test`. `src/api/middleware/integration_auth.py` defaults integration authentication to `disabled` and, in disabled or optional fallback paths, can populate identity from headers.

No customer pilot or external exposure is allowed until production configuration fails closed, authenticated identity is unambiguous, and negative tests prove anonymous or forged-header requests cannot reach protected analysis, administration, feedback, or write surfaces.

## 6. What to stop building

The following are not authorized without a customer-backed design lock:

- a generic chatbot or additional model-provider matrix;
- generative CAD;
- another PLM, PDM, workflow engine, or system of record;
- broad "serve ERP/MES/every system" adapters without a named consumer;
- point-cloud, digital-twin, or B-Rep breadth without a paid use case;
- unmounted APIs, speculative SDKs, enterprise-pattern scaffolds, or dashboards without an operator;
- autonomous feature generation sourced only from roadmap gaps.

Autonomous development may take work only from customer evidence, production failures, evaluation-integrity gaps, or explicit owner-ratified decisions. With none of those inputs, the correct cadence result is a no-op.

## 7. Operating model

### 7.1 Rigor levels

- **L1**: docs, copy, isolated maintenance. One PR and normal tests.
- **L2**: bounded product behavior. Design note when needed, focused tests, and regression verification.
- **L3**: authentication, authorization, customer data, AI output entering a sink, model-release gates, lifecycle write-back, or cross-system writes. Design lock, independent critic, fail-first golden, observed RED where feasible, default-off rollout, and explicit enablement decision.

Touching auth, audit ledgers, model-provider calls, customer drawing egress, canonical write-back, or AI output sinks automatically promotes work to L3. Authors may not self-downgrade it.

### 7.2 Definition of done

`merged != enabled != safe to enable`.

A capability is complete only when:

- its product owner and user are named;
- its contract and failure behavior are explicit;
- the relevant goldens execute in required CI;
- production configuration and rollback are tested;
- telemetry measures user outcome, not only service health;
- enablement has an owner, environment, date, and kill switch.

## 8. Ninety-day authorization

Merging this document authorizes only the following new product work. Phase 0 cleanup and CI hardening already authorized by [the 2026-07-06 design record](PLATFORM_POSITIONING_AND_ROADMAP_DESIGN_20260706.md) may continue, but it is engineering hygiene, not market validation.

### 8.1 Track E: evaluation-integrity-v2

Deliver a reproducible independent evaluation pack before improving models:

1. content-hash and normalized-family split enforcement, not path-only checks;
2. conflict quarantine for identical content with inconsistent labels;
3. customer-family or time-based holdout so revisions of one drawing cannot straddle train and validation;
4. per-class, macro, calibration, false-duplicate, and missed-reuse metrics;
5. separate reporting for real, synthetic, and augmented data;
6. a versioned manifest with source, license, provenance, family, hash, split, and label authority;
7. path-filtered dry-run in open PRs before making the new gate blocking.

Exit condition: a fresh clone can reproduce the evaluation result, and changing a split or reintroducing duplicate content makes the required gate red.

### 8.2 Track C: two customer discovery or paid-pilot attempts

Run two real discovery tracks in parallel with evaluation work. Each track must have:

- a named manufacturer or design partner;
- a sample archive that can legally be evaluated;
- a baseline for search time, duplicate creation, or reuse rate;
- a reviewer and a release/reuse workflow;
- an agreed pilot success metric and commercial next step.

Do not promise a full platform. Offer one vertical slice: ingest archive, find candidates, review evidence, decide reuse/revise/new, and export or write back the decision through the customer's authority.

Exit condition by day 90: at least one partner agrees to a measured pilot with data access and named reviewers. Otherwise pause feature work and reassess the wedge.

### 8.3 Pilot release gates

Before any pilot handles customer drawings:

- remove default test credentials from production mode;
- require authenticated tenant and user identity;
- verify customer-data storage, retention, deletion, and provider-egress policy;
- demonstrate kill switch, backup, rollback, and audit export;
- expose provider spend, budget alerts, and a fail-closed cost cap before external AI calls are enabled;
- run the candidate model on an independent customer holdout;
- document all unsupported formats and failure states.

### 8.4 Thirty, sixty, and ninety days

| Window | Required outcome | Stop signal |
|---|---|---|
| Days 0-30 | Freeze the evaluation-integrity-v2 contract; inventory and contact at least ten qualified manufacturers; obtain two legal sample-data conversations. | No one will share a bounded sample or name the costly workflow. Revisit the customer profile before building. |
| Days 31-60 | Run the reproducible evaluator on the first real archive; close production-auth defaults; demonstrate the reuse/revise/new review flow without canonical write-back. | Independent holdout quality or review usefulness is below the agreed pilot threshold. Improve evidence, not feature breadth. |
| Days 61-90 | Secure at least one measured pilot with named reviewers, baseline metrics, data policy, and commercial next step. | No measured pilot commitment. Pause product features and decide whether to fold the engine into the chosen CAD/PLM product. |

## 9. Three-year roadmap

### Year 1: prove one workflow and one buyer

- ship evaluation-integrity-v2;
- make production defaults fail closed;
- deliver one reuse/revise/new workflow end to end;
- onboard two or three design partners;
- convert at least one to a paid pilot;
- measure accepted reuse and false-duplicate cost, not model demo quality.

**Kill criterion**: if no customer pays or commits contractually by month 6, stop positioning this as an independent product. Keep the engine only as a component of the chosen CAD/PLM product.

### Year 2: make deployment repeatable

- standardize CAD/PLM adapters around the stable decision contract;
- provide an on-premise/private deployment package and operations runbook;
- reduce archive onboarding and customer calibration to two weeks;
- maintain customer-specific holdouts, thresholds, and rollback;
- add B-Rep only where customer pull and release-grade evidence justify it.

### Year 3: compound workflow evidence

- connect reuse decisions to BOM, supplier, cost, quality, and change history through the owning system of record;
- use accumulated human decisions and version-family links to improve retrieval and calibration;
- add natural-language explanation only over cited, permission-checked evidence;
- call the system a platform only after at least three paying consumers use a proven common core.

## 10. Scorecard and decision cadence

Review monthly during the first 90 days, then quarterly. Product metrics are:

- top-5 candidate usefulness and acceptance;
- false-duplicate and missed-reuse rates;
- drawings or parts reused;
- duplicate parts or drawings prevented;
- median time to find and review a candidate;
- active human reviewers and governed corrections;
- archive onboarding time;
- pilot conversion, renewal, and expansion.

PR count, test count, model count, endpoint count, and lines of code are not product KPIs.

## 11. Document hierarchy

1. **This file** is the current repo-level product strategy.
2. [`PLATFORM_POSITIONING_AND_ROADMAP_DESIGN_20260706.md`](PLATFORM_POSITIONING_AND_ROADMAP_DESIGN_20260706.md) is the historical code inventory, engineering convergence record, and Phase 0 authorization.
3. [PR #507](https://github.com/zensgit/cad-ml-platform/pull/507) is a pending cross-repository portfolio proposal. If ratified, it decides the product shell and system of record; it does not weaken this engine's evaluation, safety, or customer-evidence gates.
4. Dated development and verification documents prove individual slices. They do not silently change this strategy.

Any conflicting roadmap must explicitly amend this file. Adding another dated "current roadmap" is not sufficient.

## Appendix A: evidence notes

The key local findings were rechecked at `origin/main@0af2fa80`:

- manifest sizes: `data/manifests/golden_train_set.csv`, `data/manifests/golden_val_set.csv`;
- path-only overlap gate: `scripts/ci/check_training_data_governance.py`;
- path-only regression test: `tests/unit/test_training_data_governance.py`;
- auth defaults: `src/api/dependencies.py`, `src/api/middleware/integration_auth.py`;
- feedback placeholder: `src/api/v1/feedback.py`;
- visual recall boundary: `src/api/v1/dedup.py`, `src/core/dedupcad_vision.py`;
- deterministic precision boundary: `src/core/dedupcad_precision/`;
- B-Rep tracked surface: `config/brep_golden_manifest*.json`, `scripts/validate_brep_golden_manifest.py`, `.github/workflows/brep-golden-eval.yml`.

The content-overlap audit hashed the bytes at each manifest's resolved `file_path`. It intentionally did not treat different path strings as independent samples.
