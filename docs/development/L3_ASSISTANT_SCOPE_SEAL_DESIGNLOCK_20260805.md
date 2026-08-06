# L3 Design-Lock — CAD Assistant: Safe-Park Seal + Scope Reconciliation

**Date**: 2026-08-05 · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1 — model-provider calls, potential customer-drawing egress)
**Grounded on**: `origin/main@048172e7` (post-#528, post-#532)
**Authority**: `PRODUCT_STRATEGY.md` §6 (stop-building list), §3.3 (stable decision contract), §4 (AI safety rules), §2.1 (private-deployment customer profile) · `#530` (repository closeout / safe-park plan, `docs/development/REPO_CLOSEOUT_SAFE_PARK_PLAN_20260721.md` — safe-park definition: "Phase A complete + every dangerous path sealed + honest posture documented at a pinned SHA")

> **This is a proposal, not an implementation.** It changes no runtime, calls no model, moves no code. Solo-maintainer L3 review protocol (`L3_MODEL_ACTIVATION_MEMBRANE_DESIGNLOCK_20260712.md` §"Solo-maintainer L3 review protocol") applies verbatim: an isolated critic supplies evidence; the human owner alone ratifies and pins a head.

---

## 0. Why this exists — an undocumented L3 surface, found outside #530's scope

`#530` drives the repository toward **safe-park**: three end states (product continues / folds into a component / mothballs) decided later by calendar + customer evidence, with one shared prerequisite — *every dangerous path sealed*, needing no new customer data or model environment. `#530`'s Layer 0–4 structure (13-PR disposition, C2–C6 sequence, §5 honest-posture inventory, Track E/Phase B left unscheduled, #507 fold-in deferred) was written **2026-07-21**, before this gap was found. It does not mention `src/core/assistant/` anywhere.

**Empirical findings, this worktree, `048172e7`:**

| # | Finding | Evidence |
|---|---|---|
| 1 | `src/core/assistant/` is a large, live subsystem: 44 files, **14,587 lines** — RAG retrieval, function-calling with 9 domain tools, multi-tenant + RBAC, streaming, explainability, quality evaluation, conversation memory | `find src/core/assistant -name '*.py' \| wc -l` (corrected 2026-08-05 review — an earlier pass over-counted at 47) |
| 2 | **Predates the ratified strategy by ~6 months**: added 2026-01-29 (`feat: add RAG-based CAD intelligent assistant module (#44)`); still being actively touched 2026-07-10 — 2 days before `PRODUCT_STRATEGY.md` ratified 2026-07-12 | `git log --follow -- src/core/assistant/assistant.py` |
| 3 | **Mounted and live**: `assistant = _import_router("assistant", "src.api.v1.assistant")` at `/assistant/*` | `src/api/__init__.py:266,494-500` |
| 4 | **Deployed via CD**: `.github/workflows/cd.yml:77` builds and pushes `./docker/assistant/Dockerfile` | `cd.yml:75-79` |
| 5 | **6 concrete LLM provider classes** — `ClaudeProvider`, `OpenAIProvider`, `QwenProvider`, `OllamaProvider`, `VLLMProvider`, `OfflineProvider` — this is what `PRODUCT_STRATEGY.md` §6 names: *"a generic chatbot or additional model-provider matrix"* is not authorized without a customer-backed design lock | `src/core/assistant/llm_providers.py:28,54,95,138,182,227,386` |
| 6 | `ClaudeProvider`/`OpenAIProvider` lazily `from anthropic import Anthropic` / `from openai import OpenAI` — **neither `anthropic` nor `openai` appears in any `requirements*.txt`** (grep with `fastapi` positive control confirmed the search itself is live) | `llm_providers.py:65,108`; `grep -rn '^anthropic\|^openai' requirements*.txt` → empty |
| 7 | `domain_embedding_provider.py` (this same subsystem) degrades to a **TF-IDF fallback** — `sentence-transformers` is commented out in `requirements-assistant.txt:22-23` — the same defect class already disclosed in `PHASE0_A3_HONEST_EMBEDDING_DEGRADATION_DESIGN_20260708.md` (#503) | codebase audit this session; `requirements-assistant.txt` |
| 8 | `PRODUCT_STRATEGY.md` contains **zero** mentions of "assistant" — not authorized, not exempted, not addressed | `grep -in assistant docs/PRODUCT_STRATEGY.md` → empty |
| 9 | `api/v1/assistant.py` already contains `_decision_contract_from_metadata` / `_decision_evidence_from_metadata` / `_knowledge_citations_from_decision_evidence` — evidence the module was *aimed* at explaining this platform's own §3.3 decision contract, not built as a general-purpose chatbot from scratch | `src/api/v1/assistant.py:188-320` |
| 10 | The 9 `tools/` (e.g. `similarity_tool.py`: `"在向量库中搜索与指定图纸相似的零件"`) wrap the platform's **own** existing capabilities (similarity search, classification, point-cloud, cost, quality) via function-calling — confirmed by direct read, not assumed | `src/core/assistant/tools/similarity_tool.py:1-30` |
| 11 | **The fallback chain escalates to hosted providers regardless of the caller's original provider choice.** `_fallback_generate` iterates `["claude", "openai", "qwen", "vllm", "ollama"]` **in that literal order** on any primary-provider error — three hosted providers are attempted *before* the one local candidate (`ollama`) is ever tried. A no-opt-in / local-only deployment whose primary attempt errors (network blip, local-model timeout) can still egress to a hosted provider if a hosted API key happens to be configured, with no opt-in check at this layer at all. (Owner review, 2026-08-05) | `src/core/assistant/assistant.py:493-498` (`_fallback_generate`) |
| 12 | **Tool-execution failure returns a success-shaped payload, with no uniform structured failure marker.** Severity varies by tool, all confirmed by direct read: `cost_tool.py` fabricates a full CNY cost breakdown (material/machining/setup/overhead/total) from hardcoded per-material base rates on **any** exception, with **zero disclosure field in the returned dict** — only a `logger.warning` line, which a payload consumer never sees; `feature_tool.py` returns hardcoded default dimensions (17/22) plus a free-text `note` field; `similarity_tool.py` returns an empty result set plus a free-text `note` (safer — no fabricated candidates — but a silent `count: 0` is itself indistinguishable from "verified no duplicates" without reading the note); `classify_tool.py` returns `label: "unknown", confidence: 0.0` plus a free-text `note` — a shape a consumer could plausibly conflate with a genuine low-confidence classification. None of the four use a structured, schema-checkable status field. (Owner review, 2026-08-05) | `src/core/assistant/tools/cost_tool.py:62-89`; `feature_tool.py:40-63`; `similarity_tool.py:56-85`; `classify_tool.py:40-64` |

**Owner ruling (2026-08-05, this design-lock's originating conversation):** the absence from `PRODUCT_STRATEGY.md` is an **oversight, not an exemption**. This module is to be narrowed/reviewed under §6's spirit — the question this lock answers is *how*, in a way consistent with #530's safe-park constraints (no new customer data, no new model environment, no-regrets under all three end states).

**Owner precise-head review (2026-08-05, this PR's head `61187de7`):** findings 11–12 above are additions from that review. Verdict: direction confirmed ("assistant should only become an Evidence Pack explainability layer, not WorkBuddy/a CAD ChatGPT" — consistent with §3 below), but **not seal-complete** — findings 11–12 (P1) and the tightened egress allowlist (§2.B, P2) must be reflected as acceptance criteria in this lock before it is treated as ready to implement against. CI green on a docs-only PR proves the document parses; it says nothing about runtime safety.

---

## 1. Framing: seal now, redesign later — two different asks, not one

Finding #5–#7 above describe a **dangerous path**, in #530's own vocabulary: a live, deployed, mounted service that can route arbitrary user input to hosted third-party LLM providers, with no documented redaction boundary, sitting directly adjacent to customer-drawing data (via its own tool layer, finding #10) and a degraded retrieval substrate (finding #7) it does not honestly disclose per-response.

Two asks follow, and they are **not the same size or the same gate**:

- **§2 (SEAL) — a no-regrets action, in scope for safe-park now.** Fail-closed defaults, an explicit egress boundary. Needs no new customer data, no new model environment, no product decision about the assistant's fate. Correct under *all three* of #530's end states — sealing a dangerous default doesn't cost anything if the module is later mothballed, folded in, or kept.
- **§3 (REDESIGN) — a product decision, explicitly NOT authorized by this lock.** Narrowing the module's *purpose* to "explain this platform's own decision evidence" is only worth doing if the "continues as an independent product" end-state wins. It needs a named owner and a customer-facing rationale `PRODUCT_STRATEGY.md` §7.2's definition-of-done requires ("its product owner and user are named"). This lock describes the target shape so a *future*, separately-ratified implementation has something to build against — it does not authorize starting that work now.

---

## 2. SEAL — the immediate, no-regrets target contract

Mirrors the pattern already used for `/model/reload` (#513/#516) and production-identity defaults (#517): **fail-closed first, redesign later, never a flag that silently re-opens the sealed path.**

### A. Provider default: local-only unless explicitly opted in — including on the failure/fallback path
- With no explicit, per-deployment configuration, provider resolution **MUST** land on `OfflineProvider` (or `OllamaProvider` pointed at a local endpoint) — never a hosted provider by default. Consistent with §2.1's on-premise/private-network requirement.
- Selecting `ClaudeProvider` / `OpenAIProvider` / `QwenProvider` / any remote `VLLMProvider` **MUST** require an explicit opt-in flag documented as an external-provider-egress event under the same governance as any other external AI call (§4's "customer drawings stay private by default; external-provider transmission is explicit opt-in" rule, and §8.3's pilot release gates) — not merely "configure an API key and it works."
- **The seal covers the fallback path, not only the initial default (finding #11).** When the explicitly-opted-in (or default local) provider errors, the fallback chain **MUST NOT** attempt any hosted provider that was not itself part of the explicit opt-in. A no-opt-in / local-only deployment whose primary attempt fails **MUST** fail closed (or retry local-only candidates) — it **MUST NOT** walk a hosted-first priority list (`claude → openai → qwen → vllm → ollama`, the current order) merely because a hosted API key happens to be present. The opt-in decision governs every attempt for a request, not just the first one.
- A provider class whose required SDK (`anthropic`, `openai`) is not installed **MUST** fail closed to `OfflineProvider` at selection time, not raise an unhandled `ImportError` or silently degrade mid-request (verify current behavior — not yet checked; this lock flags it as a thing to verify, not asserts it either way).

### B. Egress boundary: a field-level allowlist schema, not a category description
- A payload sent to a hosted provider **MUST** be validated against an explicit, field-level allowlist schema — not merely "computed §3.3 fields are OK." §3.3's own field list (candidate id, provenance, evidence/rejection reasons, drawing numbers implied by provenance, supplier/material/process notes) can itself carry customer-sensitive identifiers even though every value is "computed," not raw. The schema **MUST**, by default: redact or hash candidate/file identifiers (no plaintext IDs that trace back to a customer's part/drawing numbering scheme); exclude supplier names, drawing numbers, and free-text process/material notes unless a field is explicitly allowlisted per-deployment; permit only the minimum needed for the explainability purpose (scores, confidence, rejection-reason *category*, not verbatim customer-authored text).
- Raw drawing bytes, raw OCR text, and file paths remain categorically excluded (unchanged from the original draft) — the schema requirement above is additive, not a replacement.
- A tool call (`similarity_tool`, `classify_tool`, etc.) whose result is destined for a hosted-provider payload **MUST** pass through the same schema-validated redaction boundary before the network call, not after.

### C. Honest degradation disclosure: structured status, not a log line or a free-text note (extends #503; closes finding #12)
- Any computation that did not complete as designed — the embedding/retrieval TF-IDF fallback (already governed by `PHASE0_A3_HONEST_EMBEDDING_DEGRADATION_DESIGN_20260708.md` / #503), **and every assistant tool's exception path** (finding #12: `cost_tool`, `feature_tool`, `similarity_tool`, `classify_tool` today) — **MUST** carry a structured `status` field with one of `failed` / `unavailable` / `degraded`, not only a log line or a free-text `note`. A free-text note is easy for downstream response-assembly (or an LLM composing an answer) to silently drop while still consuming the numeric/label fields as if they were valid.
- A degraded/failed tool result **MUST NOT** be citable as decision evidence: no fabricated business-looking values (e.g. `cost_tool`'s current hardcoded CNY cost breakdown on any exception) may be returned in place of a clearly-marked absence. Where a placeholder value is structurally required, it **MUST** be paired with the `status` field above, not stand alone.
- This is one invariant, not two: the same "never let a degraded/failed computation look like a real result" principle governs both the retrieval substrate and the tool layer. An implementation must not treat them as separate problems with separate fixes.

### D. Deployment posture during review
- Owner decision, not assumed here (see §5): pause `docker/assistant`'s CD deployment while this lock is under review, or leave it live with SEAL items tracked as the acceptance bar for the next PR touching it. Either is compatible with safe-park; this lock does not pick one.

---

## 3. REDESIGN target (described, not authorized) — for a later, separately-gated lock

If a future decision (owner-named, customer-facing, per §7.2) keeps this module as part of the product:

- **Scope narrows from general-purpose to own-evidence explainability.** Every response about a specific candidate/decision traces to a §3.3 decision-contract field; general-purpose content generation unrelated to a specific evidence bundle (reports, slides, free-form chat) is out of scope — that capability already exists in the market (Tencent WorkBuddy and peers) and competing on it contradicts this platform's narrow-wedge bet, not just §6's letter.
- **Placement**: assistant output surfaces inside the existing evidence-review flow (§3.1's boundary: recall → verification → evidence bundle → human decision), not as a separate destination requiring a context switch — unverified whether this is already true; a question for implementation, not this lock.
- **No new remote-dispatch/IM-bot/notification-channel integration** (WeCom/DingTalk/Slack-style) is authorized by any future work here without a named customer request — §6's "no adapters without a named consumer" applies at full force.
- **RAG substrate fix** (replacing the TF-IDF fallback with real embeddings) is its own evidence-integrity work, gated the same way as Track E, not bundled into this module's redesign.

---

## 4. What this lock does NOT authorize

No code change. No new provider onboarding. No RAG/embedding fix. No IM/notification integration. No decision on whether `docker/assistant` keeps deploying. No decision on the module's ultimate fate (continue / fold-in / mothball) — that is #530's decision ladder, unaffected by this lock. No claim that finding #6 (missing SDKs) means the providers are currently non-functional in production — that would require checking whether `anthropic`/`openai` are installed via some path this lock did not check (e.g. a base image layer); flagged as unverified, not asserted.

---

## 5. Open questions for the owner (ratification checklist)

- Does §2 (SEAL) fold into `#530` Layer 2 (honest-posture inventory) as an added item, or run as its own line? It fits Layer 2's definition (a §5-class gap: undocumented capability, not yet honestly parked) but was not in scope when Layer 0–4 were drafted.
- Deployment posture (§2.D): pause `docker/assistant` during review, or seal-forward without pausing?
- Is there a named consumer for `/assistant/*` today that this investigation did not surface, which would change any of the above?
- Does §3 (REDESIGN) wait for #530's own decision ladder (Day-90 / Month-6), or can it be scoped independently once SEAL lands, given it's a different kind of decision (product scope, not calendar/customer-evidence gate)?
- This lock is not yet ratified as seal-complete (owner verdict, 2026-08-05): findings #11–#12 and the tightened §2.B schema are now reflected as acceptance criteria (§2, §6). Does the revised contract satisfy the review, or does it need another pass before an implementation PR may cite it as its target?

---

## 6. Verification contract (what SEAL implementation must make impossible)

At contract altitude, for the eventual SEAL implementation PR:

1. No explicit opt-in configured → a request that would otherwise route to `ClaudeProvider`/`OpenAIProvider`/`QwenProvider` → **fails closed** to `OfflineProvider`, not silently succeeds against a hosted endpoint.
2. **(closes finding #11)** Force the opted-in-or-default primary provider to error → assert the resulting fallback attempt sequence contains **zero** hosted-provider calls when no hosted provider was part of the explicit opt-in — not merely that the *final* response came from a local provider (an implementation could satisfy that while still having attempted, and partially transmitted a prompt to, a hosted provider along the way; the discriminator must catch the attempt, not just the outcome).
3. A hosted-provider call payload is validated against the field-level allowlist schema (§2.B) → **rejected before the network call** if it contains raw image bytes / raw OCR text / file-path strings, an un-redacted candidate/file identifier, or any field not on the explicit per-deployment allowlist.
4. **(closes finding #12)** Inject a forced exception into each of `cost_tool`, `feature_tool`, `similarity_tool`, `classify_tool` in turn → assert the returned payload carries `status` ∈ `{failed, unavailable, degraded}` as a structured field (not only a log line or a free-text `note`), and that response-assembly code refuses to present that result as decision evidence (e.g. it cannot populate a `_decision_evidence_from_metadata`-style citation from a `status`-marked result).
5. A response whose retrieval used the TF-IDF fallback and does **not** carry the disclosed-fallback marker → **flagged as a regression** against the #503 invariant (same structured-status requirement as #4, applied to the retrieval substrate).
6. `anthropic`/`openai` uninstalled (the current, verified state) → provider selection resolves to `OfflineProvider` deterministically, not an unhandled exception surfaced to the caller.

---

*Independent-context evidence for the owner's ratification decision. `merged != enabled != safe to enable`. The drafting agent does not ratify or merge.*
