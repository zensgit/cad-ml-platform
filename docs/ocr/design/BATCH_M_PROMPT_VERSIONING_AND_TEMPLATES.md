# Batch M — Prompt Versioning and Templates

Scope
- Centralize provider prompts and embed `PROMPT_VERSION` for cache compatibility.
- Ensure DeepSeek prompt follows strict JSON → fenced JSON → regex degradation path.

Design
- Add `src/core/ocr/utils/prompt_templates.py` with `deepseek_ocr_json_prompt()`.
- Prompt includes schema hints (dimensions, symbols, title_block) and instructs fenced block fallback.
- `PROMPT_VERSION` is read from `src/core/ocr/config.py` and implicitly part of the cache key.

Rationale
- Changing prompts modifies output distribution; tying to versioning lets us manage compatibility, rollbacks, and cache invalidation.
- Single source avoids prompt drift between providers/environments.

Acceptance
- DeepSeek provider imports and uses the centralized prompt.
- Unit tests still pass; cache keys include `PROMPT_VERSION`.
- Docs updated in CHANGELOG and TODO already referencing prompt versioning.

Risks
- Real model behavior may require additional guardrails or examples; we keep the prompt concise to reduce token cost and variance.

