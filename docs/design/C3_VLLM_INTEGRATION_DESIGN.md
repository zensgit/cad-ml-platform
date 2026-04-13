# C3: vLLM End-to-End Integration Design

## Overview

Phase C3 integrates the VLLMProvider (C1) and benchmark framework (C2) end-to-end with the vision, OCR, and assistant pipelines. All new functionality is gated behind feature flags for safe, granular rollout.

## End-to-End Data Flow

```
DXF File Upload
    │
    ▼
┌─────────────┐
│  OCR Layer   │ ── PaddleOCR extracts raw text
│ (manager.py) │
└──────┬──────┘
       │ raw OCR text
       ▼
┌──────────────────┐    feature flag: vllm_ocr_enhancement_enabled
│ vLLM OCR Enhancer│ ── Post-process with local LLM for structured extraction
│ (optional)       │    Extracts: part_name, material, drawing_number, etc.
└──────┬───────────┘
       │ structured fields merged into OcrResult.title_block
       ▼
┌─────────────────┐     feature flag: vllm_vision_enabled
│ Vision Analysis │ ── VLLMVisionProvider describes drawing content
│ (optional)      │    Falls back to DeepSeekStub if unavailable
└──────┬──────────┘
       │
       ▼
┌──────────────────┐    feature flag: vllm_enabled
│ Classification   │ ── LLM-assisted classification (optional)
│ + RAG Assistant  │    Uses classification_prompt.py template
└──────────────────┘
```

## Provider Selection Flowchart

```
Start
  │
  ▼
vllm_enabled flag ON? ──No──► Standard fallback chain
  │                           (Claude > OpenAI > Qwen > Ollama > Offline)
  Yes
  │
  ▼
vLLM server healthy? ──No──► Standard fallback chain
  │
  Yes
  │
  ▼
Use VLLMProvider as primary
  │
  ▼
vLLM call succeeds? ──No──► Auto-fallback:
  │                          Claude > OpenAI > Qwen > Ollama > Offline
  Yes
  │
  ▼
Return vLLM response
```

## Prompt Engineering Strategy

### Why shorter prompts for local models

| Factor | Cloud API (GPT-4/Claude) | Local 7B (vLLM) |
|--------|--------------------------|------------------|
| Context window | 128K+ tokens | 4K-8K effective |
| Instruction following | Excellent | Good with clear structure |
| Few-shot learning | Handles many examples | 1-2 examples optimal |
| System prompt budget | ~1000 tokens | ~200-500 tokens |
| Response format | Flexible | JSON with explicit schema |

### Prompt design principles

1. **Concise system prompts**: < 500 tokens, role + rules + domain hints
2. **Explicit output format**: Always specify JSON schema in prompt
3. **Few-shot examples**: 1-2 examples only (not 5+)
4. **Bilingual**: Chinese primary (matches CAD domain), English fallback
5. **Minimal variants**: `max_tokens < 2000` triggers even shorter prompts

### Prompt templates

| Template | Purpose | Token budget |
|----------|---------|-------------|
| `cad_system_prompt.py` | General CAD assistant system prompt | ~150-300 tokens |
| `ocr_extraction_prompt.py` | Title block field extraction from OCR text | ~300-500 tokens |
| `classification_prompt.py` | Drawing type classification with reasoning | ~300-400 tokens |

## Feature Flag Rollout Plan

### Recommended rollout order

1. **`vllm_enabled`** (Phase C1) -- Enable vLLM as LLM provider
   - Prerequisite: vLLM server deployed and benchmarked (C2)
   - Validation: Compare response quality with cloud API via A/B metrics
   - Rollback: Disable flag, system falls back to cloud API

2. **`vllm_ocr_enhancement_enabled`** (Phase C3) -- Enable OCR post-processing
   - Prerequisite: `vllm_enabled` is on and stable
   - Validation: Compare title_block extraction accuracy with/without enhancement
   - Risk: Low (enhancement only fills missing fields, never overwrites)

3. **`vllm_vision_enabled`** (Phase C3) -- Enable vision analysis
   - Prerequisite: `vllm_enabled` is on and stable
   - Validation: Compare VisionDescription quality with stub
   - Risk: Low (falls back to stub on any failure)

### Flag dependencies

```
vllm_enabled (master switch)
  ├── vllm_ocr_enhancement_enabled
  └── vllm_vision_enabled
```

## Latency Comparison Framework

All LLM calls now include latency tracking via `time.monotonic()`:

```python
# In assistant._call_llm():
start = time.monotonic()
result = provider.generate(system_prompt, user_prompt)
latency_ms = (time.monotonic() - start) * 1000
logger.info("llm.generate", extra={"provider": name, "latency_ms": latency_ms})
```

### Expected latency targets

| Provider | Target p50 | Target p99 |
|----------|-----------|-----------|
| vLLM (local, AWQ) | < 100ms | < 500ms |
| Claude API | 500-2000ms | 3000ms |
| OpenAI API | 500-2000ms | 3000ms |
| Ollama (local) | 200-1000ms | 2000ms |

### Monitoring

- Log-based: `llm.generate` and `llm.fallback.success` structured log events
- Health endpoint: `assistant.llm_health_status()` returns active provider + status
- vLLM-specific: `VLLMProvider.health_check()` returns model list and server state

## Fallback Scenarios and Degradation Behavior

| Scenario | Behavior | User Impact |
|----------|----------|-------------|
| vLLM server down | Auto-fallback to cloud API chain | Slightly higher latency |
| vLLM slow (>timeout) | Timeout + fallback | Single request delayed |
| vLLM returns invalid JSON | Parse fallback (raw text as summary) | Reduced structure |
| All LLM providers down | OfflineProvider (knowledge base only) | Degraded but functional |
| OCR enhancement fails | Skip enhancement, return raw OCR result | No regression |
| Vision vLLM fails | Return stub VisionDescription | Fixed response |

### Key invariant

> If vLLM is completely unavailable, the system behaves exactly as it did before C3.
> No feature flag being ON can break the system if vLLM is DOWN.

## Security: Data Locality

With vLLM as primary provider:

- **No sensitive data leaves the local network**: All inference happens on-premise
- **Drawing content stays local**: OCR text, image descriptions processed by local GPU
- **API keys not required**: vLLM uses no authentication by default
- **Audit trail**: All LLM calls logged with provider identity

This is a significant advantage for manufacturing environments with IP-sensitive CAD data.

## Verification Results

### Test coverage

- `tests/integration/test_vllm_integration.py`: 20+ test cases covering:
  - vLLM vision provider with mocked endpoint
  - OCR enhancer pipeline (PaddleOCR -> vLLM)
  - Prompt templates (format validation, token budget)
  - Fallback chain (vLLM unavailable -> next provider)
  - Feature flag gating (disabled -> skip vLLM)

### Files changed/added

| File | Action | Description |
|------|--------|-------------|
| `src/core/vision/providers/vllm_vision.py` | **New** | vLLM-backed vision provider |
| `src/core/ocr/providers/vllm_ocr_enhancer.py` | **New** | vLLM OCR post-processor |
| `src/core/assistant/prompts/__init__.py` | **New** | Prompt templates package |
| `src/core/assistant/prompts/cad_system_prompt.py` | **New** | CAD domain system prompt |
| `src/core/assistant/prompts/ocr_extraction_prompt.py` | **New** | OCR extraction prompt |
| `src/core/assistant/prompts/classification_prompt.py` | **New** | Classification prompt |
| `src/core/vision/providers/__init__.py` | **Modified** | Register VLLMVisionProvider |
| `src/core/ocr/manager.py` | **Modified** | Add vLLM enhancement step |
| `src/core/assistant/assistant.py` | **Modified** | vLLM-first selection + fallback + latency |
| `config/feature_flags.json` | **Modified** | Add 2 new flags |
| `tests/integration/test_vllm_integration.py` | **New** | Integration tests |
| `docs/design/C3_VLLM_INTEGRATION_DESIGN.md` | **New** | This document |
