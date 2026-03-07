# Assistant Evidence Output Validation

## Scope

- Branch line: assistant stable explainability enhancement
- Conflict strategy: only touch `src/core/assistant/*`, `src/api/v1/assistant.py`, and targeted tests/docs
- Non-goals: no edits to `src/api/v1/analyze.py` or `src/ml/hybrid_classifier.py`

## Capability

The assistant now returns a deterministic `evidence` array alongside the
existing `sources` strings. Each evidence item exposes:

- `reference_id`: stable ordinal reference (`E1`, `E2`, ...)
- `source`: normalized knowledge module name
- `summary`: retrieval summary
- `relevance`: normalized relevance score
- `match_type`: retrieval hit type such as `direct`, `keyword`, or `semantic`
- `key_facts`: compact fact list extracted from structured knowledge data

This keeps the free-form `answer` backward compatible while adding an
explainable, UI-friendly grounding payload that is stable across providers.

## Response Example

```json
{
  "success": true,
  "answer": "根据知识库查询结果...",
  "confidence": 0.95,
  "sources": [
    "threads: M10 coarse thread ..."
  ],
  "evidence": [
    {
      "reference_id": "E1",
      "source": "threads",
      "summary": "M10 coarse thread specification",
      "relevance": 0.95,
      "match_type": "direct",
      "key_facts": [
        "螺纹规格: M10",
        "公称直径: 10 mm",
        "螺距: 1.5 mm",
        "攻丝底孔: 8.5 mm"
      ]
    }
  ]
}
```

## Verification

### Static checks

```bash
python3 -m py_compile \
  src/core/assistant/explainability.py \
  src/core/assistant/assistant.py \
  src/core/assistant/api_service.py \
  src/core/assistant/__init__.py \
  src/api/v1/assistant.py \
  tests/unit/assistant/test_explainability.py \
  tests/unit/assistant/test_llm_api.py \
  tests/unit/assistant/test_api_service.py

flake8 \
  src/core/assistant/explainability.py \
  src/core/assistant/assistant.py \
  src/core/assistant/api_service.py \
  src/core/assistant/__init__.py \
  src/api/v1/assistant.py \
  tests/unit/assistant/test_explainability.py \
  tests/unit/assistant/test_llm_api.py \
  tests/unit/assistant/test_api_service.py
```

Result:

- `py_compile`: passed
- `flake8`: passed

### Targeted tests

```bash
python3 -m pytest \
  tests/unit/assistant/test_explainability.py \
  tests/unit/assistant/test_llm_api.py \
  tests/unit/assistant/test_api_service.py \
  -q

python3 -m pytest tests/unit/assistant/test_assistant.py -q
```

Result:

- `tests/unit/assistant/test_explainability.py`
- `tests/unit/assistant/test_llm_api.py`
- `tests/unit/assistant/test_api_service.py`
  - `70 passed, 2 skipped in 5.29s`
- `tests/unit/assistant/test_assistant.py`
  - `35 passed in 2.81s`
