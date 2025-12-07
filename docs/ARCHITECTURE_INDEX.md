# CAD ML Platform - Architecture Index

**Comprehensive File Reference for Enhancement Planning**

---

## Core Feature Extraction Pipeline

### Primary Files

| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `src/core/feature_extractor.py` | 644 | Feature extraction engine | `FeatureExtractor`, `compute_shape_entropy()`, `compute_surface_count()` |
| `src/core/feature_manifest.py` | - | Version registry | `expected_length_for_version()` |
| `src/core/invariant_features.py` | - | v5/v6 computation | `compute_normalized_shape_signature()`, moment invariants |
| `src/ml/metric_embedder.py` | 78 | Metric learning wrapper | `MetricEmbedder.embed()` |

### Integration Pattern
```
CadDocument → FeatureExtractor.extract() 
  → v1-v4: Basic geometric/semantic
  → v5-v6: Invariant features
  → [NEW v7]: Visual CNN features
  → MetricEmbedder (optional)
  → {geometric, semantic} dict
  → Vector register
```

---

## Vector Storage & Search

### Primary Files

| File | Lines | Purpose | Key Classes/Functions |
|------|-------|---------|----------------------|
| `src/core/similarity.py` | 1188 | Dual-backend vector store | `InMemoryVectorStore`, `FaissVectorStore`, `register_vector()` |
| `src/utils/cache.py` | - | Redis client | `get_client()`, `init_redis()` |

### Architecture
```
┌─ Memory (Dict) ────────────┐
│  _VECTOR_STORE             │
│  _VECTOR_META              │
└────────────────────────────┘
    ↕ Sync
┌─ Faiss ANN ───────────────┐
│  IndexFlatIP               │
│  _FAISS_ID_MAP             │
└────────────────────────────┘
    ↕ Persist
┌─ Redis (TTL) ─────────────┐
│  vector:{doc_id} hash      │
└────────────────────────────┘
```

### Key Functions
- `register_vector(doc_id, vector, meta)` - Add with validation
- `InMemoryVectorStore.query()` - Top-k cosine similarity
- `FaissVectorStore.query()` - ANN search with normalization
- `get_vector_store(backend)` - Factory with fallback logic

---

## Knowledge-Based Classification

### Primary Files

| File | Lines | Purpose | Key Classes |
|------|-------|---------|------------|
| `src/core/knowledge/__init__.py` | - | Module exports | - |
| `src/core/knowledge/part_knowledge.py` | - | Base classifier | `MechanicalPartKnowledgeBase` |
| `src/core/knowledge/material_knowledge.py` | - | Material rules | `MaterialKnowledgeBase` |
| `src/core/knowledge/precision_knowledge.py` | - | Tolerance rules | `PrecisionKnowledgeBase` |
| `src/core/knowledge/standards_knowledge.py` | - | Standards | `StandardsKnowledgeBase` |
| `src/core/knowledge/functional_knowledge.py` | - | Feature roles | `FunctionalKnowledgeBase` |
| `src/core/knowledge/assembly_knowledge.py` | - | Fit types | `AssemblyKnowledgeBase` |
| `src/core/knowledge/manufacturing_knowledge.py` | - | Machining | `ManufacturingKnowledgeBase` |
| `src/core/knowledge/enhanced_classifier.py` | - | Multi-module fusion | `EnhancedPartClassifier` |

### Integration Pattern
```
Geometric Features ──┐
OCR Data ───────────→ EnhancedPartClassifier
Entity Counts ───┘
                     ├→ MaterialKnowledgeBase
                     ├→ PrecisionKnowledgeBase
                     ├→ StandardsKnowledgeBase
                     ├→ FunctionalKnowledgeBase
                     ├→ AssemblyKnowledgeBase
                     └→ ManufacturingKnowledgeBase
                     
                     ↓ Fusion
                     
                     {part_type, confidence}
```

---

## OCR & Dimension Parsing

### Primary Files

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `src/core/ocr/manager.py` | - | OCR orchestration | `OcrManager.extract()` |
| `src/core/ocr/providers/paddle.py` | - | Paddle provider | - |
| `src/core/ocr/providers/deepseek_hf.py` | - | DeepSeek provider | - |
| `src/core/ocr/parsing/dimension_parser.py` | - | Regex parsing | `parse_dimensions()`, `parse_symbols()` |
| `src/core/ocr/base.py` | - | Common types | `DimensionInfo`, `SymbolInfo` |

### Dimension Parser Patterns
```
Input: "Φ 20mm ±0.02" → DIAMETER_PATTERN
Output: {"type": "diameter", "value": 20.0, "unit": "mm", "tolerance": ±0.02}

Input: "M10×1.5" → THREAD_PATTERN
Output: {"type": "thread", "pitch": 1.5}

Input: "Ra 3.2 μm" → ROUGHNESS_PATTERN
Output: {"type": "roughness", "value": 3.2}
```

---

## Vision Module (Stubs)

### Primary Files

| File | Lines | Purpose | Key Classes |
|------|-------|---------|------------|
| `src/core/vision_analyzer.py` | - | Multi-provider | `VisionAnalyzer` |
| `src/core/vision/base.py` | - | Models & protocols | `VisionAnalyzeRequest/Response`, `VisionProvider` (ABC) |
| `src/core/vision/manager.py` | - | Vision orchestration | `VisionManager` |
| `src/core/vision/providers/deepseek_stub.py` | - | Stub provider | `DeepSeekStubProvider` |

### Status: Mostly stubs, ready for implementation
- OpenAI GPT-4 Vision: Client initialized, not integrated
- Anthropic Claude Vision: Available, not used
- Azure Computer Vision: Framework, not configured
- Google Cloud Vision: Framework, not configured
- Local models: Framework ready

---

## API Endpoints

### Analyze Router

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/v1/analyze/extract-features` | POST | Extract vector | CAD file | Feature dict |
| `/v1/analyze/classify` | POST | Classify part | Features | Part type |
| `/v1/analyze/reason` | POST | [NEW] Reason classification | Analysis ID | Reasoning chain |
| `/v1/analyze/check-quality` | POST | Quality assessment | Features | Issues/suggestions |
| `/v1/analyze/recommend-process` | POST | Manufacturing recommendation | Features | Process type |
| `/v1/analyze/batch-similarity` | POST | Multi-reference matching | Vector + refs | Ranked matches |

### Vectors Router

| Endpoint | Method | Purpose | New in Phase 3 |
|----------|--------|---------|---|
| `/v1/vectors` | GET | List vectors | - |
| `/v1/vectors/delete` | POST | Remove vector | - |
| `/v1/vectors/hybrid-search` | POST | Multi-modal search | ✅ NEW |
| `/v1/vectors/similarity` | POST | Single pair similarity | - |

### Feedback Router [NEW]

| Endpoint | Method | Purpose | Phase |
|----------|--------|---------|-------|
| `/v1/feedback/submit` | POST | Record correction | Phase 4 |
| `/v1/feedback/uncertain` | GET | List uncertain examples | Phase 4 |

### Reasoning Router [NEW]

| Endpoint | Method | Purpose | Phase |
|----------|--------|---------|-------|
| `/v1/analyze/reason` | POST | LLM reasoning | Phase 2 |

---

## Configuration & Settings

### Primary Files

| File | Purpose | Key Vars |
|------|---------|----------|
| `src/core/config.py` | Runtime settings | `DEBUG`, `REDIS_URL`, `FEATURE_EMBEDDER_BACKEND` |
| `.env.dev` | Dev environment | [All above + local overrides] |

### Feature Version Control
```
FEATURE_VERSION=v1|v2|v3|v4|v5|v6|v7  # Active version
FEATURE_VERSION_STRICT=0|1             # Enforce dimension match
FEATURE_L2_NORMALIZE=0|1               # Normalize v5/v6 output
FEATURE_EMBEDDER_BACKEND=none|ml_embed_v1
METRIC_MODEL_PATH=models/metric_learning/best_model.pth
```

### Vector Store Control
```
VECTOR_STORE_BACKEND=memory|faiss|redis
VECTOR_TTL_SECONDS=0                   # 0=disabled
FAISS_INDEX_PATH=data/faiss_index.bin
FAISS_RECOVERY_INTERVAL_SECONDS=300
```

---

## Testing Structure

### Test Organization

| Directory | Purpose | Pattern |
|-----------|---------|---------|
| `tests/unit/test_*.py` | Unit tests | Pytest |
| `tests/unit/test_v*.py` | Version tests | Feature version validation |
| `tests/unit/test_*_metric.py` | Metrics tests | Prometheus metric correctness |
| `tests/unit/test_*_resilience.py` | Resilience tests | Fallback/recovery scenarios |

### Key Test Files to Review
- `tests/unit/test_faiss_degraded_batch.py` - Faiss fallback testing
- `tests/unit/test_model_security_validation.py` - Model reload security
- `tests/unit/test_vector_migration_*.py` - Version migration tests

---

## Metrics & Observability

### Primary File

| File | Purpose |
|------|---------|
| `src/utils/analysis_metrics.py` | Prometheus metric registration |

### Key Metrics by Phase

**Phase 1 (Visual)**
```
feature_v7_visual_present_ratio: gauge
feature_extraction_latency_seconds{version="v7"}: histogram
feature_vector_l2_norm{version="v7"}: histogram
```

**Phase 2 (LLM)**
```
llm_inference_seconds: histogram
llm_parsing_errors_total{reason}: counter
llm_cost_usd_total: counter
```

**Phase 3 (Hybrid)**
```
hybrid_search_latency_seconds: histogram
fusion_score_distribution: histogram
vector_query_backend_total{backend}: counter
```

**Phase 4 (Active Learning)**
```
feedback_submission_rate: gauge
prediction_correction_rate_by_type{type}: gauge
retraining_trigger_events_total: counter
```

---

## Error Handling

### Primary File

| File | Purpose | Key Enum |
|------|---------|----------|
| `src/core/errors_extended.py` | Extended error codes | `ErrorCode` enum |

### Key Error Codes
```
DIMENSION_MISMATCH           # Feature vector size mismatch
MODEL_NOT_FOUND              # Classification model file missing
MODEL_LOAD_ERROR             # Model deserialization failed
FEATURE_VERSION_MISMATCH     # Vector version conflict
ANALYSIS_FAILED              # End-to-end analysis error
```

---

## Dependencies Roadmap

### Current
- fastapi, pydantic, prometheus-client
- faiss-cpu (or faiss-gpu)
- redis, sqlalchemy
- torch, transformers (for metric learning)
- paddle-ocr, pillow

### To Add (Phase 1)
- torchvision (pre-trained models)

### To Add (Phase 2)
- openai (or anthropic)
- litellm (multi-provider abstraction)

### To Add (Phase 4)
- psycopg2 (PostgreSQL)
- alembic (DB migrations)

---

## Quick Navigation by Role

### For Feature Engineers
1. Start: `src/core/feature_extractor.py` (644 lines)
2. See: `src/core/invariant_features.py` (v5/v6 implementation)
3. Test: `tests/unit/test_v*` files
4. New: Create v7 branch in feature_extractor

### For ML Engineers
1. Start: `src/ml/classifier.py` (649 lines)
2. See: `src/core/knowledge/` (8 modules)
3. New: Add LLM reasoner integration

### For Search/Retrieval Engineers
1. Start: `src/core/similarity.py` (1188 lines)
2. Key sections: Lines 434-549 (InMemoryVectorStore), 645-883 (FaissVectorStore)
3. New: Implement HybridSearchEngine

### For DevOps/Operations
1. Start: `src/main.py` (lifespan management)
2. Metrics: `src/utils/analysis_metrics.py`
3. Config: `src/core/config.py`

---

## Integration Checklist for Each Phase

### Phase 1: Visual CNN (v7)
- [ ] Understand: `feature_extractor.py` (lines 1-100, 360-495)
- [ ] Review: Version schema (SLOTS_V6, upgrade_vector)
- [ ] Implement: `visual_features.py` (VisualFeatureExtractor)
- [ ] Integrate: v7 branch in extract()
- [ ] Test: Feature dimension, CNN inference, upgrade paths
- [ ] Deploy: Shadow mode, fallback to v6

### Phase 2: LLM Reasoning
- [ ] Understand: `analyzer.py` (classify_part method)
- [ ] Review: `knowledge/enhanced_classifier.py` (fusion logic)
- [ ] Implement: `llm_reasoner.py` (LLMReasoner class)
- [ ] Integrate: `/v1/analyze/reason` endpoint
- [ ] Test: Prompt generation, JSON parsing, memory buffer
- [ ] Deploy: Beta with feature flag

### Phase 3: Hybrid Search
- [ ] Understand: `similarity.py` (lines 434-549, 789-833)
- [ ] Review: RRF fusion algorithms
- [ ] Implement: `hybrid_search.py` (HybridSearchEngine)
- [ ] Integrate: `/v1/vectors/hybrid-search` endpoint
- [ ] Test: Fusion correctness, filtering, ranking
- [ ] Deploy: Gradual weight rollout

### Phase 4: Active Learning
- [ ] Understand: Feedback loop requirements
- [ ] Review: Model reload security (`classifier.py` lines 141-609)
- [ ] Implement: `active_learning.py` + PostgreSQL schema
- [ ] Integrate: `/v1/feedback/*` endpoints
- [ ] Test: Feedback storage, uncertainty sampling, retraining
- [ ] Deploy: Gradual with validation gates

