# CAD ML Platform - Improvement Plan Summary

**Quick Reference Guide for Integration Planning**

---

## Current Architecture Strengths

### Feature Extraction (v1-v6)
- Multi-version feature progression with strict versioning
- Optional metric learning integration (24→64 embeddings)
- Geometric + semantic components (rotation/scale invariant in v5-v6)
- Production-grade observability (latency, L2 norm, dimension metrics)

### Vector Similarity Search
- Memory + Faiss (ANN) + Redis dual-backend architecture
- Automatic degradation detection with recovery loop
- TTL pruning, material drift tracking, version mismatch buffering
- Metadata-aware filtering (material, complexity)

### Knowledge-Based Classification
- 8 specialized modules: Material, Precision, Standards, Functional, Assembly, Manufacturing
- EnhancedPartClassifier with multi-module fusion
- Pattern-matching rules on geometric + OCR features

### OCR & Vision Pipeline
- Multi-provider OCR (Paddle + DeepSeek) with fallback chains
- Regex-based dimension parser (Diameter, Thread, Roughness, GD&T)
- Vision module framework (stubs for OpenAI, Anthropic, Azure, Google, Local)

### API & Infrastructure
- RESTful async/await throughout
- Structured error handling with ErrorCode enums
- Prometheus metrics, CORS, authentication
- Hot-reload support for models

---

## Planned Improvements

### Phase 1: Visual CNN Features (v7) ← START HERE
- **Goal**: Extract 10 visual features from CAD renders (edge density, symmetry, texture)
- **Integration**: VisualFeatureExtractor → feature_extractor.py v7 branch
- **Dimensions**: v6 (32) + visual (10) = 42 total
- **Expected Impact**: 15-25% accuracy improvement on shape classification
- **Timeline**: Week 1 (4 days)
- **Files Modified**: 
  - NEW: `src/core/visual_features.py`
  - MODIFY: `src/core/feature_extractor.py` (add v7 branch)
  - MODIFY: `src/core/similarity.py` (visual_available metadata)

### Phase 2: LLM Reasoning Engine
- **Goal**: Add semantic reasoning layer for explainable classification
- **Integration**: LLMReasoner with rolling memory buffer (10-part history)
- **Features**: Reasoning chain, alternative hypotheses, uncertainty bounds
- **Expected Impact**: Better handling of ambiguous cases, contextual classification
- **Timeline**: Week 2 (4 days)
- **Files Modified**:
  - NEW: `src/core/llm_reasoner.py`
  - NEW: `src/api/v1/reasoning.py`
  - MODIFY: `src/core/config.py` (LLM settings)

### Phase 3: Hybrid Search System
- **Goal**: Multi-modal similarity (visual + geometric + semantic)
- **Integration**: HybridSearchEngine with Reciprocal Rank Fusion
- **Features**: Weighted ensemble scoring, semantic re-ranking, configurable fusion
- **Expected Impact**: 20-30% better recall for related parts
- **Timeline**: Week 3 (4 days)
- **Files Modified**:
  - NEW: `src/core/hybrid_search.py`
  - MODIFY: `src/api/v1/vectors.py` (hybrid-search endpoint)
  - MODIFY: `src/core/similarity.py` (dual visual index)

### Phase 4: Active Learning Loop
- **Goal**: Capture user feedback to continuously improve predictions
- **Integration**: ActiveLearner with feedback storage (Redis + PostgreSQL)
- **Features**: Uncertainty sampling, retraining batch prep, hot-reload pipeline
- **Expected Impact**: 10-15% monthly accuracy improvement
- **Timeline**: Week 4 (4 days)
- **Files Modified**:
  - NEW: `src/core/active_learning.py`
  - NEW: `src/api/v1/feedback.py`
  - NEW: `scripts/retrain_from_feedback.py`
  - NEW: PostgreSQL schema (feedback, uncertainty_samples tables)

---

## Critical Integration Points

### 1. Feature Extractor
**Current**: `FeatureExtractor.extract(doc: CadDocument) → {geometric, semantic}`
**Enhancement**: Add optional `render_bytes` parameter → also compute visual features

```python
# Current flow
extract(doc) → [entity_count, bbox_metrics, ..., complexity_flag]

# Enhanced flow
extract(doc, render_bytes=None) → {
  "geometric": [32 dims v6 + 10 dims v7 visual],
  "semantic": [complexity, layer_count],
  "visual_available": bool,
  "render_hash": str  # For dedup
}
```

### 2. Vector Storage
**Current**: InMemoryVectorStore (memory) + FaissVectorStore (ANN)
**Enhancement**: Add separate visual index, sync register operations

```python
# Register flow
register_vector(doc_id, geometric_vector, visual_vector, metadata)
  → InMemoryVectorStore.add(doc_id, geometric)
  → FaissVectorStore.add(doc_id, visual)  # if v7
  → Redis.hset(f"vector:{doc_id}", ...)  # if enabled
```

### 3. Classification
**Current**: EnhancedPartClassifier.classify(geometric, entity_counts, ocr)
**Enhancement**: Add LLM reasoner as post-processor

```python
# Classification flow
1. Enhanced knowledge base → {part_type, confidence, signals}
2. LLM reasoner (optional) → {reasoning_chain, alternatives, uncertainty}
3. Merge → final output with explainability
```

### 4. Search
**Current**: Single-vector cosine similarity (geometric only)
**Enhancement**: Hybrid search with fusion

```python
# Search flow
hybrid_search(query_geo, query_visual, context)
  → Parallel: geometric.query(top_k*2), visual.query(top_k*2)
  → RRF fusion with weights (geo:0.6, visual:0.4)
  → LLM re-rank (optional)
  → Return top_k
```

### 5. Feedback Loop
**Current**: One-shot analysis (no feedback)
**Enhancement**: Capture corrections, drive retraining

```python
# Feedback flow
1. User submits correction → ActiveLearner.capture_feedback()
2. Store in Redis (recent) + PostgreSQL (historical)
3. Detect uncertainty signals for expert review
4. Batch retraining when 100+ corrections accumulated
5. Hot-reload models → `/v1/model/reload`
```

---

## Key Files to Understand

### Architecture & Flow
- `src/main.py` - FastAPI app initialization, lifespan management
- `src/api/__init__.py` - Router composition (analyze, vectors, vision, feedback)
- `src/core/analyzer.py` - Basic rule-based classification (replaceable)

### Feature Pipeline
- `src/core/feature_extractor.py` - 644 lines, multi-version branching
- `src/core/feature_manifest.py` - Version registry & expected lengths
- `src/core/invariant_features.py` - v5/v6 computation (shape signature, topological)

### Search & Similarity
- `src/core/similarity.py` - 1188 lines, dual-backend vector store
- `src/core/similarity.py` lines 629-1188 - FaissVectorStore implementation
- `src/utils/cache.py` - Redis client management

### Classification
- `src/core/knowledge/__init__.py` - Module exports
- `src/core/knowledge/part_knowledge.py` - Base patterns
- `src/core/knowledge/enhanced_classifier.py` - Multi-module fusion

### API & Observability
- `src/api/v1/analyze.py` - Main `/v1/analyze/*` endpoints
- `src/api/v1/vectors.py` - Vector management endpoints
- `src/utils/analysis_metrics.py` - Prometheus metrics registration

---

## Dependencies to Add

### Phase 1 (Visual)
```
torch>=2.0.0  # or tensorflow>=2.12.0
torchvision>=0.15.0  # Pre-trained models
pillow>=9.0.0  # Image processing
```

### Phase 2 (LLM)
```
openai>=1.3.0  # or anthropic, litellm
python-json-logger>=2.0.0  # Structured logging
```

### Phase 3 (Hybrid)
```
# No new dependencies (uses existing faiss, numpy)
```

### Phase 4 (Active Learning)
```
psycopg2-binary>=2.9.0  # PostgreSQL driver
sqlalchemy>=2.0.0  # ORM
redis>=4.3.0  # Already installed
```

---

## Configuration Parameters to Add

```
# Phase 1
VISUAL_FEATURE_MODEL_PATH=models/resnet50-pretrained.pth
VISUAL_FEATURE_ENABLED=true
VISUAL_RENDER_TIMEOUT=10  # seconds
FEATURE_VERSION=v7  # or v1-v6 for backward compat

# Phase 2
LLM_PROVIDER=openai  # or anthropic, replicate
LLM_MODEL=gpt-4-turbo-preview
LLM_API_KEY=sk-...
LLM_TEMPERATURE=0.3
LLM_TIMEOUT=30  # seconds
LLM_REASONING_ENABLED=true

# Phase 3
HYBRID_SEARCH_ENABLED=true
FUSION_WEIGHT_GEOMETRIC=0.6
FUSION_WEIGHT_VISUAL=0.4
SEMANTIC_RERANK_ENABLED=false  # Phase 3b

# Phase 4
ACTIVE_LEARNING_ENABLED=true
FEEDBACK_STORE_BACKEND=postgres  # or redis
FEEDBACK_TTL_DAYS=30
RETRAINING_MIN_SAMPLES=100
RETRAINING_SCHEDULE=daily  # or on_demand
```

---

## Success Metrics

### Accuracy
- Baseline (current): ~60-70% top-1 accuracy
- After Phase 1: 75-85% (visual disambiguation)
- After Phase 2: 80-90% (reasoning handles edge cases)
- After Phase 4: 85-95% (feedback loop convergence)

### Latency (p95)
- Feature extraction: <500ms (v7 adds ~100ms for CNN)
- Classification: <1s (LLM reasoning adds ~500ms)
- Hybrid search: <2s (parallel + fusion)
- Feedback submission: <100ms

### Operational
- Vector store size: Currently ~10K vectors → target ~100K with growth
- LLM cost: ~$0.01-0.05 per analysis (budget)
- Feedback volume: Target >100 corrections/week for retraining
- Model improvement rate: 1-2% monthly from feedback

---

## Risk Mitigation Strategies

1. **Visual CNN failures** → Fallback to v6 (detect dimension mismatch)
2. **LLM API downtime** → Circuit breaker + cached responses
3. **Fusion weight imbalance** → A/B test gradual rollout (10% → 50% → 100%)
4. **Feedback noise** → Validation rules + expert review gate before retraining
5. **Cost explosion** → Token budgets, sampling, caching, rate limiting

---

## Next Immediate Steps

1. ✅ Review this comprehensive architecture document
2. ⏳ Create `src/core/visual_features.py` (skeleton)
3. ⏳ Extend `src/core/feature_extractor.py` with v7 branch
4. ⏳ Write test: `tests/unit/test_v7_visual_features.py`
5. ⏳ Deploy to dev environment for validation
6. ⏳ Move to Phase 2 after v7 validation

---

**Document**: `COMPREHENSIVE_IMPROVEMENT_PLAN.md` (873 lines)  
**Last Updated**: 2025-11-30  
**Author**: Codebase Analysis
