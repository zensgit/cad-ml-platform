# CAD ML Platform - Comprehensive Improvement Plan

**Document Version**: v1.0  
**Date**: 2025-11-30  
**Platform**: CAD ML Feature Extraction & Classification System  

---

## Executive Summary

The CAD ML platform is a sophisticated production system with multi-layered architecture spanning feature extraction, vector similarity search, knowledge-based classification, and OCR analysis. This document provides a complete architectural analysis to guide integration of four major enhancements: Visual CNN Features (v7), LLM Reasoning Engine, Hybrid Search System, and Active Learning Loop.

**Current System Status**:
- Feature versions: v1 (basic) → v6 (moment invariants with metric learning)
- Vector storage: Memory + Faiss ANN + Redis persistence
- Classification: Rule-based + Knowledge base (8 specialized modules)
- OCR: Paddle + DeepSeek providers with fallback chains
- API: RESTful with async support, production metrics & resilience

---

## 1. CURRENT ARCHITECTURE ANALYSIS

### 1.1 Feature Extraction Pipeline

**File**: `src/core/feature_extractor.py` (644 lines)

**Current Implementation**:
```
Feature Versions Progression:
├── v1: Basic (7 dims)     → entity_count, bbox metrics, layer_count, complexity_flag
├── v2: Normalized (12 dims) → normalized dimensions, aspect ratios
├── v3: Entity stats (22 dims) → solids, facets, entity distribution frequencies
├── v4: Real features (24 dims) → surface_count, shape_entropy
├── v5: Invariants (26 dims) → rotation/scale invariant geometric + topological
└── v6: Moment invariants (32 dims) → v5 + inertia tensor eigenvalues + distribution stats
```

**Key Classes & Methods**:
- `FeatureExtractor.extract()` - Async feature computation with multi-version branching
- `FeatureExtractor.upgrade_vector()` - Version migration with deterministic padding
- `FeatureExtractor.rehydrate()` - Reconstruct geometric/semantic components
- `compute_shape_entropy()` - Shannon entropy with Laplace smoothing
- `compute_surface_count()` - Multi-strategy surface detection

**Current Capabilities**:
- ✅ Metric Learning Integration: Optional MetricEmbedder wrapping (v5/v6 geometric vectors)
- ✅ Feature Versioning: Strict validation with mismatch tracking
- ✅ Observable: Latency, L2 norm, vector dimension metrics
- ✅ Resilient: Graceful fallback to raw features if embedding fails

**Integration Points**:
1. **CadDocument Input**: Entity kinds, bounding boxes, layers, complexity scoring
2. **MetricEmbedder**: Optional 24→64 embedding transformation
3. **Vector Store**: Register with feature version metadata
4. **Classification**: Feed geometric+semantic components to knowledge base

**Limitations for Improvement**:
- ❌ No visual/image analysis (images must be provided separately)
- ❌ No multimodal fusion (geometry-only)
- ❌ Limited semantic reasoning (heuristic shape_entropy only)
- ❌ No active learning signal (no feedback loop for hard examples)

---

### 1.2 Vector Similarity Search

**File**: `src/core/similarity.py` (1188 lines)

**Architecture**:
```
┌─ Memory Store ──────────────────┐
│  _VECTOR_STORE: Dict[str, Vec]  │  ← Fast in-process access
│  _VECTOR_META: Dict[str, Meta]  │  ← Feature version tracking
└─────────────────────────────────┘
           ↕ (dual-backend)
┌─ Faiss Index (ANN) ─────────────┐
│  IndexFlatIP (normalized)       │  ← GPU-accelerated similarity
│  _FAISS_ID_MAP, _REVERSE_MAP    │  ← ID mapping
│  Auto-rebuild on pending deletes│  ← Garbage collection
└─────────────────────────────────┘
           ↕ (optional)
┌─ Redis Persistence ─────────────┐
│  vector:{doc_id} hash           │  ← Distributed storage
│  Recovery state (backoff)       │  ← Fault tolerance
└─────────────────────────────────┘
```

**Key Classes & Functions**:
- `InMemoryVectorStore` - Protocol implementation with TTL pruning
- `FaissVectorStore` - IndexFlatIP with normalization, rebuild backoff
- `register_vector()` - Dimension enforcement, feature version validation
- `compute_similarity()` - Cosine distance single-pair calculation
- Query methods with material/complexity filtering

**Advanced Features**:
- ✅ Degradation Detection: Automatic fallback from Faiss → Memory
- ✅ Recovery Loop: Exponential backoff with flapping suppression
- ✅ Material Drift Tracking: Histogram of registered materials
- ✅ Version Mismatch Ring Buffer: Recent 50 events stored
- ✅ TTL-based Pruning: Configurable vector expiration
- ✅ Cold Access Detection: Idle vector removal

**Limitations for Improvement**:
- ❌ Cosine-only similarity (no learned distance metrics)
- ❌ Single-dense-vector search (no hierarchical/product quantization)
- ❌ No filtering at vector retrieval level (post-hoc only)
- ❌ Manual rebuild trigger (no online learning feedback)

---

### 1.3 Knowledge Base & Classification

**Directory**: `src/core/knowledge/` (7 specialized modules)

**Architecture**:
```
MechanicalPartKnowledgeBase
├── Geometric Pattern Rules (shape, compactness, sphericity)
├── Entity Distribution Heuristics
├── Layer/complexity scoring

MaterialKnowledgeBase
├── Material-property mappings
├── Corrosion, machinability profiles

PrecisionKnowledgeBase
├── Tolerance class recognition
├── GD&T symbol interpretation

StandardsKnowledgeBase
├── ISO, ANSI standard patterns
├── Thread specs, surface finish classes

FunctionalKnowledgeBase
├── Feature role classification (bearing, seal, fastener)
├── Assembly constraint detection

AssemblyKnowledgeBase
├── Fit type classification (clearance, interference)
├── Part-to-part relationships

ManufacturingKnowledgeBase
├── Machining operation suggestion
├── Heat treatment recommendations

EnhancedPartClassifier
└── Multi-module fusion with confidence scoring
```

**Key Methods**:
- `MechanicalPartKnowledgeBase.classify()` - Pattern matching on geometric features
- `EnhancedPartClassifier.classify()` - Orchestrates all 6 knowledge modules
- Pattern rules: Custom `PatternRule` objects with scoring logic

**Current Integration**:
- Input: `geometric_features`, `entity_counts`, `ocr_data` dicts
- Output: `ClassificationResult` with `part_type`, `confidence`, `characteristics`
- API Endpoint: `/v1/analyze/classify` (optional via `enhanced_classification` flag)

**Limitations for Improvement**:
- ❌ No LLM-backed reasoning (rules are hand-crafted)
- ❌ Static rule definitions (YAML or hardcoded)
- ❌ No uncertainty quantification beyond single confidence score
- ❌ No dynamic knowledge updates (requires code deploy)

---

### 1.4 OCR & Vision

**OCR**: `src/core/ocr/` (multi-provider architecture)
- Paddle OCR: Local, fast, lower quality
- DeepSeek HF: API-based, higher quality, slower
- Fallback chain: JSON → Markdown → Regex

**Dimension Parser**: `src/core/ocr/parsing/dimension_parser.py`
- Regex patterns for: Diameter, Radius, Thread, Roughness, Dual tolerance
- GD&T symbol detection: Perpendicular, Parallel, Flatness, Concentricity, etc.
- Heuristic unit normalization (mm, cm, μm)

**Vision**: `src/core/vision/` (stub implementation)
- Provider enum: OpenAI, Anthropic, Azure, Google, Local
- Client initialization framework (most unimplemented)
- `VisionAnalyzeRequest/Response` models ready

**Current Gaps**:
- ❌ Vision providers mostly stubs
- ❌ No CNN feature extraction from CAD renders
- ❌ No image preprocessing/enhancement pipeline
- ❌ Vision results not integrated into classification

---

### 1.5 API Structure

**Router Organization**:
```
/v1/analyze
├── /extract-features (POST) → feature vectors
├── /classify (POST) → part classification
├── /check-quality (POST) → design quality assessment
├── /recommend-process (POST) → manufacturing recommendation
└── /batch-similarity (POST) → multi-reference matching

/v1/vectors
├── /vectors (GET) → list all vectors
└── /delete (POST) → remove vector

/v1/vision
├── /analyze (POST) → image analysis (stub)

/v1/ocr
├── /extract-dimensions (POST) → OCR parsing

/v1/knowledge
├── /classify (POST) → enhanced classification

/v1/model
├── /reload (POST) → hot reload classifier
├── /info (GET) → model metadata
└── /opcode-audit (GET) → security audit
```

**Middleware & Infrastructure**:
- ✅ CORS, TrustedHost
- ✅ API key authentication via dependencies
- ✅ Prometheus metrics export
- ✅ Structured error responses with ErrorCode enums
- ✅ Async/await throughout

---

## 2. INTEGRATION POINTS & LIMITATIONS

### 2.1 Feature Extraction Gaps

| Limitation | Why It Blocks v7 | Solution Preview |
|-----------|-----------------|-----------------|
| Geometry-only features | No visual CNN extraction | Add `VisualFeatureExtractor` consuming renders |
| Manual version management | Hard to add CNN dims | Extend version schema with CNN metadata |
| No learned metrics | Cosine is fixed | Gate metric learning via embedder backend |
| Single-modality fusion | Can't combine image + geometry | Multi-modal fusion layer in classifier |

### 2.2 Classification Gaps

| Limitation | Why It Blocks LLM | Solution Preview |
|-----------|-----------------|-----------------|
| Rule-based only | No reasoning chain | LLM prompt engineering with in-context examples |
| Static knowledge | Can't adapt to feedback | Dynamic knowledge store + hot update API |
| Confidence heuristic | No uncertainty bounds | Ensemble methods + confidence calibration |
| No context chaining | Single-shot inference | Memory buffer for part family history |

### 2.3 Vector Search Gaps

| Limitation | Why It Blocks Hybrid | Solution Preview |
|-----------|-------------------|-----------------|
| Single similarity metric | Can't combine visual+geometric | Weighted ensemble of similarity scores |
| Post-hoc filtering | Can't narrow search scope | Metadata-aware index partitioning |
| No re-ranking | Top-k is fixed quality | LLM-based semantic re-ranking of results |
| Passive indexing | No active learning signal | Feedback-driven importance weighting |

### 2.4 Data Pipeline Gaps

| Limitation | Why It Blocks Active Learning | Solution Preview |
|-----------|------------------------------|-----------------|
| One-shot analysis | No uncertainty sampling | Add confidence threshold triggering re-analysis |
| No feedback capture | Can't track predictions | Create feedback endpoint + historical store |
| Static training | Models never evolve | Stream learning examples to retraining pipeline |
| No user loop | Pure ML system | Add human-in-the-loop review workflow |

---

## 3. DETAILED IMPROVEMENT ROADMAP

### 3.1 Phase 1: Visual CNN Features (v7)

**Objective**: Add rotation/scale-invariant visual features from CAD renders

**Architecture Design**:
```python
# New class in src/core/visual_features.py
class VisualFeatureExtractor:
    def __init__(self, model_name="resnet50-pretrained"):
        self.model = load_pretrained_cnn(model_name)
        self.transforms = get_standard_transforms()
    
    async def extract_from_render(self, cad_document, render_image_bytes):
        # Normalize render (view angle, lighting)
        normalized = self._normalize_render(render_image_bytes)
        # Forward pass → pooled activations
        visual_features = self.model(normalized)
        # Extract invariant features
        return {
            "edge_density": compute_edge_density(visual_features),
            "shape_complexity_visual": compute_visual_entropy(visual_features),
            "symmetry_score": compute_symmetry(visual_features),
            "texture_uniformity": compute_texture_stats(visual_features),
            "size_distribution": compute_size_distribution(visual_features),
            "raw_embedding": visual_features.detach().cpu(),  # 2048-dim for v7
        }
```

**Integration Strategy**:
1. **Add to FeatureExtractor.extract()**:
   - Accept optional `render_bytes` parameter
   - Branch: if render available → extract_visual, else → return None
   - Concatenate to geometric vector (v7 = v6_32 + visual_10 = 42 dims)

2. **Update Version Schema**:
   ```python
   SLOTS_V7 = SLOTS_V6 + [
       ("edge_density", "visual"),
       ("shape_complexity_visual", "visual"),
       ("symmetry_score", "visual"),
       ("texture_uniformity", "visual"),
       ("size_distribution", "visual"),
       ("cnn_embedding_norm", "visual"),
       ("corner_sharpness", "visual"),
       ("line_continuity", "visual"),
       ("surface_roughness_visual", "visual"),
       ("feature_density", "visual"),
   ]
   ```

3. **Vector Dimension Evolution**:
   - Feature registry tracks "visual_available" boolean per vector
   - Upgrade path: v6 (32) → v7_geometric_only (42 with visual_nans) or v7_visual (42 with values)
   - Query routing: visual filters apply only to v7_visual vectors

**Expected Improvements**:
- ✅ 15-25% accuracy boost on shape classification (visual confirms/contradicts geometric)
- ✅ Better discrimination of symmetric parts (symmetry_score)
- ✅ Rotation invariance from CNN pooling

**Dependencies**:
- PyTorch or TensorFlow for CNN inference
- PIL for render preprocessing
- New config: `VISUAL_FEATURE_MODEL_PATH`, `VISUAL_RENDER_TIMEOUT`

---

### 3.2 Phase 2: LLM Reasoning Engine

**Objective**: Add semantic reasoning layer for part classification

**Architecture Design**:
```python
# New class in src/core/llm_reasoner.py
class LLMReasoner:
    def __init__(self, model_name="gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=...)
        self.memory_buffer = RollingMemory(max_parts=10)  # Part classification history
    
    async def reason_classification(
        self,
        geometric_features: dict,
        visual_features: dict | None,
        ocr_data: dict,
        knowledge_results: dict,  # From EnhancedPartClassifier
        similar_parts: list[tuple[str, float]],  # From vector search
    ) -> ReasoningResult:
        # Build prompt with few-shot examples
        prompt = self._build_classification_prompt(
            geometric_features,
            visual_features,
            ocr_data,
            knowledge_results,
            similar_parts,
            self.memory_buffer.get_recent_examples(),
        )
        
        # Call LLM with structured output
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert CAD/mechanical design analyst..."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Deterministic
        )
        
        reasoning = json.loads(response.choices[0].message.content)
        self.memory_buffer.add(reasoning)
        return reasoning
```

**Integration Strategy**:
1. **New API Endpoint**: `/v1/analyze/reason`
   - Input: analysis_id (vector already registered)
   - Output: `ReasoningResult` with logic chain + alternative hypotheses

2. **Prompt Engineering**:
   ```
   System: Expert CAD analyst role definition
   Context: 
     - {geometric_features_summary}
     - {visual_analysis} (if available)
     - {ocr_dimensions_and_symbols}
     - {knowledge_base_scores}
     - {similar_parts_from_search}
     - {recent_classifications_from_memory}
   
   Task: Classify this part and explain reasoning
   Output JSON: {
     "primary_type": "...",
     "confidence": 0.9,
     "reasoning_chain": ["observation 1", "inference 1", ...],
     "alternative_types": [{"type": "...", "likelihood": 0.1}],
     "uncertain_aspects": ["..."],
     "recommended_manual_review": bool
   }
   ```

3. **Memory Management**:
   - Sliding window of last 10 parts in same product family
   - Extract: part_type, key_features, confidence
   - Include in few-shot context

**Expected Improvements**:
- ✅ Better handling of ambiguous cases (via uncertainty expression)
- ✅ Contextual classification (parts in assembly have related features)
- ✅ Explainable predictions (reasoning chain)

**Dependencies**:
- OpenAI API or compatible (Claude, Llama 2 via Replicate)
- Cost: ~$0.01-0.05 per classification (input+output tokens)
- Config: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_TEMPERATURE`

---

### 3.3 Phase 3: Hybrid Search System

**Objective**: Multi-modal similarity search combining visual + geometric + semantic

**Architecture Design**:
```python
# New class in src/core/hybrid_search.py
class HybridSearchEngine:
    def __init__(self, geometric_store, visual_store, semantic_ranker):
        self.geometric = geometric_store  # InMemoryVectorStore (existing)
        self.visual = visual_store  # FaissVectorStore for CNN embeddings
        self.semantic_ranker = semantic_ranker  # LLM-based re-ranker
    
    async def search(
        self,
        query_geometric: list[float],
        query_visual: list[float] | None,
        query_semantic_context: str | None,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[tuple[str, float, dict]]:  # (doc_id, score, metadata)
        
        # Stage 1: Parallel retrieval
        geometric_results = await self.geometric.query_async(
            query_geometric, top_k=top_k*2, **filters
        )
        visual_results = []
        if query_visual:
            visual_results = await self.visual.query_async(
                query_visual, top_k=top_k*2
            )
        
        # Stage 2: Fusion (Reciprocal Rank Fusion)
        fused_scores = self._rrf_fusion(
            geometric_results,
            visual_results,
            weights={"geometric": 0.6, "visual": 0.4},
        )
        
        # Stage 3: Semantic re-ranking (optional)
        if query_semantic_context:
            final_scores = await self.semantic_ranker.rerank(
                fused_scores, query_semantic_context, top_k=top_k
            )
        else:
            final_scores = fused_scores[:top_k]
        
        return final_scores
    
    def _rrf_fusion(self, geom_results, vis_results, weights):
        """Reciprocal Rank Fusion combining multiple ranked lists"""
        combined = {}
        
        for rank, (doc_id, score) in enumerate(geom_results, 1):
            combined.setdefault(doc_id, 0)
            combined[doc_id] += weights["geometric"] * (1 / (rank + 60))  # RRF constant
        
        for rank, (doc_id, score) in enumerate(vis_results, 1):
            combined.setdefault(doc_id, 0)
            combined[doc_id] += weights["visual"] * (1 / (rank + 60))
        
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

**Integration Strategy**:
1. **Dual Index Management**:
   - Geometric: existing `InMemoryVectorStore` (v1-v6)
   - Visual: new `FaissVectorStore` for CNN embeddings (v7 only)
   - Sync on register: if v7 available, add to both indices

2. **New API Endpoint**: `/v1/vectors/hybrid-search`
   - Input: `query_id` (existing vector to find similar)
   - Input: `include_visual` (bool), `fusion_weights` (dict)
   - Output: Ranked results with fusion scores

3. **Metadata-Aware Filtering**:
   - Pre-filter by material, complexity before scoring
   - Filter by "has_visual_features" for visual-only results

**Expected Improvements**:
- ✅ 20-30% better recall for related parts (visual catches what geometry misses)
- ✅ Robust to single-modality noise
- ✅ Flexible weighting for different use cases

**Implementation Phases**:
- Phase 3a: Basic RRF fusion (geometric + visual scores)
- Phase 3b: LLM-based semantic re-ranking
- Phase 3c: Learned fusion weights (from user feedback)

---

### 3.4 Phase 4: Active Learning Loop

**Objective**: Capture user feedback to improve future predictions

**Architecture Design**:
```python
# New class in src/core/active_learning.py
class ActiveLearner:
    def __init__(self, feedback_store, uncertainty_sampler):
        self.feedback_store = feedback_store  # Redis + PostgreSQL
        self.uncertainty_sampler = uncertainty_sampler
        self.retraining_queue = PriorityQueue()  # High-uncertainty examples first
    
    async def capture_feedback(
        self,
        analysis_id: str,
        predicted_type: str,
        actual_type: str | None,
        user_confidence: float,
        notes: str | None,
    ) -> FeedbackResult:
        """Record user correction/validation"""
        feedback = {
            "analysis_id": analysis_id,
            "predicted": predicted_type,
            "actual": actual_type,
            "user_confidence": user_confidence,
            "timestamp": time.time(),
            "notes": notes,
            "was_correct": predicted_type == actual_type,
        }
        
        # Store feedback
        await self.feedback_store.save(feedback)
        
        # If prediction was wrong, add to retraining queue
        if not feedback["was_correct"]:
            # Compute priority by prediction confidence
            priority = -confidence  # Higher confidence errors get higher priority
            self.retraining_queue.put((priority, feedback))
        
        # Update model performance metrics
        await self._update_metrics(feedback)
        
        return FeedbackResult(
            feedback_id=generate_id(),
            status="recorded",
            retraining_triggered=not feedback["was_correct"],
        )
    
    async def get_uncertain_examples(self, top_k: int = 10):
        """Identify predictions likely to be wrong for expert review"""
        # Query recent analyses with:
        # - Low confidence scores
        # - Conflicting knowledge base signals
        # - Single-modality agreement (e.g., only geometry, no visual)
        uncertain = await self.uncertainty_sampler.sample(top_k)
        return uncertain
    
    async def prepare_retraining_batch(self, min_samples: int = 50):
        """Prepare corrected examples for model retraining"""
        corrected = await self.feedback_store.get_corrections(limit=min_samples)
        
        if len(corrected) < min_samples:
            return None  # Not enough data yet
        
        batch = {
            "geometric_vectors": [...],
            "visual_vectors": [...],
            "ocr_data": [...],
            "labels": [...],
            "confidences": [...],
            "feedback_count": len(corrected),
        }
        
        # Export to training pipeline
        return batch
```

**Integration Strategy**:
1. **New API Endpoint**: `/v1/feedback/submit`
   - Input: `analysis_id`, `actual_type`, `user_confidence`, `notes`
   - Output: `FeedbackResult` with retraining status

2. **Uncertainty Sampling**:
   ```python
   def score_uncertainty(prediction_confidence, knowledge_agreement, modality_coverage):
       # Multi-factor uncertainty
       # - Low confidence directly
       # - Knowledge modules disagree on type
       # - Missing visual or OCR data
       return (1 - prediction_confidence) * 0.6 \
            + (1 - knowledge_agreement) * 0.3 \
            + (1 - modality_coverage) * 0.1
   ```

3. **Feedback Storage**:
   - Redis: Recent feedback (last 1000 examples, TTL 30 days)
   - PostgreSQL: Historical corrections (indexed by part type)
   - Metrics: accuracy/precision by part type, feedback rate

4. **Retraining Pipeline**:
   - Trigger: Every 100 new corrections or daily batch
   - Input: Corrected examples from feedback store
   - Output: Updated knowledge rules or fine-tuned embeddings
   - Deploy: Hot-reload via `/v1/model/reload`

**Expected Improvements**:
- ✅ 10-15% monthly accuracy improvement (from user feedback)
- ✅ Automated uncertainty detection (cases needing review)
- ✅ Closed-loop system (predictions → feedback → improvement)

**Dependencies**:
- PostgreSQL for historical storage
- Message queue (Redis streams) for retraining pipeline
- Config: `ACTIVE_LEARNING_ENABLED`, `FEEDBACK_TTL_DAYS`, `RETRAINING_MIN_SAMPLES`

---

## 4. IMPLEMENTATION CHECKLIST

### 4.1 Phase 1: Visual CNN (v7)

- [ ] **src/core/visual_features.py** (NEW)
  - [ ] `VisualFeatureExtractor` class
  - [ ] Render normalization pipeline
  - [ ] CNN model loading (ResNet50 pretrained)
  - [ ] Feature pooling & extraction

- [ ] **src/core/feature_extractor.py** (MODIFY)
  - [ ] Add `v7` branch in `extract()`
  - [ ] Accept optional `render_bytes` parameter
  - [ ] Extend `SLOTS_V7` constant (10 new visual dims)
  - [ ] Update `upgrade_vector()` for v7 migrations
  - [ ] Add version 7 tests

- [ ] **src/core/similarity.py** (MODIFY)
  - [ ] Track "visual_available" in vector metadata
  - [ ] Filter by visual presence in query
  - [ ] Metrics: `feature_v7_visual_present_ratio`

- [ ] **Tests**: `tests/unit/test_v7_visual_features.py`
  - [ ] Render preprocessing
  - [ ] Feature extraction consistency
  - [ ] Invariance to rotation/scale
  - [ ] Upgrade path from v6

### 4.2 Phase 2: LLM Reasoning

- [ ] **src/core/llm_reasoner.py** (NEW)
  - [ ] `LLMReasoner` class with multi-provider support
  - [ ] Prompt template engineering
  - [ ] Rolling memory buffer for context
  - [ ] Structured output parsing

- [ ] **src/api/v1/reasoning.py** (NEW)
  - [ ] `/v1/analyze/reason` endpoint
  - [ ] `ReasoningRequest`, `ReasoningResponse` models
  - [ ] Error handling for LLM failures

- [ ] **src/core/config.py** (MODIFY)
  - [ ] Add LLM settings (provider, model, API key)
  - [ ] Temperature, max_tokens, timeout

- [ ] **Tests**: `tests/unit/test_llm_reasoner.py`
  - [ ] Prompt generation
  - [ ] Output parsing (with invalid JSON handling)
  - [ ] Memory buffer rotation
  - [ ] Fallback to rule-based if LLM fails

### 4.3 Phase 3: Hybrid Search

- [ ] **src/core/hybrid_search.py** (NEW)
  - [ ] `HybridSearchEngine` class
  - [ ] RRF fusion implementation
  - [ ] Semantic re-ranking integration
  - [ ] Configurable weights

- [ ] **src/api/v1/vectors.py** (MODIFY)
  - [ ] Add `/v1/vectors/hybrid-search` endpoint
  - [ ] `HybridSearchRequest`, `HybridSearchResponse` models

- [ ] **src/core/similarity.py** (MODIFY)
  - [ ] Create separate visual vector store
  - [ ] Sync register operations (geometric + visual)
  - [ ] Dual-index query support

- [ ] **Tests**: `tests/unit/test_hybrid_search.py`
  - [ ] RRF scoring correctness
  - [ ] Visual + geometric fusion
  - [ ] Filtering during hybrid search
  - [ ] Ranking consistency

### 4.4 Phase 4: Active Learning

- [ ] **src/core/active_learning.py** (NEW)
  - [ ] `ActiveLearner` class
  - [ ] Uncertainty sampling
  - [ ] Feedback aggregation
  - [ ] Retraining batch preparation

- [ ] **src/api/v1/feedback.py** (NEW)
  - [ ] `/v1/feedback/submit` endpoint
  - [ ] `/v1/feedback/uncertain` endpoint
  - [ ] `FeedbackRequest`, `UncertaintyResponse` models

- [ ] **Database Schema** (PostgreSQL)
  - [ ] `feedback` table (corrections history)
  - [ ] `uncertainty_samples` table (flagged predictions)
  - [ ] Indexes on part_type, timestamp

- [ ] **Retraining Pipeline** (NEW)
  - [ ] `scripts/retrain_from_feedback.py`
  - [ ] Load corrected examples
  - [ ] Fine-tune embedder or update rules
  - [ ] Validation split evaluation
  - [ ] Hot-reload via `/v1/model/reload`

- [ ] **Tests**: `tests/unit/test_active_learning.py`
  - [ ] Feedback capture & storage
  - [ ] Uncertainty sampling
  - [ ] Retraining batch format
  - [ ] Metrics tracking

---

## 5. INTEGRATION SEQUENCE

**Week 1: Foundational**
- Day 1: Vision feature extraction (v7 structure)
- Day 2: Feature extractor integration
- Day 3: Tests & validation
- Day 4: Deploy v7 → production (shadow mode)

**Week 2: Intelligence**
- Day 1: LLM reasoner implementation
- Day 2: API endpoint & prompt tuning
- Day 3: Tests & cost optimization
- Day 4: Deploy reasoning → beta

**Week 3: Search**
- Day 1: Hybrid search engine
- Day 2: Dual-index synchronization
- Day 3: Tests & performance tuning
- Day 4: Deploy hybrid search

**Week 4: Feedback Loop**
- Day 1: Active learning framework
- Day 2: Feedback API & storage
- Day 3: Retraining pipeline
- Day 4: End-to-end testing & deployment

---

## 6. KEY DESIGN DECISIONS

| Decision | Rationale | Alternative |
|----------|-----------|-------------|
| **CNN as feature extractor (v7)** | Pre-trained models avoid retraining | Train from scratch (costly) |
| **LLM via API** | Avoid model hosting | Self-hosted LLM (infrastructure) |
| **RRF for fusion** | Rank-based, no threshold tuning | Learned weighting (requires supervision) |
| **Async/await everywhere** | Non-blocking for long ops | Sync + thread pool (simpler but slower) |
| **PostgreSQL for feedback** | Transactional, queryable | File-based (hard to analyze) |
| **Hot-reload for updates** | No downtime | Full restart (breaks continuity) |

---

## 7. METRICS & MONITORING

### 7.1 Feature Quality

```
feature_v7_visual_present_ratio: gauge
  - Tracks % of v7 vectors with visual features
  
feature_extraction_latency_seconds{version="v7"}
  - Monitors CNN inference time
  
feature_vector_l2_norm{version="v7"}
  - Detects scaling issues
```

### 7.2 LLM Reasoning

```
llm_inference_seconds: histogram
  - Token efficiency tracking
  
llm_parsing_errors_total{reason}
  - Malformed JSON, API errors
  
llm_cost_usd_total: counter
  - Budget tracking
```

### 7.3 Hybrid Search

```
hybrid_search_latency_seconds: histogram
  - Geo + visual query time
  
fusion_score_distribution: histogram
  - Monitor RRF weight balance
  
hybrid_vs_baseline_ndcg: gauge
  - Search quality improvement
```

### 7.4 Active Learning

```
feedback_submission_rate: gauge
  - User engagement
  
prediction_correction_rate_by_type: gauge
  - Which types need improvement
  
retraining_trigger_events_total: counter
  - Auto-retrain frequency
```

---

## 8. RISK MITIGATION

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CNN rendering failures | v7 unavailable | Fallback to v6, graceful degradation |
| LLM API downtime | Reasoning fails | Circuit breaker, cached responses |
| Fusion weight imbalance | Poor recall | A/B testing, gradual rollout |
| Feedback data quality | Bad retraining | Validation rules, expert review gate |
| Cost explosion (LLM) | Budget overrun | Token limits, caching, sampling |

---

## 9. CONCLUSION

The CAD ML platform has a solid foundation with production-grade reliability, multi-version feature management, and modular knowledge bases. The proposed improvements (v7 Visual Features, LLM Reasoning, Hybrid Search, Active Learning) build naturally on this architecture without breaking changes.

**Expected Outcomes**:
- **Accuracy**: 25-35% improvement via multi-modal fusion
- **Explainability**: Reasoning chains for auditable decisions
- **Adaptability**: Automatic improvement via user feedback
- **Robustness**: Graceful degradation if any component fails

**Next Steps**:
1. Prioritize Phase 1 (v7) for quick wins
2. Prototype LLM integration in parallel
3. Validate on golden test set before production
4. Monitor metrics closely during rollout

