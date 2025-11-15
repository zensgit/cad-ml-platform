# Vision Module - Week 1-2 Implementation Plan

**Goal**: Build end-to-end Vision + OCR pipeline with MVP-first approach

**Status**: ✅ Phase 1 Complete (Stub provider + basic structure)

---

## Phase 1: MVP Foundation ✅ COMPLETE

### Completed Tasks (2025-01-15)

- [x] Create Vision module structure (`src/core/vision/`)
- [x] Define Pydantic models (VisionAnalyzeRequest, VisionAnalyzeResponse)
- [x] Implement VisionProvider ABC
- [x] Create DeepSeekStubProvider (fixed responses, simulated latency)
- [x] Implement VisionManager (orchestrates Vision + OCR)
- [x] Create `/api/v1/vision/analyze` endpoint
- [x] Create `/api/v1/vision/health` endpoint
- [x] Write basic tests (8 test cases, 5/8 passing)

**Files Created**:
- `src/core/vision/base.py` (177 lines)
- `src/core/vision/providers/deepseek_stub.py` (108 lines)
- `src/core/vision/manager.py` (175 lines)
- `src/api/v1/vision.py` (136 lines)
- `tests/vision/test_vision_endpoint.py` (276 lines)

**Current Capabilities**:
- ✅ Accepts base64-encoded images
- ✅ Returns fixed vision description (stub)
- ✅ Manager ready for OCR integration (placeholder)
- ✅ FastAPI endpoint with proper error handling
- ✅ Health check endpoint

**Known Limitations**:
- ⚠️ image_url not yet implemented (only image_base64 works)
- ⚠️ OCR integration placeholder (manager.ocr_manager = None)
- ⚠️ 3 tests fail due to sklearn dependency (not blocking)

---

## Phase 2: OCR Integration ✅ COMPLETE (Week 1 Day 2-3)

### Priority 1: Connect Vision + OCR ✅

- [x] **Inject OCRManager into VisionManager** ✅
  - File: `src/api/v1/vision.py:27-59` (updated get_vision_manager())
  - Import OCRManager from src.core.ocr.manager
  - Create OCRManager instance with empty providers (graceful degradation)
  - Pass to VisionManager constructor

- [x] **Enable include_ocr=True flow** ✅
  - File: `src/core/vision/manager.py:142-192` (_extract_ocr method)
  - Implemented OCRManager.extract() call with Pydantic → dict conversion
  - Handle OcrError exceptions gracefully (vision description still returns)
  - Aggregate vision description + OCR results

- [x] **Create integration test** ✅
  - File: `tests/vision/test_vision_ocr_integration.py` (294 lines)
  - Test: vision description + OCR extraction together ✅
  - Test: OCR failure doesn't break vision description ✅
  - Test: include_ocr=False skips OCR correctly ✅
  - Test: ocr_manager=None handles gracefully ✅

**Success Criteria**: ✅
- Vision + OCR returns both description and dimensions ✅
- Request with include_ocr=True triggers OCRManager.extract() ✅
- 4 integration tests passing (exceeded target of 3) ✅

---

### Priority 2: Image URL Support ✅

- [x] **Implement image URL downloading** ✅
  - File: `src/core/vision/manager.py:137-194` (_load_image method)
  - Added httpx AsyncClient with 5s timeout
  - Download image from URL with follow_redirects=True
  - Handle HTTP errors (404, 403, 500+)
  - File size limit check (50MB) via content-length and actual size

- [x] **Add URL validation** ✅
  - Validate URL format (http:// or https://) using urlparse
  - Reject file://, ftp://, and other protocols
  - Empty image (0 bytes) rejection

- [x] **Create tests** ✅
  - File: `tests/vision/test_image_loading.py` (330 lines)
  - Test: valid URL downloads successfully ✅
  - Test: invalid URL scheme returns 400 error ✅
  - Test: timeout returns appropriate error ✅
  - Test: large file rejected ✅
  - Test: HTTP 404 error ✅
  - Test: HTTP 403 error ✅
  - Test: empty image rejected ✅
  - Test: network error handling ✅
  - Test: redirects followed ✅

**Success Criteria**: ✅
- image_url parameter works end-to-end ✅
- Proper error handling for network issues ✅
- 9 new tests passing (exceeded target of 4) ✅

---

## Phase 3: Real DeepSeek-VL Provider (Week 1 Day 4-5)

### Priority 1: DeepSeek-VL Integration

- [ ] **Create DeepSeekVLProvider**
  - File: `src/core/vision/providers/deepseek_vl.py`
  - Inherit from VisionProvider ABC
  - Initialize transformers model (DeepSeek-VL-7B-chat)
  - Implement analyze_image() with actual inference
  - Handle model loading errors

- [ ] **Add model caching**
  - Cache model weights (avoid re-download)
  - Lazy loading (load on first request)
  - GPU detection and fallback to CPU

- [ ] **Create provider factory**
  - File: `src/core/vision/providers/__init__.py`
  - Add create_deepseek_vl_provider() factory
  - Environment variable: VISION_PROVIDER (stub|deepseek_vl)
  - Auto-select based on GPU availability

**Success Criteria**:
- Real DeepSeek-VL model generates descriptions
- Model loads without errors (on GPU if available)
- Fallback to stub if model not available

---

### Priority 2: Prompt Engineering

- [ ] **Design vision analysis prompt**
  - File: `src/core/vision/prompts.py`
  - Engineering drawing-specific prompt
  - Request: part type, features, dimensions visibility
  - Output format: structured JSON or natural language

- [ ] **Add prompt versioning**
  - Environment variable: VISION_PROMPT_VERSION (v1, v2, etc.)
  - Cache invalidation on version change
  - Document prompt changes in CHANGELOG.md

**Success Criteria**:
- Prompt generates relevant descriptions for CAD drawings
- Version system works (cache keys include prompt version)

---

## Phase 4: Golden Dataset Evaluation (Week 2 Day 1-2)

### Priority 1: Create Vision Golden Dataset

- [ ] **Prepare evaluation dataset**
  - Directory: `tests/vision/golden/`
  - 10 sample CAD drawings (easy/medium/hard)
  - Ground truth annotations (JSON):
    - expected_category (mechanical_part, assembly, etc.)
    - expected_features (list of key features)
    - expected_description_keywords (must-include words)

- [ ] **Create evaluation script**
  - File: `scripts/evaluate_vision_golden.py`
  - Load golden dataset
  - Run vision analysis on each sample
  - Calculate metrics:
    - Description relevance (keyword matching)
    - Category accuracy
    - Feature detection recall

**Success Criteria**:
- 10 annotated CAD drawing samples
- Evaluation script runs without errors
- Baseline metrics established

---

### Priority 2: Vision + OCR Combined Evaluation

- [ ] **Design combined metrics**
  - Vision description quality (subjective → keyword matching)
  - OCR dimension recall (from OCR golden dataset)
  - Combined confidence score
  - End-to-end latency (vision + OCR)

- [ ] **Create combined evaluation report**
  - File: `reports/vision_ocr_combined_evaluation.md`
  - Compare:
    - Vision-only results
    - OCR-only results
    - Vision + OCR combined results
  - Identify synergy opportunities

**Success Criteria**:
- Combined evaluation script working
- Report shows vision + OCR better than either alone
- Latency within acceptable range (< 5s)

---

## Phase 5: Performance Optimization (Week 2 Day 3-4)

### Priority 1: Latency Optimization

- [ ] **Profile end-to-end request**
  - Use cProfile or pyinstrument
  - Identify bottlenecks (model inference, OCR, etc.)
  - Target: < 2s for vision, < 3s for vision+OCR

- [ ] **Implement optimizations**
  - Model quantization (FP16 or INT8)
  - Batch processing for multiple images
  - Parallel vision + OCR (if independent)

- [ ] **Add performance tests**
  - File: `tests/vision/test_performance.py`
  - Test: latency under 3s for P95
  - Test: throughput > 5 requests/second

**Success Criteria**:
- P50 latency < 1.5s, P95 < 3s
- Throughput meets target
- Performance tests passing

---

### Priority 2: Memory Optimization

- [ ] **Profile memory usage**
  - Track model memory footprint
  - Identify memory leaks
  - Target: < 6GB VRAM for DeepSeek-VL

- [ ] **Implement optimizations**
  - Model offloading (CPU/GPU)
  - Image preprocessing optimization
  - Clear image cache after processing

**Success Criteria**:
- VRAM usage < 6GB
- No memory leaks after 100 requests
- OOM errors handled gracefully

---

## Phase 6: Production Readiness (Week 2 Day 5)

### Priority 1: Error Handling & Resilience

- [ ] **Comprehensive error taxonomy**
  - File: `src/core/vision/exceptions.py`
  - VisionError codes (VISION_001-999)
  - Provider-specific errors
  - Integration error mapping

- [ ] **Add circuit breaker**
  - Protect against provider failures
  - Auto-fallback to stub on repeated errors
  - Metrics: vision_circuit_breaker_state

- [ ] **Implement retries**
  - Retry transient errors (timeout, 429)
  - Exponential backoff
  - Max 3 retries

**Success Criteria**:
- All error codes documented
- Circuit breaker prevents cascading failures
- Retry logic tested

---

### Priority 2: Monitoring & Observability

- [ ] **Add Prometheus metrics**
  - vision_requests_total{provider, status}
  - vision_processing_duration_seconds{provider}
  - vision_description_length{provider}
  - vision_confidence_distribution{provider}

- [ ] **Create Grafana dashboard**
  - File: `grafana/vision_dashboard.json`
  - Panels:
    - Request rate and status
    - Latency distribution (P50/P95/P99)
    - Provider comparison
    - Error rate

**Success Criteria**:
- 4+ vision-specific metrics
- Grafana dashboard imported successfully
- Metrics visible in real-time

---

## Phase 7: Documentation & Handoff (Week 2 Final)

### Priority 1: API Documentation

- [ ] **Update OpenAPI spec**
  - Add `/api/v1/vision/analyze` schema
  - Add `/api/v1/vision/health` schema
  - Example requests and responses
  - Error response documentation

- [ ] **Create usage guide**
  - File: `docs/vision/VISION_API_GUIDE.md`
  - Quick start examples
  - cURL commands
  - Python client examples
  - Error handling guide

**Success Criteria**:
- OpenAPI spec complete and validated
- Usage guide with 5+ examples
- Postman collection available

---

### Priority 2: Integration Guide

- [ ] **Document Vision + OCR integration**
  - File: `docs/vision/VISION_OCR_INTEGRATION.md`
  - Architecture diagram
  - Data flow explanation
  - Provider selection logic
  - Fallback strategy

- [ ] **Create troubleshooting guide**
  - File: `docs/vision/TROUBLESHOOTING.md`
  - Common errors and solutions
  - Performance tuning tips
  - Model loading issues

**Success Criteria**:
- Integration guide with architecture diagram
- Troubleshooting guide with 10+ scenarios
- Team can deploy without support

---

## Quick Reference

### Current Status (End of Phase 1)
✅ Stub provider working
✅ Basic endpoint live
✅ Tests passing (5/8)
⚠️ OCR not yet connected
⚠️ Real DeepSeek-VL not yet integrated

### Next Immediate Steps (Phase 2)
1. Connect OCRManager to VisionManager
2. Test vision + OCR integration
3. Implement image URL support

### Success Metrics by End of Week 2
- [ ] Vision + OCR pipeline end-to-end
- [ ] Real DeepSeek-VL provider working
- [ ] Golden dataset evaluation complete
- [ ] Performance targets met (< 3s latency)
- [ ] Comprehensive documentation
- [ ] Production-ready error handling

---

## Dependencies

### External Dependencies
- transformers (for DeepSeek-VL)
- torch (GPU support)
- httpx or aiohttp (for image URL downloading)
- PIL/Pillow (image preprocessing)

### Internal Dependencies
- OCRManager (from src.core.ocr.manager)
- Existing metrics infrastructure
- Existing error handling patterns

---

## Risk Mitigation

### High Risk
1. **DeepSeek-VL model size** (>10GB)
   - Mitigation: Model quantization, lazy loading
2. **GPU availability**
   - Mitigation: CPU fallback, stub provider fallback
3. **Integration complexity**
   - Mitigation: Phase-by-phase approach, comprehensive testing

### Medium Risk
1. **Latency targets**
   - Mitigation: Profiling, optimization iterations
2. **Memory constraints**
   - Mitigation: Model offloading, image preprocessing limits

### Low Risk
1. **API design changes**
   - Mitigation: Versioning, backward compatibility
2. **Dependency conflicts**
   - Mitigation: Virtual environment, requirements.txt pinning

---

**Last Updated**: 2025-01-15
**Next Review**: 2025-01-17 (after Phase 2 completion)
