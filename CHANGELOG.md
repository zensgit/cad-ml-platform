# Changelog

All notable changes to the CAD ML Platform OCR integration will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- UV-Net graph dry-run guardrails (fail on empty graphs), expanded STEP fixtures, and PR-only workflow trigger.
- Materials API: cost search `include_estimated` flag and cost compare `missing` list for unknown grades.

### Planned
- vLLM provider for high throughput scenarios
- Geometric alignment with R-tree spatial indexing
- Advanced title block extraction

---

## OCR Module Versions

### PROMPT_VERSION History

#### [v1] - 2025-01-14
**Initial prompt templates**
- DeepSeek structured JSON prompt for engineering drawings
- Dimension/tolerance/symbol extraction focus
- Title block field extraction
- Cache key: includes `prompt_version=v1`
- **Version format**: `v1` (integer only, no patch version for prompts)

```python
# Example prompt (abbreviated)
"""<image>
<|grounding|>Extract dimensions/tolerances/surface-roughness/threads as strict JSON:
{
  "dimensions": [{"type":"diameter|radius|length|thread", "value":float, "unit":"mm", "tolerance":float, "bbox":{}}],
  "symbols": [{"type":"surface_roughness|perpendicular|parallel", "value":str, "bbox":{}}],
  "title_block": {"drawing_number":str, "material":str, "part_name":str, "scale":str}
}"""
```

**Fallback format (Markdown-fenced JSON)** - When model struggles with pure JSON:
```markdown
The analysis results:

```json
{
  "dimensions": [...],
  "symbols": [...],
  "title_block": {...}
}
```

Additional notes: ...
```

**Expected behavior changes**:
- Fallback rate may increase 5-10% during first week after version change (model adaptation period)
- Cache invalidation will cause temporary latency spike (~20% increase for 1-2 hours)
- Monitor `ocr_fallback_rate` metric closely for 48h post-deployment

### DATASET_VERSION History

#### [v1.0] - 2025-01-14
**Initial golden dataset**
- Categories: easy (10), medium (10), hard (5), edge (5)
- Total samples: 30
- **Metadata location**: `tests/ocr/golden/metadata.yaml` (lines 1-30)
- Annotation schema:
  - dimensions: {type, value, tolerance, unit, bbox}
  - symbols: {type, value, normalized_form, bbox}
  - title_block: {drawing_number, material, part_name}

**Evaluation metrics formulas**:

```python
# Dimension Recall
dimension_recall = matched_dimensions / ground_truth_dimensions
# Match condition: abs(value_pred - value_gt) <= max(0.05 * value_gt, tolerance_gt_if_present)

# Symbol Recall
symbol_recall = matched_symbols / ground_truth_symbols
# Match condition: normalized_form(pred) == normalized_form(gt)

# Edge-F1 (OCR bounding box quality)
TP_edge = count(boxes where IoU(pred_bbox, gt_bbox) >= 0.5 AND text_similarity >= 0.8)
FP_edge = count(pred_boxes with no matching gt_box)
FN_edge = count(gt_boxes with no matching pred_box)

precision_edge = TP_edge / (TP_edge + FP_edge)
recall_edge = TP_edge / (TP_edge + FN_edge)
edge_f1 = 2 * (precision_edge * recall_edge) / (precision_edge + recall_edge)

# Brier Score (confidence calibration quality)
brier_score = mean((confidence_i - correctness_i)^2) for all predictions
# Lower is better; target < 0.20 (Week1), < 0.15 (Week2)
```

**Matching rules details**:
- **Dimension matching**:
  - Unit normalization: all converted to mm first
  - Tolerance-aware: if GT has tolerance, use it; otherwise use 5% of value
  - Thread pitch: for M10×1.5, match major_diameter (10mm) and pitch (1.5mm) separately
- **Symbol matching**:
  - Canonical form: Ra3.2 → surface_roughness_3.2
  - Perpendicular: ⟂/⊥ → perpendicular
  - Parallel: ∥/‖ → parallel
- **Unit normalization**: 20毫米 → 20mm, 2cm → 20mm

---

## Platform Changes

### OCR Module

#### [Unreleased] - 2025-01-15

##### Added
- 🔭 **Vision Module MVP** (End-to-End Pipeline Foundation):
  - `src/core/vision/` - Core vision module structure
    - `base.py`: Pydantic models (VisionAnalyzeRequest, VisionAnalyzeResponse, VisionDescription)
    - `manager.py`: VisionManager orchestrates Vision + OCR
    - `providers/deepseek_stub.py`: Stub provider for testing and MVP
  - `/api/v1/vision/analyze` - Vision + OCR analysis endpoint
    - Accept base64-encoded images
    - Return vision description + OCR extraction (OCR integration pending)
    - Proper error handling and validation
  - `/api/v1/vision/health` - Health check endpoint
  - `tests/vision/test_vision_endpoint.py`: 8 test cases (5/8 passing)
  - `docs/ocr/VISION_WEEK1_WEEK2_PLAN.md`: Detailed 7-phase implementation plan

- 🧪 **Test Infrastructure Enhancements**:
  - `test_dimension_matching.py`: Complete matching formula validation
    - Unit normalization tests (mm/cm/m/in/毫米/厘米)
    - Thread matching with separate diameter and pitch validation
    - Recall calculation simulation with 7 test scenarios
    - 30 parametrized test cases, all passing
  - `test_cache_key.py`: Enhanced cache key validation
    - SHA256 hash (replacing MD5 for collision resistance)
    - Version format validation (v1 for prompts, v1.0 for datasets)
    - Large file (>10MB) no-cache policy tests
  - `test_fallback.py`: Comprehensive fallback strategy tests
    - Thread pitch extraction (M10×1.5 → diameter + pitch)
    - Bidirectional tolerance parsing (Φ20 +0.02 -0.01)
    - Markdown fence case-insensitive matching (```JSON/```json/``` json)
    - BOM and mixed content handling
    - Dynamic performance thresholds (50ms + 10ms/KB)
    - Schema deep validation
  - Fixtures directory: 5 mock DeepSeek outputs
    - `valid_json.txt`: Level 1 (JSON_STRICT) test case
    - `markdown_fence.txt`: Level 2 (MARKDOWN_FENCE) test case
    - `malformed_json.txt`: Syntax error recovery test
    - `text_only.txt`: Level 3 (TEXT_REGEX) test with Chinese text
    - `bom_mixed.txt`: UTF-8 BOM handling test
    - Fixtures README with integration examples

- 🛠️ **Development Tools**:
  - `scripts/verify_environment.py` enhancements:
    - MIME dual package support (`magic` or `python_magic` with mimetypes fallback)
    - Enhanced GPU detection with nvidia-smi diagnostics (driver/CUDA separation)
    - PIL MAX_IMAGE_PIXELS dynamic configuration from MAX_RESOLUTION env var
    - Concurrency validation against CPU cores with warnings
    - DEEPSEEK_ENABLED + no-GPU conflict detection
  - `scripts/dump_metrics_example.py`: Prometheus metrics reference
    - 21 metric definitions with types, labels, and examples
    - Multiple output formats: table/json/prometheus
    - Filter by metric name substring
    - Histogram bucket specifications

##### Changed
- `FallbackParser._parse_markdown_fence()`: Regex pattern updated
  - Now supports spaces around "json" keyword: ```\s*json\s*
  - Fixes test_markdown_fence_case_insensitive failure
  - Maintains case-insensitive matching (JSON/json/Json)

##### Fixed
- Markdown fence parser failing on ``` json  (with spaces)
- Missing formula documentation for Edge-F1 and Brier Score
- Cache key collision risk with MD5 → SHA256 migration

##### Developer Experience
- All test suites passing (48 tests across 3 files)
- Clear test fixtures with usage documentation
- Executable environment verification script with diagnostics
- Metrics reference tool for Prometheus integration

---

#### [1.1.0-ocr] - 2025-01-14

##### Added
- 🎯 OCR integration module with multi-provider support (Paddle, DeepSeek HF)
- 📊 Comprehensive metrics (recall, Edge-F1, latency percentiles, Brier score)
- 🔐 Security layer (MIME whitelist, PDF sanitization, resolution limits)
- 🧪 Golden dataset with versioning (v1.0, 30 samples)
- 📂 Unified error taxonomy (OCR_001-999)
- 🔄 Three-level fallback strategy:
  1. Strict JSON parsing
  2. Markdown-fenced JSON extraction
  3. Text regex patterns (Φ/R/M/Ra/⊥/∥)
- 💾 Multi-key caching with prompt_version and dataset_version
- 📈 Prometheus metrics and Grafana dashboards
- 🧮 Structured parsers for:
  - Dimensions (diameter, radius, thread with pitch)
  - Symbols (surface roughness, GD&T basics)
  - Title blocks (drawing_number, material, part_name)

##### Changed
- Cache key formula: `ocr:{image_hash}:{provider}:{prompt_v}:{crop_cfg_hash}:{dataset_v}`
- Error responses standardized with OcrError taxonomy
- Latency tracking split by stage (preprocess|inference|parse|normalize)

##### Security
- MIME type validation: image/png, image/jpeg, application/pdf only
- Maximum resolution: 2048px (auto-resize with warning, not reject)
- PDF security: reject encrypted/JS-embedded/XFA PDFs
- PII protection: only hash logged, sensitive fields redacted
- File size limit: 50MB (HTTP 413)
- Page limit: 20 pages for PDFs (HTTP 422)

##### Metrics
See `docs/DEEPSEEK_OCR_TODO_LIST.md` section "监控指标" for complete list.

Key metrics:
- `ocr_requests_total{provider, status}`
- `ocr_errors_total{type}`
- `ocr_processing_duration_seconds{provider, stage}`
- `ocr_confidence_score` (histogram)
- `ocr_field_recall{field_type}` (gauge, updated after evaluation)

##### Known Limitations (Week 1)
- No vLLM support (Week 3 feature)
- No geometric alignment (Week 3 feature)
- Confidence calibration requires 50+ samples for accuracy

---

### Core Platform

#### [1.1.0] - 2025-01-14

##### Changed
- Extended `/api/v1/analyze` endpoint with OCR support
  - New parameter: `enable_ocr: bool = False`
  - New parameter: `ocr_provider: str = "auto"`
  - Backward compatible: existing calls unchanged
- Redis cache extended for OCR results (1-hour TTL)

#### [1.0.0] - 2025-01-10

##### Added
- Initial CAD ML Platform release
- Core analysis capabilities for 8 part types
- 95-dimensional feature extraction
- DXF/STEP/IGES format support
- Assembly understanding module
- Vision analysis with multi-provider support

---

## Migration Notes

### Prompt Version Changes
When updating PROMPT_VERSION:
1. Update `PROMPT_VERSION` environment variable
2. Cache will auto-invalidate (new keys)
3. Run regression tests on golden dataset
4. Document changes in this file

### Dataset Version Changes
When updating DATASET_VERSION:
1. Update `tests/ocr/golden/metadata.yaml`
2. Update `DATASET_VERSION` environment variable
3. Re-run all evaluations for new baseline
4. Archive previous version results
5. Document annotation changes

### Breaking Changes
- Cache keys will change when versions update
- Clients should handle cache misses gracefully
- Monitor fallback rates after version changes

---

## Version Compatibility Matrix

| Platform Version | PROMPT_VERSION | DATASET_VERSION | Min Python | Min CUDA |
|-----------------|----------------|-----------------|------------|----------|
| 1.1.0           | v1.0           | v1.0            | 3.9        | 11.8     |
| 1.0.0           | -              | -               | 3.8        | 11.6     |

---

## Upcoming Changes (Next Release)

### PROMPT_VERSION v2 (Planned)
- Enhanced assembly drawing understanding
- Multi-view correlation
- BOM extraction optimization
- Estimated release: Week 3-4

### DATASET_VERSION v1.1 (Planned)
- Add assembly drawings (10 samples)
- Add handwritten annotations subset (5 samples)
- Improve hard category diversity
- Estimated release: After Week 2 evaluation

### vLLM Provider Integration (Week 3 - Optional)

**Performance comparison matrix** (placeholder, to be filled after implementation):

| Metric | DeepSeek HF (Baseline) | DeepSeek vLLM | Target |
|--------|------------------------|---------------|--------|
| **Throughput** (tokens/s) | TBD | TBD | ≥ 2x HF |
| **P95 Latency** (s) | TBD | TBD | < 3s |
| **Batch Size** | 1 | TBD | 4-8 |
| **VRAM Usage** (GB) | TBD | TBD | < 8GB |
| **Dimension Recall** | TBD | TBD | Δ < 1% |
| **Edge-F1** | TBD | TBD | Δ < 2% |
| **Cost per 1K requests** | TBD | TBD | < 80% HF |
| **Cold Start** (s) | TBD | TBD | < 30s |

**Expected trade-offs**:
- ✅ Higher throughput for batch processing
- ✅ Lower latency per request when batched
- ⚠️ May require separate service deployment
- ⚠️ Additional complexity in batching logic
- ⚠️ Network latency if not co-located

**Evaluation criteria**:
- Throughput improvement ≥ 100% (2x)
- Accuracy degradation < 1% (dimension recall)
- Deployment complexity acceptable (1-2 days setup)
- Cost reduction ≥ 20% for sustained loads

---

## Deprecation Policy

- Prompt versions supported for 3 months after deprecation
- Dataset versions archived but not actively maintained after 2 major versions
- Cache invalidation happens immediately on version change
- Fallback to previous version not supported (forward-only)

---

## Contact

For questions about version changes:
- Technical: tech-team@cad-ml-platform.com
- Dataset annotations: ml-team@cad-ml-platform.com
