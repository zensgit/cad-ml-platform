# C2: vLLM Model Selection & Quantization Benchmark Design

**Status:** Approved  
**Author:** CAD-ML Platform Team  
**Date:** 2026-04-13  
**Depends on:** C1 (VLLMProvider)

---

## 1. Objective

Build the infrastructure to systematically evaluate, compare, and select the best
open-source LLM for CAD/manufacturing domain tasks running on vLLM. The benchmark
suite answers three questions:

1. **Which model** gives the best balance of domain accuracy and speed?
2. **Which quantization** minimises VRAM while preserving quality?
3. **What concurrency** can we sustain within our latency budget?

Performance targets:
- P95 latency < 100 ms (single request, short prompt)
- Throughput > 20 req/s at operating concurrency
- Quality: >= 80% keyword coverage on CAD domain prompts

---

## 2. Candidate Model Comparison Matrix

| Model | Params | Architecture | VRAM FP16 | VRAM AWQ | Context | Chinese | Code | License |
|-------|--------|-------------|-----------|----------|---------|---------|------|---------|
| DeepSeek-Coder-6.7B | 6.7B | Decoder | 13 GB | 4 GB | 16K | Good | Excellent | DeepSeek |
| Qwen2-7B-Instruct | 7B | Decoder | 14 GB | 4.5 GB | 32K | Excellent | Good | Apache-2.0 |
| Llama-3-8B-Instruct | 8B | Decoder | 16 GB | 5 GB | 8K | Weak | Moderate | Llama 3 |
| DeepSeek-V2-Lite | 16B (MoE) | MoE | 10 GB | 6 GB | 32K | Good | Good | DeepSeek |

**Initial recommendation:** Start with **DeepSeek-Coder-6.7B AWQ** for its small
footprint, strong code understanding, and adequate Chinese support. If domain
accuracy is insufficient, upgrade to **Qwen2-7B AWQ** (best Chinese model with
Apache-2.0 license).

---

## 3. Benchmark Methodology

### 3.1 Metrics Collected

| Metric | Description | Target |
|--------|-------------|--------|
| P50 / P95 / P99 latency | End-to-end request latency in seconds | P95 < 100ms |
| Throughput (req/s) | Requests completed per second | > 20 req/s |
| Tokens/sec (TPS) | Output token generation rate | > 30 tok/s |
| Quality score | Keyword coverage on domain prompts (0-1) | >= 0.80 |
| Hallucination count | Responses with incorrect domain keywords | 0 |
| Relevance rate | % of responses judged relevant | >= 90% |
| VRAM usage | Peak GPU memory during benchmark | Fits 24 GB |
| Optimal concurrency | Highest concurrency meeting latency target | Reported |

### 3.2 Test Suite

The benchmark uses **22 CAD-domain prompts** across 7 categories:

| Category | Count | Languages | Example |
|----------|-------|-----------|---------|
| Part Classification | 4 | zh/en | "这个零件是什么类型？描述：法兰盘，DN150，PN16" |
| Material Analysis | 4 | zh | "Q235B钢适合什么加工工艺？" |
| Drawing Interpretation | 3 | zh | "标题栏显示：零件名称-轴承座，材料-45#钢" |
| Technical QA | 4 | zh/en | "什么是粗糙度Ra3.2？适用于什么场景？" |
| Process Planning | 3 | zh | "设计一个φ100mm、长500mm的45#钢轴的加工工艺路线" |
| Tolerance & Fit | 3 | zh | "H7/g6配合在φ50mm时的间隙范围是多少？" |
| Complex / Multi-step | 2 | zh | "设计液压缸端盖，工作压力16MPa" |

Each prompt carries:
- **expected_keywords**: domain terms that should appear in a correct answer
- **negative_keywords**: terms indicating hallucination or wrong-domain answers
- **language tag**: zh or en

### 3.3 Quality Evaluation

```
keyword_score = matched_keywords / total_expected_keywords
hallucination  = any(negative_keyword in response)
is_relevant    = keyword_score >= 0.4 AND len(response) >= 20
```

A model is considered **CAD-domain ready** when:
- Average keyword_score >= 0.80 across all prompts
- Zero hallucinations
- Relevance rate >= 90%

### 3.4 Stress Test Protocol

Concurrency ramp: 1 -> 5 -> 10 -> 20 -> 50 concurrent requests.

For each level the suite measures P50/P95/P99 latency and throughput.
The **optimal concurrency** is the highest level where P95 stays under 100 ms.

---

## 4. Quantization Trade-off Analysis

| Method | VRAM Factor | Speed Factor | Quality Impact | Maturity |
|--------|------------|-------------|----------------|----------|
| FP16 | 1.0x | 1.0x (baseline) | None | Stable |
| AWQ (4-bit) | 0.3x | 1.5-2.0x | Minimal (< 1%) | Stable |
| GPTQ (4-bit) | 0.35x | 1.3-1.8x | Minimal-Moderate | Stable |
| INT8 | 0.5x | 1.2-1.5x | Minimal | Stable |

**Recommendation:** AWQ is the default quantization method. It delivers the best
VRAM reduction (70%) with negligible quality loss and the fastest inference among
4-bit methods. Use FP16 only for MoE models (DeepSeek-V2-Lite) where 4-bit
quantization is less mature.

The `--quantization-sweep` mode benchmarks all four methods on a single model to
produce empirical speed-vs-quality data for the specific hardware.

---

## 5. Hardware Requirements

| Model + Quant | Min VRAM | Recommended GPU | Est. Cost/hr |
|---------------|----------|-----------------|-------------|
| DeepSeek-Coder-6.7B AWQ | 4 GB | RTX 3090 / A5000 | $0.80 |
| Qwen2-7B AWQ | 4.5 GB | RTX 3090 / A5000 | $0.80 |
| Llama-3-8B AWQ | 5 GB | RTX 4090 / A5000 | $1.00 |
| DeepSeek-V2-Lite FP16 | 10 GB | RTX 4090 / A5000 | $1.00 |
| Any model FP16 (13-16 GB) | 16 GB | A100-40GB | $2.50 |

All AWQ models fit on a single 24 GB consumer GPU with room for batching.

---

## 6. Benchmark Components

### 6.1 Files

| File | Purpose |
|------|---------|
| `scripts/benchmark_vllm_quantization.py` | Main benchmark engine (async, quality eval, dry-run) |
| `scripts/run_vllm_benchmark_suite.sh` | Shell runner: Docker lifecycle + GPU metrics |
| `config/vllm_models.yaml` | Model metadata and hardware profiles |
| `tests/unit/test_vllm_benchmark.py` | Unit tests for prompt suite, scoring, reporting |

### 6.2 CLI Interface

```bash
# CI dry-run (no GPU needed)
python3 scripts/benchmark_vllm_quantization.py --dry-run

# Single model
python3 scripts/benchmark_vllm_quantization.py \
    --model deepseek-coder-6.7b --endpoint http://localhost:8100

# Compare all candidates
python3 scripts/benchmark_vllm_quantization.py --compare-all --dry-run

# Quantization sweep
python3 scripts/benchmark_vllm_quantization.py \
    --model deepseek-coder-6.7b --quantization-sweep

# Stress test
python3 scripts/benchmark_vllm_quantization.py \
    --model deepseek-coder-6.7b --stress

# Full suite via shell runner (Docker + GPU metrics)
./scripts/run_vllm_benchmark_suite.sh --compare-all --stress
```

---

## 7. Expected Results Format

### JSON Report (sample)

```json
{
  "generated_at": "2026-04-13T10:00:00",
  "prompt_count": 22,
  "models": [
    {
      "model_name": "deepseek-coder-6.7b",
      "quantization": "awq",
      "summary": {
        "avg_quality_score": 0.82,
        "hallucination_count": 0,
        "relevance_rate": 0.95,
        "optimal_concurrency": 20,
        "throughput_plateau_rps": 28.5
      },
      "concurrency_results": [
        {"concurrency": 1,  "p95_latency": 0.035, "req_per_sec": 28.5},
        {"concurrency": 10, "p95_latency": 0.068, "req_per_sec": 26.1},
        {"concurrency": 50, "p95_latency": 0.142, "req_per_sec": 22.3}
      ]
    }
  ]
}
```

### Human-Readable Table (sample)

```
==============================================================================================================
vLLM Model Benchmark Comparison Report
==============================================================================================================

Model                        Quant  P50(ms)   P95(ms)   P99(ms)   RPS      Quality   Hallu  Relev%
--------------------------------------------------------------------------------------------------------------
deepseek-coder-6.7b          awq    32.1      68.2      85.4      26.1     82.00%    0      95%
qwen2-7b                     awq    38.5      78.9      95.2      22.4     88.00%    0      96%
llama3-8b                    awq    41.2      85.3      102.1     20.1     71.00%    1      82%
deepseek-v2-lite             fp16   29.8      62.1      78.3      30.2     79.00%    0      91%
--------------------------------------------------------------------------------------------------------------

Performance Targets: P95 < 100ms, Throughput > 20 req/s
```

---

## 8. Decision Framework

To select the production model, evaluate results in this priority order:

1. **Quality gate:** avg_quality_score >= 0.80 AND hallucination_count == 0
2. **Latency gate:** P95 < 100ms at concurrency=10
3. **Throughput gate:** req/s >= 20 at operating concurrency
4. **VRAM efficiency:** fits target GPU with room for batching
5. **License:** prefer Apache-2.0 for commercial deployment

If multiple models pass all gates, prefer:
- Qwen2-7B for **Chinese-primary** deployments (best Chinese, Apache-2.0)
- DeepSeek-Coder-6.7B for **code-heavy** tasks (G-code, scripts, structured output)
- DeepSeek-V2-Lite for **throughput-critical** scenarios (MoE efficiency)

---

## 9. CI/CD Integration

### Dry-Run in CI

The benchmark runs in `--dry-run` mode on every PR to validate:
- Prompt suite integrity (all 22+ prompts load, IDs unique)
- Quality evaluation logic (scoring, hallucination detection)
- Report generation (JSON valid, table renders)

```yaml
# .github/workflows/ci.yml (excerpt)
- name: vLLM benchmark dry-run
  run: python3 scripts/benchmark_vllm_quantization.py --dry-run --compare-all
```

### Nightly GPU Benchmark

On a GPU-equipped runner, the full suite runs nightly:

```yaml
# .github/workflows/nightly-benchmark.yml (excerpt)
- name: vLLM model benchmark
  run: ./scripts/run_vllm_benchmark_suite.sh --compare-all --stress
  env:
    VLLM_IMAGE: vllm/vllm-openai:latest
```

Reports are uploaded as artifacts and compared against previous runs to detect
performance regressions.

---

## 10. Cost Analysis

| Configuration | GPU | $/hr | Monthly (8h/day) | Notes |
|---------------|-----|------|-------------------|-------|
| DeepSeek-6.7B AWQ | RTX 3090 | $0.80 | $176 | Cheapest option |
| Qwen2-7B AWQ | RTX 4090 | $1.00 | $220 | Best quality/price |
| DeepSeek-V2-Lite FP16 | RTX 4090 | $1.00 | $220 | MoE efficiency |
| Any FP16 large | A100-40GB | $2.50 | $550 | For ablation study only |

Cloud GPU pricing based on typical spot instance rates (2026 Q1).

---

## 11. Starting Recommendation

**Phase 1 deployment:** `deepseek-ai/deepseek-coder-6.7b-instruct` with AWQ quantization.

Rationale:
- Smallest VRAM footprint (4 GB) — runs on any modern GPU
- Excellent code understanding for CAD scripts and structured output
- Good Chinese language support
- Already configured as the default in VLLMProvider (C1)

**Upgrade path:** If domain quality benchmarks show avg_quality_score < 0.80
on material analysis or drawing interpretation prompts, switch to
`Qwen/Qwen2-7B-Instruct` AWQ which has superior Chinese domain knowledge
at only +0.5 GB VRAM cost.
