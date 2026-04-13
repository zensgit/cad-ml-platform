#!/usr/bin/env python3
from __future__ import annotations

"""
vLLM Model Selection & Quantization Benchmark Suite.

Production-grade benchmarking for CAD/manufacturing domain LLM inference.
Tests multiple models and quantization methods to find the optimal balance
of latency, throughput, memory usage, and response quality.

Usage:
    # Dry-run mode (CI, no vLLM server needed):
    python3 scripts/benchmark_vllm_quantization.py --dry-run

    # Single model benchmark:
    python3 scripts/benchmark_vllm_quantization.py --model deepseek-coder-6.7b

    # Full model comparison:
    python3 scripts/benchmark_vllm_quantization.py --compare-all

    # Stress test:
    python3 scripts/benchmark_vllm_quantization.py --model deepseek-coder-6.7b --stress

    # Quantization comparison for one model:
    python3 scripts/benchmark_vllm_quantization.py --model deepseek-coder-6.7b --quantization-sweep
"""

import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vllm_benchmark")

try:
    import aiohttp
    import numpy as np
except ImportError:
    logger.warning("aiohttp and/or numpy not available; install with: pip install aiohttp numpy")
    aiohttp = None  # type: ignore
    np = None  # type: ignore

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
REPORTS_DIR = ROOT_DIR / "reports"

# ---------------------------------------------------------------------------
# CAD-Domain Test Suite  (20+ bilingual prompts)
# ---------------------------------------------------------------------------

@dataclass
class TestPrompt:
    """A benchmark test prompt with quality evaluation criteria."""

    id: str
    category: str  # part_classification | material_analysis | drawing_interpretation | technical_qa | process_planning | tolerance_fit
    prompt: str
    expected_keywords: List[str]  # keywords that SHOULD appear in a good response
    negative_keywords: List[str] = field(default_factory=list)  # hallucination indicators
    language: str = "zh"  # primary language


CAD_TEST_PROMPTS: List[TestPrompt] = [
    # --- Part Classification (零件分类) ---
    TestPrompt(
        id="pc-01",
        category="part_classification",
        prompt="这个零件是什么类型？描述：法兰盘，DN150，PN16，8个螺栓孔",
        expected_keywords=["法兰", "管道", "连接", "压力", "密封"],
    ),
    TestPrompt(
        id="pc-02",
        category="part_classification",
        prompt="根据以下特征分类零件：外径50mm，内径30mm，厚度10mm，外圆有滚花。",
        expected_keywords=["套", "环", "衬套", "轴套"],
    ),
    TestPrompt(
        id="pc-03",
        category="part_classification",
        prompt="Classify this part: hexagonal head, threaded shaft M12x1.75, Grade 8.8 steel.",
        expected_keywords=["bolt", "fastener", "hex", "thread"],
        language="en",
    ),
    TestPrompt(
        id="pc-04",
        category="part_classification",
        prompt="零件特征：阶梯轴，总长200mm，三段直径分别为φ30/φ40/φ50，有键槽。",
        expected_keywords=["阶梯轴", "传动", "键槽", "轴"],
    ),

    # --- Material Analysis (材料分析) ---
    TestPrompt(
        id="ma-01",
        category="material_analysis",
        prompt="Q235B钢适合什么加工工艺？列出推荐的切削参数。",
        expected_keywords=["碳钢", "焊接", "切削", "车削", "铣削"],
        negative_keywords=["不锈钢", "铝合金"],
    ),
    TestPrompt(
        id="ma-02",
        category="material_analysis",
        prompt="304不锈钢和316不锈钢有什么区别？在海洋环境中应该选哪个？",
        expected_keywords=["耐腐蚀", "钼", "海洋", "316", "氯离子"],
    ),
    TestPrompt(
        id="ma-03",
        category="material_analysis",
        prompt="6061-T6铝合金的机械性能是什么？适合CNC加工吗？",
        expected_keywords=["铝合金", "抗拉强度", "硬度", "CNC", "切削"],
    ),
    TestPrompt(
        id="ma-04",
        category="material_analysis",
        prompt="45#钢和40Cr钢在齿轮制造中如何选择？需要热处理吗？",
        expected_keywords=["淬火", "调质", "齿轮", "硬度", "耐磨"],
    ),

    # --- Drawing Interpretation (图纸解读) ---
    TestPrompt(
        id="di-01",
        category="drawing_interpretation",
        prompt="标题栏显示：零件名称-轴承座，材料-45#钢，数量-2，比例1:1。请解读关键信息。",
        expected_keywords=["轴承座", "45", "钢", "数量", "比例"],
    ),
    TestPrompt(
        id="di-02",
        category="drawing_interpretation",
        prompt="图纸标注：φ50H7/g6，Ra1.6，形位公差⊥0.02|A。请解释这些技术要求。",
        expected_keywords=["配合", "间隙", "粗糙度", "垂直度", "基准"],
    ),
    TestPrompt(
        id="di-03",
        category="drawing_interpretation",
        prompt="图纸注释：未注公差按GB/T 1804-m执行，未注形位公差按GB/T 1184-K执行。这意味着什么？",
        expected_keywords=["公差", "国标", "精度", "等级"],
    ),

    # --- Technical QA (技术问答) ---
    TestPrompt(
        id="tq-01",
        category="technical_qa",
        prompt="什么是粗糙度Ra3.2？适用于什么场景？",
        expected_keywords=["表面粗糙度", "Ra", "微米", "加工"],
    ),
    TestPrompt(
        id="tq-02",
        category="technical_qa",
        prompt="什么是GD&T中的位置度公差？和传统坐标公差有什么区别？",
        expected_keywords=["位置度", "公差带", "圆形", "坐标"],
    ),
    TestPrompt(
        id="tq-03",
        category="technical_qa",
        prompt="CNC加工中，顺铣和逆铣的区别是什么？各自适用什么场景？",
        expected_keywords=["顺铣", "逆铣", "切削力", "表面质量"],
    ),
    TestPrompt(
        id="tq-04",
        category="technical_qa",
        prompt="What is the difference between casting and forging? When to use each?",
        expected_keywords=["casting", "forging", "strength", "shape", "grain"],
        language="en",
    ),

    # --- Process Planning (工艺规划) ---
    TestPrompt(
        id="pp-01",
        category="process_planning",
        prompt="设计一个φ100mm、长500mm的45#钢轴的加工工艺路线。",
        expected_keywords=["下料", "车削", "热处理", "磨削", "检验"],
    ),
    TestPrompt(
        id="pp-02",
        category="process_planning",
        prompt="铝合金薄壁零件（壁厚2mm）加工变形如何控制？",
        expected_keywords=["装夹", "切削力", "变形", "余量", "冷却"],
    ),
    TestPrompt(
        id="pp-03",
        category="process_planning",
        prompt="如何选择M10螺纹的底孔钻头直径？内螺纹和外螺纹分别是多少？",
        expected_keywords=["底孔", "8.5", "螺距", "钻头"],
    ),

    # --- Tolerance & Fit (公差配合) ---
    TestPrompt(
        id="tf-01",
        category="tolerance_fit",
        prompt="H7/g6配合在φ50mm时的间隙范围是多少？属于什么类型的配合？",
        expected_keywords=["间隙配合", "上偏差", "下偏差", "微米"],
    ),
    TestPrompt(
        id="tf-02",
        category="tolerance_fit",
        prompt="IT7公差等级在基本尺寸25mm时的公差值是多少？",
        expected_keywords=["公差", "IT7", "微米", "25"],
    ),
    TestPrompt(
        id="tf-03",
        category="tolerance_fit",
        prompt="轴承内圈与轴的配合一般推荐什么公差？为什么？",
        expected_keywords=["过盈", "k", "m", "n", "轴承", "配合"],
    ),

    # --- Complex / Multi-step ---
    TestPrompt(
        id="cx-01",
        category="complex",
        prompt="我需要设计一个液压缸端盖，工作压力16MPa，介质为液压油。请推荐材料、密封方式和关键尺寸公差。",
        expected_keywords=["密封", "O形圈", "压力", "材料", "公差"],
    ),
    TestPrompt(
        id="cx-02",
        category="complex",
        prompt="比较线切割、电火花和激光切割三种特种加工方法的精度、效率和适用场景。",
        expected_keywords=["线切割", "电火花", "激光", "精度", "效率"],
    ),
]


def get_test_prompts(category: Optional[str] = None) -> List[TestPrompt]:
    """Return filtered test prompts."""
    if category:
        return [p for p in CAD_TEST_PROMPTS if p.category == category]
    return list(CAD_TEST_PROMPTS)


# ---------------------------------------------------------------------------
# Model Configuration Loader
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model metadata loaded from config/vllm_models.yaml."""

    name: str
    hf_id: str
    vram_fp16: str
    vram_awq: str
    strengths: List[str]
    recommended_quantization: str


def load_model_configs() -> List[ModelConfig]:
    """Load model configurations from YAML (or return defaults)."""
    yaml_path = CONFIG_DIR / "vllm_models.yaml"
    if yaml_path.exists():
        try:
            import yaml

            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            configs = []
            _fields = {f.name for f in ModelConfig.__dataclass_fields__.values()}
            for m in data.get("models", []):
                filtered = {k: v for k, v in m.items() if k in _fields}
                configs.append(ModelConfig(**filtered))
            return configs
        except Exception as e:
            logger.warning(f"Failed to load {yaml_path}: {e}")

    # Fallback defaults matching the YAML spec
    return [
        ModelConfig("deepseek-coder-6.7b", "deepseek-ai/deepseek-coder-6.7b-instruct",
                     "13GB", "4GB", ["code_understanding", "structured_output", "chinese"], "awq"),
        ModelConfig("qwen2-7b", "Qwen/Qwen2-7B-Instruct",
                     "14GB", "4.5GB", ["chinese", "general_knowledge", "long_context"], "awq"),
        ModelConfig("llama3-8b", "meta-llama/Meta-Llama-3-8B-Instruct",
                     "16GB", "5GB", ["reasoning", "english", "tool_use"], "awq"),
        ModelConfig("deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite",
                     "10GB", "6GB", ["efficiency", "chinese", "code"], "fp16"),
    ]


# ---------------------------------------------------------------------------
# Quality Evaluation
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """Quality evaluation for a single response."""

    prompt_id: str
    keyword_hits: int
    keyword_total: int
    keyword_score: float  # 0-1
    negative_hits: int
    has_hallucination: bool
    response_length: int
    is_relevant: bool  # overall relevance flag


def evaluate_response_quality(prompt: TestPrompt, response: str) -> QualityScore:
    """Evaluate response quality against expected keywords."""
    response_lower = response.lower()
    hits = sum(1 for kw in prompt.expected_keywords if kw.lower() in response_lower)
    total = len(prompt.expected_keywords)
    keyword_score = hits / total if total > 0 else 0.0

    neg_hits = sum(1 for kw in prompt.negative_keywords if kw.lower() in response_lower)
    has_hallucination = neg_hits > 0

    # A response is considered relevant if it matches >= 40% keywords and
    # is at least 20 characters long
    is_relevant = keyword_score >= 0.4 and len(response) >= 20

    return QualityScore(
        prompt_id=prompt.id,
        keyword_hits=hits,
        keyword_total=total,
        keyword_score=keyword_score,
        negative_hits=neg_hits,
        has_hallucination=has_hallucination,
        response_length=len(response),
        is_relevant=is_relevant,
    )


# ---------------------------------------------------------------------------
# Benchmark Result Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SingleRequestResult:
    """Result of a single benchmark request."""

    error: bool
    latency: float  # seconds
    tokens_out: float  # estimated
    tps: float  # tokens per second
    response: str = ""


@dataclass
class ConcurrencyResult:
    """Aggregated results for one concurrency level."""

    concurrency: int
    total_requests: int
    successful: int
    failed: int
    p50_latency: float
    p95_latency: float
    p99_latency: float
    avg_tps: float
    total_duration: float
    req_per_sec: float


@dataclass
class ModelBenchmarkResult:
    """Complete benchmark result for one model+quantization combination."""

    model_name: str
    quantization: str
    endpoint: str
    timestamp: str
    concurrency_results: List[ConcurrencyResult]
    quality_scores: List[QualityScore]
    avg_quality_score: float
    hallucination_count: int
    relevance_rate: float
    # Stress test fields (optional)
    optimal_concurrency: Optional[int] = None
    throughput_plateau_rps: Optional[float] = None


# ---------------------------------------------------------------------------
# Async Benchmark Engine
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> SingleRequestResult:
    """Send a single request to vLLM and measure latency."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a mechanical engineering AI assistant specializing in CAD, manufacturing, and materials."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start_time = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Request failed ({response.status}): {text[:200]}")
                return SingleRequestResult(error=True, latency=0, tokens_out=0, tps=0)

            result = await response.json()
            end_time = time.perf_counter()

            content = result["choices"][0]["message"]["content"]
            # Approximate token count (CJK ~1.5 chars/token, English ~4 chars/token, blend ~3.5)
            out_tokens = len(content) / 3.5
            latency = end_time - start_time

            return SingleRequestResult(
                error=False,
                latency=latency,
                tokens_out=out_tokens,
                tps=out_tokens / latency if latency > 0 else 0,
                response=content,
            )
    except Exception as e:
        logger.error(f"Request exception: {e}")
        return SingleRequestResult(error=True, latency=0, tokens_out=0, tps=0)


async def run_concurrency_benchmark(
    endpoint: str,
    model: str,
    concurrency: int,
    prompts: List[str],
) -> ConcurrencyResult:
    """Run benchmark at a specified concurrency level."""
    url = f"{endpoint}/v1/chat/completions"
    logger.info(f"  Concurrency={concurrency} ...")

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(concurrency):
            prompt = prompts[i % len(prompts)]
            tasks.append(send_request(session, url, model, prompt))

        start_total = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_total = time.perf_counter()

    latencies = [r.latency for r in results if not r.error]
    tps_list = [r.tps for r in results if not r.error]

    if not latencies:
        logger.error("All requests failed at this concurrency level.")
        return ConcurrencyResult(
            concurrency=concurrency, total_requests=concurrency,
            successful=0, failed=concurrency,
            p50_latency=0, p95_latency=0, p99_latency=0,
            avg_tps=0, total_duration=end_total - start_total, req_per_sec=0,
        )

    arr = np.array(latencies)
    return ConcurrencyResult(
        concurrency=concurrency,
        total_requests=concurrency,
        successful=len(latencies),
        failed=concurrency - len(latencies),
        p50_latency=float(np.percentile(arr, 50)),
        p95_latency=float(np.percentile(arr, 95)),
        p99_latency=float(np.percentile(arr, 99)),
        avg_tps=float(np.mean(tps_list)),
        total_duration=end_total - start_total,
        req_per_sec=len(latencies) / (end_total - start_total) if (end_total - start_total) > 0 else 0,
    )


async def run_quality_evaluation(
    endpoint: str,
    model: str,
    prompts: List[TestPrompt],
) -> List[Tuple[QualityScore, str]]:
    """Run quality evaluation: send each prompt and score the response."""
    url = f"{endpoint}/v1/chat/completions"
    results = []

    async with aiohttp.ClientSession() as session:
        for prompt in prompts:
            result = await send_request(session, url, model, prompt.prompt, max_tokens=512)
            if not result.error:
                score = evaluate_response_quality(prompt, result.response)
                results.append((score, result.response))
            else:
                # Score as zero if request failed
                score = QualityScore(
                    prompt_id=prompt.id, keyword_hits=0, keyword_total=len(prompt.expected_keywords),
                    keyword_score=0, negative_hits=0, has_hallucination=False,
                    response_length=0, is_relevant=False,
                )
                results.append((score, ""))

    return results


async def run_stress_test(
    endpoint: str,
    model: str,
    prompts: List[str],
    levels: Optional[List[int]] = None,
) -> Tuple[List[ConcurrencyResult], int, float]:
    """Ramp up concurrency and find the throughput plateau.

    Returns:
        (results, optimal_concurrency, plateau_rps)
    """
    if levels is None:
        levels = [1, 5, 10, 20, 50]

    results = []
    for c in levels:
        cr = await run_concurrency_benchmark(endpoint, model, c, prompts)
        results.append(cr)
        logger.info(f"    c={c} -> RPS={cr.req_per_sec:.2f}, P95={cr.p95_latency:.4f}s")

    # Find optimal: highest RPS before P95 > 100ms threshold
    TARGET_P95 = 0.100  # 100ms
    best_rps = 0.0
    optimal_c = 1
    for cr in results:
        if cr.p95_latency <= TARGET_P95 and cr.req_per_sec > best_rps:
            best_rps = cr.req_per_sec
            optimal_c = cr.concurrency

    # If no level met the threshold, pick the one with lowest P95
    if best_rps == 0 and results:
        best = min(results, key=lambda r: r.p95_latency if r.successful > 0 else float("inf"))
        optimal_c = best.concurrency
        best_rps = best.req_per_sec

    return results, optimal_c, best_rps


# ---------------------------------------------------------------------------
# Dry-run simulation
# ---------------------------------------------------------------------------

def _simulate_concurrency_result(concurrency: int) -> ConcurrencyResult:
    """Generate simulated benchmark results for dry-run mode."""
    base_latency = 0.035 + concurrency * 0.002
    jitter = random.uniform(-0.005, 0.005)
    return ConcurrencyResult(
        concurrency=concurrency,
        total_requests=concurrency,
        successful=concurrency,
        failed=0,
        p50_latency=base_latency + jitter,
        p95_latency=base_latency * 1.8 + jitter,
        p99_latency=base_latency * 2.5 + jitter,
        avg_tps=45.0 - concurrency * 0.5,
        total_duration=base_latency * 1.2,
        req_per_sec=concurrency / (base_latency * 1.2) if base_latency > 0 else 0,
    )


def _simulate_quality_scores(prompts: List[TestPrompt]) -> List[QualityScore]:
    """Generate simulated quality scores for dry-run mode."""
    scores = []
    for p in prompts:
        hits = max(1, len(p.expected_keywords) - random.randint(0, 2))
        total = len(p.expected_keywords)
        scores.append(QualityScore(
            prompt_id=p.id,
            keyword_hits=hits,
            keyword_total=total,
            keyword_score=hits / total if total > 0 else 0,
            negative_hits=0,
            has_hallucination=False,
            response_length=random.randint(100, 500),
            is_relevant=True,
        ))
    return scores


def run_dry_run_benchmark(
    model_name: str = "deepseek-coder-6.7b",
    quantization: str = "awq",
    concurrency_levels: Optional[List[int]] = None,
    stress: bool = False,
) -> ModelBenchmarkResult:
    """Run a simulated benchmark in dry-run mode (no vLLM server needed)."""
    if concurrency_levels is None:
        concurrency_levels = [1, 10, 50]

    logger.info(f"[DRY-RUN] Simulating benchmark for {model_name} ({quantization})")

    c_results = [_simulate_concurrency_result(c) for c in concurrency_levels]
    prompts = get_test_prompts()
    q_scores = _simulate_quality_scores(prompts)

    avg_quality = statistics.mean(s.keyword_score for s in q_scores)
    hallucinations = sum(1 for s in q_scores if s.has_hallucination)
    relevance = sum(1 for s in q_scores if s.is_relevant) / len(q_scores) if q_scores else 0

    optimal_c = None
    plateau_rps = None
    if stress:
        stress_levels = [1, 5, 10, 20, 50]
        stress_results = [_simulate_concurrency_result(c) for c in stress_levels]
        c_results.extend(stress_results)
        optimal_c = 10
        plateau_rps = 25.0

    return ModelBenchmarkResult(
        model_name=model_name,
        quantization=quantization,
        endpoint="dry-run://localhost",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        concurrency_results=c_results,
        quality_scores=q_scores,
        avg_quality_score=avg_quality,
        hallucination_count=hallucinations,
        relevance_rate=relevance,
        optimal_concurrency=optimal_c,
        throughput_plateau_rps=plateau_rps,
    )


# ---------------------------------------------------------------------------
# Live Benchmark Orchestrator
# ---------------------------------------------------------------------------

async def run_full_benchmark(
    endpoint: str,
    model: str,
    quantization: str = "unknown",
    concurrency_levels: Optional[List[int]] = None,
    stress: bool = False,
) -> ModelBenchmarkResult:
    """Run the full benchmark suite against a live vLLM endpoint."""
    if concurrency_levels is None:
        concurrency_levels = [1, 10, 50]

    logger.info(f"Benchmarking {model} ({quantization}) at {endpoint}")
    prompts = get_test_prompts()
    prompt_texts = [p.prompt for p in prompts]

    # 1. Concurrency benchmarks
    c_results = []
    for c in concurrency_levels:
        cr = await run_concurrency_benchmark(endpoint, model, c, prompt_texts)
        c_results.append(cr)

    # 2. Quality evaluation
    logger.info("Running quality evaluation ...")
    quality_data = await run_quality_evaluation(endpoint, model, prompts)
    q_scores = [qd[0] for qd in quality_data]

    avg_quality = statistics.mean(s.keyword_score for s in q_scores) if q_scores else 0
    hallucinations = sum(1 for s in q_scores if s.has_hallucination)
    relevance = sum(1 for s in q_scores if s.is_relevant) / len(q_scores) if q_scores else 0

    # 3. Stress test (optional)
    optimal_c = None
    plateau_rps = None
    if stress:
        logger.info("Running stress test ...")
        stress_results, optimal_c, plateau_rps = await run_stress_test(
            endpoint, model, prompt_texts
        )
        c_results.extend(stress_results)

    return ModelBenchmarkResult(
        model_name=model,
        quantization=quantization,
        endpoint=endpoint,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        concurrency_results=c_results,
        quality_scores=q_scores,
        avg_quality_score=avg_quality,
        hallucination_count=hallucinations,
        relevance_rate=relevance,
        optimal_concurrency=optimal_c,
        throughput_plateau_rps=plateau_rps,
    )


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def result_to_dict(result: ModelBenchmarkResult) -> Dict[str, Any]:
    """Convert benchmark result to serializable dict."""
    return {
        "model_name": result.model_name,
        "quantization": result.quantization,
        "endpoint": result.endpoint,
        "timestamp": result.timestamp,
        "summary": {
            "avg_quality_score": round(result.avg_quality_score, 4),
            "hallucination_count": result.hallucination_count,
            "relevance_rate": round(result.relevance_rate, 4),
            "optimal_concurrency": result.optimal_concurrency,
            "throughput_plateau_rps": result.throughput_plateau_rps,
        },
        "concurrency_results": [asdict(cr) for cr in result.concurrency_results],
        "quality_scores": [asdict(qs) for qs in result.quality_scores],
    }


def format_comparison_table(results: List[ModelBenchmarkResult]) -> str:
    """Format results as a human-readable comparison table."""
    lines = []
    lines.append("")
    lines.append("=" * 110)
    lines.append("vLLM Model Benchmark Comparison Report")
    lines.append("=" * 110)
    lines.append("")

    # Header
    header = f"{'Model':<28} {'Quant':<6} {'P50(ms)':<9} {'P95(ms)':<9} {'P99(ms)':<9} {'RPS':<8} {'Quality':<9} {'Hallu':<6} {'Relev%':<8}"
    lines.append(header)
    lines.append("-" * 110)

    for r in results:
        # Use the first concurrency result with concurrency=10, or first available
        cr = None
        for c in r.concurrency_results:
            if c.concurrency == 10:
                cr = c
                break
        if cr is None and r.concurrency_results:
            cr = r.concurrency_results[0]

        p50 = f"{cr.p50_latency * 1000:.1f}" if cr else "N/A"
        p95 = f"{cr.p95_latency * 1000:.1f}" if cr else "N/A"
        p99 = f"{cr.p99_latency * 1000:.1f}" if cr else "N/A"
        rps = f"{cr.req_per_sec:.1f}" if cr else "N/A"
        quality = f"{r.avg_quality_score:.2%}"
        hallu = str(r.hallucination_count)
        relev = f"{r.relevance_rate:.0%}"

        lines.append(f"{r.model_name:<28} {r.quantization:<6} {p50:<9} {p95:<9} {p99:<9} {rps:<8} {quality:<9} {hallu:<6} {relev:<8}")

    lines.append("-" * 110)

    # Performance targets
    lines.append("")
    lines.append("Performance Targets: P95 < 100ms, Throughput > 20 req/s")

    # Stress test summary
    stress_models = [r for r in results if r.optimal_concurrency is not None]
    if stress_models:
        lines.append("")
        lines.append("Stress Test Results:")
        for r in stress_models:
            lines.append(f"  {r.model_name}: optimal concurrency={r.optimal_concurrency}, plateau RPS={r.throughput_plateau_rps:.1f}")

    # Quality breakdown by category
    lines.append("")
    lines.append("Quality Scores by Category:")
    if results:
        r = results[0]
        categories: Dict[str, List[float]] = {}
        for qs in r.quality_scores:
            pid = qs.prompt_id
            cat = pid.split("-")[0]
            cat_map = {"pc": "Part Classification", "ma": "Material Analysis",
                       "di": "Drawing Interp.", "tq": "Technical QA",
                       "pp": "Process Planning", "tf": "Tolerance & Fit",
                       "cx": "Complex"}
            cat_name = cat_map.get(cat, cat)
            categories.setdefault(cat_name, []).append(qs.keyword_score)
        for cat_name, scores in categories.items():
            avg = statistics.mean(scores)
            lines.append(f"  {cat_name:<22}: {avg:.2%}")

    lines.append("")
    lines.append("=" * 110)
    return "\n".join(lines)


def save_report(results: List[ModelBenchmarkResult], output_dir: Optional[str] = None) -> str:
    """Save benchmark report as JSON and print human-readable table.

    Returns:
        Path to the saved JSON report.
    """
    report_dir = Path(output_dir) if output_dir else REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    json_path = report_dir / f"vllm_benchmark_{ts}.json"

    report_data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_count": len(CAD_TEST_PROMPTS),
        "models": [result_to_dict(r) for r in results],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Print table
    table = format_comparison_table(results)
    print(table)

    logger.info(f"JSON report saved to {json_path}")
    return str(json_path)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="vLLM Model Selection & Quantization Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (CI mode, no vLLM needed):
  python3 scripts/benchmark_vllm_quantization.py --dry-run

  # Benchmark a specific model:
  python3 scripts/benchmark_vllm_quantization.py --model deepseek-coder-6.7b --endpoint http://localhost:8100

  # Compare all candidate models (requires each to be served in turn):
  python3 scripts/benchmark_vllm_quantization.py --compare-all --dry-run

  # Stress test:
  python3 scripts/benchmark_vllm_quantization.py --model deepseek-coder-6.7b --stress

  # Quantization sweep:
  python3 scripts/benchmark_vllm_quantization.py --model deepseek-coder-6.7b --quantization-sweep --dry-run
        """,
    )

    parser.add_argument("--endpoint", default="http://localhost:8100", help="vLLM API endpoint")
    parser.add_argument("--model", help="Model name (as served by vLLM)")
    parser.add_argument("--quantization", default="awq", help="Quantization method label (fp16, awq, gptq, int8)")
    parser.add_argument("--scenarios", default="1,10,50", help="Concurrency levels (comma separated)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate benchmark without a live vLLM server")
    parser.add_argument("--compare-all", action="store_true", help="Compare all candidate models")
    parser.add_argument("--stress", action="store_true", help="Run stress test with ramping concurrency")
    parser.add_argument("--quantization-sweep", action="store_true", help="Test FP16, AWQ, GPTQ, INT8 for one model")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--list-prompts", action="store_true", help="List all test prompts and exit")

    args = parser.parse_args()

    # --- List prompts ---
    if args.list_prompts:
        print(f"\nCAD Domain Test Suite ({len(CAD_TEST_PROMPTS)} prompts):\n")
        for p in CAD_TEST_PROMPTS:
            print(f"  [{p.id}] ({p.category}/{p.language}) {p.prompt}")
            print(f"         Expected: {', '.join(p.expected_keywords)}")
            if p.negative_keywords:
                print(f"         Negative: {', '.join(p.negative_keywords)}")
            print()
        return

    concurrency_levels = [int(x) for x in args.scenarios.split(",")]
    all_results: List[ModelBenchmarkResult] = []

    # --- Dry-run mode ---
    if args.dry_run:
        if args.compare_all:
            configs = load_model_configs()
            for mc in configs:
                result = run_dry_run_benchmark(
                    model_name=mc.name,
                    quantization=mc.recommended_quantization,
                    concurrency_levels=concurrency_levels,
                    stress=args.stress,
                )
                all_results.append(result)
        elif args.quantization_sweep:
            model_name = args.model or "deepseek-coder-6.7b"
            for quant in ["fp16", "awq", "gptq", "int8"]:
                result = run_dry_run_benchmark(
                    model_name=model_name,
                    quantization=quant,
                    concurrency_levels=concurrency_levels,
                )
                all_results.append(result)
        else:
            model_name = args.model or "deepseek-coder-6.7b"
            result = run_dry_run_benchmark(
                model_name=model_name,
                quantization=args.quantization,
                concurrency_levels=concurrency_levels,
                stress=args.stress,
            )
            all_results.append(result)

        report_path = save_report(all_results, args.output_dir)
        print(f"\n[DRY-RUN] Report saved to {report_path}")
        return

    # --- Live benchmark mode ---
    if not args.model and not args.compare_all:
        parser.error("--model is required for live benchmarks (or use --dry-run / --compare-all)")

    if args.compare_all:
        configs = load_model_configs()
        for mc in configs:
            try:
                result = asyncio.run(run_full_benchmark(
                    endpoint=args.endpoint,
                    model=mc.hf_id,
                    quantization=mc.recommended_quantization,
                    concurrency_levels=concurrency_levels,
                    stress=args.stress,
                ))
                all_results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed for {mc.name}: {e}")
    elif args.quantization_sweep:
        for quant in ["fp16", "awq", "gptq", "int8"]:
            try:
                result = asyncio.run(run_full_benchmark(
                    endpoint=args.endpoint,
                    model=args.model,
                    quantization=quant,
                    concurrency_levels=concurrency_levels,
                ))
                all_results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed for {args.model} ({quant}): {e}")
    else:
        try:
            result = asyncio.run(run_full_benchmark(
                endpoint=args.endpoint,
                model=args.model,
                quantization=args.quantization,
                concurrency_levels=concurrency_levels,
                stress=args.stress,
            ))
            all_results.append(result)
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

    if all_results:
        report_path = save_report(all_results, args.output_dir)
        print(f"\nReport saved to {report_path}")
    else:
        logger.error("No benchmark results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
