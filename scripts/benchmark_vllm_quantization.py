#!/usr/bin/env python3
"""
vLLM Quantization Benchmark Script.

This script benchmarks different quantization methods (AWQ, GPTQ, SqueezeLLM)
supported by vLLM to find the optimal balance between latency and throughput
for the CAD ML Platform.

Usage:
    python3 scripts/benchmark_vllm_quantization.py --model deepseek-ai/deepseek-coder-6.7b-instruct
"""

import argparse
import time
import json
import logging
import sys
import os
import asyncio
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm_benchmark")

try:
    import aiohttp
    import numpy as np
except ImportError:
    logger.error("aiohttp and numpy are required. pip install aiohttp numpy")
    sys.exit(1)

# Test prompts relevant to CAD domain
TEST_PROMPTS = [
    "Analyze this CAD part: Hexagonal head, threaded shaft M12x1.5. Material: Steel 8.8. Classify and suggest usage.",
    "Extract geometric features from the following description: A cylindrical shaft with a keyway slot 5mm wide and 3mm deep.",
    "Identify the manufacturing standard for a washer with inner diameter 10.5mm, outer diameter 20mm, thickness 2mm.",
    "Explain the difference between a blind hole and a through hole in the context of CNC machining.",
    "Suggest a material for a high-temperature exhaust manifold gasket."
]

async def send_request(session: aiohttp.ClientSession, url: str, model: str, prompt: str) -> Dict[str, Any]:
    """Send a single request to vLLM and measure latency."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a mechanical engineering AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 256
    }
    
    start_time = time.perf_counter()
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            text = await response.text()
            logger.error(f"Request failed: {text}")
            return {"error": True, "latency": 0}
        
        result = await response.json()
        end_time = time.perf_counter()
        
        # Calculate tokens per second (approximate)
        content = result["choices"][0]["message"]["content"]
        # Rough token count estimate (char / 4)
        out_tokens = len(content) / 4
        
        latency = end_time - start_time
        return {
            "error": False,
            "latency": latency,
            "tokens": out_tokens,
            "tps": out_tokens / latency if latency > 0 else 0
        }

async def run_benchmark(endpoint: str, model: str, concurrency: int = 10) -> Dict[str, float]:
    """Run benchmark with specified concurrency."""
    url = f"{endpoint}/v1/chat/completions"
    logger.info(f"Benchmarking {model} at {endpoint} with concurrency {concurrency}...")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        # Generate requests
        for _ in range(concurrency):
            prompt = TEST_PROMPTS[0] # Use same prompt for consistency or rotate
            tasks.append(send_request(session, url, model, prompt))
            
        start_total = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_total = time.perf_counter()
        
    # Analyze results
    latencies = [r["latency"] for r in results if not r["error"]]
    tps_list = [r["tps"] for r in results if not r["error"]]
    
    if not latencies:
        logger.error("All requests failed.")
        return {}
        
    metrics = {
        "p50_latency": np.percentile(latencies, 50),
        "p95_latency": np.percentile(latencies, 95),
        "p99_latency": np.percentile(latencies, 99),
        "avg_tps": np.mean(tps_list),
        "total_duration": end_total - start_total,
        "req_per_sec": len(latencies) / (end_total - start_total)
    }
    
    logger.info(f"Results: P95 Latency={metrics['p95_latency']:.4f}s, TPS={metrics['avg_tps']:.2f}, RPS={metrics['req_per_sec']:.2f}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM Quantization")
    parser.add_argument("--endpoint", default="http://localhost:8000", help="vLLM API endpoint")
    parser.add_argument("--model", required=True, help="Model name deployed in vLLM")
    parser.add_argument("--scenarios", default="1,10,50", help="Concurrency levels to test (comma separated)")
    
    args = parser.parse_args()
    
    concurrency_levels = [int(x) for x in args.scenarios.split(",")]
    
    print(f"=== Benchmarking Model: {args.model} ===")
    print(f"Endpoint: {args.endpoint}")
    
    all_results = {}
    
    for c in concurrency_levels:
        try:
            metrics = asyncio.run(run_benchmark(args.endpoint, args.model, c))
            all_results[f"concurrency_{c}"] = metrics
        except Exception as e:
            logger.error(f"Benchmark failed for concurrency {c}: {e}")
            
    # Save report
    report_path = f"reports/vllm_benchmark_{int(time.time())}.json"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nBenchmark complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
