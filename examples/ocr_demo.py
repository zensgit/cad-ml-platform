#!/usr/bin/env python3
"""
OCR End-to-End Demo
====================

This script demonstrates the OCR extraction pipeline:
1. Load a CAD image (or use synthetic data)
2. Call OcrManager directly
3. Display structured results
4. Show evaluation metrics

Run: python examples/ocr_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sample_image() -> bytes:
    """Create a simple 1x1 PNG image for testing."""
    # Minimal valid PNG (1x1 white pixel)
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8O"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


async def demo_ocr_extraction():
    """Demo: Direct OcrManager usage."""
    print("=" * 60)
    print("OCR Extraction Demo")
    print("=" * 60)

    # Import OCR components
    from src.core.ocr.manager import OcrManager
    from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider
    from src.core.ocr.providers.paddle import PaddleOcrProvider

    # Initialize manager with providers
    print("\n1. Initializing OCR Manager...")
    manager = OcrManager(confidence_fallback=0.85)
    manager.register_provider("paddle", PaddleOcrProvider())
    manager.register_provider("deepseek_hf", DeepSeekHfProvider())
    print("   - Registered providers: paddle, deepseek_hf")
    print("   - Confidence fallback threshold: 0.85")

    # Create sample image
    print("\n2. Loading sample image...")
    image_bytes = create_sample_image()
    print(f"   - Image size: {len(image_bytes)} bytes")

    # Run OCR extraction
    print("\n3. Running OCR extraction (strategy=auto)...")
    result = await manager.extract(image_bytes, strategy="auto", trace_id="demo-001")

    # Display results
    print("\n4. Extraction Results:")
    print("-" * 40)
    print(f"   Provider:            {result.provider}")
    print(f"   Confidence:          {result.confidence}")
    print(f"   Calibrated Conf:     {result.calibrated_confidence}")
    print(f"   Completeness:        {result.completeness}")
    print(f"   Fallback Level:      {result.fallback_level}")
    print(f"   Extraction Mode:     {result.extraction_mode}")
    print(f"   Processing Time:     {result.processing_time_ms} ms")
    print(f"   Image Hash:          {result.image_hash}")
    print(f"   Trace ID:            {result.trace_id}")

    # Dimensions
    print(f"\n5. Extracted Dimensions ({len(result.dimensions)} found):")
    print("-" * 40)
    if result.dimensions:
        for i, dim in enumerate(result.dimensions, 1):
            tol_str = ""
            if dim.tolerance:
                tol_str = f" +/-{dim.tolerance}"
            elif dim.tol_pos and dim.tol_neg:
                tol_str = f" +{dim.tol_pos}/-{dim.tol_neg}"
            pitch_str = f" (pitch={dim.pitch})" if dim.pitch else ""
            print(f"   {i}. {dim.type.value}: {dim.value}{dim.unit}{tol_str}{pitch_str}")
            if dim.raw:
                print(f"      Raw: '{dim.raw}'")
    else:
        print("   (No dimensions found)")

    # Symbols
    print(f"\n6. Extracted Symbols ({len(result.symbols)} found):")
    print("-" * 40)
    if result.symbols:
        for i, sym in enumerate(result.symbols, 1):
            print(f"   {i}. {sym.type.value}: {sym.value}")
    else:
        print("   (No symbols found)")

    # Title Block
    print("\n7. Title Block:")
    print("-" * 40)
    tb = result.title_block
    print(f"   Drawing Number: {tb.drawing_number or '(not found)'}")
    print(f"   Material:       {tb.material or '(not found)'}")
    print(f"   Part Name:      {tb.part_name or '(not found)'}")
    print(f"   Scale:          {tb.scale or '(not found)'}")

    # Stage latencies
    print("\n8. Stage Latencies:")
    print("-" * 40)
    if result.stages_latency_ms:
        for stage, ms in result.stages_latency_ms.items():
            print(f"   {stage:15s}: {ms:6d} ms")
    else:
        print("   (No stage timing data)")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    return result


async def demo_golden_evaluation():
    """Demo: Running golden evaluation."""
    print("\n" + "=" * 60)
    print("Golden Evaluation Demo")
    print("=" * 60)

    print("\n1. Running golden evaluation script...")
    print("   Command: python tests/ocr/run_golden_evaluation.py")

    # Import and run evaluation
    import subprocess

    result = subprocess.run(
        ["python3", "tests/ocr/run_golden_evaluation.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    print("\n2. Evaluation Results:")
    print("-" * 40)
    for line in result.stdout.strip().split("\n"):
        parts = line.split("=")
        if len(parts) == 2:
            metric, value = parts
            print(f"   {metric:20s}: {value}")

    # Check thresholds
    print("\n3. Week 1 Threshold Comparison:")
    print("-" * 40)
    thresholds = {"dimension_recall": 0.70, "brier_score": 0.20, "edge_f1": 0.60}

    for line in result.stdout.strip().split("\n"):
        metric, value = line.split("=")
        val = float(value)
        if metric in thresholds:
            threshold = thresholds[metric]
            if metric == "brier_score":
                passed = val < threshold
                comp = "<"
            else:
                passed = val >= threshold
                comp = ">="
            status = "PASS" if passed else "FAIL"
            print(f"   {metric}: {val:.3f} {comp} {threshold:.2f} [{status}]")

    print("\n" + "=" * 60)
    print("Golden Evaluation Complete!")
    print("=" * 60)


async def demo_idempotency():
    """Demo: Idempotency key support."""
    print("\n" + "=" * 60)
    print("Idempotency Key Demo")
    print("=" * 60)

    from src.utils.idempotency import build_idempotency_key, check_idempotency, store_idempotency

    print("\n1. Building idempotency key...")
    idem_key = build_idempotency_key("demo-request-123", endpoint="ocr")
    print(f"   Key: {idem_key}")

    print("\n2. Checking cache (should be empty)...")
    cached = await check_idempotency("demo-request-123", endpoint="ocr")
    print(f"   Cached: {cached}")

    print("\n3. Simulating response storage...")
    sample_response = {
        "provider": "paddle",
        "confidence": 0.88,
        "fallback_level": None,
        "processing_time_ms": 150,
        "dimensions": [{"type": "diameter", "value": 20.0}],
        "symbols": [],
        "title_block": {},
    }
    # Note: This would store in Redis if connected
    # await store_idempotency("demo-request-123", sample_response, endpoint="ocr")
    print("   (Skipped - Redis not connected in demo)")

    print("\n4. API Usage Example:")
    print("-" * 40)
    print(
        """
    curl -X POST \\
      -H "Idempotency-Key: unique-request-123" \\
      -F "file=@drawing.png" \\
      "http://localhost:8000/api/v1/ocr/extract"

    # Second request with same key returns cached response
    curl -X POST \\
      -H "Idempotency-Key: unique-request-123" \\
      -F "file=@drawing.png" \\
      "http://localhost:8000/api/v1/ocr/extract"
    """
    )

    print("=" * 60)
    print("Idempotency Demo Complete!")
    print("=" * 60)


async def main():
    """Run all demos."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 20 + "OCR DEMO SUITE" + " " * 22 + "*")
    print("*" * 60)

    # Demo 1: OCR Extraction
    await demo_ocr_extraction()

    # Demo 2: Golden Evaluation
    await demo_golden_evaluation()

    # Demo 3: Idempotency
    await demo_idempotency()

    print("\n" + "*" * 60)
    print("*" + " " * 17 + "ALL DEMOS COMPLETE!" + " " * 20 + "*")
    print("*" * 60)
    print("\nNext Steps:")
    print("  1. Install PaddleOCR for real OCR: pip install paddleocr")
    print("  2. Start API server: make run")
    print("  3. Test endpoint: curl -F 'file=@image.png' localhost:8000/api/v1/ocr/extract")
    print("  4. Run full test suite: pytest tests/ocr/ -v")
    print("  5. View metrics: http://localhost:8000/metrics")
    print()


if __name__ == "__main__":
    asyncio.run(main())
