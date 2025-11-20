"""PaddleOCR parameter A/B tuning harness.

Runs multiple configurations on the same synthetic / placeholder image bytes
to compare dimension recall proxy (regex extraction), symbol recall and latency.

Note: For MVP we use a stub image; integrate real samples when available.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Any, List

from src.core.ocr.manager import OcrManager
from src.core.ocr.providers.paddle import PaddleOcrProvider
from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider


@dataclass
class TrialResult:
    name: str
    latency_ms: float
    dimensions_count: int
    symbols_count: int
    extraction_mode: str


CONFIGS = [
    ("baseline", {"lang": "ch", "use_angle_cls": True, "use_gpu": False}),
    ("higher_box_thresh", {"lang": "ch", "use_angle_cls": True, "use_gpu": False, "det_db_box_thresh": 0.7}),
    ("lower_box_thresh", {"lang": "ch", "use_angle_cls": True, "use_gpu": False, "det_db_box_thresh": 0.3}),
    ("unclip_ratio_high", {"lang": "ch", "use_angle_cls": True, "use_gpu": False, "det_db_unclip_ratio": 2.0}),
]


async def run_trial(name: str, paddle_kwargs: Dict[str, Any]) -> TrialResult:
    manager = OcrManager(providers={
        "paddle": PaddleOcrProvider(enable_preprocess=True, **paddle_kwargs),
        "deepseek_hf": DeepSeekHfProvider(),
    })
    image_bytes = b"ab_tune_stub_image"
    start = time.time()
    result = await manager.extract(image_bytes, strategy="paddle")
    return TrialResult(
        name=name,
        latency_ms=(time.time() - start) * 1000.0,
        dimensions_count=len(result.dimensions),
        symbols_count=len(result.symbols),
        extraction_mode=result.extraction_mode or "unknown",
    )


async def main():
    results: List[TrialResult] = []
    for name, cfg in CONFIGS:
        try:
            tr = await run_trial(name, cfg)
            results.append(tr)
        except Exception as e:
            results.append(TrialResult(name=name, latency_ms=-1, dimensions_count=-1, symbols_count=-1, extraction_mode=f"error:{e}"))
    print("name,latency_ms,dimensions_count,symbols_count,extraction_mode")
    for r in results:
        print(f"{r.name},{r.latency_ms:.2f},{r.dimensions_count},{r.symbols_count},{r.extraction_mode}")


if __name__ == "__main__":
    asyncio.run(main())

