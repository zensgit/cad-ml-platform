from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.api.dependencies import get_api_key
from src.api.v1.analyze_live_models import BatchClassifyResponse
from src.core.analysis_batch_pipeline import run_batch_analysis
from src.core.classification import run_batch_classify_pipeline


def build_batch_router(
    *,
    analyze_file_fn: Callable[..., Awaitable[object]],
    logger_instance: logging.Logger,
) -> APIRouter:
    router = APIRouter()

    @router.post("/batch")
    async def batch_analyze(
        files: list[UploadFile] = File(..., description="多个CAD文件"),
        options: str = Form(default='{"extract_features": true}'),
        api_key: str = Depends(get_api_key),
    ):
        """批量分析CAD文件"""
        return await run_batch_analysis(
            files=files,
            options=options,
            api_key=api_key,
            analyze_file_fn=analyze_file_fn,
        )

    @router.post("/batch-classify", response_model=BatchClassifyResponse)
    async def batch_classify(
        files: list[UploadFile] = File(..., description="CAD文件列表(DXF/DWG)"),
        max_workers: Optional[int] = Form(default=None, description="并行工作线程数"),
        api_key: str = Depends(get_api_key),
    ):
        """
        批量分类CAD文件

        使用V16超级集成分类器并行处理多个文件，支持DXF和DWG格式。
        相比逐个调用，批量处理可获得约3倍性能提升。
        """
        result = await run_batch_classify_pipeline(
            files=files,
            max_workers=max_workers,
            logger=logger_instance,
        )
        return BatchClassifyResponse(**result)

    return router


__all__ = [
    "build_batch_router",
    "run_batch_analysis",
    "run_batch_classify_pipeline",
]
