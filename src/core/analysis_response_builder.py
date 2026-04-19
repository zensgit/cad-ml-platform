from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Type


FinalizeAnalysisSuccess = Callable[..., Awaitable[Dict[str, Any]]]


async def build_analysis_response(
    *,
    result_model_cls: Type[Any],
    finalize_analysis_success_fn: FinalizeAnalysisSuccess,
    analysis_id: str,
    start_time: Any,
    file_name: str,
    file_format: str,
    results: Dict[str, Any],
    doc: Any,
    stage_times: Dict[str, float],
    analysis_cache_key: str,
    vector_context: Dict[str, Any],
    material: str | None,
    unified_data: Dict[str, Any],
    logger_instance: Any,
) -> Any:
    response_payload = await finalize_analysis_success_fn(
        analysis_id=analysis_id,
        start_time=start_time,
        file_name=file_name,
        file_format=file_format,
        results=results,
        doc=doc,
        stage_times=stage_times,
        analysis_cache_key=analysis_cache_key,
        vector_context=vector_context,
        material=material,
        unified_data=unified_data,
        logger_instance=logger_instance,
    )
    return result_model_cls(**response_payload)
