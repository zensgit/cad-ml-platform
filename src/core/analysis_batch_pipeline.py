"""Shared batch analyze helper for analyze flows."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Sequence

AnalyzeFileFn = Callable[..., Awaitable[Any]]


def _serialize_batch_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


async def run_batch_analysis(
    *,
    files: Sequence[Any],
    options: str,
    api_key: str,
    analyze_file_fn: AnalyzeFileFn,
) -> Dict[str, Any]:
    results: List[Any] = []

    for file in files:
        try:
            result = await analyze_file_fn(
                file=file,
                options=options,
                api_key=api_key,
            )
            results.append(_serialize_batch_result(result))
        except Exception as exc:
            results.append({"file_name": file.filename, "error": str(exc)})

    successful = len([item for item in results if "error" not in item])
    failed = len(results) - successful
    return {
        "total": len(files),
        "successful": successful,
        "failed": failed,
        "results": results,
    }


__all__ = ["run_batch_analysis"]
