"""Default live recall: private dedupcad-vision search → candidate hits.

Only used when ``REVIEW_REUSE_LIVE_DEDUP`` is enabled. Failures surface as
empty/raise for the adapter to map to insufficient_evidence reasons.
Does not call training paths or eval_integrity_gate.
"""

from __future__ import annotations

import asyncio
import mimetypes
from typing import Any, Dict, List, Optional


def vision_response_to_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Map dedup_2d / vision search payload to adapter hit dicts."""
    hits: List[Dict[str, Any]] = []
    if not isinstance(response, dict):
        return hits

    for bucket, default_state in (
        ("duplicates", "duplicate"),
        ("similar", "similar"),
    ):
        rows = response.get(bucket) or []
        if not isinstance(rows, list):
            continue
        for match in rows:
            if not isinstance(match, dict):
                continue
            verdict = str(match.get("verdict") or default_state)
            visual = match.get("visual_similarity")
            if visual is None:
                visual = match.get("similarity")
            geom = match.get("precision_score")
            if geom is None:
                geom = match.get("similarity")
            methods = ["dedup2d-vision"]
            levels = match.get("levels") or {}
            if isinstance(levels, dict) and levels.get("l4"):
                methods.append("precision-l4")
            hits.append(
                {
                    "candidate_id": str(
                        match.get("file_hash")
                        or match.get("drawing_id")
                        or match.get("file_name")
                        or match.get("id")
                        or f"live-{len(hits)}"
                    ),
                    "candidate_source": "archive",
                    "state": verdict,
                    "scores": {
                        "geometric": geom,
                        "semantic": visual,
                        "visual": visual,
                    },
                    "match_level": match.get("match_level", 0),
                    "methods": methods,
                    "verification": {
                        "verdict": verdict,
                        "level": match.get("match_level", 0),
                        "methods": methods,
                    },
                    "decision_source": match.get("decision_source") or "dedup2d-vision",
                    "rejection_reasons": list(match.get("rejection_reasons") or []),
                }
            )
    return hits


def _guess_content_type(file_name: str) -> str:
    ctype, _ = mimetypes.guess_type(file_name or "")
    return ctype or "application/octet-stream"


def default_live_recall(
    file_name: str, file_bytes: bytes, content_sha: str
) -> List[Dict[str, Any]]:
    """Sync entry: run vision search_2d (async under the hood).

    Raises on hard failure so ``recall_candidates`` can map to
    ``external_service_unavailable``. Empty list → tool_unavailable.
    """
    del content_sha  # used by caller for provenance only

    async def _search() -> Dict[str, Any]:
        from src.core.dedupcad_vision import DedupCadVisionClient

        client = DedupCadVisionClient()
        return await client.search_2d(
            file_name=file_name or "upload.bin",
            file_bytes=file_bytes,
            content_type=_guess_content_type(file_name),
            mode="balanced",
            max_results=20,
            compute_diff=False,
            enable_ml=False,
            enable_geometric=False,
        )

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Nested loop (e.g. already in async context): use a worker thread.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    response = pool.submit(lambda: asyncio.run(_search())).result(
                        timeout=120
                    )
            else:
                response = loop.run_until_complete(_search())
        except RuntimeError:
            response = asyncio.run(_search())
    except Exception:
        raise

    return vision_response_to_hits(response if isinstance(response, dict) else {})


def ensure_default_live_hook() -> None:
    """Install default vision recall when live mode is on and no hook set."""
    from .dedup_adapter import get_live_recall_hook, live_dedup_enabled, set_live_recall_hook

    if live_dedup_enabled() and get_live_recall_hook() is None:
        set_live_recall_hook(default_live_recall)
