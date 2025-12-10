"""Arq worker definitions for CAD analysis tasks.

This module defines the worker tasks that process jobs from the queue.
Run with: arq src.core.tasks.worker.WorkerSettings

Example:
    # Start worker
    $ arq src.core.tasks.worker.WorkerSettings

    # Or programmatically
    >>> from arq import run_worker
    >>> run_worker(WorkerSettings)
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Conditional import for arq
try:
    from arq import cron
    from arq.connections import RedisSettings

    ARQ_AVAILABLE = True
except ImportError:
    ARQ_AVAILABLE = False
    cron = None
    RedisSettings = None


# ============================================================================
# Task Functions
# ============================================================================


async def analyze_cad_file(
    ctx: dict[str, Any],
    file_path: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Analyze a CAD file.

    This task replaces the synchronous analysis pipeline,
    enabling distributed processing across multiple workers.

    Args:
        ctx: Worker context with initialized dependencies.
        file_path: Path to the CAD file.
        options: Analysis options.

    Returns:
        Analysis results dictionary.
    """
    logger.info(f"Analyzing CAD file: {file_path}")

    try:
        analyzer = ctx.get("analyzer")
        if analyzer is None:
            # Lazy initialization if not in context
            from src.core.cad_analyzer import CADAnalyzer
            analyzer = CADAnalyzer()

        result = await analyzer.analyze(file_path, **options)

        logger.info(f"Analysis complete for {file_path}")
        return {
            "status": "success",
            "file_path": file_path,
            "result": result.to_dict() if hasattr(result, "to_dict") else result,
        }
    except Exception as e:
        logger.error(f"Analysis failed for {file_path}: {e}")
        return {
            "status": "error",
            "file_path": file_path,
            "error": str(e),
        }


async def extract_features(
    ctx: dict[str, Any],
    document_id: str,
    feature_types: list[str],
) -> dict[str, Any]:
    """Extract features from a CAD document.

    Args:
        ctx: Worker context.
        document_id: Document identifier.
        feature_types: Types of features to extract.

    Returns:
        Extracted features dictionary.
    """
    logger.info(f"Extracting features for document: {document_id}")

    try:
        extractor = ctx.get("extractor")
        if extractor is None:
            from src.core.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor()

        features = await extractor.extract(document_id, feature_types=feature_types)

        return {
            "status": "success",
            "document_id": document_id,
            "features": features,
        }
    except Exception as e:
        logger.error(f"Feature extraction failed for {document_id}: {e}")
        return {
            "status": "error",
            "document_id": document_id,
            "error": str(e),
        }


async def search_similar(
    ctx: dict[str, Any],
    query_vector: list[float],
    top_k: int,
    filter_conditions: dict[str, Any],
) -> dict[str, Any]:
    """Search for similar vectors.

    Args:
        ctx: Worker context.
        query_vector: Query feature vector.
        top_k: Number of results.
        filter_conditions: Metadata filters.

    Returns:
        Search results dictionary.
    """
    logger.info(f"Searching for {top_k} similar vectors")

    try:
        vector_store = ctx.get("vector_store")
        if vector_store is None:
            from src.core.vector_stores import get_vector_store
            vector_store = get_vector_store()

        results = await vector_store.search_similar(
            query_vector=query_vector,
            top_k=top_k,
            filter_conditions=filter_conditions if filter_conditions else None,
        )

        return {
            "status": "success",
            "results": [
                {"id": r.id, "score": r.score, "metadata": r.metadata}
                for r in results
            ],
        }
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


async def batch_register_vectors(
    ctx: dict[str, Any],
    vectors: list[tuple[str, list[float], dict[str, Any] | None]],
) -> dict[str, Any]:
    """Batch register vectors to the store.

    Args:
        ctx: Worker context.
        vectors: List of (id, vector, metadata) tuples.

    Returns:
        Registration result.
    """
    logger.info(f"Batch registering {len(vectors)} vectors")

    try:
        vector_store = ctx.get("vector_store")
        if vector_store is None:
            from src.core.vector_stores import get_vector_store
            vector_store = get_vector_store()

        count = await vector_store.register_vectors_batch(vectors)

        return {
            "status": "success",
            "registered_count": count,
        }
    except Exception as e:
        logger.error(f"Batch registration failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


# ============================================================================
# Scheduled Tasks (Cron)
# ============================================================================


async def cleanup_expired_vectors(ctx: dict[str, Any]) -> dict[str, Any]:
    """Periodic cleanup of expired vectors.

    This runs as a cron job to remove stale data.
    """
    logger.info("Running vector cleanup task")

    try:
        # Implement cleanup logic here
        return {"status": "success", "cleaned": 0}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"status": "error", "error": str(e)}


async def health_check(ctx: dict[str, Any]) -> dict[str, Any]:
    """Periodic health check task."""
    return {"status": "healthy", "worker_id": ctx.get("worker_id", "unknown")}


# ============================================================================
# Worker Configuration
# ============================================================================


class WorkerSettings:
    """Arq worker settings.

    This class configures the worker behavior including:
    - Available task functions
    - Redis connection settings
    - Concurrency limits
    - Retry behavior
    - Startup/shutdown hooks
    """

    # Task functions available to workers
    functions = [
        analyze_cad_file,
        extract_features,
        search_similar,
        batch_register_vectors,
        health_check,
    ]

    # Cron jobs (scheduled tasks)
    cron_jobs = [
        # Run cleanup every hour
        # cron(cleanup_expired_vectors, hour={0, 6, 12, 18}, minute=0),
    ] if ARQ_AVAILABLE and cron else []

    # Redis connection
    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        database=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD"),
    ) if ARQ_AVAILABLE else None

    # Concurrency settings
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", "10"))
    job_timeout = timedelta(seconds=int(os.getenv("ARQ_JOB_TIMEOUT", "300")))

    # Retry settings
    max_tries = int(os.getenv("ARQ_MAX_RETRIES", "3"))
    retry_jobs = os.getenv("ARQ_RETRY_JOBS", "true").lower() == "true"

    # Queue name
    queue_name = os.getenv("ARQ_QUEUE_NAME", "arq:queue")

    @staticmethod
    async def on_startup(ctx: dict[str, Any]) -> None:
        """Initialize worker context on startup.

        This is called once when the worker starts. Use it to
        initialize expensive resources like database connections.
        """
        logger.info("Worker starting up...")

        # Initialize shared resources
        try:
            # Vector store (lazy - will be created on first use)
            ctx["vector_store"] = None

            # Feature extractor (lazy)
            ctx["extractor"] = None

            # CAD analyzer (lazy)
            ctx["analyzer"] = None

            logger.info("Worker startup complete")
        except Exception as e:
            logger.error(f"Worker startup failed: {e}")
            raise

    @staticmethod
    async def on_shutdown(ctx: dict[str, Any]) -> None:
        """Cleanup worker context on shutdown.

        This is called when the worker is shutting down.
        Use it to close connections and release resources.
        """
        logger.info("Worker shutting down...")

        # Close resources
        if ctx.get("vector_store"):
            try:
                ctx["vector_store"].close()
            except Exception as e:
                logger.warning(f"Error closing vector store: {e}")

        logger.info("Worker shutdown complete")

    @staticmethod
    async def on_job_start(ctx: dict[str, Any]) -> None:
        """Called at the start of each job."""
        logger.debug(f"Starting job: {ctx.get('job_id', 'unknown')}")

    @staticmethod
    async def on_job_end(ctx: dict[str, Any]) -> None:
        """Called at the end of each job."""
        logger.debug(f"Completed job: {ctx.get('job_id', 'unknown')}")
