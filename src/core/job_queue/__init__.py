"""Job Queue Module.

Provides distributed job queue capabilities:
- Priority-based job scheduling
- Retry with backoff
- Dead letter handling
- Worker management
"""

from src.core.job_queue.core import (
    JobState,
    JobPriority,
    JobOptions,
    Job,
    create_job,
    JobQueue,
    QueueStats,
    JobHandler,
    FunctionHandler,
    HandlerRegistry,
)
from src.core.job_queue.backends import (
    InMemoryJobQueue,
    RedisJobQueue,
)
from src.core.job_queue.worker import (
    WorkerConfig,
    WorkerStats,
    Worker,
    WorkerPool,
    setup_signal_handlers,
)

__all__ = [
    # Core
    "JobState",
    "JobPriority",
    "JobOptions",
    "Job",
    "create_job",
    "JobQueue",
    "QueueStats",
    "JobHandler",
    "FunctionHandler",
    "HandlerRegistry",
    # Backends
    "InMemoryJobQueue",
    "RedisJobQueue",
    # Worker
    "WorkerConfig",
    "WorkerStats",
    "Worker",
    "WorkerPool",
    "setup_signal_handlers",
]
