"""
Pipeline orchestrator for managing E2E ML workflows.

Provides execution control, state management, and error handling.
"""

from __future__ import annotations

import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.ml.pipeline.stages import PipelineStage, StageResult, StageStatus

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    name: str = "default_pipeline"
    description: str = ""
    stop_on_error: bool = True
    save_intermediate: bool = False
    checkpoint_dir: Optional[str] = None
    max_retries: int = 0
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    verbose: bool = True


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_name: str
    status: PipelineStatus
    stage_results: List[StageResult] = field(default_factory=list)
    final_output: Any = None
    total_time: float = 0.0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def success(self) -> bool:
        return self.status == PipelineStatus.COMPLETED

    @property
    def failed_stages(self) -> List[StageResult]:
        return [r for r in self.stage_results if r.status == StageStatus.FAILED]

    @property
    def completed_stages(self) -> List[StageResult]:
        return [r for r in self.stage_results if r.status == StageStatus.COMPLETED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "total_time": round(self.total_time, 3),
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "stages": [r.to_dict() for r in self.stage_results],
            "summary": {
                "total_stages": len(self.stage_results),
                "completed": len(self.completed_stages),
                "failed": len(self.failed_stages),
            },
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save result to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class Pipeline:
    """
    Pipeline orchestrator for E2E ML workflows.

    Manages stage execution, error handling, checkpointing,
    and state persistence.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self._config = config or PipelineConfig()
        self._stages: List[PipelineStage] = []
        self._context: Dict[str, Any] = {}
        self._status = PipelineStatus.PENDING
        self._result: Optional[PipelineResult] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "on_start": [],
            "on_stage_start": [],
            "on_stage_complete": [],
            "on_complete": [],
            "on_error": [],
        }

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def stages(self) -> List[PipelineStage]:
        return self._stages.copy()

    @property
    def status(self) -> PipelineStatus:
        return self._status

    @property
    def context(self) -> Dict[str, Any]:
        return self._context

    @property
    def result(self) -> Optional[PipelineResult]:
        return self._result

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """
        Add a stage to the pipeline.

        Args:
            stage: Pipeline stage to add

        Returns:
            Self for chaining
        """
        self._stages.append(stage)
        return self

    def insert_stage(self, index: int, stage: PipelineStage) -> "Pipeline":
        """
        Insert a stage at a specific position.

        Args:
            index: Position to insert
            stage: Pipeline stage to insert

        Returns:
            Self for chaining
        """
        self._stages.insert(index, stage)
        return self

    def remove_stage(self, name: str) -> bool:
        """
        Remove a stage by name.

        Args:
            name: Stage name

        Returns:
            True if stage was removed
        """
        for i, stage in enumerate(self._stages):
            if stage.name == name:
                self._stages.pop(i)
                return True
        return False

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get stage by name."""
        for stage in self._stages:
            if stage.name == name:
                return stage
        return None

    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> "Pipeline":
        """
        Register an event callback.

        Args:
            event: Event name (on_start, on_stage_start, on_stage_complete, on_complete, on_error)
            callback: Callback function

        Returns:
            Self for chaining
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        return self

    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """Trigger event callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def set_context(self, key: str, value: Any) -> "Pipeline":
        """Set context value."""
        self._context[key] = value
        return self

    def update_context(self, values: Dict[str, Any]) -> "Pipeline":
        """Update multiple context values."""
        self._context.update(values)
        return self

    def run(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        start_from: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            input_data: Initial input data
            start_from: Stage name to start from (for resuming)

        Returns:
            PipelineResult
        """
        if not self._stages:
            raise ValueError("Pipeline has no stages")

        self._status = PipelineStatus.RUNNING
        start_time = time.time()
        started_at = datetime.now()

        input_data = input_data or {}
        stage_results = []
        current_data = input_data

        # Find starting stage
        start_index = 0
        if start_from:
            for i, stage in enumerate(self._stages):
                if stage.name == start_from:
                    start_index = i
                    break

        self._trigger_callbacks("on_start", self, input_data)

        if self._config.verbose:
            logger.info(f"Starting pipeline: {self._config.name}")
            logger.info(f"Stages: {[s.name for s in self._stages]}")

        try:
            for i, stage in enumerate(self._stages[start_index:], start=start_index):
                if self._status == PipelineStatus.CANCELLED:
                    break

                self._trigger_callbacks("on_stage_start", stage, current_data)

                # Execute stage with retries
                result = self._execute_with_retry(stage, current_data)
                stage_results.append(result)

                self._trigger_callbacks("on_stage_complete", stage, result)

                # Save checkpoint if enabled
                if self._config.save_intermediate and self._config.checkpoint_dir:
                    self._save_checkpoint(stage.name, current_data, result)

                # Handle failure
                if result.status == StageStatus.FAILED:
                    if self._config.stop_on_error:
                        self._status = PipelineStatus.FAILED
                        self._trigger_callbacks("on_error", stage, result.error)
                        break
                    else:
                        logger.warning(f"Stage {stage.name} failed, continuing...")

                # Update current data with output
                if result.output is not None:
                    if isinstance(result.output, dict):
                        current_data = result.output
                    else:
                        current_data["stage_output"] = result.output

            # Determine final status
            if self._status == PipelineStatus.RUNNING:
                failed = any(r.status == StageStatus.FAILED for r in stage_results)
                self._status = PipelineStatus.FAILED if failed else PipelineStatus.COMPLETED

            total_time = time.time() - start_time

            self._result = PipelineResult(
                pipeline_name=self._config.name,
                status=self._status,
                stage_results=stage_results,
                final_output=current_data,
                total_time=total_time,
                started_at=started_at,
                completed_at=datetime.now(),
            )

            self._trigger_callbacks("on_complete", self, self._result)

            if self._config.verbose:
                logger.info(f"Pipeline {self._config.name} completed: {self._status.value}")
                logger.info(f"Total time: {total_time:.2f}s")

        except Exception as e:
            self._status = PipelineStatus.FAILED
            total_time = time.time() - start_time

            self._result = PipelineResult(
                pipeline_name=self._config.name,
                status=PipelineStatus.FAILED,
                stage_results=stage_results,
                error=str(e),
                total_time=total_time,
                started_at=started_at,
                completed_at=datetime.now(),
            )

            self._trigger_callbacks("on_error", None, str(e))
            logger.error(f"Pipeline failed: {e}")

        return self._result

    def _execute_with_retry(
        self,
        stage: PipelineStage,
        input_data: Dict[str, Any],
    ) -> StageResult:
        """Execute stage with retry logic."""
        last_result = None

        for attempt in range(self._config.max_retries + 1):
            result = stage.execute(input_data, self._context)

            if result.status != StageStatus.FAILED:
                return result

            last_result = result

            if attempt < self._config.max_retries:
                logger.warning(
                    f"Stage {stage.name} failed (attempt {attempt + 1}), "
                    f"retrying in {self._config.retry_delay}s..."
                )
                time.sleep(self._config.retry_delay)

        return last_result or result

    def _save_checkpoint(
        self,
        stage_name: str,
        data: Dict[str, Any],
        result: StageResult,
    ) -> None:
        """Save checkpoint for stage."""
        if not self._config.checkpoint_dir:
            return

        checkpoint_dir = Path(self._config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "stage": stage_name,
            "result": result.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_{stage_name}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def pause(self) -> None:
        """Pause pipeline execution."""
        if self._status == PipelineStatus.RUNNING:
            self._status = PipelineStatus.PAUSED
            logger.info("Pipeline paused")

    def resume(self) -> None:
        """Resume paused pipeline."""
        if self._status == PipelineStatus.PAUSED:
            self._status = PipelineStatus.RUNNING
            logger.info("Pipeline resumed")

    def cancel(self) -> None:
        """Cancel pipeline execution."""
        if self._status in (PipelineStatus.RUNNING, PipelineStatus.PAUSED):
            self._status = PipelineStatus.CANCELLED
            logger.info("Pipeline cancelled")

    def reset(self) -> None:
        """Reset pipeline state."""
        self._status = PipelineStatus.PENDING
        self._result = None
        self._context.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary."""
        return {
            "name": self._config.name,
            "description": self._config.description,
            "status": self._status.value,
            "stages": [
                {"name": s.name, "status": s.status.value}
                for s in self._stages
            ],
            "has_result": self._result is not None,
        }

    def __repr__(self) -> str:
        return f"Pipeline({self._config.name}, stages={len(self._stages)}, status={self._status.value})"

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self):
        return iter(self._stages)
