"""Graph2D temperature scaling loader.

This module is intentionally torch-free so it can be used from lightweight
code paths such as health payload builders without importing the full 2D
vision stack.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_graph2d_temperature_settings(
    *,
    env_temperature: Optional[str] = None,
    env_calibration_path: Optional[str] = None,
) -> Tuple[float, Optional[str]]:
    """Load effective Graph2D temperature scaling settings.

    Precedence:
    1) `GRAPH2D_TEMPERATURE` (if valid > 0)
    2) `GRAPH2D_TEMPERATURE_CALIBRATION_PATH` JSON file (key: `temperature`)
    3) default: 1.0

    Returns:
        (temperature, source)

        `source` values:
        - "env" when loaded from `GRAPH2D_TEMPERATURE`
        - "<path>" when loaded from calibration file
        - None when using default (or when configuration is invalid)
    """
    temp_raw = (
        env_temperature
        if env_temperature is not None
        else os.getenv("GRAPH2D_TEMPERATURE")
    )
    if temp_raw:
        try:
            temp = float(str(temp_raw).strip())
        except (TypeError, ValueError):
            logger.warning("Invalid GRAPH2D_TEMPERATURE=%s", temp_raw)
        else:
            if temp > 0:
                return temp, "env"
            logger.warning("GRAPH2D_TEMPERATURE must be > 0; got %s", temp_raw)
        return 1.0, None

    calibration_path = (
        env_calibration_path
        if env_calibration_path is not None
        else os.getenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH")
    )
    if not calibration_path:
        return 1.0, None

    path = Path(str(calibration_path))
    if not path.exists():
        logger.warning("Graph2D calibration file not found: %s", path)
        return 1.0, None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse Graph2D calibration file: %s", exc)
        return 1.0, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read Graph2D calibration file: %s", exc)
        return 1.0, None

    if not isinstance(payload, dict):
        logger.warning(
            "Invalid Graph2D calibration payload (expected object): %s",
            type(payload),
        )
        return 1.0, None

    temp = payload.get("temperature")
    try:
        temp_val = float(temp)
    except (TypeError, ValueError):
        logger.warning("Invalid temperature in calibration file: %s", temp)
        return 1.0, None

    if temp_val <= 0:
        logger.warning("Calibration temperature must be > 0; got %s", temp_val)
        return 1.0, None

    return temp_val, str(path)
