"""Structured logging setup for OCR subsystem."""

import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Common structured fields emitted by OCR subsystem
        for attr in [
            "provider",
            "image_hash",
            "latency_ms",
            "fallback_level",
            "error_code",
            "stage",
            "trace_id",
            "extraction_mode",
            "completeness",
            "calibrated_confidence",
            "dimensions_count",
            "symbols_count",
            "stages_latency_ms",
        ]:
            if hasattr(record, attr):
                data[attr] = getattr(record, attr)
        return json.dumps(data, ensure_ascii=False)


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
