from __future__ import annotations

import logging

from src.core.analysis_manufacturing_summary import (
    attach_manufacturing_decision_summary,
)


def test_attach_manufacturing_decision_summary_writes_summary():
    results = {
        "quality": {"manufacturability": "high"},
        "process": {"primary_recommendation": {"process": "turning"}},
        "cost_estimation": {"total_unit_cost": 15.0},
    }
    captured: dict[str, object] = {}

    def _builder(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {"feasibility": "high"}

    summary = attach_manufacturing_decision_summary(
        results=results,
        summary_builder=_builder,
        logger_instance=logging.getLogger("test"),
    )

    assert summary == {"feasibility": "high"}
    assert results["manufacturing_decision"] == {"feasibility": "high"}
    assert captured == {
        "quality_payload": {"manufacturability": "high"},
        "process_payload": {"primary_recommendation": {"process": "turning"}},
        "cost_payload": {"total_unit_cost": 15.0},
    }


def test_attach_manufacturing_decision_summary_logs_and_returns_none(caplog):
    results = {
        "quality": {"manufacturability": "medium"},
        "process": {"primary_recommendation": {"process": "milling"}},
    }

    def _builder(**_kwargs):  # noqa: ANN003, ANN201
        raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING):
        summary = attach_manufacturing_decision_summary(
            results=results,
            summary_builder=_builder,
            logger_instance=logging.getLogger("test"),
        )

    assert summary is None
    assert "manufacturing_decision" not in results
    assert "Manufacturing decision summary failed: boom" in caplog.text
