"""Tests for similarity degraded/restored metrics instrumentation.

Verifies that similarity_degraded_total counter increments on degraded and
restored events when requesting Faiss backend via get_vector_store().
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_state():
    from src.core.similarity import reset_default_store

    reset_default_store()
    yield
    reset_default_store()


def _read_metric_counter(counter) -> int:
    # Helper to read Prometheus counter value robustly
    try:
        return int(getattr(counter, "_value").get())  # type: ignore[attr-defined]
    except Exception:
        # Fallback; some dummy counters may expose simple attribute
        try:
            return int(counter._value.get())  # type: ignore
        except Exception:
            return 0


def _skip_if_metrics_disabled(counter) -> None:
    if not hasattr(counter, "_value"):
        pytest.skip("Metrics disabled or prometheus_client unavailable")


def test_similarity_degraded_metric_increment_on_degrade():
    """Faiss unavailable should increment degraded event metric."""
    from src.core.similarity import get_vector_store
    from src.utils.analysis_metrics import similarity_degraded_total

    degraded_counter = similarity_degraded_total.labels(event="degraded")
    _skip_if_metrics_disabled(degraded_counter)
    before = _read_metric_counter(degraded_counter)

    os.environ["VECTOR_STORE_BACKEND"] = "faiss"
    try:
        with patch("src.core.similarity.FaissVectorStore") as MockFaiss:
            inst = MagicMock()
            inst._available = False
            MockFaiss.return_value = inst
            get_vector_store("faiss")
        after = _read_metric_counter(similarity_degraded_total.labels(event="degraded"))
        assert after == before + 1
    finally:
        os.environ.pop("VECTOR_STORE_BACKEND", None)


def test_similarity_restored_metric_increment_on_recovery():
    """Faiss available after degradation should increment restored event metric."""
    from src.core.similarity import get_vector_store
    from src.utils.analysis_metrics import similarity_degraded_total

    # First trigger degradation
    os.environ["VECTOR_STORE_BACKEND"] = "faiss"
    try:
        with patch("src.core.similarity.FaissVectorStore") as MockFaiss:
            inst = MagicMock()
            inst._available = False
            MockFaiss.return_value = inst
            get_vector_store("faiss")

        degraded_counter = similarity_degraded_total.labels(event="degraded")
        restored_counter = similarity_degraded_total.labels(event="restored")
        _skip_if_metrics_disabled(degraded_counter)
        _skip_if_metrics_disabled(restored_counter)
        degraded_before_restore = _read_metric_counter(degraded_counter)
        restored_before = _read_metric_counter(restored_counter)

        # Now simulate successful availability (restoration)
        with patch("src.core.similarity.FaissVectorStore") as MockFaiss:
            inst2 = MagicMock()
            inst2._available = True
            MockFaiss.return_value = inst2
            get_vector_store("faiss")

        degraded_after_restore = _read_metric_counter(
            similarity_degraded_total.labels(event="degraded")
        )
        restored_after = _read_metric_counter(similarity_degraded_total.labels(event="restored"))

        # Degraded count unchanged; restored incremented by 1
        assert degraded_after_restore == degraded_before_restore
        assert restored_after == restored_before + 1
    finally:
        os.environ.pop("VECTOR_STORE_BACKEND", None)


def test_similarity_no_restore_without_previous_degrade():
    """If Faiss is available initially, restored metric should not increment."""
    from src.core.similarity import get_vector_store, reset_default_store
    from src.utils.analysis_metrics import similarity_degraded_total

    os.environ["VECTOR_STORE_BACKEND"] = "faiss"
    try:
        restored_counter = similarity_degraded_total.labels(event="restored")
        _skip_if_metrics_disabled(restored_counter)
        before_restored = _read_metric_counter(restored_counter)
        with patch("src.core.similarity.FaissVectorStore") as MockFaiss:
            inst = MagicMock()
            inst._available = True
            MockFaiss.return_value = inst
            get_vector_store("faiss")
        after_restored = _read_metric_counter(similarity_degraded_total.labels(event="restored"))
        assert after_restored == before_restored  # No change
    finally:
        os.environ.pop("VECTOR_STORE_BACKEND", None)
