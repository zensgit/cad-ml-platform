"""Unit tests for the manufacturing cost estimation module."""

from __future__ import annotations

import pytest

from src.ml.cost.estimator import CostEstimator
from src.ml.cost.models import CostEstimateRequest, CostEstimateResponse


@pytest.fixture()
def estimator() -> CostEstimator:
    """Return a CostEstimator loaded from the project's default config."""
    return CostEstimator(config_path="config/cost_model.yaml")


# ------------------------------------------------------------------
# Basic estimate sanity
# ------------------------------------------------------------------

def test_basic_steel_estimate(estimator: CostEstimator) -> None:
    """A steel part with known volume and entity count produces a positive
    total that equals the sum of its cost components."""
    req = CostEstimateRequest(
        material="steel",
        bounding_volume_mm3=1000.0,
        entity_count=10,
    )
    resp = estimator.estimate(req)

    assert resp.estimate.total > 0
    expected_total = (
        resp.estimate.material_cost
        + resp.estimate.machining_cost
        + resp.estimate.setup_cost
        + resp.estimate.overhead
    )
    assert resp.estimate.total == pytest.approx(expected_total, abs=0.02)
    assert resp.estimate.currency == "CNY"


# ------------------------------------------------------------------
# Batch-size effect on setup cost
# ------------------------------------------------------------------

def test_batch_size_effect(estimator: CostEstimator) -> None:
    """Setup cost per part should be roughly 100x lower for batch=100
    compared to batch=1 (same absolute setup cost amortised)."""
    base = CostEstimateRequest(
        material="steel",
        bounding_volume_mm3=5000.0,
        entity_count=20,
        batch_size=1,
    )
    batch = CostEstimateRequest(
        material="steel",
        bounding_volume_mm3=5000.0,
        entity_count=20,
        batch_size=100,
    )
    resp_single = estimator.estimate(base)
    resp_batch = estimator.estimate(batch)

    ratio = resp_single.estimate.setup_cost / max(resp_batch.estimate.setup_cost, 1e-9)
    assert 95 <= ratio <= 105  # allow small rounding drift


# ------------------------------------------------------------------
# Material price ordering
# ------------------------------------------------------------------

def test_material_price_ordering(estimator: CostEstimator) -> None:
    """Titanium > stainless_steel > steel > plastic_abs in material cost
    when all other parameters are held constant."""
    def mat_cost(material: str) -> float:
        req = CostEstimateRequest(
            material=material,
            bounding_volume_mm3=10000.0,
            entity_count=50,
        )
        return estimator.estimate(req).estimate.material_cost

    titanium = mat_cost("titanium")
    stainless = mat_cost("stainless_steel")
    steel = mat_cost("steel")
    plastic = mat_cost("plastic_abs")

    assert titanium > stainless > steel > plastic


# ------------------------------------------------------------------
# Tolerance grade effect
# ------------------------------------------------------------------

def test_tolerance_effect(estimator: CostEstimator) -> None:
    """Tighter tolerance grades (IT6) should cost more than looser ones (IT12)."""
    def total_for_grade(grade: str) -> float:
        req = CostEstimateRequest(
            material="steel",
            bounding_volume_mm3=5000.0,
            entity_count=30,
            tolerance_grade=grade,
        )
        return estimator.estimate(req).estimate.total

    it6 = total_for_grade("IT6")
    it8 = total_for_grade("IT8")
    it12 = total_for_grade("IT12")

    assert it6 > it8 > it12


# ------------------------------------------------------------------
# Missing volume still works
# ------------------------------------------------------------------

def test_missing_volume(estimator: CostEstimator) -> None:
    """When bounding_volume_mm3 is zero / default the estimator should
    still return a valid response (machining + setup dominate)."""
    req = CostEstimateRequest(material="aluminum", entity_count=5)
    resp = estimator.estimate(req)

    assert resp.estimate.total > 0
    assert resp.estimate.material_cost == 0.0  # no volume -> no material


# ------------------------------------------------------------------
# Confidence calculation
# ------------------------------------------------------------------

def test_confidence_calculation(estimator: CostEstimator) -> None:
    """Providing all optional fields should yield high confidence;
    providing almost nothing should yield low confidence."""
    full = CostEstimateRequest(
        material="steel",
        bounding_volume_mm3=10000.0,
        entity_count=100,
        complexity_score=0.6,
        tolerance_grade="IT7",
        surface_finish="Ra1.6",
    )
    minimal = CostEstimateRequest()

    resp_full = estimator.estimate(full)
    resp_min = estimator.estimate(minimal)

    assert resp_full.confidence > 0.8
    assert resp_min.confidence < 0.5


# ------------------------------------------------------------------
# Reasoning list
# ------------------------------------------------------------------

def test_reasoning_not_empty(estimator: CostEstimator) -> None:
    """Every estimate should include at least one reasoning string."""
    req = CostEstimateRequest(
        material="steel",
        bounding_volume_mm3=500.0,
        entity_count=3,
    )
    resp = estimator.estimate(req)
    assert len(resp.reasoning) > 0
    assert all(isinstance(r, str) for r in resp.reasoning)


# ------------------------------------------------------------------
# Optimistic / pessimistic ordering
# ------------------------------------------------------------------

def test_optimistic_pessimistic(estimator: CostEstimator) -> None:
    """Optimistic total < estimate total < pessimistic total."""
    req = CostEstimateRequest(
        material="aluminum",
        bounding_volume_mm3=8000.0,
        entity_count=40,
    )
    resp = estimator.estimate(req)

    assert resp.optimistic.total < resp.estimate.total
    assert resp.estimate.total < resp.pessimistic.total
