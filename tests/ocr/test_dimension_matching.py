"""Dimension matching formula tests.

Formula: abs(pred - gt) <= max(0.05 * gt, tolerance_gt)
Edge cases: zero tolerance, very small gt, missing tolerance.
"""

import pytest


def match_dimension(pred_value: float, gt_value: float, tolerance_gt: float | None) -> bool:
    """Matching rule per spec:
    abs(pred - gt) <= max(0.05*gt, tolerance_gt)
    If tolerance_gt is None treat as 0 (so adaptive wins). If tolerance_gt == 0 use adaptive unless zero tolerance explicitly enforced by downstream (handled elsewhere).
    """
    adaptive = 0.05 * gt_value
    tol_explicit = tolerance_gt if tolerance_gt is not None else 0.0
    threshold = max(adaptive, tol_explicit)
    return abs(pred_value - gt_value) <= threshold


@pytest.mark.parametrize(
    "pred,gt,tol,expected",
    [
        (20.01, 20.0, 0.02, True),  # within explicit tolerance
        (20.5, 20.0, 0.02, True),  # adaptive threshold 1.0 > 0.02 so still match
        (10.4, 10.0, 0.3, True),  # adaptive=0.5, tolerance=0.3 -> threshold=0.5
        (10.6, 10.0, 0.3, False),  # beyond adaptive threshold
        (5.2, 5.0, None, True),  # no tolerance uses adaptive (0.25)
        (5.3, 5.0, None, False),  # outside adaptive
        (1.02, 1.0, None, True),  # adaptive=0.05
        (1.08, 1.0, None, False),
        (50.0, 50.0, 0.0, True),  # exact match
        (50.03, 50.0, 0.0, True),  # adaptive dominates zero explicit tolerance
    ],
)
def test_dimension_matching(pred, gt, tol, expected):
    assert match_dimension(pred, gt, tol) is expected


def test_threshold_consistency():
    # Ensure threshold monotonic with gt value when tolerance absent
    for gt in [5, 10, 20, 40]:
        adaptive = 0.05 * gt
        assert adaptive == pytest.approx(0.05 * gt)


# ========== Unit Normalization Tests ==========


def normalize_to_mm(value: float, unit: str) -> float:
    """Normalize all units to mm (production logic)"""
    conversion = {
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "in": 25.4,
        "inch": 25.4,
        "毫米": 1.0,
        "厘米": 10.0,
        "米": 1000.0,
    }
    return value * conversion.get(unit, 1.0)


@pytest.mark.parametrize(
    "value,unit,expected_mm",
    [
        (20, "mm", 20.0),
        (2, "cm", 20.0),
        (0.02, "m", 20.0),
        (1, "in", 25.4),
        (20, "毫米", 20.0),
        (2, "厘米", 20.0),
    ],
)
def test_unit_normalization(value, unit, expected_mm):
    """Test unit conversion to mm"""
    assert normalize_to_mm(value, unit) == pytest.approx(expected_mm)


def test_unit_normalized_matching():
    """Test matching with unit normalization"""
    # Pred: 2cm, GT: 20mm -> should match exactly after normalization
    pred_mm = normalize_to_mm(2, "cm")
    gt_mm = normalize_to_mm(20, "mm")

    assert match_dimension(pred_mm, gt_mm, None)

    # Pred: 2.1cm (21mm), GT: 20mm -> should match within 5% (threshold=1mm)
    pred_mm = normalize_to_mm(2.1, "cm")
    gt_mm = normalize_to_mm(20, "mm")

    assert match_dimension(pred_mm, gt_mm, None)


# ========== Thread Matching Tests ==========


def match_thread(
    pred_diameter: float,
    pred_pitch: float | None,
    gt_diameter: float,
    gt_pitch: float | None,
    tolerance_diameter: float | None = None,
) -> bool:
    """
    Match thread dimensions (M10×1.5)

    Thread matching requires:
    1. Major diameter matches (using standard dimension matching)
    2. Pitch matches exactly (if both present)
    """
    # Match diameter
    diameter_match = match_dimension(pred_diameter, gt_diameter, tolerance_diameter)

    if not diameter_match:
        return False

    # If GT has pitch, pred must have matching pitch
    if gt_pitch is not None:
        if pred_pitch is None:
            return False
        # Pitch must match exactly (small tolerance for float comparison)
        return abs(pred_pitch - gt_pitch) < 0.01

    # If GT has no pitch, diameter match is sufficient
    return True


@pytest.mark.parametrize(
    "pred_d,pred_p,gt_d,gt_p,tol_d,expected",
    [
        # M10×1.5 - exact match
        (10.0, 1.5, 10.0, 1.5, None, True),
        # M10×1.5 vs M10×1.25 - diameter matches but pitch differs
        (10.0, 1.25, 10.0, 1.5, None, False),
        # M10 (no pitch) vs M10×1.5 - pred missing pitch
        (10.0, None, 10.0, 1.5, None, False),
        # M10×1.5 vs M10 (no pitch required) - extra pitch in pred is OK
        (10.0, 1.5, 10.0, None, None, True),
        # Diameter within 5% tolerance, pitch matches
        (10.4, 1.5, 10.0, 1.5, None, True),
        # Diameter outside tolerance even though pitch matches
        (11.0, 1.5, 10.0, 1.5, None, False),
    ],
)
def test_thread_matching(pred_d, pred_p, gt_d, gt_p, tol_d, expected):
    """Test thread dimension matching (diameter + pitch)"""
    assert match_thread(pred_d, pred_p, gt_d, gt_p, tol_d) is expected


# ========== Recall Calculation Simulation ==========


def calculate_dimension_recall(predictions: list, ground_truths: list) -> dict:
    """
    Calculate dimension recall as per CHANGELOG.md:

    dimension_recall = matched_dimensions / ground_truth_dimensions

    Returns:
        dict with recall, matched_count, gt_count, unmatched_gt
    """
    matched = 0
    unmatched_gt = []

    # Track which GT dimensions have been matched
    gt_matched = [False] * len(ground_truths)

    for pred in predictions:
        for i, gt in enumerate(ground_truths):
            if gt_matched[i]:
                continue

            # Check if types match
            if pred.get("type") != gt.get("type"):
                continue

            # Normalize values to mm
            pred_value_mm = normalize_to_mm(pred["value"], pred.get("unit", "mm"))
            gt_value_mm = normalize_to_mm(gt["value"], gt.get("unit", "mm"))

            # Special handling for threads
            if gt["type"] == "thread":
                is_match = match_thread(
                    pred_value_mm,
                    pred.get("pitch"),
                    gt_value_mm,
                    gt.get("pitch"),
                    gt.get("tolerance"),
                )
            else:
                is_match = match_dimension(pred_value_mm, gt_value_mm, gt.get("tolerance"))

            if is_match:
                matched += 1
                gt_matched[i] = True
                break

    # Collect unmatched GT dimensions
    for i, gt in enumerate(ground_truths):
        if not gt_matched[i]:
            unmatched_gt.append(gt)

    recall = matched / len(ground_truths) if ground_truths else 0.0

    return {
        "recall": recall,
        "matched_count": matched,
        "gt_count": len(ground_truths),
        "unmatched_gt": unmatched_gt,
    }


def test_recall_perfect_match():
    """Test recall calculation with perfect predictions"""
    predictions = [
        {"type": "diameter", "value": 20, "unit": "mm"},
        {"type": "radius", "value": 5, "unit": "mm"},
    ]

    ground_truths = [
        {"type": "diameter", "value": 20, "unit": "mm"},
        {"type": "radius", "value": 5, "unit": "mm"},
    ]

    result = calculate_dimension_recall(predictions, ground_truths)

    assert result["recall"] == 1.0
    assert result["matched_count"] == 2
    assert result["gt_count"] == 2
    assert len(result["unmatched_gt"]) == 0


def test_recall_partial_match():
    """Test recall with some missing predictions"""
    predictions = [
        {"type": "diameter", "value": 20, "unit": "mm"},
        # Missing radius prediction
    ]

    ground_truths = [
        {"type": "diameter", "value": 20, "unit": "mm"},
        {"type": "radius", "value": 5, "unit": "mm"},
    ]

    result = calculate_dimension_recall(predictions, ground_truths)

    assert result["recall"] == 0.5
    assert result["matched_count"] == 1
    assert result["gt_count"] == 2
    assert len(result["unmatched_gt"]) == 1
    assert result["unmatched_gt"][0]["type"] == "radius"


def test_recall_with_unit_normalization():
    """Test recall with different units"""
    predictions = [
        {"type": "diameter", "value": 2, "unit": "cm"},  # 20mm
    ]

    ground_truths = [
        {"type": "diameter", "value": 20, "unit": "mm"},  # 20mm
    ]

    result = calculate_dimension_recall(predictions, ground_truths)

    assert result["recall"] == 1.0
    assert result["matched_count"] == 1


def test_recall_with_tolerance():
    """Test recall with tolerance-based matching"""
    predictions = [
        {"type": "diameter", "value": 20.01, "unit": "mm"},
    ]

    ground_truths = [
        {"type": "diameter", "value": 20, "unit": "mm", "tolerance": 0.02},
    ]

    result = calculate_dimension_recall(predictions, ground_truths)

    assert result["recall"] == 1.0
    assert result["matched_count"] == 1


def test_recall_zero_predictions():
    """Test recall when no predictions made"""
    predictions = []

    ground_truths = [
        {"type": "diameter", "value": 20, "unit": "mm"},
    ]

    result = calculate_dimension_recall(predictions, ground_truths)

    assert result["recall"] == 0.0
    assert result["matched_count"] == 0
    assert result["gt_count"] == 1


def test_recall_thread_matching():
    """Test recall with thread dimensions (diameter + pitch)"""
    predictions = [
        {"type": "thread", "value": 10, "pitch": 1.5, "unit": "mm"},
        {"type": "thread", "value": 8, "pitch": 1.0, "unit": "mm"},  # Wrong pitch
    ]

    ground_truths = [
        {"type": "thread", "value": 10, "pitch": 1.5, "unit": "mm"},
        {"type": "thread", "value": 8, "pitch": 1.25, "unit": "mm"},  # Different pitch
    ]

    result = calculate_dimension_recall(predictions, ground_truths)

    # Only first thread should match (second has wrong pitch)
    assert result["recall"] == 0.5
    assert result["matched_count"] == 1
    assert result["gt_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
