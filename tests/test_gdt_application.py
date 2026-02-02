import pytest

from src.core.knowledge.gdt import (
    GDTCharacteristic,
    ToleranceModifier,
    interpret_feature_control_frame,
)


def test_interpret_feature_control_frame_chinese():
    fcf = interpret_feature_control_frame("位置度 0.2 M A B C")
    assert fcf is not None
    assert fcf.characteristic == GDTCharacteristic.POSITION
    assert fcf.tolerance_value == pytest.approx(0.2)
    assert fcf.tolerance_modifier == ToleranceModifier.MMC
    assert fcf.primary_datum == "A"
    assert fcf.secondary_datum == "B"
    assert fcf.tertiary_datum == "C"
    assert fcf.notes == []


def test_interpret_feature_control_frame_diameter_and_lmc():
    fcf = interpret_feature_control_frame("位置度 Ø0.1 L A")
    assert fcf is not None
    assert fcf.characteristic == GDTCharacteristic.POSITION
    assert fcf.tolerance_value == pytest.approx(0.1)
    assert fcf.tolerance_modifier == ToleranceModifier.LMC
    assert fcf.primary_datum == "A"


def test_interpret_feature_control_frame_unknown():
    assert interpret_feature_control_frame("UNKNOWN 0.1 A") is None
