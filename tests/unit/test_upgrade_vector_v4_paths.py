from src.core.feature_extractor import SLOTS_V1, SLOTS_V2, SLOTS_V3, SLOTS_V4, FeatureExtractor


def make_len(version: str) -> int:
    base = len(SLOTS_V1)
    if version in {"v2", "v3", "v4"}:
        base += len(SLOTS_V2)
    if version in {"v3", "v4"}:
        base += len(SLOTS_V3)
    if version == "v4":
        base += len(SLOTS_V4)
    return base


def test_upgrade_downgrade_cycles_v4():
    # Start with v4 zero vector
    v4_vec = [1.0] * make_len("v4")
    # Downgrade to v2 then back to v4
    fx_v2 = FeatureExtractor(feature_version="v2")
    v2_vec = fx_v2.upgrade_vector(v4_vec)
    assert len(v2_vec) == make_len("v2")
    fx_v4 = FeatureExtractor(feature_version="v4")
    v4_vec2 = fx_v4.upgrade_vector(v2_vec)
    assert len(v4_vec2) == make_len("v4")
    # New tail should be zero padded
    assert v4_vec2[-len(SLOTS_V4) :] == [0.0] * len(SLOTS_V4)


def test_upgrade_error_on_invalid_length():
    invalid = [0.0] * (make_len("v1") + 3)  # not matching any expected set
    fx = FeatureExtractor(feature_version="v4")
    import pytest

    with pytest.raises(ValueError):
        fx.upgrade_vector(invalid)
