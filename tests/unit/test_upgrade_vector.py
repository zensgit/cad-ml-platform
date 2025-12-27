from fastapi.testclient import TestClient

from src.core.feature_extractor import SLOTS_V2, SLOTS_V3, FeatureExtractor
from src.main import app


def test_upgrade_v1_to_v2_padding():
    # v1 vector length = 5 geometric + 2 semantic
    v1 = list(range(7))
    fx = FeatureExtractor(feature_version="v2")
    upgraded = fx.upgrade_vector(v1)
    assert len(upgraded) == 7 + len(SLOTS_V2)
    # Padding zeros for new slots
    assert upgraded[7:] == [0.0] * len(SLOTS_V2)


def test_upgrade_v1_to_v3_padding():
    v1 = list(range(7))
    fx = FeatureExtractor(feature_version="v3")
    upgraded = fx.upgrade_vector(v1)
    assert len(upgraded) == 7 + len(SLOTS_V2) + len(SLOTS_V3)
    assert upgraded[7 : 7 + len(SLOTS_V2)] == [0.0] * len(SLOTS_V2)
    assert upgraded[7 + len(SLOTS_V2) :] == [0.0] * len(SLOTS_V3)


def test_upgrade_v2_to_v3_padding():
    # v2 length = 7 + len(SLOTS_V2)
    v2 = list(range(7 + len(SLOTS_V2)))
    fx = FeatureExtractor(feature_version="v3")
    upgraded = fx.upgrade_vector(v2)
    assert len(upgraded) == 7 + len(SLOTS_V2) + len(SLOTS_V3)
    assert upgraded[-len(SLOTS_V3) :] == [0.0] * len(SLOTS_V3)


def test_upgrade_v3_to_v2_truncate():
    v3 = list(range(7 + len(SLOTS_V2) + len(SLOTS_V3)))
    fx = FeatureExtractor(feature_version="v2")
    downgraded = fx.upgrade_vector(v3)
    assert len(downgraded) == 7 + len(SLOTS_V2)
    # Last element of downgraded should equal original position before v3 extension
    assert downgraded[-1] == v3[7 + len(SLOTS_V2) - 1]
