from src.core.feature_extractor import FeatureExtractor


def test_flatten_rehydrate_round_trip_v3():
    extractor = FeatureExtractor(feature_version="v3")
    geometric = [
        1.0, 2.0, 3.0, 4.0, 5.0,  # base geometric
        10.0, 11.0, 12.0, 13.0, 14.0,  # v2 extension
        20.0, 21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0, 29.0,  # v3 extension (10)
    ]
    semantic = [99.0, 0.0]
    combined = extractor.flatten({"geometric": geometric, "semantic": semantic})
    rehydrated = extractor.rehydrate(combined, "v3")
    assert rehydrated["geometric"] == geometric
    assert rehydrated["semantic"] == semantic


def test_reorder_legacy_vector_v2():
    extractor = FeatureExtractor()
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    ext = [6.0, 7.0, 8.0, 9.0, 10.0]
    semantic = [99.0, 0.0]
    legacy = base + ext + semantic
    reordered = extractor.reorder_legacy_vector(legacy, "v2")
    assert reordered == base + semantic + ext


def test_reorder_legacy_vector_invalid_length():
    extractor = FeatureExtractor()
    legacy = [0.0] * 9
    import pytest
    with pytest.raises(ValueError):
        extractor.reorder_legacy_vector(legacy, "v1")
