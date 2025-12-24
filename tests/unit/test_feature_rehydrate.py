from src.core.feature_extractor import FeatureExtractor


def test_rehydrate_v1():
    extractor = FeatureExtractor()
    # mock vector: 5 geometric + 2 semantic
    combined = [1, 2, 3, 4, 5, 10, 0]
    reh = extractor.rehydrate(combined, version="v1")
    assert len(reh["geometric"]) == 5
    assert len(reh["semantic"]) == 2


def test_rehydrate_v2():
    extractor = FeatureExtractor()
    # v2 adds 5 geometric extension slots
    combined = [1, 2, 3, 4, 5, 10, 0, 0.1, 0.2, 0.3, 1.5, 2.5]
    reh = extractor.rehydrate(combined, version="v2")
    assert len(reh["geometric"]) == 5 + 5
    assert len(reh["semantic"]) == 2


def test_rehydrate_v3():
    extractor = FeatureExtractor()
    # v3 adds v2 + 10 enrichment slots
    combined = [
        1,
        2,
        3,
        4,
        5,
        10,
        0,
        0.1,
        0.2,
        0.3,
        1.5,
        2.5,
        5,
        6,
        7,
        0.1,
        0.2,
        0.3,
        0.4,
        0.0,
        0.0,
        0.0,
    ]
    reh = extractor.rehydrate(combined, version="v3")
    assert len(reh["geometric"]) == 5 + 5 + 10
    assert len(reh["semantic"]) == 2
