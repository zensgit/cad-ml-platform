from src.core.feature_extractor import FeatureExtractor, SLOTS_V4, SLOTS_V3, SLOTS_V2, SLOTS_V1


class DummyBBox:
    def __init__(self):
        self.width = 10.0
        self.height = 5.0
        self.depth = 2.0
        self.volume_estimate = 100.0


class DummyEntity:
    def __init__(self, kind: str):
        self.kind = kind


class DummyDoc:
    def __init__(self, kinds):
        self.bounding_box = DummyBBox()
        self.entities = [DummyEntity(k) for k in kinds]
        self.layers = ["L1", "L2"]
        self.metadata = {"solids": 3, "facets": 7}

    def entity_count(self):  # v1 base slot expects this
        return len(self.entities)

    def complexity_bucket(self):  # semantic flag
        return "high" if len(self.entities) > 2 else "low"


def test_feature_extractor_v4_basic():
    doc = DummyDoc(["SOLID", "SOLID", "FACE", "EDGE", "EDGE", "FACE"])  # diverse kinds
    fx = FeatureExtractor(feature_version="v4")
    data = __import__("asyncio").run(fx.extract(doc))
    geometric = data["geometric"]
    semantic = data["semantic"]
    # lengths: v1(7)+v2(5)+v3(11)+v4(2)=25
    assert len(geometric) == 7 + 5 + 11 + 2
    assert len(semantic) == 2
    surface_count = geometric[-2]
    shape_entropy = geometric[-1]
    assert surface_count == 3 + 7  # solids + facets placeholder
    assert 0.0 <= shape_entropy <= 1.0


def test_feature_extractor_v4_entropy_zero_for_single_kind():
    doc = DummyDoc(["SOLID", "SOLID", "SOLID"])  # single kind only
    fx = FeatureExtractor(feature_version="v4")
    data = __import__("asyncio").run(fx.extract(doc))
    geometric = data["geometric"]
    shape_entropy = geometric[-1]
    assert shape_entropy == 0.0


def test_upgrade_vector_to_v4_from_v1():
    from_v1_len = len(SLOTS_V1)  # 7
    base_v1 = [0.1] * from_v1_len
    fx = FeatureExtractor(feature_version="v4")
    upgraded = fx.upgrade_vector(base_v1)
    assert len(upgraded) == len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3) + len(SLOTS_V4)
    # padded sections should be zeros after original length
    assert upgraded[from_v1_len:] == [0.0] * (len(upgraded) - from_v1_len)


def test_upgrade_vector_to_v4_from_v3_partial():
    # Construct a fake v3 vector with correct length
    v3_len = len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3)
    v3_vec = [0.2] * v3_len
    fx = FeatureExtractor(feature_version="v4")
    upgraded = fx.upgrade_vector(v3_vec)
    assert len(upgraded) == v3_len + len(SLOTS_V4)
    assert upgraded[-2:] == [0.0, 0.0]


def test_rehydrate_v4_vector():
    total_len = len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3) + len(SLOTS_V4)
    vec = [0.3] * total_len
    fx = FeatureExtractor(feature_version="v4")
    rehydrated = fx.rehydrate(vec, "v4")
    assert len(rehydrated["geometric"]) == total_len
    assert len(rehydrated["semantic"]) == 2

