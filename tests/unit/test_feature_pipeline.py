import asyncio
import hashlib

from src.core.feature_pipeline import run_feature_pipeline


class _DummyFeatureCache:
    def __init__(self, initial=None):
        self._store = dict(initial or {})

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def size(self):
        return len(self._store)


class _DummyGeometryCache:
    def __init__(self, cached=None):
        self.cached = cached
        self.last_key = None

    def generate_key(self, content, version):
        self.last_key = f"{version}:{len(content)}"
        return self.last_key

    def get(self, key):
        return self.cached

    def set(self, key, value):
        self.cached = value


class _DummyFeatureExtractor:
    def __init__(self):
        self.extract_calls = 0
        self.rehydrate_calls = 0

    async def extract(self, doc, brep_features=None):
        self.extract_calls += 1
        return {"geometric": [1.0, 2.0], "semantic": [3.0]}

    def flatten(self, features):
        return list(features["geometric"]) + list(features["semantic"])

    def rehydrate(self, vector, version="v1"):
        self.rehydrate_calls += 1
        return {"geometric": list(vector[:2]), "semantic": list(vector[2:])}

    def slots(self, version):
        return {"geometric": 2, "semantic": 1}


def test_run_feature_pipeline_uses_2d_cache_hit_and_rehydrates(monkeypatch):
    monkeypatch.setenv("FEATURE_VERSION", "v_test")
    cached_vector = [9.0, 8.0, 7.0]
    cache_key = f"{hashlib.sha256(b'abc').hexdigest()}:v_test:layout_v2"
    feature_cache = _DummyFeatureCache(
        {cache_key: cached_vector}
    )
    extractor = _DummyFeatureExtractor()

    result = asyncio.run(
        run_feature_pipeline(
            extract_features=True,
            file_format="dxf",
            file_name="sample.dxf",
            content=b"abc",
            doc=object(),
            started_at=0.0,
            stage_times={"parse": 0.2},
            feature_extractor_factory=lambda: extractor,
            feature_cache_factory=lambda: feature_cache,
        )
    )

    assert result["features"] == {"geometric": [9.0, 8.0], "semantic": [7.0]}
    assert result["features_3d"] == {}
    assert extractor.extract_calls == 0
    assert extractor.rehydrate_calls == 1
    assert result["results_patch"]["features"]["combined"] == cached_vector
    assert result["results_patch"]["features"]["cache_hit"] is True
    assert result["results_patch"]["features"]["feature_version"] == "v_test"
    assert result["features_stage_duration"] is not None


def test_run_feature_pipeline_extracts_3d_features_and_embedding_result():
    feature_cache = _DummyFeatureCache()
    geometry_cache = _DummyGeometryCache()
    extractor = _DummyFeatureExtractor()

    class _DummyGeometryEngine:
        def load_step(self, content, file_name=""):
            assert file_name == "sample.step"
            return object()

        def extract_brep_features(self, shape):
            return {"surface_area": 12.0}

        def extract_dfm_features(self, shape):
            return {"thin_walls_detected": False}

    class _DummyEncoder:
        def encode(self, features_3d):
            return [0.1, 0.2, 0.3]

    result = asyncio.run(
        run_feature_pipeline(
            extract_features=True,
            file_format="step",
            file_name="sample.step",
            content=b"step-content",
            doc=object(),
            started_at=0.0,
            stage_times={"parse": 0.1},
            feature_extractor_factory=lambda: extractor,
            feature_cache_factory=lambda: feature_cache,
            geometry_cache_factory=lambda: geometry_cache,
            geometry_engine_factory=lambda: _DummyGeometryEngine(),
            encoder_3d_factory=lambda: _DummyEncoder(),
        )
    )

    assert result["features_3d"]["surface_area"] == 12.0
    assert result["features_3d"]["thin_walls_detected"] is False
    assert result["features_3d"]["embedding_vector"] == [0.1, 0.2, 0.3]
    assert result["results_patch"]["features_3d"]["surface_area"] == 12.0
    assert result["results_patch"]["features_3d"]["embedding_dim"] == 3
    assert result["features_3d_stage_duration"] is not None
    assert result["results_patch"]["features"]["cache_hit"] is False
